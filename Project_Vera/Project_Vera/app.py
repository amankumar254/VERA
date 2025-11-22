# app.py — Project Vera backend (integrated FalseNewsChecker + cleaner routes)
import os
import logging
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from false_image_check import serpapi_reverse_image_search, evaluate_fake_image
from flask_cors import CORS



# Third-party utils used by routes
import requests
from bs4 import BeautifulSoup
from newspaper import Article

# -----------------------
# CONFIG / KEYS
# -----------------------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

# -----------------------
# IMPORT PROJECT MODULES (best-effort)
# -----------------------
try:
    from news_fetcher import get_news
except Exception:
    get_news = None

try:
    from hybrid_recommender import generate_dataset, recommend, DEFAULT_CSV
except Exception:
    generate_dataset = None
    recommend = None
    DEFAULT_CSV = "hybrid_news_dataset.csv"

try:
    from news_verifier import analyze_news
except Exception:
    analyze_news = None

# optional image reverse search wrapper
try:
    from false_image_check import serpapi_reverse_image_search, evaluate_fake_image
except Exception:
    serpapi_reverse_image_search = None
    evaluate_fake_image = None

# sentiment modules
try:
    from sentiment_dynamic import analyze_sentiment_dynamic
except Exception:
    analyze_sentiment_dynamic = None

try:
    from sentiment import predict_sentiment as fallback_predict_sentiment
except Exception:
    fallback_predict_sentiment = None

# False news checker (the new realistic implementation)
try:
    from falseNewsChecker import FalseNewsChecker
except Exception:
    FalseNewsChecker = None

# story clusterer
try:
    from story_clusterer_semantic import cluster_articles
except Exception:
    cluster_articles = None

# -----------------------
# FLASK SETUP
# -----------------------
app = Flask(__name__, static_folder=".", template_folder=".")
CORS(app)
logging.basicConfig(level=logging.INFO)

# instantiate false-news checker once (if available)
checker = None
if FalseNewsChecker is not None:
    try:
        checker = FalseNewsChecker()
    except Exception:
        logging.exception("Failed to instantiate FalseNewsChecker")
        checker = None

# -----------------------
# CONSTANTS
# -----------------------
CATEGORIES = [
    "Home", "Business", "Sports", "Science", "Entertainment",
    "World", "Health", "Lifestyle", "Technology"
]

# -----------------------
# HELPERS
# -----------------------
def safe_get_news(category_name: str):
    if get_news is None:
        logging.warning("news_fetcher.get_news not available.")
        return []
    try:
        return get_news(category_name)
    except Exception:
        logging.exception("Error in get_news() for category %s", category_name)
        return []

def extract_article_text(url: str):
    if not url or not isinstance(url, str):
        return None
    try:
        art = Article(url, language="en")
        art.download()
        art.parse()
        text = art.text.strip() if art.text else ""
        if len(text) > 50:
            return text
    except Exception:
        logging.debug("Newspaper3k extraction failed for URL: %s", url)
    # fallback with requests+bs4
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=12, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        extracted = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs if p.get_text(strip=True)])
        if len(extracted) > 50:
            return extracted
    except Exception:
        logging.debug("BeautifulSoup extraction failed for URL: %s", url)
    return None

# -----------------------
# PAGES
# -----------------------
@app.route("/")
def index():
    try:
        articles = safe_get_news("general")
    except Exception:
        articles = []
    current_date = datetime.now().strftime("%B %d, %Y")
    return render_template("index.html", articles=articles, current_category="Home", categories=CATEGORIES, current_date=current_date)

@app.route("/category/<category>")
def category_page(category):
    cat_clean = category.replace("-", " ").title()
    map_lookup = {
        "business": "business", "sports": "sports", "science": "science",
        "entertainment": "entertainment", "world": "general", "health": "health",
        "lifestyle": "general", "home": "general", "technology": "technology"
    }
    news_cat = map_lookup.get(category.lower(), "general")
    articles = safe_get_news(news_cat)
    current_date = datetime.now().strftime("%B %d, %Y")
    return render_template("index.html", articles=articles, current_category=cat_clean, categories=CATEGORIES, current_date=current_date)

# -----------------------
# API: SUMMARY
# -----------------------
# summary.smart_summarize expected in summary.py
try:
    from summary import smart_summarize
except Exception:
    smart_summarize = None

@app.route("/api/summary", methods=["POST"])
def api_summary():
    try:
        if smart_summarize is None:
            return jsonify({"error": "Summary module not available."}), 500
        data = request.get_json(force=True)
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"error": "No URL provided"}), 400
        final_summary = smart_summarize(url)
        if final_summary.startswith("❌"):
            return jsonify({"error": final_summary}), 400
        return jsonify({"summary": final_summary, "method": "smart_pipeline"})
    except Exception as e:
        logging.exception("api_summary error")
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

# -----------------------
# API: RECOMMENDATION
# -----------------------
from flask import request, jsonify
from hybrid_recommender import recommend_news


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"status": "error", "message": "Query cannot be empty"}), 400

        results = recommend_news(query)

        if not results:
            return jsonify({
                "status": "ok",
                "recs": [],
                "message": "No matching news found"
            })

        return jsonify({
            "status": "ok",
            "recs": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# -------------------------
# SENTIMENT ANALYZER API
# -------------------------
@app.route("/api/sentiment", methods=["POST"])
def api_sentiment():
    try:
        data = request.get_json()
        text = (data.get("text") or "").strip()

        if not text:
            return jsonify({"status": "error", "message": "No text provided"}), 400

        from sentiment import analyze_sentiment
        result = analyze_sentiment(text)

        return jsonify({
            "status": "ok",
            "sentiment": result["sentiment"],
            "emoji": result["emoji"],
            "confidence": result["confidence"],
            "method": result["method"],
            "notes": result["notes"]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------
# API: News verifier wrapper
# -----------------------
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if analyze_news is None:
        return jsonify({"status": "error", "message": "News verifier module not available."}), 500
    try:
        data = request.get_json(force=True)
        user_input = (data.get("input") or "").strip()
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
        result = analyze_news(user_input)
        return jsonify({"status": "ok", "result": result})
    except Exception:
        logging.exception("api_analyze error")
        return jsonify({"status": "error", "message": "Analyze failed."}), 500

# -----------------------
# API: Image check (SerpAPI wrapper)
# -----------------------
@app.route("/api/image_check", methods=["POST"])
def api_image_check():
    try:
        if serpapi_reverse_image_search is None or evaluate_fake_image is None:
            print("\n❌ ERROR: reverse search module missing\n")
            return jsonify({"status": "error", "message": "Image search module missing"}), 500

        # If no file uploaded
        if "image" not in request.files:
            print("\n❌ ERROR: No image uploaded\n")
            return jsonify({"status": "error", "message": "No image uploaded"}), 400

        f = request.files["image"]

        # TEMP PATH FIX FOR WINDOWS
        tmp_dir = os.path.join(os.getcwd(), "tmp_images")
        os.makedirs(tmp_dir, exist_ok=True)

        tmp_path = os.path.join(tmp_dir, f.filename)
        f.save(tmp_path)

        print("\n📸 Saved image to:", tmp_path)

        # Reverse Search
        try:
            print("\n🔍 Running reverse search…")
            results = serpapi_reverse_image_search(tmp_path)
        except Exception as e:
            print("\n===== REVERSE SEARCH ERROR =====")
            traceback.print_exc()
            print("=================================\n")
            try: os.remove(tmp_path)
            except: pass
            return jsonify({"status": "error", "message": f"Reverse search failed: {str(e)}"}), 500

        # Evaluation
        try:
            print("\n🧪 Evaluating result…")
            evaluation = evaluate_fake_image(results)
        except Exception as e:
            print("\n===== EVALUATION ERROR =====")
            traceback.print_exc()
            print("================================\n")
            evaluation = {"result": "Unknown"}

        try:
            os.remove(tmp_path)
        except:
            pass

        return jsonify({
            "status": "ok",
            "results": results,
            "evaluation": evaluation
        })

    except Exception as e:
        print("\n===== FINAL API ERROR =====")
        traceback.print_exc()
        print("================================\n")
        return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------
# API: Story clustering
# -----------------------
@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    try:
        if cluster_articles is None:
            return jsonify({"error": "Story clusterer not available."}), 500
        data = request.get_json(force=True)
        articles = data.get("articles", [])
        if len(articles) < 2:
            return jsonify({"error": "At least 2 articles required"}), 400
        html = cluster_articles(articles)
        return jsonify({"html": html})
    except Exception as e:
        logging.exception("api_cluster error")
        return jsonify({"error": str(e)}), 500

# -----------------------
# API: Article Comparison AI
# -----------------------
from article_compare import analyze_articles

@app.route("/api/compare_articles", methods=["POST"])
def compare_articles_api():
    try:
        data = request.get_json(force=True)
        urls = data.get("urls", [])

        if not urls or len(urls) < 2:
            return jsonify({"error": "Provide at least two article URLs"}), 400

        result = analyze_articles(urls)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ------------------------------------------
# Fetch News API Latest News
# ------------------------------------------

@app.route("/api/latest_news", methods=["GET"])
def api_latest_news():
    try:
        category = request.args.get("category", "general")
        articles = safe_get_news(category)
        return jsonify({"articles": articles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ------------------------------------------
# HELPER: Fetch Related Verified Articles
# ------------------------------------------
def get_related_articles(query):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 5,
            "apiKey": NEWSAPI_KEY
        }

        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        related = []
        for a in data.get("articles", []):
            related.append({
                "title": a.get("title"),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name")
            })

        return related
    except Exception:
        return []





# -----------------------
# API: FALSE NEWS (URL or RAW TEXT)
# -----------------------

# Load the upgraded checker
from falseNewsChecker import FalseNewsChecker
checker = FalseNewsChecker()

# -----------------------
# API: FALSE NEWS (URL or TEXT)
# -----------------------
@app.route("/api/false_news", methods=["POST"])
def api_false_news():
    try:
        url = request.form.get("url", "").strip()
        text = request.form.get("text", "").strip()

        # choose text first, otherwise url
        input_val = text if text else url

        if not input_val:
            return jsonify({"error": "Provide URL or text"}), 400

        # NEW check signature → only one argument
        result = checker.check(input_val)

        return jsonify({"status": "ok", "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================
# FALSE IMAGE CHECK API
# ===========================
from false_image_check import check_false_image
import os

@app.route("/api/false_image", methods=["POST"])
def api_false_image():
    print("\n[DEBUG] /api/false_image HIT")

    try:
        # -----------------------------
        # 1. Check if image file exists
        # -----------------------------
        if "image" in request.files:
            img = request.files["image"]
            print("[DEBUG] File received:", img.filename)

            temp_path = os.path.join("uploads", img.filename)
            img.save(temp_path)
            print("[DEBUG] Saved to:", temp_path)

            from false_image_check import check_false_image
            result = check_false_image(temp_path)

            print("[DEBUG] RESULT:", result)

            os.remove(temp_path)
            return jsonify({"status": "ok", "result": result})

        # -----------------------------
        # 2. Otherwise check URL input
        # -----------------------------
        url = request.form.get("url", "").strip()
        print("[DEBUG] URL received:", url)

        if url:
            temp_path = "temp_download.jpg"
            r = requests.get(url, timeout=5)
            with open(temp_path, "wb") as f:
                f.write(r.content)

            from false_image_check import check_false_image
            result = check_false_image(temp_path)

            print("[DEBUG] RESULT:", result)

            os.remove(temp_path)
            return jsonify({"status": "ok", "result": result})

        # -----------------------------
        # 3. No input
        # -----------------------------
        print("[DEBUG] No file or URL")
        return jsonify({"status": "error", "message": "No file or URL provided"}), 400

    except Exception as e:
        print("[ERROR] Exception:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500




# -----------------------
# STATIC FILES (optional)
# -----------------------
@app.route("/<path:filename>")
def serve_static(filename):
    if os.path.exists(filename):
        return send_from_directory(".", filename)
    return "Not found", 404

# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    logging.info("Starting Project Vera backend (app.py)")
    app.run(host="0.0.0.0", port=5000, debug=True)
