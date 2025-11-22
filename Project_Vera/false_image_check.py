# false_image_check.py  
import requests
import os

SERPAPI_KEY = "279cac01758ff057128d7f4c30f86aa2750b0bbf59ea53287298dbddc7681a75"   # <-- your SerpAPI key


# ------------------------------
# 1. Upload image to file.io
# ------------------------------
def upload_image(image_path):
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            url = "https://file.io/?expires=1d"
            r = requests.post(url, files=files)

        if r.status_code == 200 and r.json().get("success"):
            return r.json().get("link")

        return None
    except Exception:
        return None


# ------------------------------
# 2. Reverse Image Search (SerpAPI)
# ------------------------------
def serpapi_reverse_image_search(image_url):
    if not SERPAPI_KEY:
        return {"error": "Missing SERPAPI key", "matches": [], "articles": []}

    try:
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": SERPAPI_KEY,
        }

        r = requests.get("https://serpapi.com/search", params=params)
        if r.status_code != 200:
            return {"error": "API Error", "matches": [], "articles": []}

        data = r.json()

        # Extract results
        matches = data.get("visually_similar_images", [])
        news_results = data.get("news_results", [])
        page_results = data.get("pages_with_matching_images", [])

        articles = []
        for a in (news_results or page_results):
            articles.append({
                "title": a.get("title"),
                "source": a.get("source"),
                "url": a.get("link") or a.get("url")
            })

        return {
            "matches": matches,
            "articles": articles,
        }

    except Exception as e:
        return {"error": str(e), "matches": [], "articles": []}


# ------------------------------
# 3. Main wrapper for Flask route
# ------------------------------
def check_false_image(local_image_path):
    """Used in Flask. Uploads → Reverse search → Returns structured output."""
    print("\n[DEBUG] Received file:", local_image_path)

    # Upload image
    uploaded_url = upload_image(local_image_path)
    print("[DEBUG] Uploaded URL:", uploaded_url)

    if not uploaded_url:
        return {
            "status": "error",
            "reason": "Failed to upload image.",
            "matches": 0,
            "articles": []
        }

    # Reverse search
    results = serpapi_reverse_image_search(uploaded_url)
    print("[DEBUG] SerpAPI Results:", results)

    matches = results.get("matches", [])
    articles = results.get("articles", [])

    status = "Verified" if matches else "Unknown"

    return {
        "status": status,
        "reason": results.get("error", "-"),
        "matches": len(matches),
        "articles": articles
    }

def evaluate_fake_image(results):
    """Very simple evaluator until stronger logic is added."""
    matches = results.get("matches", [])
    articles = results.get("articles", [])

    if len(matches) == 0 and len(articles) == 0:
        return {"status": "Unknown", "reason": "-", "matches": 0}

    return {
        "status": "Found",
        "reason": "Matches detected",
        "matches": len(matches),
    }