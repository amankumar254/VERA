import os
import requests

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "279cac01758ff057128d7f4c30f86aa2750b0bbf59ea53287298dbddc7681a75")

def serpapi_reverse_image_search(image_path):
    """Reverse image search using SerpAPI."""
    
    if not SERP_API_KEY:
        return {
            "error": "Missing SERPAPI Key",
            "matches": [],
            "articles": []
        }

    try:
        files = {
            "file": open(image_path, "rb")
        }
        params = {
            "engine": "google_reverse_image",
            "api_key": SERP_API_KEY
        }

        res = requests.post("https://serpapi.com/search", params=params, files=files)
        data = res.json()

        # Extract visually similar images + used-in-pages
        matches = data.get("image_results", [])
        articles = data.get("news_results", []) or data.get("pages_with_matching_images", [])

        # Normalize result
        normalized = {
            "matches": matches,
            "articles": [
                {
                    "title": a.get("title", "No title"),
                    "source": a.get("source", "Unknown"),
                    "url": a.get("link") or a.get("url")
                }
                for a in articles
            ]
        }
        return normalized

    except Exception as e:
        return {
            "error": str(e),
            "matches": [],
            "articles": []
        }


def evaluate_fake_image(results):
    """Generate a clean evaluation result."""
    
    if results.get("error"):
        return {
            "status": "error",
            "reason": results["error"],
            "articles": [],
            "match_count": 0
        }

    articles = results.get("articles", [])
    status = "Verified (No suspicious use found)" if len(articles) == 0 else "Warning (Image reused)"
    
    return {
        "status": status,
        "reason": "Based on pattern of reuse in online news.",
        "articles": articles,
        "match_count": len(articles)
    }
