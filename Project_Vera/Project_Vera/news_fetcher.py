# news_fetcher.py

from newsapi import NewsApiClient

API_KEY = "443db7d311184d94a11b6c42bf5376a2"

try:
    newsapi = NewsApiClient(api_key=API_KEY)
except:
    newsapi = None


def get_news(category, country="us", pagesize=50):
    if not newsapi:
        return [{
            "title": "News API Not Initialised",
            "url": "#",
            "source": "System",
            "description": "Check your API key.",
            "image_url": None,
            "category": category
        }]

    mapping = {
        "business": "business",
        "entertainment": "entertainment",
        "health": "health",
        "science": "science",
        "sports": "sports",
        "technology": "technology",
        "world": "general",
        "lifestyle": "general",
        "general": "general",
        "home": "general"
    }

    api_category = mapping.get(category.lower(), "general")

    try:
        data = newsapi.get_top_headlines(
            category=api_category,
            language="en",
            country=country,
            page_size=pagesize
        )

        articles = []
        for a in data.get("articles", []):
            img = a.get("urlToImage")

            if img and img.startswith("http://"):
                img = img.replace("http://", "https://")

            articles.append({
                "title": a.get("title", "No Title"),
                "url": a.get("url", "#"),
                "source": a.get("source", {}).get("name", "Unknown"),
                "description": a.get("description") or a.get("content") or "",
                "image_url": img,
                "category": category.lower()
            })

        return articles

    except Exception as e:
        return [{
            "title": f"Error loading {category} news",
            "url": "#",
            "source": "Error",
            "description": str(e),
            "image_url": None,
            "category": category
        }]
