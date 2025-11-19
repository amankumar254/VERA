# hybrid_recommender.py
"""
Simple Google News RSS + TF-IDF + Cosine Similarity Recommender
(Replacement for the previous hybrid recommender)
"""

import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_google_news_rss(query: str):
    """Fetch Google News RSS feed for a search query"""
    q = "+".join(query.split())
    url = f"https://news.google.com/rss/search?q={q}"
    return feedparser.parse(url)


def extract_news(feed):
    """Extract title, summary, link from Google News RSS entries"""
    articles = []
    for item in feed.entries:
        title = item.get("title", "")
        summary = item.get("summary", "")
        link = item.get("link", "")

        articles.append({
            "title": title,
            "summary": summary,
            "link": link
        })
    return articles


def recommend_news(query: str, top_n: int = 10):
    """Rank articles with TF-IDF cosine similarity and return top N."""

    # Fetch RSS data
    feed = get_google_news_rss(query)
    articles = extract_news(feed)

    if not articles:
        return []

    # Prepare documents
    docs = [a["title"] + " " + a["summary"] for a in articles]

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(docs + [query])

    query_vec = tfidf[-1]
    article_vecs = tfidf[:-1]

    scores = cosine_similarity(article_vecs, query_vec.reshape(1, -1)).flatten()

    # Sort by highest similarity
    ranked = np.argsort(scores)[::-1]

    results = []
    for idx in ranked[:top_n]:
        a = articles[idx]
        results.append({
            "title": a["title"],
            "summary": a["summary"],
            "link": a["link"],
            "score": float(scores[idx])
        })

    return results
