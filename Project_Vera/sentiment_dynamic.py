# sentiment_dynamic.py — Live-news powered dynamic sentiment analysis

import os
import re
import time
import traceback
from typing import Dict, Any, List

import numpy as np
import requests
from bs4 import BeautifulSoup
import feedparser

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Weak Lexicon Models
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TEXTBLOB_AVAILABLE = False


# -----------------------------
# URL → extract article text
# -----------------------------
def extract_text_from_url(url: str) -> str:
    url = url.strip()
    if not url:
        return ""

    # Try newspaper3k
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        if len(art.text) > 80:
            return art.text
    except:
        pass

    # Fallback: BeautifulSoup
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        ps = soup.find_all("p")
        return " ".join([p.get_text().strip() for p in ps])
    except:
        return ""


# -----------------------------
# Fetch related articles
# -----------------------------
def fetch_related_articles(query: str, max_items: int = 40):
    q = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={q}"

    feed = feedparser.parse(url)
    articles = []

    for item in feed.entries[:max_items]:
        articles.append({
            "title": item.get("title", ""),
            "description": item.get("summary", ""),
            "url": item.get("link", "")
        })

    return articles


# -----------------------------
# Clean text
# -----------------------------
def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# -----------------------------
# Weak Sentiment Label
# -----------------------------
def weak_label_text(text: str) -> int:
    text = text or ""

    # Improved thresholds (stronger detection)
    if VADER_AVAILABLE:
        s = vader.polarity_scores(text)
        c = s["compound"]
        if c >= 0.05: return 1
        if c <= -0.05: return -1
        return 0

    if TEXTBLOB_AVAILABLE:
        p = TextBlob(text).sentiment.polarity
        if p > 0.05: return 1
        if p < -0.05: return -1
        return 0

    # Keyword fallback
    low = text.lower()
    pos = ["good","great","gain","win","rise","profit","up","improve","boost"]
    neg = ["loss","drop","fall","weak","accident","crash","down","bad"]
    if any(w in low for w in pos): return 1
    if any(w in low for w in neg): return -1
    return 0


# -----------------------------
# Main dynamic sentiment engine
# -----------------------------
def analyze_sentiment_dynamic(input_text_or_url: str, fetch_count: int = 30) -> Dict[str, Any]:
    try:
        start = time.time()
        original = input_text_or_url.strip()

        # URL → extract text
        if original.startswith(("http://", "https://")):
            extracted = extract_text_from_url(original)
            text = extracted if len(extracted) > 80 else original
        else:
            text = original

        # Build search query
        tokens = re.findall(r"\w+", text.lower())
        query = " ".join(tokens[:10]) if tokens else text

        # Fetch related news
        articles = fetch_related_articles(query, max_items=fetch_count)

        related_texts = []
        for a in articles:
            combined = clean_text(a.get("title", "") + " " + a.get("description", ""))
            if combined:
                related_texts.append(combined)

        if len(related_texts) < 5:
            return {
                "sentiment": None,
                "emoji": None,
                "confidence": 0,
                "method": "fallback",
                "notes": "Not enough related articles"
            }

        corpus = related_texts + [clean_text(text)]
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=8000)
        X = vec.fit_transform(corpus)

        labels = np.array([weak_label_text(t) for t in related_texts])
        pos = (labels == 1).sum()
        neg = (labels == -1).sum()

        # -------------------------
        # 1) Naive Bayes (preferred)
        # -------------------------
        if (pos + neg) >= 3:    # lower threshold = better accuracy
            clf = MultinomialNB()
            clf.fit(X[:len(related_texts)], labels)

            prob = clf.predict_proba(X[-1])[0]
            idx = prob.argmax()
            pred_label = int(clf.classes_[idx])
            confidence = float(prob[idx])
            method = "dynamic_nb"

        else:
            # -------------------------
            # 2) KMeans fallback
            # -------------------------
            km = KMeans(n_clusters=3, random_state=42)
            km.fit(X)

            sims = cosine_similarity(X[-1], km.cluster_centers_).flatten()
            cluster = sims.argmax()

            members = np.where(km.labels_[:len(related_texts)] == cluster)[0]
            cluster_score = np.mean([weak_label_text(related_texts[i]) for i in members])

            if cluster_score > 0.1: pred_label = 1
            elif cluster_score < -0.1: pred_label = -1
            else: pred_label = 0

            confidence = float(sims.max())
            method = "dynamic_kmeans"

        # -------------------------
        # Convert label to text
        # -------------------------
        if pred_label == 1:
            return {"sentiment": "Positive", "emoji": "😊",
                    "confidence": confidence, "method": method}

        if pred_label == -1:
            return {"sentiment": "Negative", "emoji": "😡",
                    "confidence": confidence, "method": method}

        return {"sentiment": "Neutral", "emoji": "😐",
                "confidence": confidence, "method": method}

    except Exception as e:
        traceback.print_exc()
        return {
            "sentiment": None,
            "emoji": None,
            "confidence": 0,
            "method": "error",
            "notes": str(e)
        }
