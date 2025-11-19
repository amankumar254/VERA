from newspaper import Article
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import re

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

def extract_text(url_or_text):
    url_or_text = url_or_text.strip()

    # --- It's a URL ---
    if url_or_text.startswith("http"):
        try:
            art = Article(url_or_text, language="en")
            art.download()
            art.parse()
            if len(art.text) > 100:
                return art.text
        except:
            pass

        # fallback BS4
        try:
            r = requests.get(url_or_text, headers={"User-Agent": UA}, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            text = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
            if len(text) > 100:
                return text
        except:
            pass

    # --- Raw text input ---
    return url_or_text


def cluster_articles(article_inputs):
    # 1. Extract article text for each input (URL or raw text)
    articles = []
    raw_map = []

    for a in article_inputs:
        text = extract_text(a)
        articles.append(text)
        raw_map.append(a)

    # 2. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(articles)

    # 3. DBSCAN clustering
    model = DBSCAN(eps=0.45, min_samples=1, metric='cosine')
    labels = model.fit_predict(X)

    # 4. Group results
    clusters = {}
    for label, original_input in zip(labels, raw_map):
        clusters.setdefault(label, []).append(original_input)

    # 5. Return HTML for your Glass UI frontend
    html = ""

    for label, items in clusters.items():
        cname = f"Cluster {label+1}" if label != -1 else "Unique/Noise"
        html += f"<div class='cluster-box'><h3>{cname} ({len(items)} items)</h3><ul>"
        for it in items:
            html += f"<li>{it}</li>"
        html += "</ul></div>"

    return html
