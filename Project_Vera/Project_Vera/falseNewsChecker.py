# falseNewsChecker.py
import re
import numpy as np
import pandas as pd
from GoogleNews import GoogleNews
from newspaper import Article
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

class FalseNewsChecker:

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------------------------------------------------------
    # SAFE ARTICLE EXTRACTOR
    # ---------------------------------------------------------
    def extract_article(self, url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except:
            return ""

    # ---------------------------------------------------------
    # GOOGLE NEWS RSS SEARCH
    # ---------------------------------------------------------
    def fetch_related_news(self, query, max_results=10):
        try:
            gn = GoogleNews(lang="en")
            gn.search(query)
            results = gn.result()[:max_results]
            gn.clear()

            # Only real article links
            cleaned = [
                n for n in results
                if n.get("link", "").startswith("http")
            ]
            return cleaned
        except:
            return []

    # ---------------------------------------------------------
    # FEATURE EXTRACTION
    # ---------------------------------------------------------
    def extract_features(self, input_text, related_texts):
        features = {}

        if related_texts:
            vectorizer = TfidfVectorizer().fit([input_text] + related_texts)
            vectors = vectorizer.transform([input_text] + related_texts).toarray()
            sims = cosine_similarity([vectors[0]], vectors[1:])[0]

            features["max_sim"] = float(sims.max())
            features["avg_sim"] = float(sims.mean())
            features["supporting"] = int((sims > 0.5).sum())
        else:
            features["max_sim"] = 0
            features["avg_sim"] = 0
            features["supporting"] = 0

        sentiment = TextBlob(input_text).sentiment
        features["polarity"] = float(sentiment.polarity)
        features["subjectivity"] = float(sentiment.subjectivity)

        features["length"] = len(input_text.split())
        return features

    # ---------------------------------------------------------
    # KEYWORD MATCH SCORE
    # ---------------------------------------------------------
    def keyword_score(self, input_text, related_texts):
        keywords = re.findall(r"\b[A-Z][a-z]+\b", input_text)

        if not keywords:
            return 0

        matches = sum(
            1 for k in keywords
            if any(k.lower() in rt.lower() for rt in related_texts)
        )

        return (matches / max(1, len(keywords))) * 100

    # ---------------------------------------------------------
    # SNIPPET GENERATOR
    # ---------------------------------------------------------
    def highlight_snippet(self, main_text, related_text):
        main_words = set(re.findall(r"\w+", main_text.lower()))
        words = re.findall(r"\w+", related_text.lower())
        matches = [w for w in words if w in main_words]
        return " ".join(matches[:12]) + ("..." if len(matches) > 12 else "")

    # ---------------------------------------------------------
    # MAIN CHECK FUNCTION
    # ---------------------------------------------------------
    def check(self, input_url_or_text):

        # If input is a URL
        if input_url_or_text.startswith("http"):
            input_text = self.extract_article(input_url_or_text)
        else:
            input_text = input_url_or_text

        if not input_text or len(input_text.split()) < 5:
            return {
                "error": "Could not extract enough article text.",
                "verdict": "UNVERIFIABLE"
            }

        # Build search query
        query = " ".join(input_text.split()[:10])
        related = self.fetch_related_news(query)

        # If no news found
        if not related:
            return {
                "max_similarity": 0,
                "reliability_score": 0,
                "verdict": "❌ Extremely Fake News",
                "related_articles": [],
                "snippets": []
            }

        # Extract related news text
        related_texts = []
        related_clean = []

        for n in related:
            try:
                txt = self.extract_article(n["link"])
                if txt:
                    related_texts.append(txt)
                    related_clean.append({
                        "title": n.get("title"),
                        "link": n.get("link"),
                        "media": n.get("media")
                    })
            except:
                continue

        # Calculate features
        feats = self.extract_features(input_text, related_texts)
        kscore = self.keyword_score(input_text, related_texts)

        reliability_score = (feats["max_sim"] * 60) + (kscore * 0.4)
        reliability_score = min(reliability_score, 100)

        # Verdicts
        if feats["max_sim"] < 0.2 and kscore < 20:
            verdict = "❌ Extremely Fake News"
        elif reliability_score < 50:
            verdict = "❌ Fake News"
        else:
            verdict = "✅ Reliable News"

        # Create snippets
        snippets = [
            {
                "title": related_clean[i]["title"],
                "snippet": self.highlight_snippet(input_text, related_texts[i])
            }
            for i in range(min(3, len(related_texts)))
        ]

        return {
            "max_similarity": round(feats["max_sim"], 3),
            "avg_similarity": round(feats["avg_sim"], 3),
            "reliability_score": round(reliability_score, 2),
            "verdict": verdict,
            "related_articles": related_clean,
            "snippets": snippets
        }
