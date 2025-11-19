# sentiment.py — Unified wrapper for the dynamic sentiment engine

from sentiment_dynamic import analyze_sentiment_dynamic

def analyze_sentiment(text: str):
    """
    Clean wrapper so Flask always receives consistent
    JSON structure for frontend.
    """
    if not text or not text.strip():
        return {
            "sentiment": None,
            "emoji": None,
            "confidence": 0.0,
            "method": "error",
            "notes": "Empty input"
        }

    result = analyze_sentiment_dynamic(text)

    # Normalize output for frontend
    return {
        "sentiment": result.get("sentiment"),
        "emoji": result.get("emoji"),
        "confidence": float(result.get("confidence", 0)),
        "method": result.get("method", "dynamic"),
        "notes": result.get("notes", "")
    }
