"""
Updated article_compare.py
- Adds Bloomberg-specific extractor (JSON-LD) before other fallbacks
- Keeps spaCy sentence-splitting optional
- Keeps paraphraser optional (transformers) and Playwright optional for dynamic rendering
- Robust extraction chain with logging and clear failures for paywall/CAPTCHA

Uploaded assets reference (if you want to inspect original upload):
/mnt/data/7bd7d723-42c5-4ae4-8b2a-82db46016070.zip

Usage:
    from article_compare import analyze_articles
    result = analyze_articles([url1, url2], use_paraphrase=False)

"""

import re
import json
import logging
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Optional heavy dependencies (lazy/optional)
SPACY_AVAILABLE = False
PARAPHRASER_AVAILABLE = False
PLAYWRIGHT_ENABLED = False
_nlp = None
paraphraser_pipeline = None
sync_playwright = None

# Try spaCy for reliable sentence segmentation
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("spaCy loaded — sentence splitting enabled")
except Exception:
    _nlp = None
    SPACY_AVAILABLE = False
    logger.info("spaCy not available — falling back to regex sentence splitting")

# Try paraphraser (transformers) only if installed — OPTIONAL heavy
try:
    from transformers import pipeline
    try:
        # Do not force download in production; if model not present this may try to download.
        # User may replace 't5-base' with a local small model for faster inference.
        paraphraser_pipeline = pipeline("text2text-generation", model="t5-base")
        PARAPHRASER_AVAILABLE = True
        logger.info("Paraphraser pipeline ready")
    except Exception:
        paraphraser_pipeline = None
        PARAPHRASER_AVAILABLE = False
        logger.info("Paraphraser pipeline not available")
except Exception:
    PARAPHRASER_AVAILABLE = False
    paraphraser_pipeline = None
    logger.info("transformers not installed — paraphraser disabled")

# Optional Playwright for JS-heavy pages
try:
    from playwright.sync_api import sync_playwright
    sync_playwright  # silence linter
    PLAYWRIGHT_ENABLED = True
    logger.info("Playwright available — dynamic extraction enabled")
except Exception:
    PLAYWRIGHT_ENABLED = False
    sync_playwright = None
    logger.info("Playwright not available — dynamic extraction disabled")

# SentenceTransformer (single instance)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Utilities
# ---------------------------

def _regex_split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s and len(s.strip()) > 10]


def split_sentences(text):
    """Return list of sentences using spaCy if available, otherwise a simple regex splitter."""
    if SPACY_AVAILABLE and _nlp is not None:
        try:
            doc = _nlp(text)
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        except Exception:
            return _regex_split_sentences(text)
    else:
        return _regex_split_sentences(text)


def _looks_like_captcha_or_blocked(html_text):
    low = (html_text or "").lower()
    blockers = [
        "not a robot", "enable javascript", "please verify", "access denied",
        "are you human", "complete the security check", "cloudflare", "captcha"
    ]
    return any(phrase in low for phrase in blockers)

# ---------------------------
# Bloomberg-specific extractor
# ---------------------------

def extract_bloomberg(url):
    """
    Try to extract full article text from Bloomberg by reading JSON-LD (<script type="application/ld+json">).
    This often contains an "articleBody" field with the full text, bypassing many JS/cookie walls.
    Returns article text string or None.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=12)
        html = r.text or ""

        if _looks_like_captcha_or_blocked(html):
            logger.debug("Bloomberg page looks blocked by captcha/banner")
            return None

        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script", {"type": "application/ld+json"})

        for script in scripts:
            try:
                data = json.loads(script.string)
            except Exception:
                # sometimes script contents aren't pure json, try to sanitize
                try:
                    text = script.string or ""
                    # remove leading/trailing non-json
                    start = text.find('{')
                    end = text.rfind('}')
                    if start != -1 and end != -1:
                        data = json.loads(text[start:end+1])
                    else:
                        continue
                except Exception:
                    continue

            # data may be a dict or a list
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "articleBody" in item:
                        body = item.get("articleBody")
                        if body and len(body) > 200:
                            return body
            elif isinstance(data, dict):
                if "articleBody" in data:
                    body = data.get("articleBody")
                    if body and len(body) > 200:
                        return body

        # fallback: try to heuristically assemble from <div> with article text
        # Some Bloomberg pages use <section> or <div> with data-type or paragraph tags
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        if paras:
            combined = " ".join([p for p in paras if p])
            if len(combined) > 400:
                return combined

    except Exception:
        logger.exception("extract_bloomberg failed for %s", url)
    return None

# ---------------------------
# Other extraction methods
# ---------------------------

def extract_newspaper(url):
    try:
        art = Article(url)
        art.download()
        art.parse()
        txt = (art.text or "").strip()
        if len(txt) > 200:
            return txt
    except Exception:
        logger.debug("newspaper3k failed for %s", url)
    return None


def extract_manual(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=12)
        text = r.text or ""
        if _looks_like_captcha_or_blocked(text):
            logger.debug("manual extract blocked for %s", url)
            return None

        soup = BeautifulSoup(text, "html.parser")

        # Meta/title/description
        meta_parts = []
        title_tag = soup.find("meta", {"property": "og:title"})
        if title_tag and title_tag.get("content"):
            meta_parts.append(title_tag.get("content"))
        desc_tag = soup.find("meta", {"name": "description"})
        if desc_tag and desc_tag.get("content"):
            meta_parts.append(desc_tag.get("content"))
        og_desc = soup.find("meta", {"property": "og:description"})
        if og_desc and og_desc.get("content"):
            meta_parts.append(og_desc.get("content"))

        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        paras_text = " ".join([p for p in paras if p])

        candidate = (" ".join(meta_parts) + " " + paras_text).strip()
        if len(candidate) > 200:
            return candidate
    except Exception:
        logger.debug("manual extraction failed for %s", url)
    return None


def extract_textise(url):
    try:
        textise_url = f"https://textise.net/showtext.aspx?strURL={url}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(textise_url, headers=headers, timeout=12)
        if r.status_code != 200:
            return None
        if _looks_like_captcha_or_blocked(r.text):
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        t = soup.get_text(" ", strip=True)
        if len(t) > 200:
            return t
    except Exception:
        logger.debug("textise extraction failed for %s", url)
    return None


def extract_dynamic(url):
    if not PLAYWRIGHT_ENABLED:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_navigation_timeout(20000)
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            html = page.content()
            browser.close()

        if _looks_like_captcha_or_blocked(html):
            return None

        soup = BeautifulSoup(html, "html.parser")
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        t = " ".join([p for p in paras if p])
        if len(t) > 200:
            return t
    except Exception:
        logger.exception("dynamic extraction failed for %s", url)
    return None

# ---------------------------
# Master extraction flow
# ---------------------------

def extract_article(url):
    """
    Master extraction flow (ordered):
      0) Bloomberg-specific extractor (fast, reliable)
      1) newspaper3k
      2) manual HTML extraction
      3) textise.net
      4) Playwright dynamic render (heavy, last resort)
    """
    url = url.strip()
    logger.info("Extracting: %s", url)

    # 0) Bloomberg special case
    if "bloomberg.com" in url:
        txt = extract_bloomberg(url)
        if txt:
            logger.info("Bloomberg extractor succeeded for %s (len=%d)", url, len(txt))
            return txt
        else:
            logger.info("Bloomberg extractor failed or blocked for %s, continuing fallbacks", url)

    # 1) newspaper3k
    txt = extract_newspaper(url)
    if txt:
        logger.info("newspaper3k succeeded for %s (len=%d)", url, len(txt))
        return txt

    # 2) manual
    txt = extract_manual(url)
    if txt:
        logger.info("manual extraction succeeded for %s (len=%d)", url, len(txt))
        return txt

    # 3) textise.net
    txt = extract_textise(url)
    if txt:
        logger.info("textise succeeded for %s (len=%d)", url, len(txt))
        return txt

    # 4) dynamic (Playwright)
    txt = extract_dynamic(url)
    if txt:
        logger.info("dynamic extraction succeeded for %s (len=%d)", url, len(txt))
        return txt

    logger.warning("All extraction methods failed for %s", url)
    return None

# ---------------------------
# Summarization / paraphrase
# ---------------------------

def generate_summary(text, sentence_count=4, use_paraphrase=False):
    sentences = [s for s in split_sentences(text) if len(s) > 25]
    if not sentences:
        return "Summary not available."

    try:
        embeddings = model.encode(sentences, convert_to_tensor=True)
        centroid = embeddings.mean(dim=0)
        sims = util.cos_sim(embeddings, centroid).squeeze()
        top_idx = sims.argsort(descending=True)[:sentence_count]
        summary = " ".join([sentences[i] for i in top_idx])
    except Exception:
        summary = " ".join(sentences[:sentence_count])

    # Optional paraphrase/grammar correction
    if use_paraphrase and PARAPHRASER_AVAILABLE and paraphraser_pipeline is not None:
        try:
            prompt = f"paraphrase: {summary}"
            out = paraphraser_pipeline(prompt, max_length=300, truncation=True)
            if isinstance(out, list) and len(out) > 0:
                # different pipelines return different keys
                if "generated_text" in out[0]:
                    summary = out[0]["generated_text"].strip()
                elif "summary_text" in out[0]:
                    summary = out[0]["summary_text"].strip()
        except Exception:
            logger.debug("paraphraser failed, returning raw summary")

    return summary

# ---------------------------
# Semantic utilities
# ---------------------------

def compute_similarity(text1, text2):
    try:
        e1 = model.encode(text1, convert_to_tensor=True)
        e2 = model.encode(text2, convert_to_tensor=True)
        sim = float(util.cos_sim(e1, e2))
        if sim > 1.0: sim = 1.0
        if sim < -1.0: sim = -1.0
        return max(0.0, sim)
    except Exception:
        logger.exception("compute_similarity error")
        return 0.0


def common_summary(texts, max_sentences=4, use_paraphrase=False):
    all_sents = []
    for t in texts:
        all_sents.extend([s for s in split_sentences(t) if len(s) > 20])

    if not all_sents:
        return "No common summary possible."

    try:
        emb = model.encode(all_sents, convert_to_tensor=True)
        centroid = emb.mean(dim=0)
        sims = util.cos_sim(emb, centroid).squeeze()
        top_idx = sims.argsort(descending=True)[:max_sentences]
        summary = " ".join([all_sents[i] for i in top_idx])
    except Exception:
        summary = " ".join(all_sents[:max_sentences])

    if use_paraphrase and PARAPHRASER_AVAILABLE and paraphraser_pipeline is not None:
        try:
            prompt = f"paraphrase: {summary}"
            out = paraphraser_pipeline(prompt, max_length=400, truncation=True)
            if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
                summary = out[0]["generated_text"].strip()
        except Exception:
            logger.debug("common_summary paraphrase failed")

    return summary

# ---------------------------
# Main public function
# ---------------------------

def analyze_articles(urls, use_paraphrase=False):
    if not isinstance(urls, (list, tuple)) or len(urls) < 2:
        return {"error": "Provide at least two article URLs."}

    results = []
    texts = []

    for url in urls:
        u = (url or "").strip()
        if not u:
            results.append({"url": url, "error": "Empty URL"})
            continue

        try:
            txt = extract_article(u)
        except Exception:
            logger.exception("extract_article exception for %s", u)
            txt = None

        if not txt:
            results.append({"url": u, "error": "Unable to extract article (paywall/CAPTCHA/blocked)"})
            continue

        summary = generate_summary(txt, sentence_count=4, use_paraphrase=(use_paraphrase and PARAPHRASER_AVAILABLE))

        results.append({
            "url": u,
            "summary": summary,
            "text_length": len(txt)
        })
        texts.append(txt)

    if len(texts) < 2:
        return {"error": "Need at least 2 valid articles", "articles": results}

    # compute primary similarity between first two
    sim = compute_similarity(texts[0], texts[1])

    # similarity matrix for all provided articles (optional)
    try:
        embeds = model.encode(texts, convert_to_tensor=True)
        sim_matrix = util.cos_sim(embeds, embeds).cpu().numpy().tolist()
    except Exception:
        sim_matrix = []

    verdict = (
        "Same Topic ✔" if sim >= 0.70 else
        "Related Topic ✔" if sim >= 0.40 else
        "Different Topic ✖"
    )

    shared = common_summary(texts, max_sentences=4, use_paraphrase=(use_paraphrase and PARAPHRASER_AVAILABLE))

    return {
        "verdict": verdict,
        "similarity": sim,
        "similarity_percent": round(sim * 100, 2),
        "similarity_matrix": sim_matrix,
        "common_summary": shared,
        "articles": results
    }


if __name__ == "__main__":
    test_urls = [
        "https://www.firstpost.com/explainers/pinkfong-baby-shark-400-million-ipo-rise-viral-video-13952019.html",
        "https://www.bloomberg.com/news/articles/2025-11-17/-baby-shark-creator-pinkfong-set-for-seoul-debut-after-popular-ipo"
    ]
    out = analyze_articles(test_urls, use_paraphrase=False)
    print(json.dumps(out, indent=2))
