import os
import re
import logging
from typing import List, Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from newspaper import Article

# Optional heavier deps
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    sync_playwright = None
    PLAYWRIGHT_AVAILABLE = False

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    _nlp = None
    SPACY_AVAILABLE = False

# ML libs
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import pairwise_distances

# Logging
logger = logging.getLogger("story_clusterer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
logger.addHandler(ch)

# Model — load once
MODEL_NAME = "all-MiniLM-L6-v2"
logger.info("Loading embedding model: %s", MODEL_NAME)
EMBED_MODEL = SentenceTransformer(MODEL_NAME)

# Tunable thresholds
MIN_CHARS = 120
TITLE_SIM_GATE = 0.35      # require at least this title similarity (cosine) to avoid penalising
SAME_TOPIC = 0.70
RELATED_TOPIC = 0.45
DBSCAN_EPS = 0.36         # cosine distance (1 - cosine sim) threshold
DBSCAN_MIN_SAMPLES = 2
SPECTRAL_N_CLUSTERS = None  # if None, estimate using affinity eigen-gap heuristics

# ------------------------- Extraction helpers -------------------------

def _looks_blocked(html: str) -> bool:
    if not html:
        return True
    low = html.lower()
    blockers = ["not a robot", "please enable javascript", "please verify", "access denied", "are you human", "cloudflare", "blocked by"]
    return any(x in low for x in blockers)


def _clean_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "iframe", "noscript", "header", "footer", "nav", "aside", "svg"]):
        try:
            tag.decompose()
        except Exception:
            pass
    # remove ad-like nodes
    for bad in soup.select("[class*='ad'], [id*='ad'], [class*='promo'], [class*='subscribe']"):
        try:
            bad.decompose()
        except Exception:
            pass
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_newspaper(url: str) -> Optional[Tuple[str, str]]:
    try:
        art = Article(url)
        art.download()
        art.parse()
        full = (art.title or "") + "\n\n" + (art.meta_description or "") + "\n\n" + (art.text or "")
        if full and len(full) >= MIN_CHARS:
            return (art.title or "", full)
    except Exception as e:
        logger.debug("newspaper3k failed for %s: %s", url, e)
    return None


def extract_bs4(url: str) -> Optional[Tuple[str, str]]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return None
        if _looks_blocked(r.text):
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()
        # look for article containers
        selectors = ["article", "[itemprop='articleBody']", ".article-body", ".article__content", ".story-body", ".content__article-body", ".post-content", ".entry-content", ".story-content"]
        paras = []
        for sel in selectors:
            el = soup.select_one(sel)
            if el:
                paras = [p.get_text(" ", strip=True) for p in el.find_all("p")]
                break
        if not paras:
            paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paras if p])
        candidate = f"{title}\n\n{text}".strip()
        candidate = re.sub(r"\s{2,}", " ", candidate)
        if candidate and len(candidate) >= MIN_CHARS:
            return (title, candidate)
    except Exception as e:
        logger.debug("bs4 extraction failed for %s: %s", url, e)
    return None


def extract_textise(url: str) -> Optional[Tuple[str, str]]:
    try:
        textise_url = f"https://textise.net/showtext.aspx?strURL={requests.utils.requote_uri(url)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(textise_url, headers=headers, timeout=12)
        if r.status_code != 200 or _looks_blocked(r.text):
            return None
        txt = _clean_text_from_html(r.text)
        if txt and len(txt) >= MIN_CHARS:
            # best-effort title: first line
            first = (txt.splitlines()[0] or "").strip()
            return (first, txt)
    except Exception as e:
        logger.debug("textise failed for %s: %s", url, e)
    return None


def extract_playwright(url: str) -> Optional[Tuple[str, str]]:
    if not PLAYWRIGHT_AVAILABLE:
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
        if _looks_blocked(html):
            return None
        txt = _clean_text_from_html(html)
        if not txt or len(txt) < MIN_CHARS:
            return None
        soup = BeautifulSoup(html, "html.parser")
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()
        return (title, txt)
    except Exception as e:
        logger.debug("playwright extraction failed for %s: %s", url, e)
    return None


# Master extraction
def extract_article(url_or_text: str) -> Tuple[Optional[str], Optional[str], str]:
    url_or_text = (url_or_text or "").strip()
    # local file
    if url_or_text.startswith("file://") or os.path.exists(url_or_text):
        try:
            path = url_or_text.replace("file://", "")
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            return ("", txt if len(txt) >= MIN_CHARS else None, "localfile")
        except Exception as e:
            return (None, None, f"local_error:{e}")

    # raw text
    if not (url_or_text.startswith("http://") or url_or_text.startswith("https://")):
        if len(url_or_text) >= MIN_CHARS:
            # first short line as title if available
            first_line = url_or_text.splitlines()[0].strip()
            title = first_line if len(first_line) < 200 else ""
            return (title, url_or_text, "raw_text")
        else:
            return ("", None, "raw_too_short")

    url = url_or_text
    # pipeline
    for fn in (extract_newspaper, extract_bs4, extract_textise, extract_playwright):
        try:
            out = fn(url)
            if out and out[1] and len(out[1]) >= MIN_CHARS:
                return (out[0] or "", out[1], fn.__name__)
        except Exception:
            continue
    return ("", None, "extraction_failed")


# ------------------------- Sentence splitting -------------------------

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    if SPACY_AVAILABLE and _nlp:
        try:
            doc = _nlp(text)
            return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
        except Exception:
            pass
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    return sents


# ------------------------- Embedding & similarity -------------------------

def embed_texts(texts: List[str]):
    if not texts:
        return []
    return EMBED_MODEL.encode(texts, convert_to_tensor=True)


def compute_similarity_matrix(texts: List[str]):
    try:
        embs = embed_texts(texts)
        sim = util.cos_sim(embs, embs).cpu().numpy()
        # clamp
        sim = sim.clip(0.0, 1.0)
        return sim
    except Exception as e:
        logger.warning("Similarity matrix computation failed: %s", e)
        n = len(texts)
        return [[0.0]*n for _ in range(n)]


def title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        emb = EMBED_MODEL.encode([a, b], convert_to_tensor=True)
        return float(util.cos_sim(emb[0], emb[1]))
    except Exception:
        return 0.0


# ------------------------- Clustering -------------------------

def hybrid_cluster(sim_matrix, min_cluster_size=2):
    """
    Hybrid approach:
      1) Build affinity (from sim_matrix)
      2) Run SpectralClustering to get initial groups (if >2 items)
      3) Run DBSCAN within each spectral cluster to mark noise / small groups

    Returns: cluster_labels (list int), where -1 indicates noise.
    """
    import numpy as np
    n = len(sim_matrix)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # affinity = sim (already in 0..1)
    affinity = sim_matrix.copy()

    # If spectral cannot be used (small n), fallback to DBSCAN on embeddings distances
    try:
        if n >= 3:
            # Estimate number of clusters by looking for gaps in eigenvalues of affinity Laplacian
            # Use sklearn SpectralClustering with guessed K (clamp 2..n)
            # Quick heuristic: try K=2..min(6,n) and pick that which maximises silhouette-like score on affinity
            best_labels = None
            best_score = -1
            max_k = min(6, n)
            from sklearn.metrics import silhouette_score
            # convert affinity to distance for silhouette (1 - sim)
            dist = 1.0 - affinity
            for k in range(2, max_k + 1):
                try:
                    sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
                    labels = sc.fit_predict(affinity)
                    # silhouette expects feature vectors; we supply distance matrix via precomputed metric
                    # compute a pseudo-score using within-cluster avg similarity - between-cluster avg similarity
                    intra = 0.0
                    inter = 0.0
                    intra_count = inter_count = 0
                    for i in range(n):
                        for j in range(i+1, n):
                            if labels[i] == labels[j]:
                                intra += affinity[i][j]; intra_count += 1
                            else:
                                inter += affinity[i][j]; inter_count += 1
                    if intra_count == 0:
                        continue
                    intra_avg = intra / intra_count
                    inter_avg = inter / inter_count if inter_count > 0 else 0.0
                    score = intra_avg - inter_avg
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                except Exception:
                    continue
            if best_labels is None:
                # fallback to DBSCAN
                logger.info("Spectral failed to choose K; falling back to DBSCAN")
            else:
                labels = list(best_labels)
                # Post-process each cluster with DBSCAN to separate noise
                final_labels = [-1]*n
                next_label = 0
                for cluster_id in sorted(set(labels)):
                    members = [i for i,l in enumerate(labels) if l==cluster_id]
                    if len(members) < min_cluster_size:
                        # mark as small cluster (assign as separate cluster)
                        for m in members:
                            final_labels[m] = next_label
                        next_label += 1
                        continue
                    # sub-sim matrix for members
                    sub_aff = affinity.__array__()[members][:,members] if hasattr(affinity, '__array__') else affinity
                    # apply DBSCAN on distances
                    try:
                        db = DBSCAN(eps=1-DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="precomputed")
                        # dbscan expects distance matrix, so pass (1 - sim)
                        dmat = [[1.0 - sub_aff[i][j] for j in range(len(members))] for i in range(len(members))]
                        sub_labels = db.fit_predict(dmat)
                    except Exception:
                        sub_labels = [0]*len(members)
                    # map sub_labels into final_labels space
                    unique_sub = sorted(set([x for x in sub_labels if x!=-1]))
                    if len(unique_sub) == 0:
                        # all noise -> create single cluster label
                        for idx_m, m in enumerate(members):
                            if sub_labels[idx_m] == -1:
                                final_labels[m] = -1
                            else:
                                final_labels[m] = next_label
                        next_label += 1
                    else:
                        mapping = {u: next_label + i for i,u in enumerate(unique_sub)}
                        for idx_m, m in enumerate(members):
                            if sub_labels[idx_m] == -1:
                                final_labels[m] = -1
                            else:
                                final_labels[m] = mapping[sub_labels[idx_m]]
                        next_label += len(unique_sub)
                return final_labels

        # Fallback DBSCAN on full matrix
        logger.info("Using DBSCAN fallback on full matrix")
        from sklearn.cluster import DBSCAN as _DB
        # DBSCAN requires distance matrix; we use 1 - similarity
        dmat_full = [[1.0 - affinity[i][j] for j in range(n)] for i in range(n)]
        db = _DB(eps=1-DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='precomputed')
        lab = db.fit_predict(dmat_full)
        return list(lab)
    except Exception as e:
        logger.exception("hybrid_cluster failed: %s", e)
        # as ultimate fallback, put all in single cluster
        return [0]*n

# ------------------------- Analysis API -------------------------

def analyze_and_cluster(inputs: List[str]) -> Dict:
    """High-level function returning detailed analysis used by Flask endpoints."""
    if not isinstance(inputs, (list, tuple)) or len(inputs) < 2:
        return {"error": "Provide at least two inputs (urls or raw texts)"}

    meta = []
    texts = []
    valid_indices = []

    # 1) extract
    for i, item in enumerate(inputs):
        title, text, src = extract_article(item)
        if not text:
            meta.append({"input": item, "title": title or "", "text_length": 0, "summary": "", "source": src})
            continue
        cleaned = re.sub(r"\s{2,}", " ", text.strip())
        meta.append({"input": item, "title": title or "", "text_length": len(cleaned), "text": cleaned, "source": src})
        valid_indices.append(i)
        texts.append(cleaned)

    if len(texts) < 2:
        return {"error": "Need at least 2 valid articles/texts", "articles": meta}

    # 2) similarity matrix between documents
    sim_matrix = compute_similarity_matrix(texts)

    # 3) hybrid clustering (returns labels for L = len(texts))
    labels = hybrid_cluster(sim_matrix)

    # 4) compute top pairs
    top_score = -1.0
    top_pair = (0,1)
    L = len(texts)
    for i in range(L):
        for j in range(i+1, L):
            if sim_matrix[i][j] > top_score:
                top_score = sim_matrix[i][j]
                top_pair = (i, j)

    # title gating penalty
    t_sim = title_similarity(meta[valid_indices[top_pair[0]]].get('title',''), meta[valid_indices[top_pair[1]]].get('title',''))
    adjusted_score = top_score
    if top_score >= RELATED_TOPIC and t_sim < TITLE_SIM_GATE:
        logger.info("Title gating applied: title_sim=%.3f top_score=%.3f", t_sim, top_score)
        adjusted_score = top_score * 0.85

    verdict = "Different Topic ✖"
    if adjusted_score >= SAME_TOPIC:
        verdict = "Same Topic ✔"
    elif adjusted_score >= RELATED_TOPIC:
        verdict = "Related Topic ✔"

    # 5) per-article summaries (centroid sentence selection)
    def centroid_summary_for_text(text, n_sent=3):
        sents = split_sentences(text)
        if not sents:
            return ""
        try:
            emb = EMBED_MODEL.encode(sents, convert_to_tensor=True)
            centroid = emb.mean(dim=0)
            sims = util.cos_sim(emb, centroid).squeeze()
            topk = sims.argsort(descending=True)[:n_sent]
            return " ".join([sents[int(i)] for i in topk])
        except Exception:
            return " ".join(sents[:n_sent])

    for idx_emb, orig_idx in enumerate(valid_indices):
        meta[orig_idx]["summary"] = centroid_summary_for_text(texts[idx_emb], n_sent=4)

    # 6) cluster groups mapped to original indices
    groups = {}
    for emb_idx, lab in enumerate(labels):
        orig_idx = valid_indices[emb_idx]
        groups.setdefault(lab, []).append(orig_idx)

    # 7) build full similarity matrix mapped to inputs length
    A_n = len(meta)
    full_sim = [[0.0]*A_n for _ in range(A_n)]
    for i_emb, i_orig in enumerate(valid_indices):
        for j_emb, j_orig in enumerate(valid_indices):
            full_sim[i_orig][j_orig] = float(sim_matrix[i_emb][j_emb])

    common_all = ""
    try:
        all_sents = []
        for i_emb in range(len(texts)):
            all_sents.extend(split_sentences(texts[i_emb]))
        common_all = centroid_summary_for_text(" ".join(all_sents), n_sent=4)
    except Exception:
        common_all = ""

    out = {
        "verdict": verdict,
        "top_pair": {"i": valid_indices[top_pair[0]], "j": valid_indices[top_pair[1]], "score": float(top_score)},
        "similarity": float(adjusted_score),
        "similarity_percent": round(float(adjusted_score) * 100.0, 2),
        "similarity_matrix": full_sim,
        "groups": groups,
        "common_summary": common_all,
        "articles": meta
    }
    return out


# ------------------------- HTML helper -------------------------

def cluster_articles(inputs: List[str]) -> str:
    """Flask-friendly HTML fragment (safe, minimal inline styles)."""
    try:
        analysis = analyze_and_cluster(inputs)
    except Exception as e:
        logger.exception("cluster_articles failed: %s", e)
        return f"<div style='color:red;'>Error: {e}</div>"

    if analysis.get("error"):
        return f"<div style='color:red;padding:12px;'>Error: {analysis['error']}</div>"

    html = "<div style='font-family:Inter,sans-serif;'>"
    html += f"<h2 style='color:white;'>Story Clustering Result</h2>"
    html += f"<p style='color:#cbd5e1;'>Verdict: <strong style='color:#fff;'>{analysis['verdict']}</strong> — Top similarity: <strong>{analysis['similarity_percent']}%</strong></p>"
    html += "<hr style='border-color:#222;margin:10px 0;'>"
    html += "<h4 style='color:#9ed2ff;'>Common Summary</h4>"
    html += f"<p style='color:#bfc3c8'>{analysis.get('common_summary','')}</p>"

    # groups
    html += "<h4 style='color:#9ed2ff;'>Clusters</h4>"
    groups = analysis.get('groups', {})
    # prefer numeric labels sorted; show -1 (noise) last
    sorted_keys = sorted([k for k in groups.keys() if k != -1]) + ([-1] if -1 in groups else [])
    for k in sorted_keys:
        items = groups[k]
        label = 'Unique/Noise' if k == -1 else f"Cluster {k}"
        color = '#ff6b6b' if k == -1 else '#4caf50'
        html += f"<div style='background:#111;padding:12px;border-left:4px solid {color};border-radius:8px;margin:8px 0;'>"
        html += f"<h5 style='color:{color};margin:0;padding:0;'>{label} (x{len(items)})</h5>"
        for idx in items:
            art = analysis['articles'][idx]
            summary_preview = (art.get('summary') or '')[:220]
            html += f"<p style='color:#ddd;margin:6px 0;word-break:break-all;'>• {art.get('input')}<br><small style='color:#9aa0a6;'>{summary_preview}...</small></p>"
        html += "</div>"

    html += "</div>"
    return html


# ------------------------- module quick test -------------------------
if __name__ == "__main__":
    test = [
        "https://www.firstpost.com/explainers/pinkfong-baby-shark-400-million-ipo-rise-viral-video-13952019.html",
        "https://www.bloomberg.com/news/articles/2025-11-17/-baby-shark-creator-pinkfong-set-for-seoul-debut-after-popular-ipo",
    ]
    res = analyze_and_cluster(test)
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False))
