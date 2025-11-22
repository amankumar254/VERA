# ==============================================================================
# DUAL-MODE NEWS STORY CLUSTERING - K-MEANS and DBSCAN
# ==============================================================================
# This script allows the user to select between two statistical clustering methods:
# 1. K-Means: Forces articles into a fixed number of clusters (K).
# 2. DBSCAN: Automatically finds dense clusters (redundancy) and identifies unique stories (noise).

import numpy as np
import string
import warnings
import sys
import subprocess

# Suppress warnings that are common in this kind of ML pipeline
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Installation and Imports (Colab Specific) ---
try:
    # Attempt to import necessary libraries
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans, DBSCAN
    from nltk.corpus import stopwords
    import nltk
except ImportError:
    print("Installing required packages (scikit-learn, nltk)...")
    # Use subprocess to run pip install
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "nltk"])
    # Re-import after installation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans, DBSCAN
    from nltk.corpus import stopwords
    import nltk

# --- 2. NLTK Data Download (Crucial Fix for LookupError) ---
print("\n[SETUP] Downloading NLTK 'stopwords' resource...")
try:
    nltk.download('stopwords', quiet=True)
    print("Download complete. Stopwords are now available.")
except Exception as e:
    print(f"Error during NLTK download: {e}")

# --- 3. Global Setup ---
custom_stopwords = set(stopwords.words('english') + list(string.punctuation))

# ==============================================================================
# --- CLUSTERING LOGIC FUNCTIONS ---
# ==============================================================================

def cluster_kmeans(articles, X):
    """K-Means clustering logic."""

    # 1. Determine K (Number of Clusters)
    estimated_k = max(1, int(len(articles) * 0.6))

    try:
        k_input = input(f"K-Means: Enter the estimated number of distinct stories (K). Recommended: {estimated_k}: ")
        K = int(k_input) if k_input.strip() else estimated_k
    except ValueError:
        print("Invalid input for K. Using recommended value.")
        K = estimated_k

    if K > len(articles):
        K = len(articles)

    print(f"\n[K-MEANS] Running K-Means Clustering with K={K}...")

    kmeans_model = KMeans(
        n_clusters=K,
        init='k-means++',
        max_iter=300,
        random_state=42,
        n_init='auto'
    )
    clusters = kmeans_model.fit_predict(X)
    return clusters, K

def cluster_dbscan(articles, X):
    """DBSCAN clustering logic."""

    print(f"\n[DBSCAN] Running DBSCAN Clustering (Automatic K)...")

    # Using 'cosine' distance metric for sparse TF-IDF data
    dbscan_model = DBSCAN(
        eps=0.5,           # Max distance for clustering (lower = tighter groups)
        min_samples=2,     # Min number of articles to form a core cluster
        metric='cosine',
        n_jobs=-1
    )

    # Fit the model. DBSCAN automatically assigns noise to cluster ID -1.
    clusters = dbscan_model.fit_predict(X)
    return clusters, None # K is not used for DBSCAN

# ==============================================================================
# --- EXECUTION AND DISPLAY FUNCTIONS ---
# ==============================================================================

def get_user_articles():
    """Prompts the user to input news articles."""
    articles = []
    print("\n" + "="*60)
    print("--- 📝 Enter Your News Articles for Clustering ---")
    print("Enter one article per line. Type 'DONE' when finished.")
    print("="*60)

    while True:
        article_input = input(f"Article {len(articles) + 1} (or DONE): ")
        if article_input.upper() == 'DONE':
            break
        if article_input.strip():
            articles.append(article_input.strip())

    return articles

def run_clustering_pipeline(articles):
    """Runs the full pipeline, including feature extraction and method selection."""
    if len(articles) < 2:
        print("\nNeed at least two articles to perform clustering. Exiting.")
        return

    # --- Feature Extraction (TF-IDF) ---
    print("\n[STEP 1/3] Generating Statistical Features (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        stop_words=list(custom_stopwords), # Convert set to list to resolve error
        max_df=0.85,
        min_df=1,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(articles)

    # --- Method Selection ---
    print("\n[STEP 2/3] Choose Clustering Method:")
    print("  1: K-Means (Forces all articles into K groups, useful if you know K)")
    print("  2: DBSCAN (Finds dense redundancy, marks unique stories as 'Noise')")

    method = input("Enter 1 or 2: ").strip() or '2' # Default to DBSCAN for robust results

    # --- Run Selected Clustering Model ---
    if method == '1':
        clusters, K = cluster_kmeans(articles, X)
        method_name = f"K-Means (K={K})"
    elif method == '2':
        clusters, K = cluster_dbscan(articles, X)
        method_name = "DBSCAN"
    else:
        print("Invalid choice. Defaulting to DBSCAN.")
        clusters, K = cluster_dbscan(articles, X)
        method_name = "DBSCAN (Default)"

    # --- Grouping Results ---
    print("\n[STEP 3/3] Grouping results by cluster ID...")
    clustered_stories = {}

    for i, cluster_id in enumerate(clusters):
        # DBSCAN noise group is -1, K-Means groups are 0, 1, 2...
        key = 'Unique/Noise' if cluster_id == -1 and method == '2' else cluster_id
        if key not in clustered_stories:
            clustered_stories[key] = []
        clustered_stories[key].append(articles[i])

    display_results(clustered_stories, method_name)


def display_results(clustered_stories, method_name):
    """Prints the grouped stories clearly."""

    print("\n" + "=="*30)
    print(f"      📰 REDUNDANCY / STORY CLUSTERING RESULTS ({method_name})")
    print("=="*30)

    # Handle the 'Unique/Noise' cluster separately for clear display
    noise_stories = clustered_stories.pop('Unique/Noise', [])

    # Sort detected clusters (for K-Means: all groups; for DBSCAN: only dense groups)
    sorted_clusters = sorted(clustered_stories.items(), key=lambda item: len(item[1]), reverse=True)

    cluster_count = 0
    for cluster_id, story_list in sorted_clusters:
        cluster_count += 1
        redundancy_note = " (REDUNDANCY detected!)" if len(story_list) > 1 else ""
        print(f"\n⭐ Story Cluster #{cluster_count} | Size: {len(story_list)} Articles{redundancy_note}")
        print("-" * 40)
        for story in story_list:
            print(f"  > {story[:95].strip()}...")

    # Display unique (noise) stories last
    if noise_stories:
        print(f"\n⚪ Unique/Noise Stories | Size: {len(noise_stories)} Articles")
        print("-----------------------------------")
        for story in noise_stories:
            print(f"  > {story[:95].strip()}...")

    print("\n" + "=="*30)
# -------------------------------------------------------
# NEW FUNCTION FOR FLASK BACKEND
# Converts clusters into HTML instead of console output
# -------------------------------------------------------
def cluster_articles(articles):
    """
    Flask-compatible wrapper.
    Input: list of article texts
    Output: HTML string for UI display
    """

    if len(articles) < 2:
        return "<p style='color:red;'>At least 2 articles required.</p>"

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words=list(custom_stopwords),
        max_df=0.85,
        min_df=1,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(articles)

    # Use DBSCAN automatically (best for redundancy)
    clusters, _ = cluster_dbscan(articles, X)

    # Group into dictionary
    grouped = {}
    for i, cid in enumerate(clusters):
        key = "Unique/Noise" if cid == -1 else f"Cluster {cid}"
        grouped.setdefault(key, []).append(articles[i])

    # ---- BUILD HTML ----
    html = """
    <div style='font-family:Inter, sans-serif;'>
        <h2 style='color:white; font-size:24px; margin-bottom:10px;'>Story Clustering Result</h2>
    """

    for cluster_name, items in grouped.items():
        color = "#ff4444" if cluster_name == "Unique/Noise" else "#4caf50"

        html += f"""
        <div style='background:#1e1e1e; padding:16px; border-radius:12px; 
                    margin-bottom:16px; border-left:4px solid {color};'>
            <h3 style='color:{color}; font-size:18px;'>{cluster_name} (x{len(items)})</h3>
        """

        for story in items:
            html += f"""
                <p style='color:#ccc; margin:6px 0;'>
                    • {story[:140]}...
                </p>
            """

        html += "</div>"

    html += "</div>"
    return html


# --- Main Execution Block ---
if __name__ == "__main__":

    user_articles = get_user_articles()

    if user_articles:
        run_clustering_pipeline(user_articles)
    else:
        print("No articles provided. Please run the script again and input content.")