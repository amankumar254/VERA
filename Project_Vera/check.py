import requests
from bs4 import BeautifulSoup

# --- Placeholder for the core classification logic ---
def classify_text(text: str) -> str:
    """
    Simulates a machine learning model predicting news authenticity.
    (Replace this with your actual ML model prediction or API call)
    """
    text_lower = text.lower()
    if "hoax" in text_lower or "fake" in text_lower or "satire" in text_lower:
        return "Likely FAKE/SATIRE (Placeholder Logic)"
    elif len(text) < 500:
        return "UNCERTAIN (Too little content extracted)"
    else:
        # A simple, non-scientific placeholder for demonstration
        return "Likely REAL (Placeholder Logic)" 
# ----------------------------------------------------

def get_article_text(url: str) -> str:
    """Fetches and extracts the main text content from a given URL."""
    try:
        # Ensure the URL has a scheme (http/https)
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Set a User-Agent to mimic a browser
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Common tags for article content (needs refinement for real-world use)
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
        article_text = ' '.join([p.get_text() for p in paragraphs])
        
        # Return a manageable length of text for classification
        return article_text[:5000].strip()

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def check_news_authenticity_by_url(url: str) -> str:
    """Main function to check news authenticity."""
    print("\n🔗 Attempting to retrieve and analyze article content...")
    
    article_content = get_article_text(url)
    
    if "Error fetching URL" in article_content or "An unexpected error occurred" in article_content:
        return f"**Analysis Failed:** {article_content}"
    
    if not article_content:
        return "**Analysis Failed:** Could not extract sufficient article text from the URL."
        
    # Pass the extracted text to the classification model
    result = classify_text(article_content)
    
    return f"\n\n📰 **CLASSIFICATION RESULT:**\nSource URL: {url}\nPrediction: **{result}**"

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # 1. Take URL input from the user
    user_url = input("Please enter the full news article URL you want to check: ")
    
    # 2. Validate input (basic check)
    if not user_url.strip():
        print("Error: URL cannot be empty.")
    else:
        # 3. Run the authenticity check
        final_output = check_news_authenticity_by_url(user_url)
        print(final_output)