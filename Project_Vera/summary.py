import re
import requests
from bs4 import BeautifulSoup
from requests_html import HTMLSession
from newspaper import Article
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY_HERE")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


# ----------------------------------------------------------
# LEVEL 0 — HindustanTimes hidden API bypass
# ----------------------------------------------------------
def try_hindustantimes(url):
    try:
        match = re.search(r"(\d+)\.html", url)
        if not match:
            return None

        article_id = match.group(1)
        api_url = f"https://www.hindustantimes.com/api/content/{article_id}"

        res = requests.get(api_url, headers={"User-Agent": UA})
        if res.status_code != 200:
            return None

        data = res.json()
        html_body = data.get("content", "")

        soup = BeautifulSoup(html_body, "html.parser")
        text = "\n".join([p.get_text(strip=True) for p in soup.find_all("p")])

        return text if len(text) > 200 else None
    except:
        return None


# ----------------------------------------------------------
# LEVEL 1 — Newspaper3k
# ----------------------------------------------------------
def try_newspaper(url):
    try:
        article = Article(url, headers={"User-Agent": UA})
        article.download()
        article.parse()
        article.nlp()
        return article.text, article.summary
    except:
        return None, None


# ----------------------------------------------------------
# LEVEL 2 — JS rendered extraction
# ----------------------------------------------------------
def try_htmlsession(url):
    try:
        session = HTMLSession()
        r = session.get(url, headers={"User-Agent": UA})
        r.html.render(timeout=15)

        text = "\n".join(el.text for el in r.html.find("p"))
        return text if len(text) > 200 else None
    except:
        return None


# ----------------------------------------------------------
# LEVEL 3 — BeautifulSoup HTML fallback
# ----------------------------------------------------------
def try_bs4(url):
    try:
        r = requests.get(url, headers={"User-Agent": UA})
        soup = BeautifulSoup(r.text, "html.parser")
        text = "\n".join([p.get_text(strip=True) for p in soup.find_all("p")])
        return text if len(text) > 200 else None
    except:
        return None


# ----------------------------------------------------------
# LEVEL 4 — AI summarizer
# ----------------------------------------------------------
def ai_summarize(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize this news article in 6 bullet points."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI summarization failed: {str(e)}"


# ----------------------------------------------------------
# MASTER PIPELINE
# ----------------------------------------------------------
def smart_summarize(url):

    # HINDUSTAN TIMES bypass
    text = try_hindustantimes(url)
    if text:
        return ai_summarize(text)

    # Newspaper
    text, summary = try_newspaper(url)
    if summary:
        return summary

    # JS rendered
    text = try_htmlsession(url)
    if text:
        return ai_summarize(text)

    # Pure HTML
    text = try_bs4(url)
    if text:
        return ai_summarize(text)

    return "❌ Unable to extract article text (all fallback methods failed)."
