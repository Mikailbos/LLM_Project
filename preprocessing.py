import os
import re
import json
import nltk
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from newspaper import Article

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

INPUT_DIR = "data_output"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clean raw text (punctuation, stopwords, lowercase)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special characters
    text = text.lower()
    words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(words)

# Resolve redirect URLs (placeholder if needed later)
def resolve_redirect(url):
    try:
        response = requests.get(url, timeout=5, allow_redirects=True)
        return response.url
    except:
        return url

# Extract article text using newspaper3k, with fallback using BeautifulSoup + headers
def extract_article_text(url):
    # First attempt: newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text.split()) > 20:
            return article.text
    except Exception as e:
        print(f"⚠️ newspaper3k failed: {e} on URL {url}")

    # Fallback: use BeautifulSoup with browser-like headers
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        if len(text.split()) > 20:
            return text
    except Exception as e:
        print(f"❌ Fallback failed for {url}: {e}")
    
    return ""

# Preprocess a single JSON file
def preprocess_news_file(filename):
    with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
        articles = json.load(f)

    processed_articles = []
    for entry in articles:
        real_url = resolve_redirect(entry["link"])
        article_text = extract_article_text(real_url)

        if not article_text or len(article_text.split()) < 20:
            print(f"⚠️ Skipped (too short): {real_url}")
            continue

        article_clean = clean_text(article_text)
        title_clean = clean_text(entry["title"])

        processed_articles.append({
            "original_title": entry["title"],
            "cleaned_title": title_clean,
            "article_text": article_text,
            "cleaned_article_text": article_clean,
            "link": real_url,
            "published": entry["published"]
        })

    with open(os.path.join(OUTPUT_DIR, f"processed_{filename}"), "w", encoding="utf-8") as f:
        json.dump(processed_articles, f, indent=2)

    print(f"✅ Processed: {filename} → {len(processed_articles)} articles")

# Process all *_news.json files in the input directory
def run_preprocessing():
    for file in os.listdir(INPUT_DIR):
        if file.endswith("_news.json"):
            preprocess_news_file(file)

if __name__ == "__main__":
    run_preprocessing()
