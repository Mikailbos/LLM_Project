import os
import json
import re
import spacy
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk

# Download NLTK data if not already
nltk.download("stopwords")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# ------------------ Configuration ------------------
INPUT_DIR = "processed_data"
OUTPUT_DIR = "enriched_data"
TICKER_FILE = "sp500_ticker_mapping.json"

# ------------------ Load Ticker Mapping ------------------
with open(TICKER_FILE, "r", encoding="utf-8") as f:
    raw_map = json.load(f)
    ticker_map = {k.lower(): v.upper() for k, v in raw_map.items()}


# ------------------ Helper Functions ------------------

def extract_entities(text):
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def get_sentiment(text):
    blob = TextBlob(text)
    return {
        "polarity": round(blob.polarity, 3),
        "subjectivity": round(blob.subjectivity, 3)
    }

def extract_triplet(text):
    doc = nlp(text)
    for sent in doc.sents:
        subj = verb = obj = ""
        for token in sent:
            if "subj" in token.dep_ and token.head.pos_ == "VERB":
                subj = token.text
                verb = token.head.text
                for child in token.head.children:
                    if "obj" in child.dep_:
                        obj = child.text
                        return subj, verb, obj, sent.text
    return "", "", "", ""

def match_tickers(text):
    text_lower = text.lower()
    matches = set()

    for company, ticker in ticker_map.items():
        if re.search(rf"\b{re.escape(company)}\b", text_lower):
            matches.add(ticker)

    return list(matches)

# ------------------ File Processor ------------------

def process_file(filename):
    with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
        articles = json.load(f)

    enriched = []
    for article in articles:
        text = article.get("article_text", "")
        sentiment = get_sentiment(text)
        entities = extract_entities(text)
        tickers = match_tickers(text)
        subj, verb, obj, snt = extract_triplet(text)

        enriched.append({
            **article,
            "sentiment": sentiment,
            "entities": entities,
            "tickers": tickers,
            "subject": subj,
            "verb": verb,
            "object": obj,
            "sentence": snt
        })

    out_path = os.path.join(OUTPUT_DIR, f"enriched_{filename}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"✅ NLP enriched: {filename}")

# ------------------ Main ------------------

def run_nlp():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".json"):
            process_file(file)

if __name__ == "__main__":
    run_nlp()
