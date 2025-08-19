import os
import json
import spacy
import re

# Config
INPUT_DIR = "enriched_data"
OUTPUT_DIR = "triplets_data"
TICKER_MAP_FILE = "sp500_ticker_mapping.json"

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Words to exclude as subjects or objects
BAD_SUBJECT_OBJECTS = {"inc", "co", "ltd", "corp", "corporation", "company", "group"}

# Normalize company names and entities
def clean_entity(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(inc|co|ltd|corp|corporation|company|group|plc|llc|holdings)\b", "", text)
    return text.strip()

# Load and normalize ticker map
with open(TICKER_MAP_FILE, "r", encoding="utf-8") as f:
    raw_map = json.load(f)

ticker_map = {}
for k, v in raw_map.items():
    norm_k = clean_entity(k)
    if isinstance(v, dict) and "ticker" in v:
        ticker_map[norm_k] = v["ticker"].upper()
    elif isinstance(v, str):
        ticker_map[norm_k] = v.upper()

# Detect tickers using NER
def find_tickers_in_text(text, ticker_dict):
    doc = nlp(text)
    found_tickers = set()

    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "PRODUCT"]:
            ent_clean = clean_entity(ent.text)
            if ent_clean in ticker_dict:
                found_tickers.add(ticker_dict[ent_clean])

    return list(found_tickers)

# Extract meaningful triplets
def extract_triplets(text):
    triplets = []
    doc = nlp(text)
    for sent in doc.sents:
        for token in sent:
            if token.pos_ == "VERB":
                subject = ""
                obj = ""

                for child in token.children:
                    if "subj" in child.dep_:
                        subject = child.text.lower()
                    if "obj" in child.dep_:
                        obj = child.text.lower()

                if subject and obj and subject not in BAD_SUBJECT_OBJECTS and obj not in BAD_SUBJECT_OBJECTS:
                    triplets.append({
                        "subject": subject,
                        "verb": token.lemma_,
                        "object": obj,
                        "sentence": sent.text
                    })
    return triplets

# Process one article file
def process_file(filename):
    with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
        articles = json.load(f)

    all_triplets = []

    for article in articles:
        full_text = article.get("cleaned_article_text", "")
        published = article.get("published")
        sentiment = article.get("sentiment")
        title = article.get("original_title")

        tickers = find_tickers_in_text(full_text, ticker_map)
        sentences = [sent.text for sent in nlp(full_text).sents]

        for sent in sentences:
            triplets = extract_triplets(sent)
            for t in triplets:
                t["published"] = published
                t["sentiment"] = sentiment
                t["source_title"] = title
                t["tickers"] = tickers
                all_triplets.append(t)

    out_path = os.path.join(OUTPUT_DIR, f"triplets_{filename}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_triplets, f, indent=2)

    print(f"✅ Extracted {len(all_triplets)} triplets from: {filename}")

# Run all files
def run_triplet_extraction():
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".json"):
            process_file(file)

if __name__ == "__main__":
    run_triplet_extraction()
