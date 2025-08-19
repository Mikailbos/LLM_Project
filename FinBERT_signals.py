import os
import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Paths
INPUT_DIR = "enriched_data"
OUTPUT_FILE = "data_output/finbert_signals_combined.json"  # Single final file

# Load FinBERT model
print("Loading FinBERT...")
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

finbert = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)
print(f"Using device: {device}")

# Main execution
def generate_signals():
    all_signals = []

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
            articles = json.load(f)

        for article in tqdm(articles, desc=f"Processing {filename}"):
            tickers = article.get("tickers", [])
            text = article.get("article_text", "") or article.get("cleaned_article_text", "")
            title = article.get("original_title", article.get("title", ""))

            if not tickers or not text:
                continue

            try:
                result = finbert(text[:512])[0]  # FinBERT handles max 512 tokens
                sentiment = result["label"].lower()
                confidence = round(float(result["score"]), 4)

                # One sentiment per ticker for consistency with GPT-4 output
                for ticker in tickers:
                    all_signals.append({
                        "title": title,
                        "published": article.get("published", ""),
                        "ticker": ticker,
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "justification": f'FinBERT classified: "{text[:200]}..." as {sentiment}',
                        "source_file": filename
                    })

            except Exception as e:
                print(f"⚠️ Skipping due to error: {e}")
                continue

    # Save the combined file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_signals, f, indent=2)

    print(f"\n✅ Done! Saved {len(all_signals)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_signals()
