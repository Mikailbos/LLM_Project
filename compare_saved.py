import os
import json
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher

# === Config ===
GPT_FILE = "data_output/gpt_signals_combined.json"
FINBERT_FILE = "data_output/finbert_signals_combined.json"
RESULTS_DIR = "evaluation_results"
FUZZY_MATCH_THRESHOLD = 0.9  # 90% similarity to match titles

os.makedirs(RESULTS_DIR, exist_ok=True)

# === Helpers ===
def slug_title(t: str) -> str:
    """Normalise title for comparison."""
    if not t:
        return ""
    t = t.strip().lower()
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"\s+", " ", t)
    return t

def norm_ticker(x: str) -> str:
    return (x or "").strip().upper()

def load_json(path):
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to parse {path}: {e}")
            return []

def extract_signal_map(data, key_field):
    """Return {(title_slug, ticker): {sentiment, confidence}}"""
    m = {}
    for entry in data:
        title_slug = slug_title(entry.get("title", ""))
        for sig in entry.get(key_field, []):
            ticker = norm_ticker(sig.get("ticker", ""))
            sentiment = (sig.get("sentiment", "") or "").strip().lower()
            confidence = float(sig.get("confidence", 0.0))
            if title_slug and ticker and sentiment:
                m[(title_slug, ticker)] = {
                    "sentiment": sentiment,
                    "confidence": confidence
                }
    return m

def fuzzy_match_title(t1, t2):
    """Return True if titles are similar enough."""
    return SequenceMatcher(None, t1, t2).ratio() >= FUZZY_MATCH_THRESHOLD

# === Load data ===
gpt_data = load_json(GPT_FILE)
finbert_data = load_json(FINBERT_FILE)

gpt_map = extract_signal_map(gpt_data, "gpt_signals")
finbert_map = extract_signal_map(finbert_data, "finbert_signals")

# === Compare sentiments ===
records = []
gpt_titles = {k[0] for k in gpt_map.keys()}
finbert_titles = {k[0] for k in finbert_map.keys()}

for (gpt_title, gpt_ticker), gpt_vals in gpt_map.items():
    # Try exact match first
    if (gpt_title, gpt_ticker) in finbert_map:
        fin = finbert_map[(gpt_title, gpt_ticker)]
    else:
        # Try fuzzy title matching
        match_title = None
        for f_title in finbert_titles:
            if fuzzy_match_title(gpt_title, f_title):
                if (f_title, gpt_ticker) in finbert_map:
                    match_title = f_title
                    break
        if match_title:
            fin = finbert_map[(match_title, gpt_ticker)]
        else:
            continue  # No match found

    records.append({
        "title": gpt_title,
        "ticker": gpt_ticker,
        "gpt_sentiment": gpt_vals["sentiment"],
        "finbert_sentiment": fin["sentiment"],
        "gpt_confidence": gpt_vals["confidence"],
        "finbert_confidence": fin["confidence"],
        "match": gpt_vals["sentiment"] == fin["sentiment"]
    })

df = pd.DataFrame(records)

# === Handle no overlaps ===
if df.empty:
    print("ℹ️ No overlapping (title, ticker) pairs found between GPT and FinBERT outputs.")
    pd.DataFrame([]).to_csv(f"{RESULTS_DIR}/disagreements.csv", index=False)
    pd.DataFrame([]).to_csv(f"{RESULTS_DIR}/comparison_full.csv", index=False)
    with open(f"{RESULTS_DIR}/comparison_summary.txt", "w", encoding="utf-8") as f:
        f.write("No overlaps found.\n")
    raise SystemExit(0)

# === Summary ===
total = len(df)
matches = int(df["match"].sum())
summary_lines = [
    f"✅ Total matched (title + ticker): {total}",
    f"✅ Agreement: {matches} ({matches/total:.2%})",
    f"❌ Disagreement: {total - matches} ({(total - matches)/total:.2%})\n"
]
print("\n".join(summary_lines))

# === Classification report ===
labels = ["positive", "neutral", "negative"]
y_true = df["gpt_sentiment"]
y_pred = df["finbert_sentiment"]
report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
print("\n📋 Classification Report (FinBERT vs GPT-4):")
print(report)

# === Save summary and report ===
with open(f"{RESULTS_DIR}/comparison_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
    f.write("\n📋 Classification Report (FinBERT vs GPT-4):\n")
    f.write(report)

# === Save CSVs ===
df[df["match"] == False].to_csv(f"{RESULTS_DIR}/disagreements.csv", index=False)
df.to_csv(f"{RESULTS_DIR}/comparison_full.csv", index=False)

# === Confusion matrix ===
cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("FinBERT Prediction")
plt.ylabel("GPT-4 Ground Truth")
plt.title("Confusion Matrix: FinBERT vs GPT-4")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
plt.show()

# === Violin plot ===
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df.melt(id_vars=["match"], value_vars=["gpt_confidence", "finbert_confidence"]),
    x="variable", y="value", hue="match", split=True
)
plt.title("Confidence Score Distribution by Model and Agreement")
plt.ylabel("Confidence")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confidence_violin_plot.png")
plt.show()

# === Example disagreements ===
example_disagreements = df[df["match"] == False].sample(min(5, len(df[df["match"] == False])), random_state=42)[
    ["title", "ticker", "gpt_sentiment", "finbert_sentiment"]
]
print("\n🔍 Example disagreements:")
print(example_disagreements.to_string(index=False))
example_disagreements.to_csv(f"{RESULTS_DIR}/sample_disagreements.csv", index=False)
