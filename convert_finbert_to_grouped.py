import os, json, shutil
from collections import defaultdict

IN_FILE  = "data_output/finbert_signals_combined.json"
OUT_FILE = "data_output/finbert_signals_combined.json"   # overwrite in place
BACKUP   = "data_output/finbert_signals_combined.backup.json"

def norm_ticker(t): return (t or "").strip().upper()
def norm_sent(s): return (s or "").strip().lower()

with open(IN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# If already grouped, do nothing
if isinstance(data, list) and data and isinstance(data[0], dict) and "finbert_signals" in data[0]:
    print("File already in grouped format; nothing to do.")
    raise SystemExit(0)

# Expecting a flat list of dicts
grouped = defaultdict(lambda: {"title":"", "published":"", "source":"", "url":"", "finbert_signals":[]})

for row in data:
    title = (row.get("title") or "").strip()
    published = (row.get("published") or "").strip()
    key = (title, published)

    rec = grouped[key]
    rec["title"] = title
    rec["published"] = published
    rec["source"] = rec.get("source","")
    rec["url"] = rec.get("url","")

    rec["finbert_signals"].append({
        "ticker": norm_ticker(row.get("ticker","")),
        "sentiment": norm_sent(row.get("sentiment","")),
        "confidence": float(row.get("confidence", 0.0)),
        "justification": (row.get("justification") or "").strip()
    })

result = list(grouped.values())

# Backup and write
shutil.copyfile(IN_FILE, BACKUP)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"✅ Converted flat FinBERT file to grouped format with {len(result)} articles.")
print(f"🗂  Backup saved to: {BACKUP}")
