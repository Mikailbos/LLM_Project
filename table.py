import os
import json
from pathlib import Path
import csv

# ---- Inputs (adjust paths if needed) ----
ENRICHED_DIR = "enriched_data"
CLUSTERED_FILE = "data_output/clustered_triplets.json"
GPT_SIGNALS_FILE = "data_output/gpt_signals_combined.json"

# ---- Counters ----
num_enriched_articles = 0
total_triplets_extracted = 0
num_clustered_triplets = 0
num_gpt_signals = 0

# ---- Count enriched articles and triplets ----
if os.path.isdir(ENRICHED_DIR):
    for filename in os.listdir(ENRICHED_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(ENRICHED_DIR, filename), "r", encoding="utf-8") as f:
                articles = json.load(f)
                num_enriched_articles += len(articles)
                for article in articles:
                    if all(k in article and article[k] for k in ["subject", "verb", "object"]):
                        total_triplets_extracted += 1

# ---- Count clustered triplets ----
if os.path.exists(CLUSTERED_FILE):
    with open(CLUSTERED_FILE, "r", encoding="utf-8") as f:
        clustered_triplets = json.load(f)
        num_clustered_triplets = len(clustered_triplets)

# ---- Count GPT-generated signals (sum over per-article lists) ----
if os.path.exists(GPT_SIGNALS_FILE):
    with open(GPT_SIGNALS_FILE, "r", encoding="utf-8") as f:
        gpt_data = json.load(f)
        if isinstance(gpt_data, list) and gpt_data and isinstance(gpt_data[0], dict):
            num_gpt_signals = sum(len(a.get("gpt_signals", [])) for a in gpt_data)
        else:
            num_gpt_signals = len(gpt_data)

# ---- Derived metrics (keep it simple) ----
articles = max(num_enriched_articles, 1)
triplets = max(total_triplets_extracted, 1)

avg_triplets_per_article = round(total_triplets_extracted / articles, 2)
avg_signals_per_article = round(num_gpt_signals / articles, 2)
triplet_to_cluster_pct = f"{round(100 * num_clustered_triplets / triplets, 1)}%"
triplet_to_signal_pct  = f"{round(100 * num_gpt_signals / triplets, 1)}%"
cluster_to_signal_pct  = f"{round(100 * num_gpt_signals / max(num_clustered_triplets,1), 1)}%"

# ---- Tables (plain, neat) ----
summary_rows = [
    ("Enriched articles",           num_enriched_articles),
    ("Total triplets extracted",    total_triplets_extracted),
    ("Clustered triplets",          num_clustered_triplets),
    ("GPT-4 sentiment signals",     num_gpt_signals),
]

efficiency_rows = [
    ("Avg triplets per article",    avg_triplets_per_article),
    ("Avg GPT signals per article", avg_signals_per_article),
    ("Triplet → Cluster retention", triplet_to_cluster_pct),
    ("Triplet → Signal yield",      triplet_to_signal_pct),
    ("Cluster → Signal yield",      cluster_to_signal_pct),
]

def print_table(title, rows, col1="Metric", col2="Value"):
    # compute widths
    w1 = max(len(col1), max(len(str(r[0])) for r in rows))
    w2 = max(len(col2), max(len(str(r[1])) for r in rows))
    line = f"+-{'-'*w1}-+-{'-'*w2}-+"
    print(f"\n{title}")
    print(line)
    print(f"| {col1.ljust(w1)} | {col2.ljust(w2)} |")
    print(line)
    for k,v in rows:
        print(f"| {str(k).ljust(w1)} | {str(v).ljust(w2)} |")
    print(line)

print_table("Table: Pipeline Summary (Counts)", summary_rows)
print_table("Table: Pipeline Efficiency & Conversion", efficiency_rows)

# ---- Save simple CSVs (UTF-8) ----
OUT_DIR = Path("data_output/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_DIR / "pipeline_summary_counts.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Metric","Value"])
    w.writerows(summary_rows)

with open(OUT_DIR / "pipeline_efficiency_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Metric","Value"])
    w.writerows(efficiency_rows)

print(f"\n✅ Saved CSVs to: {OUT_DIR.resolve()}")
