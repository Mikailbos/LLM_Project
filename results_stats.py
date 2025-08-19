import os
import json

# Define input directories (adjust paths as needed)
ENRICHED_DIR = "enriched_data"
CLUSTERED_FILE = "data_output/clustered_triplets.json"
GPT_SIGNALS_FILE = "data_output/gpt_signals_combined.json"

# Initialise counters
num_enriched_articles = 0
total_triplets_extracted = 0
num_clustered_triplets = 0
num_gpt_signals = 0

# Count enriched articles and triplets
for filename in os.listdir(ENRICHED_DIR):
    if filename.endswith(".json"):
        with open(os.path.join(ENRICHED_DIR, filename), "r", encoding="utf-8") as f:
            articles = json.load(f)
            num_enriched_articles += len(articles)
            for article in articles:
                if all(k in article and article[k] for k in ["subject", "verb", "object"]):
                    total_triplets_extracted += 1

# Count clustered triplets
if os.path.exists(CLUSTERED_FILE):
    with open(CLUSTERED_FILE, "r", encoding="utf-8") as f:
        clustered_triplets = json.load(f)
        num_clustered_triplets = len(clustered_triplets)

# Count GPT-generated signals
if os.path.exists(GPT_SIGNALS_FILE):
    with open(GPT_SIGNALS_FILE, "r", encoding="utf-8") as f:
        gpt_signals = json.load(f)
        num_gpt_signals = len(gpt_signals)

# Print the results
print(f"Enriched Articles: {num_enriched_articles}")
print(f"Total Triplets Extracted: {total_triplets_extracted}")
print(f"Clustered Triplets: {num_clustered_triplets}")
print(f"GPT Signals Generated: {num_gpt_signals}")




