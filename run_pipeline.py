import subprocess
import sys

scripts = [
    "preprocessing.py",
    "nlp_processing.py",
    "triplet_extraction.py",
    "embedding_and_clustering.py",
    "GPT4_signals.py",
    "FinBERT_signals.py",
    "convert_finbert_to_grouped.py",
    "results_stats.py",
    "compare_saved.py",
    "gpt_sentiment_charts.py",
    "heatmap.py"
    ]

print("🔁 Starting full dissertation pipeline...\n")

for script in scripts:
    print(f"🚀 Running: {script}")
    try:
        result = subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {script}")
        print(e)
    print()

print("🏁 Pipeline finished.")
