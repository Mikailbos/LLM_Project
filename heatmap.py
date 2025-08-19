import yfinance as yf
import json
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# Load your GPT output file
with open("data_output/gpt_signals_combined.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten signals and collect sectors
rows = []
for article in data:
    date = article.get("published", "")[:10]  # Extract YYYY-MM-DD
    for signal in article.get("gpt_signals", []):
        ticker = signal.get("ticker")
        sentiment = signal.get("sentiment")
        confidence = signal.get("confidence", 0.0)
        if not ticker or sentiment not in {"positive", "neutral", "negative"}:
            continue
        rows.append({"date": date, "ticker": ticker, "sentiment": sentiment, "confidence": confidence})

df = pd.DataFrame(rows)

# Get unique tickers
unique_tickers = df["ticker"].unique().tolist()

# Fetch sector info using yfinance
sector_map = {}
for ticker in unique_tickers:
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "Unknown")
        sector_map[ticker] = sector
    except Exception:
        sector_map[ticker] = "Unknown"

# Add sector column
df["sector"] = df["ticker"].map(sector_map)

# Group by sector and sentiment
heatmap_data = df.groupby(["sector", "sentiment"]).size().unstack(fill_value=0)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="coolwarm", linewidths=0.5, linecolor='gray')
plt.title("Sentiment Distribution per Sector")
plt.ylabel("Sector")
plt.xlabel("Sentiment")
plt.tight_layout()
plt.show()
