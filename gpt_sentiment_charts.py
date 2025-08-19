import json
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─── Helper: Robust date parsing ───────────────────────────────────────────────
def parse_published_date(published):
    for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"]:
        try:
            return datetime.strptime(published, fmt).date()
        except ValueError:
            continue
    print(f"⚠️ Date parse failed: {published}")
    return None

# ─── Load Data ─────────────────────────────────────────────────────────────────
with open("data_output/gpt_signals_combined.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ─── Containers ────────────────────────────────────────────────────────────────
ticker_counter = Counter()
sentiment_counter = Counter()
sentiment_by_date = defaultdict(Counter)

# ─── Parse and Aggregate ───────────────────────────────────────────────────────
for entry in data:
    date = parse_published_date(entry.get("published", ""))
    if not date:
        continue

    for signal in entry.get("gpt_signals", []):
        ticker = signal.get("ticker", "").strip().upper()
        sentiment = signal.get("sentiment", "").strip().capitalize()

        if ticker:
            ticker_counter[ticker] += 1
        if sentiment:
            sentiment_counter[sentiment] += 1
            sentiment_by_date[date][sentiment] += 1

# ─── Plot 1: Top 10 Most Mentioned Tickers ─────────────────────────────────────
top_tickers = ticker_counter.most_common(10)
if top_tickers:
    tickers, counts = zip(*top_tickers)
    plt.figure(figsize=(12, 6))
    plt.bar(tickers, counts, color='skyblue')
    plt.title("Top 10 Most Mentioned Tickers (GPT Signals)")
    plt.xlabel("Ticker")
    plt.ylabel("Mention Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No tickers found.")

# ─── Plot 2: Sentiment Distribution (Pie Chart) ────────────────────────────────
if sentiment_counter:
    labels = list(sentiment_counter.keys())
    sizes = list(sentiment_counter.values())
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Sentiment Distribution (GPT Signals)")
    plt.tight_layout()
    plt.show()
else:
    print("No sentiment data.")

# ─── Plot 3: Sentiment Over Time (Dynamic Line/Bar) ────────────────────────────
if sentiment_by_date:
    all_dates = sorted(sentiment_by_date)
    sentiment_types = ["Positive", "Negative", "Neutral", "Mixed"]

    if len(all_dates) == 1:
        # Bar chart for single-day data
        date = all_dates[0]
        counts = [sentiment_by_date[date].get(s, 0) for s in sentiment_types]

        plt.figure(figsize=(8, 5))
        plt.bar(sentiment_types, counts, color=["blue", "orange", "green", "red"][:len(counts)])
        plt.title(f"Sentiment Breakdown on {date}")
        plt.xlabel("Sentiment")
        plt.ylabel("Signal Count")
        plt.tight_layout()
        plt.show()
    else:
        # Line chart for multi-day data
        plt.figure(figsize=(12, 6))
        for s in sentiment_types:
            y = [sentiment_by_date[d].get(s, 0) for d in all_dates]
            if any(y):
                plt.plot(all_dates, y, label=s, marker="o")

        plt.title("Sentiment Trend Over Time (GPT Signals)")
        plt.xlabel("Date")
        plt.ylabel("Signal Count")
        plt.xticks(rotation=45)
        plt.xlim(min(all_dates), max(all_dates))
        plt.ylim(bottom=0)
        plt.legend()
        plt.tight_layout()
        plt.show()
else:
    print("No sentiment-by-date data.")

# ─── Plot 4: Signal Volume Over Time (Hourly) ──────────────────────────────────
timestamps = []
for entry in data:
    pub_date = entry.get("published", "").strip()
    try:
        dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S GMT")
        hour_key = dt.replace(minute=0, second=0, microsecond=0)
        for _ in entry.get("gpt_signals", []):
            timestamps.append(hour_key)
    except Exception:
        continue

counts = Counter(timestamps)
sorted_items = sorted(counts.items())

x = [dt for dt, _ in sorted_items]
y = [count for _, count in sorted_items]

if x:
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o')
    plt.title("Signal Volume Over Time")
    plt.xlabel("Date/Hour")
    plt.ylabel("Number of Signals")
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No valid timestamps found for volume plot.")
