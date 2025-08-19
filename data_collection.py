import os
import json
import time
import feedparser
import pandas as pd
from datetime import datetime
import yfinance as yf

OUTPUT_DIR = "data_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper to clean RSS entries
def clean_entry(entry):
    return {
        "title": entry.get("title", ""),
        "link": entry.get("link", ""),
        "published": entry.get("published", "")
    }

# Generic fetcher
def fetch_rss(name, url, filename, max_articles=40):
    print(f"\n📰 Fetching {name} RSS")
    feed = feedparser.parse(url)

    if not feed.entries:
        print(f"⚠️ {name}: 0 articles (empty feed or error)")
        return

    articles = [clean_entry(entry) for entry in feed.entries[:max_articles]]
    print(f"✅ {name}: {len(articles)} articles → {filename}")

    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2)

# Specific news fetchers
def fetch_yahoo_finance_news():     fetch_rss("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US", "yahoo_finance_news.json")
def fetch_cnbc_news():              fetch_rss("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html", "cnbc_news.json")
def fetch_marketwatch_news():      fetch_rss("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/", "marketwatch_news.json")
def fetch_investopedia_news():     fetch_rss("Investopedia", "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline", "investopedia_news.json")
def fetch_motley_fool_news():      fetch_rss("Motley Fool", "https://www.fool.com/feeds/index.aspx?type=headline", "motley_fool_news.json")

# Yahoo finance for tickers
def fetch_yahoo_finance(ticker):
    print(f"\n📈 Fetching data for: {ticker}")
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        hist = stock.history(period="5d")

        data = {
            "ticker": ticker,
            "name": info.get("shortName", ""),
            "price": info.get("currentPrice", ""),
            "previousClose": info.get("previousClose", ""),
            "sector": info.get("sector", ""),
            "history": {
                str(date): row.to_dict() for date, row in hist.iterrows()
            }
        }

        with open(os.path.join(OUTPUT_DIR, f"{ticker}_finance.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"✔️ {ticker}: {data['price']} ({data['name']})")

    except Exception as e:
        print(f"❌ {ticker}: Failed to fetch — {e}")

# Scrape S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    return tables[0]['Symbol'].tolist()

# Main runner
def run_data_collection(skip_tickers=False):
    print(f"\n🕒 Running on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # News
    fetch_yahoo_finance_news()
    fetch_cnbc_news()
    fetch_marketwatch_news()
    fetch_investopedia_news()
    fetch_motley_fool_news()

    # Ticker data
    if not skip_tickers:
        tickers = get_sp500_tickers()
        print(f"\n✅ Found {len(tickers)} tickers")
        for idx, ticker in enumerate(tickers):
            fetch_yahoo_finance(ticker)
            time.sleep(1.5)

    print("\n✅ Done! All data saved in:", OUTPUT_DIR)

# Entrypoint
if __name__ == "__main__":
    SKIP_TICKERS = os.getenv("SKIP_TICKERS", "false").lower() == "true"
    run_data_collection(skip_tickers=SKIP_TICKERS)

