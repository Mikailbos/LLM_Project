# save_sp500_mapping.py

import pandas as pd
import json
import re

# Common suffixes to remove
SUFFIXES = r"(inc|incorporated|corporation|corp|co|ltd|plc|llc|group|holdings|company|limited|class a|class b|class c)"

def normalize(name):
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)  # Remove punctuation
    name = re.sub(rf"\b{SUFFIXES}\b", "", name)
    name = re.sub(r"\s+", " ", name)  # Remove extra spaces
    return name.strip()

def generate_aliases(name):
    name = name.lower()
    base = re.sub(r"[^\w\s]", "", name)  # Remove punctuation
    base = re.sub(rf"\b{SUFFIXES}\b", "", base).strip()
    aliases = set()

    # Add base name, base without spaces, and common simplifications
    aliases.add(base)
    aliases.add(base.replace(" ", ""))
    aliases.add(re.sub(r"\s+(inc|corp|llc|ltd)$", "", base))  # tail strip
    aliases.add(base.split()[0])  # just first word

    return {a.strip() for a in aliases if a.strip()}

# Load from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
df = pd.read_html(url)[0]
df = df.dropna(subset=["Security", "Symbol"])
df["Security"] = df["Security"].str.strip()
df["Symbol"] = df["Symbol"].str.strip().str.upper()

# Build mapping: variation → ticker
variation_to_ticker = {}

for _, row in df.iterrows():
    name = row["Security"]
    ticker = row["Symbol"]

    aliases = generate_aliases(name)
    for alias in aliases:
        variation_to_ticker[alias] = ticker

# Save result
with open("sp500_ticker_mapping.json", "w", encoding="utf-8") as f:
    json.dump(variation_to_ticker, f, indent=2)

print(f"✅ Saved {len(variation_to_ticker)} normalized company name variations.")
