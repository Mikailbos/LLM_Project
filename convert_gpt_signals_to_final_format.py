import json
import re
from datetime import datetime

def extract_companies(signal_text):
    """Extract (company, sentiment) tuples from GPT signal text."""
    company_blocks = re.findall(r"(?i)(?:Company|1\.|2\.|3\.|4\.|5\.)[:\-]?\s*([^\n:]+?)[:\-]?\s*Sentiment[:\-]?\s*([^\n]+)", signal_text)
    if not company_blocks:
        # fallback to single "Company: XYZ\nSentiment: Neutral"
        match = re.search(r"Company[:\-]?\s*(.*?)\s*Sentiment[:\-]?\s*(.*?)\n", signal_text, re.IGNORECASE)
        if match:
            return [(match.group(1).strip(), match.group(2).strip())]
    return [(company.strip(), sentiment.strip()) for company, sentiment in company_blocks]

with open("data_output/gpt_company_trip_signals.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

final_signals = []

for entry in raw_data:
    article = entry.get("article_identifier", "unknown_article")
    pub_date = entry.get("published", "")
    try:
        date_obj = datetime.strptime(pub_date[:16], "%a, %d %b %Y")
        date_str = date_obj.strftime("%Y-%m-%d")
    except:
        date_str = ""

    signal_text = entry.get("signal", "")
    companies = extract_companies(signal_text)

    for company, sentiment in companies:
        final_signals.append({
            "article_id": article,
            "company": company,
            "ticker": "",  # you can optionally fill this using a mapping
            "date": date_str,
            "sentiment": sentiment.capitalize(),
            "confidence": None,
            "justification": signal_text.strip()
        })

with open("data_output/final_company_signals.json", "w", encoding="utf-8") as f:
    json.dump(final_signals, f, indent=2)

print(f"[✔] Converted {len(final_signals)} company signals.")
