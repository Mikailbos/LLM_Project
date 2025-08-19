# GPT4_signals.py
# Generates company-level sentiment signals from enriched/clustered triplet data
# Output: data_output/gpt_signals_combined.json

import os, json, re, time, hashlib
from typing import Any, Dict, List

# ── Config ─────────────────────────────────────────────────────────────
ENRICHED_DIR = "enriched_data"
CLUSTERED_FILE = "data_output/clustered_triplets.json"
DATA_DIR = "data_output"
OUTPUT_FILE = os.path.join(DATA_DIR, "gpt_signals_combined.json")
MODEL_NAME = "gpt-4o-mini"  # Change to your preferred GPT model
RATE_LIMIT_SLEEP = 0.8

os.makedirs(DATA_DIR, exist_ok=True)

# ── OpenAI client ───────────────────────────────────────────────────────
try:
    from openai import OpenAI
    client = OpenAI()  # Uses OPENAI_API_KEY from environment
except Exception as e:
    raise SystemExit("OpenAI client not available. Install `openai` and set OPENAI_API_KEY.") from e

# ── Utilities ───────────────────────────────────────────────────────────
def load_enriched_lookup() -> Dict[str, Dict[str, Any]]:
    """Loads enriched articles keyed by (title|published)."""
    lookup = {}
    if not os.path.isdir(ENRICHED_DIR):
        return lookup
    for fn in os.listdir(ENRICHED_DIR):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(ENRICHED_DIR, fn), "r", encoding="utf-8") as f:
            try:
                articles = json.load(f)
            except Exception:
                continue
            for art in articles:
                title = (art.get("title") or "").strip()
                pub = (art.get("published") or "").strip()
                key = f"{title}|{pub}" if title else hashlib.md5(json.dumps(art, sort_keys=True).encode()).hexdigest()
                if key not in lookup or len(json.dumps(art)) > len(json.dumps(lookup[key])):
                    lookup[key] = art
    return lookup

def safe_json_extract(text: str) -> Any:
    """Extract JSON object/array from GPT output text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if m:
        snippet = m.group(1)
        snippet = re.sub(r',\s*([}\]])', r'\1', snippet)
        return json.loads(snippet)
    raise ValueError("No JSON found in GPT output")

def normalise_ticker(t: str) -> str:
    return (t or "").strip().upper()

def build_prompt(item: Dict[str, Any], article_ctx: Dict[str, Any]) -> str:
    """Creates the GPT prompt."""
    title = article_ctx.get("title", "")
    pub = article_ctx.get("published", "")
    source = article_ctx.get("source", "")
    tickers = item.get("tickers") or article_ctx.get("tickers") or []
    tickers = [normalise_ticker(t) for t in tickers if t]

    triplet = item.get("triplet") or {
        "subject": item.get("subject"),
        "verb": item.get("verb"),
        "object": item.get("object")
    }
    sentence = item.get("sentence") or article_ctx.get("sentence") or ""
    cluster_label = item.get("cluster_label") or item.get("label") or ""
    textblob_polarity = item.get("polarity") or item.get("textblob_polarity")

    return (
        "You are a financial NLP analyst. Analyse the given news context and return "
        "only companies that are clearly affected.\n"
        "Return ONLY JSON: a list of objects with keys: ticker (string), sentiment "
        "(positive/neutral/negative), confidence (0-1 float), justification (<=30 words).\n\n"
        f"ArticleTitle: {title}\n"
        f"Published: {pub}\n"
        f"Source: {source}\n"
        f"CandidateTickers: {tickers}\n"
        f"Triplet: {triplet}\n"
        f"SentenceContext: {sentence}\n"
        f"ClusterLabel: {cluster_label}\n"
        f"TextBlobPolarity: {textblob_polarity}\n\n"
        "JSON output spec:\n"
        "[\n"
        '  {"ticker":"AAPL","sentiment":"positive","confidence":0.83,"justification":"<why in <=30 words>"},\n'
        "  ...\n"
        "]"
    )

def call_model(prompt: str) -> List[Dict[str, Any]]:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    data = safe_json_extract(text)

    if not isinstance(data, list):
        raise ValueError("Model did not return a JSON list.")

    cleaned = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        ticker = normalise_ticker(obj.get("ticker", ""))
        sent = (obj.get("sentiment") or "").strip().lower()
        if ticker and sent in {"positive", "neutral", "negative"}:
            conf = float(obj.get("confidence", 0.0))
            just = (obj.get("justification") or "").strip()
            cleaned.append({
                "ticker": ticker,
                "sentiment": sent,
                "confidence": conf,
                "justification": just[:200]
            })
    return cleaned

def load_existing() -> List[Dict[str, Any]]:
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def write_outputs(records: List[Dict[str, Any]]):
    tmp = OUTPUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_FILE)

def stable_key(title: str, published: str) -> str:
    return f"{title}|{published}"

# ── Load data ───────────────────────────────────────────────────────────
enriched_lookup = load_enriched_lookup()

cluster_items: List[Dict[str, Any]] = []
if os.path.exists(CLUSTERED_FILE):
    with open(CLUSTERED_FILE, "r", encoding="utf-8") as f:
        cluster_items = json.load(f)
else:
    for art in enriched_lookup.values():
        cluster_items.append({
            "title": art.get("title"),
            "published": art.get("published"),
            "tickers": art.get("tickers") or [],
            "sentence": art.get("sentence"),
            "triplet": art.get("triplet"),
            "polarity": art.get("polarity"),
        })

existing = load_existing()
existing_by_key = {stable_key(e.get("title",""), e.get("published","")): e for e in existing}

output_by_key = dict(existing_by_key)
processed = 0
created_now = 0

for item in cluster_items:
    title = (item.get("title") or "").strip()
    published = (item.get("published") or "").strip()
    key = stable_key(title, published)

    article_ctx = enriched_lookup.get(key, {"title": title, "published": published})

    if key in output_by_key and output_by_key[key].get("gpt_signals"):
        continue

    prompt = build_prompt(item, article_ctx)
    try:
        signals = call_model(prompt)
    except Exception as e:
        signals = []
        article_ctx["error"] = str(e)

    record = {
        "title": article_ctx.get("title",""),
        "published": article_ctx.get("published",""),
        "source": article_ctx.get("source",""),
        "url": article_ctx.get("url",""),
        "gpt_signals": signals
    }
    output_by_key[key] = record
    created_now += 1
    processed += 1

    if processed % 5 == 0:
        write_outputs(list(output_by_key.values()))

    time.sleep(RATE_LIMIT_SLEEP)

# Final write
final_records = list(output_by_key.values())
write_outputs(final_records)

print(f"✅ GPT signals written: {len(final_records)} total "
      f"(new: {created_now}, resumed: {len(existing)})")
print(f"📄 {OUTPUT_FILE}")
