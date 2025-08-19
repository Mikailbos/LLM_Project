import json
import re
import collections
import itertools
import csv
from pathlib import Path

STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "by", "with", "at", "from",
    "inc", "corp", "ltd", "plc", "sa", "llc", "co", "company", "group", "&"
}

def norm_token(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\-/\.]+", "", t)
    return t.strip("-./")

def split_nouns(text: str):
    toks = [norm_token(t) for t in re.split(r"\s+", text or "")]
    return [t for t in toks if t and t not in STOPWORDS and not t.isdigit()]

def get_field(d, *names, default=""):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default

def summarize_clusters(path, topk=10, out_csv="cluster_keywords_summary.csv"):
    data = json.loads(Path(path).read_text())
    verb_counts = collections.defaultdict(collections.Counter)
    noun_counts = collections.defaultdict(collections.Counter)

    for row in data:
        c = row.get("cluster", row.get("cluster_id", -1))
        v = get_field(row, "verb", "relation", "pred", default="")
        v = norm_token(v)
        if v:
            verb_counts[c][v] += 1

        subj = get_field(row, "subject", "head", "subj", default="")
        obj = get_field(row, "object", "tail", "obj", default="")
        companies = row.get("companies") or row.get("company_mentions") or []
        tickers = row.get("tickers") or []
        noun_tokens = split_nouns(subj) + split_nouns(obj)
        noun_tokens += [norm_token(x) for x in itertools.chain(companies, tickers)]
        noun_tokens = [t for t in noun_tokens if t]
        noun_counts[c].update(noun_tokens)

    clusters = sorted(set(verb_counts.keys()) | set(noun_counts.keys()))

    # Write to CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster ID", "Top Verbs", "Top Nouns"])
        for c in clusters:
            top_verbs = ", ".join([f"{w} ({n})" for w, n in verb_counts[c].most_common(topk)])
            top_nouns = ", ".join([f"{w} ({n})" for w, n in noun_counts[c].most_common(topk)])
            writer.writerow([c, top_verbs, top_nouns])

    print(f"✅ Saved cluster keyword summary to {out_csv}")

if __name__ == "__main__":
    summarize_clusters("data_output/clustered_triplets.json", topk=10)
