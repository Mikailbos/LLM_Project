import os
import json
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are available
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ---------------------- Configuration ----------------------
INPUT_DIR = "enriched_data"
OUTPUT_CLUSTERED_FILE = "data_output/clustered_triplets.json"
OUTPUT_LABELS_FILE = "data_output/cluster_labels.json"
MODEL_NAME = "all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE = 3

# ---------------------- Step 1: Load and Prepare Triplets ----------------------
triplet_texts = []
triplet_data = []

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
        articles = json.load(f)

    for article in articles:
        subj = article.get("subject", "").strip()
        verb = article.get("verb", "").strip()
        obj = article.get("object", "").strip()

        if not subj or not verb or not obj:
            continue

        triplet_text = f"{subj} {verb} {obj}"
        triplet_texts.append(triplet_text)
        triplet_data.append({
            "triplet": triplet_text,
            "subject": subj,
            "verb": verb,
            "object": obj,
            "title": article.get("original_title", ""),
            "published": article.get("published", ""),
            "tickers": article.get("tickers", []),
            "source_file": filename
        })

print(f"🔢 Loaded {len(triplet_texts)} valid triplets.")

# ---------------------- Step 2: Embedding ----------------------
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(triplet_texts, show_progress_bar=True)

# ---------------------- Step 3: UMAP Reduction ----------------------
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embeddings_2d = umap_model.fit_transform(embeddings)

# ---------------------- Step 4: HDBSCAN Clustering ----------------------
clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)
cluster_labels = clusterer.fit_predict(embeddings_2d)

num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"🧭 Found {num_clusters} clusters.")

# ---------------------- Step 5: Assign Cluster Metadata ----------------------
clustered = []
cluster_to_terms = defaultdict(list)

for i, label in enumerate(cluster_labels):
    enriched = {
        **triplet_data[i],
        "embedding_2d": embeddings_2d[i].tolist(),
        "cluster_label": int(label)
    }
    clustered.append(enriched)

    if label != -1:
        cluster_to_terms[label].extend([
            triplet_data[i]["verb"].lower(),
            triplet_data[i]["object"].lower()
        ])

# ---------------------- Step 6: Label Clusters ----------------------
cluster_labels_dict = {}

for label, terms in cluster_to_terms.items():
    filtered = [t for t in terms if t.lower() not in stop_words]
    most_common = [word for word, _ in Counter(filtered).most_common(2)]
    label_text = ", ".join(most_common) if most_common else "Unlabelled"
    cluster_labels_dict[str(label)] = {  # Ensure key is string
        "label": label_text,
        "top_terms": most_common,
        "size": len(terms) // 2  # two terms per triplet
    }

# ---------------------- Step 7: Save Output ----------------------
os.makedirs(os.path.dirname(OUTPUT_CLUSTERED_FILE), exist_ok=True)

with open(OUTPUT_CLUSTERED_FILE, "w", encoding="utf-8") as f:
    json.dump(clustered, f, indent=2)

with open(OUTPUT_LABELS_FILE, "w", encoding="utf-8") as f:
    json.dump(cluster_labels_dict, f, indent=2)

print(f"✅ Saved clustered triplets → {OUTPUT_CLUSTERED_FILE}")
print(f"✅ Saved cluster labels → {OUTPUT_LABELS_FILE}")
