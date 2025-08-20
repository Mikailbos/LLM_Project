import json
import matplotlib.pyplot as plt

# Load clustered triplets
with open("data_output/clustered_triplets.json", "r", encoding="utf-8") as f:
    triplets = json.load(f)

# Filter triplets that have 2D coordinates
x = []
y = []
labels = []

for t in triplets:
    if "embedding_2d" in t and "cluster_label" in t:
        x_val, y_val = t["embedding_2d"]
        x.append(x_val)
        y.append(y_val)
        labels.append(t["cluster_label"])

# Check if we have valid data
if not x:
    print("⚠️ No 2D embeddings found in triplets. Did you forget to save them during dimensionality reduction?")
else:
    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=labels, cmap="tab10", s=50, edgecolor='k')
    plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("2D UMAP Embeddings Colored by Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
