import torch, transformers, spacy, umap, hdbscan
from sentence_transformers import SentenceTransformer
from importlib.metadata import version, PackageNotFoundError

def pkg_ver(name, fallback="(unknown)"):
    try:
        return version(name)
    except PackageNotFoundError:
        return fallback

print("✓ torch", torch.__version__)
print("✓ transformers", transformers.__version__)
print("✓ spaCy", spacy.__version__)
print("✓ umap", getattr(umap, "__version__", "(unknown)"))
print("✓ hdbscan", pkg_ver("hdbscan"))

# Test loading a sentence transformer model
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✓ SBERT model loaded")
