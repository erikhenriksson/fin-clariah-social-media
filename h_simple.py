import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Get pickle file from command line
pkl_file = sys.argv[1]
filename_without_ext = os.path.splitext(os.path.basename(pkl_file))[0]

# Load data
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

# Extract data
embeddings = np.array([row["embed_last"] for row in data])
texts = [row["text"] for row in data]
preds = [row["preds"] for row in data]

# UMAP reduction to 50D
umap_50d = umap.UMAP(n_components=50, n_neighbors=30, min_dist=0.0, random_state=42)
embeddings_50d = umap_50d.fit_transform(embeddings)

# UMAP reduction to 2D (from original embeddings)
umap_2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, random_state=42)
embeddings_2d = umap_2d.fit_transform(embeddings)

# Find best clustering
best_score = -1
best_k = 2
best_labels = None
silhouette_scores = {}

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings_50d)
    score = silhouette_score(embeddings_50d, labels)
    silhouette_scores[k] = score
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

# Create output directory
output_dir = f"single_analyses/{filename_without_ext}"
os.makedirs(output_dir, exist_ok=True)

# Plot 2D UMAP with clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1], c=best_labels, cmap="tab10", alpha=0.6
)
plt.title(f"UMAP 2D with {best_k} clusters (silhouette: {best_score:.3f})")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches="tight")
plt.close()

# Save silhouette scores
with open(f"{output_dir}/silhouette_scores.txt", "w") as f:
    f.write(f"Overall best silhouette score: {best_score:.4f} (k={best_k})\n\n")
    f.write("Silhouette scores by k:\n")
    for k, score in silhouette_scores.items():
        f.write(f"k={k}: {score:.4f}\n")

# Save text examples from each cluster
with open(f"{output_dir}/cluster_examples.txt", "w") as f:
    for cluster_id in range(best_k):
        cluster_indices = np.where(best_labels == cluster_id)[0]
        f.write(f"\n=== CLUSTER {cluster_id} ===\n")
        f.write(f"Size: {len(cluster_indices)} samples\n\n")

        # Get up to 20 random examples
        sample_indices = np.random.choice(
            cluster_indices, min(20, len(cluster_indices)), replace=False
        )

        for i, idx in enumerate(sample_indices, 1):
            f.write(f"{i}. {texts[idx]}\n")

print(f"Analysis complete! Results saved in {output_dir}/")
print(f"Best clustering: k={best_k}, silhouette={best_score:.4f}")
