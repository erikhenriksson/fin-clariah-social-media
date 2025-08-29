import os
import pickle
import sys

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

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

# Build k-NN graph for Leiden
knn_graph = kneighbors_graph(
    embeddings_50d, n_neighbors=30, mode="connectivity", include_self=False
)
sources, targets = knn_graph.nonzero()
g = ig.Graph(directed=False)
g.add_vertices(embeddings_50d.shape[0])
g.add_edges(list(zip(sources, targets)))

# Grid search for resolutions that give k=2-8 clusters
resolutions = np.logspace(-2, 1, 50)  # 0.01 to 10
target_ks = list(range(2, 9))
best_resolutions = {}
all_results = {}

for target_k in target_ks:
    best_res = None
    best_diff = float("inf")

    for resolution in resolutions:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=42,
        )
        n_clusters = len(set(partition.membership))
        diff = abs(n_clusters - target_k)

        if diff < best_diff:
            best_diff = diff
            best_res = resolution

    best_resolutions[target_k] = best_res

# Run Leiden with best resolutions and calculate silhouette scores
best_score = -1
best_k = 2
best_labels = None
best_resolution = None

for target_k in target_ks:
    resolution = best_resolutions[target_k]
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=42,
    )
    labels = np.array(partition.membership)
    actual_k = len(set(labels))

    if actual_k > 1:
        score = silhouette_score(embeddings_50d, labels)
        all_results[target_k] = {
            "resolution": resolution,
            "actual_k": actual_k,
            "silhouette": score,
        }

        if score > best_score:
            best_score = score
            best_k = target_k
            best_labels = labels
            best_resolution = resolution

# Create output directory
output_dir = f"single_analyses/{filename_without_ext}"
os.makedirs(output_dir, exist_ok=True)

# Plot 2D UMAP with clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1], c=best_labels, cmap="tab10", alpha=0.6
)
plt.title(
    f"UMAP 2D with {len(set(best_labels))} clusters (Leiden, silhouette: {best_score:.3f})"
)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches="tight")
plt.close()

# Save silhouette scores
with open(f"{output_dir}/silhouette_scores.txt", "w") as f:
    f.write(
        f"Overall best silhouette score: {best_score:.4f} (target_k={best_k}, resolution={best_resolution:.4f})\n\n"
    )
    f.write("Results by target k:\n")
    for target_k, result in all_results.items():
        marker = " <-- BEST" if target_k == best_k else ""
        f.write(
            f"target_k={target_k}: resolution={result['resolution']:.4f}, actual_k={result['actual_k']}, silhouette={result['silhouette']:.4f}{marker}\n"
        )

# Save text examples from each cluster
with open(f"{output_dir}/cluster_examples.txt", "w") as f:
    actual_k = len(set(best_labels))
    for cluster_id in range(actual_k):
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
print(
    f"Best clustering: target_k={best_k}, actual_k={len(set(best_labels))}, resolution={best_resolution:.4f}, silhouette={best_score:.4f}"
)
