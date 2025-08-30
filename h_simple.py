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
umap_50d = umap.UMAP(n_components=100, n_neighbors=30, min_dist=0.0, random_state=42)
embeddings_50d = umap_50d.fit_transform(embeddings)

# UMAP reduction to 2D (from original embeddings)
umap_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
embeddings_2d = umap_2d.fit_transform(embeddings)

# Build k-NN graph for Leiden

knn_graph = kneighbors_graph(
    embeddings_50d, n_neighbors=30, mode="connectivity", include_self=False
)
g = ig.Graph.Adjacency(knn_graph.toarray().tolist(), mode="undirected")


def run_leiden(resolution):
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=42,
    )
    labels = np.array(partition.membership)
    n_clusters = len(set(labels))
    if n_clusters > 1:
        return labels, n_clusters, silhouette_score(embeddings_50d, labels)
    else:
        return labels, n_clusters, -1


# Smart grid search - start coarse, then refine around best
best_resolution = None
best_score = -1
best_labels = None
best_k = None
all_results = []

# Coarse search
coarse_resolutions = np.logspace(-2, 1, 20)  # 0.01 to 10
for res in coarse_resolutions:
    labels, k, score = run_leiden(res)
    all_results.append((res, k, score))
    if score > best_score:
        best_score = score
        best_resolution = res
        best_labels = labels
        best_k = k

print(
    f"Coarse search best: res={best_resolution:.4f}, k={best_k}, score={best_score:.4f}"
)

# Fine search around best
search_range = best_resolution * 0.5  # +/- 50% around best
fine_resolutions = np.linspace(
    max(0.001, best_resolution - search_range), best_resolution + search_range, 30
)

for res in fine_resolutions:
    if res in coarse_resolutions:  # Skip already tested
        continue
    labels, k, score = run_leiden(res)
    all_results.append((res, k, score))
    if score > best_score:
        best_score = score
        best_resolution = res
        best_labels = labels
        best_k = k

print(
    f"Fine search best: res={best_resolution:.4f}, k={best_k}, score={best_score:.4f}"
)

# Ultra-fine search around best
search_range = best_resolution * 0.1  # +/- 10% around best
ultra_fine_resolutions = np.linspace(
    max(0.001, best_resolution - search_range), best_resolution + search_range, 20
)

for res in ultra_fine_resolutions:
    labels, k, score = run_leiden(res)
    all_results.append((res, k, score))
    if score > best_score:
        best_score = score
        best_resolution = res
        best_labels = labels
        best_k = k

# Create output directory
output_dir = f"single_analyses/{filename_without_ext}"
os.makedirs(output_dir, exist_ok=True)

# Plot 2D UMAP with clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1], c=best_labels, cmap="tab10", alpha=0.6
)
plt.title(
    f"UMAP 2D with {best_k} clusters (Leiden res={best_resolution:.4f}, silhouette: {best_score:.3f})"
)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches="tight")
plt.close()

# Save all results sorted by silhouette score
with open(f"{output_dir}/silhouette_scores.txt", "w") as f:
    f.write(
        f"Best result: resolution={best_resolution:.6f}, k={best_k}, silhouette={best_score:.4f}\n\n"
    )
    f.write("All results (top 20 by silhouette score):\n")
    sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
    for i, (res, k, score) in enumerate(sorted_results[:20]):
        marker = " <-- BEST" if res == best_resolution else ""
        f.write(
            f"{i + 1}. resolution={res:.6f}, k={k}, silhouette={score:.4f}{marker}\n"
        )

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
print(
    f"Final best: resolution={best_resolution:.6f}, k={best_k}, silhouette={best_score:.4f}"
)
