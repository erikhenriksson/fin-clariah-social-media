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
print(f"Processing file: {pkl_file}")

# Load data
print("Loading data...")
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

# Extract data
embeddings = np.array([row["embed_last"] for row in data])
texts = [row["text"] for row in data]
preds = [row["preds"] for row in data]
print(f"Loaded {len(embeddings)} samples with {embeddings.shape[1]}D embeddings")

# UMAP reduction to 50D
print("Reducing to 100D with UMAP...")
umap_50d = umap.UMAP(n_components=100, n_neighbors=30, min_dist=0.0, random_state=42)
embeddings_50d = umap_50d.fit_transform(embeddings)
print("100D reduction complete")

# UMAP reduction to 2D (from original embeddings)
print("Reducing to 2D with UMAP...")
umap_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
embeddings_2d = umap_2d.fit_transform(embeddings)
print("2D reduction complete")

# Build k-NN graph for Leiden
print("Building k-NN graph...")
knn_graph = kneighbors_graph(
    embeddings_50d, n_neighbors=30, mode="connectivity", include_self=False
)
g = ig.Graph.Adjacency(knn_graph.toarray().tolist(), mode="undirected")
print(f"k-NN graph built with {g.vcount()} vertices and {g.ecount()} edges")


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
print("\nStarting coarse resolution search...")
coarse_resolutions = np.logspace(-2, 1, 20)  # 0.01 to 10
for i, res in enumerate(coarse_resolutions):
    labels, k, score = run_leiden(res)
    all_results.append((res, k, score))
    print(f"  {i + 1}/20: res={res:.4f}, k={k}, score={score:.4f}")
    if score > best_score:
        best_score = score
        best_resolution = res
        best_labels = labels
        best_k = k

print(
    f"Coarse search best: res={best_resolution:.4f}, k={best_k}, score={best_score:.4f}"
)

# Fine search around best
print("\nStarting fine search around best result...")
search_range = best_resolution * 0.5  # +/- 50% around best
fine_resolutions = np.linspace(
    max(0.001, best_resolution - search_range), best_resolution + search_range, 30
)

tested_count = 0
for res in fine_resolutions:
    if res in coarse_resolutions:  # Skip already tested
        continue
    labels, k, score = run_leiden(res)
    all_results.append((res, k, score))
    tested_count += 1
    print(f"  Fine {tested_count}: res={res:.4f}, k={k}, score={score:.4f}")
    if score > best_score:
        best_score = score
        best_resolution = res
        best_labels = labels
        best_k = k

print(
    f"Fine search best: res={best_resolution:.4f}, k={best_k}, score={best_score:.4f}"
)

# Ultra-fine search around best
print("\nStarting ultra-fine search...")
search_range = best_resolution * 0.1  # +/- 10% around best
ultra_fine_resolutions = np.linspace(
    max(0.001, best_resolution - search_range), best_resolution + search_range, 20
)

for i, res in enumerate(ultra_fine_resolutions):
    labels, k, score = run_leiden(res)
    all_results.append((res, k, score))
    print(f"  Ultra {i + 1}/20: res={res:.4f}, k={k}, score={score:.4f}")
    if score > best_score:
        best_score = score
        best_resolution = res
        best_labels = labels
        best_k = k

# Create output directory
output_dir = f"single_analyses/{filename_without_ext}"
os.makedirs(output_dir, exist_ok=True)
print(f"\nSaving results to {output_dir}/")

# Plot 2D UMAP with clusters
print("Creating UMAP visualization...")
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
print("UMAP plot saved")

# Save all results sorted by silhouette score
print("Saving silhouette scores...")
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
print("Saving cluster examples...")
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

print(f"\nAnalysis complete! Results saved in {output_dir}/")
print(
    f"Final best: resolution={best_resolution:.6f}, k={best_k}, silhouette={best_score:.4f}"
)
