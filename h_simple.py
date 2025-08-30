import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import silhouette_score

import hdbscan

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
n_samples = len(embeddings)
print(f"Loaded {n_samples} samples with {embeddings.shape[1]}D embeddings")

# UMAP reduction to 50D
print("Reducing to 50D with UMAP...")
umap_50d = umap.UMAP(n_components=50, n_neighbors=30, min_dist=0.0)
embeddings_50d = umap_50d.fit_transform(embeddings)
print("50D reduction complete")

# UMAP reduction to 2D (from original embeddings)
print("Reducing to 2D with UMAP...")
umap_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
embeddings_2d = umap_2d.fit_transform(embeddings)
print("2D reduction complete")

# Define min_cluster_size values as percentages of dataset
percentages = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]
min_cluster_sizes = [max(2, int(n_samples * p / 100)) for p in percentages]

print(f"\nTesting {len(min_cluster_sizes)} different min_cluster_size values:")
for p, size in zip(percentages, min_cluster_sizes):
    print(f"  {p}% = {size} samples")

# Test different min_cluster_size values
best_score = -1
best_labels = None
best_min_size = None
best_k = None
all_results = []

print("\nRunning HDBSCAN with different min_cluster_size values...")
for i, (percentage, min_size) in enumerate(zip(percentages, min_cluster_sizes)):
    print(
        f"Testing {i + 1}/{len(min_cluster_sizes)}: min_cluster_size={min_size} ({percentage}%)"
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=1,  # Minimize noise
        gen_min_span_tree=True,  # Required for DBCV calculation
    )

    labels = clusterer.fit_predict(embeddings_50d)

    # Count clusters (excluding noise points labeled as -1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)

    # Use DBCV (Density-Based Cluster Validation) - better for HDBSCAN
    if n_clusters > 1:
        score = clusterer.relative_validity_  # DBCV score
    else:
        score = -1

    all_results.append((percentage, min_size, n_clusters, n_noise, score))
    print(f"  → {n_clusters} clusters, {n_noise} noise points, DBCV: {score:.4f}")

    if score > best_score and n_clusters > 1:
        best_score = score
        best_labels = labels
        best_min_size = min_size
        best_k = n_clusters

if best_labels is None:
    print("\nWarning: No valid clustering found! All results had ≤1 cluster.")
    # Use the result with most clusters as fallback
    best_result = max(all_results, key=lambda x: x[2])
    percentage, min_size, n_clusters, n_noise, score = best_result
    print(
        f"Using fallback: {percentage}% ({min_size} samples) with {n_clusters} clusters"
    )

    # Re-run with best parameters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=1,
        gen_min_span_tree=True,  # Required for DBCV calculation
    )
    best_labels = clusterer.fit_predict(embeddings_50d)
    best_min_size = min_size
    best_k = n_clusters
    best_score = score

print(
    f"\nBest result: min_cluster_size={best_min_size}, k={best_k}, DBCV={best_score:.4f}"
)

# Create output directory
output_dir = f"single_analyses/{filename_without_ext}"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving results to {output_dir}/")

# Plot 2D UMAP with clusters
print("Creating UMAP visualization...")
plt.figure(figsize=(10, 8))

# Color noise points differently
colors = best_labels.copy()
scatter = plt.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap="tab10", alpha=0.6
)

n_noise = np.sum(best_labels == -1)
plt.title(
    f"UMAP 2D with {best_k} clusters + {n_noise} noise (HDBSCAN min_size={best_min_size}, DBCV: {best_score:.3f})"
)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches="tight")
plt.close()
print("UMAP plot saved")

# Save all results sorted by silhouette score
print("Saving results summary...")
with open(f"{output_dir}/hdbscan_results.txt", "w") as f:
    f.write(f"Dataset size: {n_samples} samples\n")
    f.write(
        f"Best result: min_cluster_size={best_min_size}, k={best_k}, DBCV={best_score:.4f}\n\n"
    )
    f.write("All results (sorted by DBCV score):\n")
    sorted_results = sorted(all_results, key=lambda x: x[4], reverse=True)
    for i, (percentage, min_size, n_clusters, n_noise, score) in enumerate(
        sorted_results
    ):
        marker = " <-- BEST" if min_size == best_min_size else ""
        f.write(
            f"{i + 1}. {percentage}% ({min_size} samples): {n_clusters} clusters, {n_noise} noise, DBCV={score:.4f}{marker}\n"
        )

# Save text examples from each cluster
print("Saving cluster examples...")
with open(f"{output_dir}/cluster_examples.txt", "w") as f:
    # Handle noise points first
    noise_indices = np.where(best_labels == -1)[0]
    if len(noise_indices) > 0:
        f.write(f"\n=== NOISE POINTS ===\n")
        f.write(f"Size: {len(noise_indices)} samples\n\n")
        sample_indices = np.random.choice(
            noise_indices, min(10, len(noise_indices)), replace=False
        )
        for i, idx in enumerate(sample_indices, 1):
            f.write(f"{i}. {texts[idx]}\n")

    # Handle regular clusters
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
    f"Final best: min_cluster_size={best_min_size}, k={best_k}, DBCV={best_score:.4f}"
)
print(
    f"Noise points: {np.sum(best_labels == -1)}/{n_samples} ({100 * np.sum(best_labels == -1) / n_samples:.1f}%)"
)
