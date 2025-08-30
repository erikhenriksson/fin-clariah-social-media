import hashlib
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import silhouette_score

import hdbscan


def get_embeddings_hash(embeddings):
    """Create a hash of the embeddings to use as cache key"""
    return hashlib.md5(embeddings.tobytes()).hexdigest()[:16]


def get_cache_filename(embeddings_hash, n_components, n_neighbors, min_dist):
    """Generate cache filename based on parameters"""
    return (
        f"umap_{n_components}d_nn{n_neighbors}_md{min_dist:.3f}_{embeddings_hash}.pkl"
    )


def load_cached_umap(cache_dir, embeddings_hash, n_components, n_neighbors, min_dist):
    """Load cached UMAP reduction if it exists"""
    cache_file = os.path.join(
        cache_dir,
        get_cache_filename(embeddings_hash, n_components, n_neighbors, min_dist),
    )

    if os.path.exists(cache_file):
        print(f"Loading cached {n_components}D UMAP from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def save_umap_cache(
    cache_dir, embeddings_hash, n_components, n_neighbors, min_dist, reduced_embeddings
):
    """Save UMAP reduction to cache"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        get_cache_filename(embeddings_hash, n_components, n_neighbors, min_dist),
    )

    print(f"Saving {n_components}D UMAP to cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(reduced_embeddings, f)


def get_or_compute_umap(
    embeddings, cache_dir, embeddings_hash, n_components, n_neighbors, min_dist
):
    """Get UMAP reduction from cache or compute it"""
    # Try to load from cache first
    cached_result = load_cached_umap(
        cache_dir, embeddings_hash, n_components, n_neighbors, min_dist
    )
    if cached_result is not None:
        return cached_result

    # Compute UMAP reduction
    print(
        f"Computing {n_components}D UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})..."
    )
    umap_reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist
    )
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    # Save to cache
    save_umap_cache(
        cache_dir,
        embeddings_hash,
        n_components,
        n_neighbors,
        min_dist,
        reduced_embeddings,
    )

    return reduced_embeddings


# Get pickle file from command line
pkl_file = sys.argv[1]
filename_without_ext = os.path.splitext(os.path.basename(pkl_file))[0]
print(f"Processing file: {pkl_file}")

# Create cache directory relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "umap_cache")
print(f"Cache directory: {cache_dir}")

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

# Generate hash for embeddings to use as cache key
embeddings_hash = get_embeddings_hash(embeddings)
print(f"Embeddings hash: {embeddings_hash}")

# UMAP reduction to 50D (with caching)
embeddings_50d = get_or_compute_umap(
    embeddings,
    cache_dir,
    embeddings_hash,
    n_components=50,
    n_neighbors=30,
    min_dist=0.0,
)
print("50D reduction complete")

# UMAP reduction to 2D (with caching)
embeddings_2d = get_or_compute_umap(
    embeddings, cache_dir, embeddings_hash, n_components=2, n_neighbors=15, min_dist=0.1
)
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
        cluster_selection_epsilon=0.0,
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
    f.write(f"Embeddings hash: {embeddings_hash}\n")
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

# Print cache statistics
cache_files = [
    f for f in os.listdir(cache_dir) if f.startswith("umap_") and f.endswith(".pkl")
]
print(f"\nCache statistics:")
print(f"Cache directory: {cache_dir}")
print(f"Total cached reductions: {len(cache_files)}")
for cache_file in sorted(cache_files):
    file_size = os.path.getsize(os.path.join(cache_dir, cache_file)) / 1024 / 1024  # MB
    print(f"  {cache_file} ({file_size:.1f} MB)")

print(f"\nAnalysis complete! Results saved in {output_dir}/")
print(
    f"Final best: min_cluster_size={best_min_size}, k={best_k}, DBCV={best_score:.4f}"
)
print(
    f"Noise points: {np.sum(best_labels == -1)}/{n_samples} ({100 * np.sum(best_labels == -1) / n_samples:.1f}%)"
)
