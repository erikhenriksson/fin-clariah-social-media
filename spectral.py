import hashlib
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import rbf_kernel


def get_embeddings_hash(embeddings):
    """Create a hash of the embeddings to use as cache key"""
    return hashlib.md5(embeddings.tobytes()).hexdigest()[:16]


def get_cache_filename(embeddings_hash, n_components, n_neighbors, min_dist):
    """Generate cache filename based on parameters"""
    return (
        f"umap_{n_components}d_nn{n_neighbors}_md{min_dist:.3f}_{embeddings_hash}.pkl"
    )


def get_spectral_cache_filename(embeddings_hash, n_clusters, gamma):
    """Generate cache filename for Spectral clustering results"""
    return f"spectral_k{n_clusters}_gamma{gamma:.6f}_{embeddings_hash}.pkl"


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


def load_cached_spectral(cache_dir, embeddings_hash, n_clusters, gamma):
    """Load cached Spectral clustering result if it exists"""
    cache_file = os.path.join(
        cache_dir, get_spectral_cache_filename(embeddings_hash, n_clusters, gamma)
    )

    if os.path.exists(cache_file):
        print(f"Loading cached Spectral result (k={n_clusters}, gamma={gamma:.6f})")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def save_spectral_cache(cache_dir, embeddings_hash, n_clusters, gamma, result_data):
    """Save Spectral clustering result to cache"""
    cache_file = os.path.join(
        cache_dir, get_spectral_cache_filename(embeddings_hash, n_clusters, gamma)
    )

    print(f"Saving Spectral result to cache (k={n_clusters}, gamma={gamma:.6f})")
    with open(cache_file, "wb") as f:
        pickle.dump(result_data, f)


def estimate_gamma(embeddings, sample_size=1000):
    """Estimate a good gamma value for RBF kernel based on data"""
    # Sample data for efficiency
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[idx]
    else:
        sample_embeddings = embeddings

    # Calculate pairwise distances
    from sklearn.metrics.pairwise import euclidean_distances

    distances = euclidean_distances(sample_embeddings)

    # Use median distance as basis for gamma
    # gamma = 1 / (2 * median_distance^2) is a common heuristic
    median_dist = np.median(distances[distances > 0])
    gamma = 1.0 / (2 * median_dist**2)

    print(f"Estimated gamma: {gamma:.6f} (based on median distance: {median_dist:.4f})")
    return gamma


def get_or_compute_spectral(cache_dir, embeddings_hash, embeddings, n_clusters, gamma):
    """Get Spectral clustering result from cache or compute it"""
    # Try to load from cache first
    cached_result = load_cached_spectral(cache_dir, embeddings_hash, n_clusters, gamma)
    if cached_result is not None:
        return cached_result

    # Compute Spectral clustering
    print(f"Computing Spectral clustering (k={n_clusters}, gamma={gamma:.6f})...")

    # Use RBF (Gaussian) kernel with estimated gamma
    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity="rbf", gamma=gamma, n_init=10, random_state=42
    )

    labels = spectral.fit_predict(embeddings)

    # Calculate metrics
    if n_clusters > 1:
        sil_score = silhouette_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
    else:
        sil_score = -1
        ch_score = -1

    # Package result
    result_data = {
        "labels": labels,
        "n_clusters": n_clusters,
        "silhouette_score": sil_score,
        "ch_score": ch_score,
        "gamma": gamma,
    }

    # Save to cache
    save_spectral_cache(cache_dir, embeddings_hash, n_clusters, gamma, result_data)

    return result_data


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

# UMAP reduction to 2D for visualization only (with caching)
embeddings_2d = get_or_compute_umap(
    embeddings, cache_dir, embeddings_hash, n_components=2, n_neighbors=15, min_dist=0.1
)
print("2D reduction complete")

# Estimate optimal gamma for RBF kernel
gamma = estimate_gamma(embeddings)

# Test different numbers of clusters (1-8)
k_values = list(range(1, 9))

print(f"\nTesting {len(k_values)} different cluster numbers: {k_values}")

# Test different k values
best_sil_score = -1
best_ch_score = -1
best_labels = None
best_k = None
all_results = []

print("\nRunning Spectral clustering with different k values...")
for i, k in enumerate(k_values):
    print(f"Testing {i + 1}/{len(k_values)}: k={k}")

    # Get Spectral clustering result from cache or compute it
    result = get_or_compute_spectral(cache_dir, embeddings_hash, embeddings, k, gamma)

    labels = result["labels"]
    sil_score = result["silhouette_score"]
    ch_score = result["ch_score"]

    all_results.append((k, sil_score, ch_score))

    if k == 1:
        print(f"  → {k} cluster (no metrics calculated)")
    else:
        print(f"  → {k} clusters, Silhouette: {sil_score:.4f}, CH: {ch_score:.2f}")

    # Track best results (skip k=1 since no meaningful metrics)
    if k > 1 and sil_score > best_sil_score:
        best_sil_score = sil_score
        best_labels = labels
        best_k = k
        best_ch_score = ch_score

# If no valid clustering found, use k=2 as fallback
if best_labels is None:
    print("\nNo valid clustering found! Using k=2 as fallback.")
    fallback_result = get_or_compute_spectral(
        cache_dir, embeddings_hash, embeddings, 2, gamma
    )
    best_labels = fallback_result["labels"]
    best_k = 2
    best_sil_score = fallback_result["silhouette_score"]
    best_ch_score = fallback_result["ch_score"]

print(
    f"\nBest result: k={best_k}, Silhouette={best_sil_score:.4f}, CH={best_ch_score:.2f}"
)

# Create output directory
output_dir = f"single_analyses/{filename_without_ext}"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving results to {output_dir}/")

# Plot 2D UMAP with clusters
print("Creating UMAP visualization...")
plt.figure(figsize=(10, 8))

# Color clusters
colors = best_labels.copy()
scatter = plt.scatter(
    embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap="tab10", alpha=0.6
)

plt.title(
    f"UMAP 2D with {best_k} clusters (Spectral k={best_k}, Silhouette: {best_sil_score:.3f})"
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
with open(f"{output_dir}/spectral_results.txt", "w") as f:
    f.write(f"Dataset size: {n_samples} samples\n")
    f.write(f"Embeddings hash: {embeddings_hash}\n")
    f.write(f"Gamma parameter: {gamma:.6f}\n")
    f.write(
        f"Best result: k={best_k}, Silhouette={best_sil_score:.4f}, Calinski-Harabasz={best_ch_score:.2f}\n\n"
    )
    f.write("All results (sorted by Silhouette score):\n")
    f.write("Rank | K | Silhouette Score | Calinski-Harabasz | Notes\n")
    f.write("-" * 55 + "\n")

    # Sort by silhouette score, but put k=1 at the end since it has no metrics
    sorted_results = sorted(
        [r for r in all_results if r[0] > 1], key=lambda x: x[1], reverse=True
    )
    k1_result = [r for r in all_results if r[0] == 1]
    sorted_results.extend(k1_result)

    for i, (k, sil_score, ch_score) in enumerate(sorted_results):
        marker = " <-- BEST" if k == best_k else ""
        if k == 1:
            f.write(f"{i + 1:4d} | {k:1d} | {'N/A':>14} | {'N/A':>15} |{marker}\n")
        else:
            f.write(
                f"{i + 1:4d} | {k:1d} | {sil_score:>14.4f} | {ch_score:>15.2f} |{marker}\n"
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
            # Convert actual newlines to literal \n characters
            text_clean = texts[idx].replace("\n", "\\n").replace("\r", "\\r")
            f.write(f"{i}. {text_clean}\n")

# Print cache statistics
umap_cache_files = [
    f for f in os.listdir(cache_dir) if f.startswith("umap_") and f.endswith(".pkl")
]
spectral_cache_files = [
    f for f in os.listdir(cache_dir) if f.startswith("spectral_") and f.endswith(".pkl")
]
print(f"\nCache statistics:")
print(f"Cache directory: {cache_dir}")
print(f"UMAP cached reductions: {len(umap_cache_files)}")
for cache_file in sorted(umap_cache_files):
    file_size = os.path.getsize(os.path.join(cache_dir, cache_file)) / 1024 / 1024  # MB
    print(f"  {cache_file} ({file_size:.1f} MB)")
print(f"Spectral cached results: {len(spectral_cache_files)}")
for cache_file in sorted(spectral_cache_files):
    file_size = os.path.getsize(os.path.join(cache_dir, cache_file)) / 1024 / 1024  # MB
    print(f"  {cache_file} ({file_size:.1f} MB)")

print(f"\nAnalysis complete! Results saved in {output_dir}/")
print(
    f"Final best: k={best_k}, Silhouette={best_sil_score:.4f}, CH={best_ch_score:.2f}"
)
print(f"Clustering method: Spectral clustering with RBF kernel (gamma={gamma:.6f})")
