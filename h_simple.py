import hashlib
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

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


def get_hdbscan_cache_filename(embeddings_hash, min_cluster_size):
    """Generate cache filename for HDBSCAN results"""
    return f"hdbscan_mcs{min_cluster_size}_{embeddings_hash}.pkl"


def load_cached_hdbscan(cache_dir, embeddings_hash, min_cluster_size):
    """Load cached HDBSCAN result if it exists"""
    cache_file = os.path.join(
        cache_dir, get_hdbscan_cache_filename(embeddings_hash, min_cluster_size)
    )

    if os.path.exists(cache_file):
        print(f"Loading cached HDBSCAN result (min_cluster_size={min_cluster_size})")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def save_hdbscan_cache(cache_dir, embeddings_hash, min_cluster_size, result_data):
    """Save HDBSCAN result to cache"""
    cache_file = os.path.join(
        cache_dir, get_hdbscan_cache_filename(embeddings_hash, min_cluster_size)
    )

    print(f"Saving HDBSCAN result to cache (min_cluster_size={min_cluster_size})")
    with open(cache_file, "wb") as f:
        pickle.dump(result_data, f)


def relabel_clusters(labels):
    """
    Convert HDBSCAN labels to our format:
    - Noise (-1) becomes cluster 0
    - Real clusters (0, 1, 2, ...) become clusters (1, 2, 3, ...)
    """
    new_labels = labels.copy()

    # Noise points (-1) become cluster 0
    new_labels[labels == -1] = 0

    # Real clusters get shifted up by 1
    unique_clusters = np.unique(labels[labels != -1])
    for i, cluster in enumerate(sorted(unique_clusters)):
        new_labels[labels == cluster] = i + 1

    return new_labels


def get_or_compute_hdbscan(
    cache_dir, embeddings_hash, embeddings_50d, min_cluster_size
):
    """Get HDBSCAN result from cache or compute it"""
    # Try to load from cache first
    cached_result = load_cached_hdbscan(cache_dir, embeddings_hash, min_cluster_size)
    if cached_result is not None:
        return cached_result

    # Compute HDBSCAN clustering
    print(f"Computing HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,  # Minimize noise
        gen_min_span_tree=True,  # Required for DBCV calculation
    )

    hdbscan_labels = clusterer.fit_predict(embeddings_50d)

    # Convert to our labeling scheme (noise=0, real clusters=1,2,3...)
    labels = relabel_clusters(hdbscan_labels)

    # Calculate metrics
    unique_labels = set(labels)
    n_clusters = len(unique_labels)  # This includes cluster 0 (noise)
    n_real_clusters = len(unique_labels) - (
        1 if 0 in unique_labels else 0
    )  # Actual clusters excluding noise
    n_noise = np.sum(labels == 0)

    # For metrics calculation, we need to exclude noise points (cluster 0) if present
    if n_real_clusters > 1:
        # Calculate metrics only on non-noise points
        non_noise_mask = labels != 0
        if np.sum(non_noise_mask) > 0:
            non_noise_embeddings = embeddings_50d[non_noise_mask]
            non_noise_labels = labels[non_noise_mask]

            dbcv_score = clusterer.relative_validity_
            ch_score = calinski_harabasz_score(non_noise_embeddings, non_noise_labels)
        else:
            dbcv_score = -1
            ch_score = -1
    else:
        dbcv_score = -1
        ch_score = -1

    # Package result
    result_data = {
        "labels": labels,
        "hdbscan_labels": hdbscan_labels,  # Original HDBSCAN labels for reference
        "n_clusters": n_clusters,  # Total clusters including noise
        "n_real_clusters": n_real_clusters,  # Actual clusters excluding noise
        "n_noise": n_noise,
        "dbcv_score": dbcv_score,
        "ch_score": ch_score,
        "clusterer": clusterer,  # Save the clusterer object for DBCV access
    }

    # Save to cache
    save_hdbscan_cache(cache_dir, embeddings_hash, min_cluster_size, result_data)

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
# percentages = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
percentages = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
min_cluster_sizes = [max(2, int(n_samples * p / 100)) for p in percentages]

# Filter out cluster sizes that would result in fewer than 100 examples per cluster
valid_params = []
for p, size in zip(percentages, min_cluster_sizes):
    if size >= 100:
        valid_params.append((p, size))
    else:
        print(f"Skipping {p}% ({size} samples) - below 100 sample threshold")

if not valid_params:
    print("ERROR: No valid cluster sizes found! All percentages produce <100 samples.")
    sys.exit(1)

percentages, min_cluster_sizes = zip(*valid_params)
percentages = list(percentages)
min_cluster_sizes = list(min_cluster_sizes)

print(f"\nTesting {len(min_cluster_sizes)} different min_cluster_size values:")
for p, size in zip(percentages, min_cluster_sizes):
    print(f"  {p}% = {size} samples")

# Test different min_cluster_size values
best_score = -1
best_labels = None
best_min_size = None
best_n_real_clusters = None
all_results = []

# DBCV threshold - if best DBCV is below this, assign everything to single cluster
DBCV_THRESHOLD = 0.3  # Adjust this threshold as needed

print("\nRunning HDBSCAN with different min_cluster_size values...")
for i, (percentage, min_size) in enumerate(zip(percentages, min_cluster_sizes)):
    print(
        f"Testing {i + 1}/{len(min_cluster_sizes)}: min_cluster_size={min_size} ({percentage}%)"
    )

    # Get HDBSCAN result from cache or compute it
    result = get_or_compute_hdbscan(
        cache_dir, embeddings_hash, embeddings_50d, min_size
    )

    labels = result["labels"]
    n_clusters = result["n_clusters"]
    n_real_clusters = result["n_real_clusters"]
    n_noise = result["n_noise"]
    dbcv_score = result["dbcv_score"]
    ch_score = result["ch_score"]

    all_results.append(
        (
            percentage,
            min_size,
            n_clusters,
            n_real_clusters,
            n_noise,
            dbcv_score,
            ch_score,
        )
    )
    print(
        f"  → {n_real_clusters} real clusters + {n_noise} noise points, DBCV: {dbcv_score:.4f}, CH: {ch_score:.2f}"
    )

    if dbcv_score > best_score and n_real_clusters > 1:
        best_score = dbcv_score
        best_labels = labels
        best_min_size = min_size
        best_n_real_clusters = n_real_clusters

if best_labels is None:
    print("\nWarning: No valid clustering found! All results had ≤1 real cluster.")
    # Use the result with most real clusters as fallback
    best_result = max(all_results, key=lambda x: x[3])  # x[3] is n_real_clusters
    percentage, min_size, n_clusters, n_real_clusters, n_noise, dbcv_score, ch_score = (
        best_result
    )
    print(
        f"Using fallback: {percentage}% ({min_size} samples) with {n_real_clusters} real clusters"
    )

    # Get the cached result for the fallback parameters
    fallback_result = get_or_compute_hdbscan(
        cache_dir, embeddings_hash, embeddings_50d, min_size
    )
    best_labels = fallback_result["labels"]
    best_min_size = min_size
    best_n_real_clusters = n_real_clusters
    best_score = dbcv_score

# Check if best DBCV score is below threshold
if best_score < DBCV_THRESHOLD:
    print(
        f"\nDBCV threshold check: Best DBCV ({best_score:.4f}) < threshold ({DBCV_THRESHOLD})"
    )
    print("Assigning all points to a single cluster due to poor clustering quality.")

    # Create single cluster assignment (all points go to cluster 1, no noise)
    best_labels = np.ones(n_samples, dtype=int)  # All points assigned to cluster 1
    best_n_real_clusters = 1
    best_min_size = "N/A (single cluster)"
    best_score = "N/A (single cluster)"
    n_noise_best = 0

    # Calculate CH score for single cluster (will be undefined, but we'll note it)
    best_ch_score = "N/A (single cluster)"

    print(f"Final result: 1 cluster with all {n_samples} samples")
else:
    # Calculate final scores for the best result
    # For CH score, exclude noise points if they exist
    non_noise_mask = best_labels != 0
    if np.sum(non_noise_mask) > 1 and best_n_real_clusters > 1:
        best_ch_score = calinski_harabasz_score(
            embeddings_50d[non_noise_mask], best_labels[non_noise_mask]
        )
    else:
        best_ch_score = -1

    # Find the number of noise points for the best result
    n_noise_best = next(
        result[4] for result in all_results if result[1] == best_min_size
    )

    print(
        f"\nBest result: min_cluster_size={best_min_size}, {best_n_real_clusters} real clusters, {n_noise_best} noise points, DBCV={best_score:.4f}, CH={best_ch_score:.2f}"
    )

# Create output directory
output_dir = f"clusters/{filename_without_ext}"
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

# Handle title formatting for single cluster case
if best_n_real_clusters == 1:
    if isinstance(best_score, str):  # Single cluster due to threshold
        title = f"UMAP 2D with {best_n_real_clusters} cluster (forced due to low DBCV < {DBCV_THRESHOLD})"
    else:
        title = f"UMAP 2D with {best_n_real_clusters} cluster"
else:
    n_noise_display = n_noise_best if "n_noise_best" in locals() else 0
    title = f"UMAP 2D with {best_n_real_clusters} clusters + {n_noise_display} noise (HDBSCAN min_size={best_min_size}, DBCV: {best_score:.3f})"

plt.title(title)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches="tight")
plt.close()
print("UMAP plot saved")

# Save all results sorted by DBCV score
print("Saving results summary...")
with open(f"{output_dir}/hdbscan_results.txt", "w") as f:
    f.write(f"Dataset size: {n_samples} samples\n")
    f.write(f"Embeddings hash: {embeddings_hash}\n")
    f.write(f"DBCV threshold: {DBCV_THRESHOLD}\n")
    f.write(f"Clustering scheme: Noise = Cluster 0, Real clusters = 1, 2, 3, ...\n")
    if isinstance(best_score, str):
        f.write(f"Best result: Single cluster (all samples), reason: {best_score}\n\n")
    else:
        f.write(
            f"Best result: min_cluster_size={best_min_size}, {best_n_real_clusters} real clusters, {n_noise_best} noise points, DBCV={best_score}, Calinski-Harabasz={best_ch_score}\n\n"
        )
    f.write("All results (sorted by DBCV score):\n")
    f.write(
        "Rank | Percentage | Min_Size | Total_Clusters | Real_Clusters | Noise_Points | DBCV Score | Calinski-Harabasz | Notes\n"
    )
    f.write("-" * 120 + "\n")
    sorted_results = sorted(
        all_results, key=lambda x: x[5], reverse=True
    )  # Sort by DBCV (index 5)
    for i, (
        percentage,
        min_size,
        n_clusters,
        n_real_clusters,
        n_noise,
        dbcv_score,
        ch_score,
    ) in enumerate(sorted_results):
        marker = " <-- BEST" if min_size == best_min_size else ""
        f.write(
            f"{i + 1:4d} | {percentage:9.1f}% | {min_size:8d} | {n_clusters:13d} | {n_real_clusters:12d} | {n_noise:11d} | {dbcv_score:10.4f} | {ch_score:17.2f} |{marker}\n"
        )

# Save text examples from each cluster
print("Saving cluster examples...")
with open(f"{output_dir}/cluster_examples.txt", "w") as f:
    unique_clusters = sorted(set(best_labels))

    for cluster_id in unique_clusters:
        cluster_indices = np.where(best_labels == cluster_id)[0]

        if cluster_id == 0:
            f.write(f"\n=== CLUSTER {cluster_id} (NOISE) ===\n")
        else:
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
hdbscan_cache_files = [
    f for f in os.listdir(cache_dir) if f.startswith("hdbscan_") and f.endswith(".pkl")
]
print(f"\nCache statistics:")
print(f"Cache directory: {cache_dir}")
print(f"UMAP cached reductions: {len(umap_cache_files)}")
for cache_file in sorted(umap_cache_files):
    file_size = os.path.getsize(os.path.join(cache_dir, cache_file)) / 1024 / 1024  # MB
    print(f"  {cache_file} ({file_size:.1f} MB)")
print(f"HDBSCAN cached results: {len(hdbscan_cache_files)}")
for cache_file in sorted(hdbscan_cache_files):
    file_size = os.path.getsize(os.path.join(cache_dir, cache_file)) / 1024 / 1024  # MB
    print(f"  {cache_file} ({file_size:.1f} MB)")

print(f"\nAnalysis complete! Results saved in {output_dir}/")
if isinstance(best_score, str):  # Single cluster case
    print(f"Final result: 1 cluster with all {n_samples} samples")
else:
    print(
        f"Final best: min_cluster_size={best_min_size}, {best_n_real_clusters} real clusters, {n_noise_best} noise points, DBCV={best_score:.4f}, CH={best_ch_score:.2f}"
    )
print(
    f"Clustering scheme: Noise points assigned to cluster 0, real clusters numbered 1, 2, 3, ..."
)
