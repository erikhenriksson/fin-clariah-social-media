import hashlib
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import calinski_harabasz_score

# Configuration
MAX_CLUSTERS = 5
EMBEDDING_TYPES = ["embed_ref", "embeds_last"]


class ClusteringConfig:
    def __init__(
        self,
        max_clusters: int = 5,
        dbcv_threshold: float = 0.3,
        min_absolute_size: int = 50,
        min_percentage: float = 0.02,
        umap_50d_neighbors: int = 30,
        umap_2d_neighbors: int = 15,
        umap_min_dist_50d: float = 0.0,
        umap_min_dist_2d: float = 0.1,
    ):
        self.max_clusters = max_clusters
        self.dbcv_threshold = dbcv_threshold
        self.min_absolute_size = min_absolute_size
        self.min_percentage = min_percentage
        self.umap_50d_neighbors = umap_50d_neighbors
        self.umap_2d_neighbors = umap_2d_neighbors
        self.umap_min_dist_50d = umap_min_dist_50d
        self.umap_min_dist_2d = umap_min_dist_2d


def get_embeddings_hash(embeddings: np.ndarray) -> str:
    """Create a hash of the embeddings to use as cache key"""
    return hashlib.md5(embeddings.tobytes()).hexdigest()[:16]


def get_cache_filename(
    embeddings_hash: str, n_components: int, n_neighbors: int, min_dist: float
) -> str:
    """Generate cache filename based on parameters"""
    return (
        f"umap_{n_components}d_nn{n_neighbors}_md{min_dist:.3f}_{embeddings_hash}.pkl"
    )


def load_cached_umap(
    cache_dir: str,
    embeddings_hash: str,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
) -> Optional[np.ndarray]:
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
    cache_dir: str,
    embeddings_hash: str,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    reduced_embeddings: np.ndarray,
) -> None:
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
    embeddings: np.ndarray,
    cache_dir: str,
    embeddings_hash: str,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    """Get UMAP reduction from cache or compute it"""
    cached_result = load_cached_umap(
        cache_dir, embeddings_hash, n_components, n_neighbors, min_dist
    )
    if cached_result is not None:
        return cached_result

    print(
        f"Computing {n_components}D UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})..."
    )
    umap_reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist
    )
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    save_umap_cache(
        cache_dir,
        embeddings_hash,
        n_components,
        n_neighbors,
        min_dist,
        reduced_embeddings,
    )
    return reduced_embeddings


def get_hdbscan_cache_filename(embeddings_hash: str, min_cluster_size: int) -> str:
    """Generate cache filename for HDBSCAN results"""
    return f"hdbscan_mcs{min_cluster_size}_{embeddings_hash}.pkl"


def load_cached_hdbscan(
    cache_dir: str, embeddings_hash: str, min_cluster_size: int
) -> Optional[Dict[str, Any]]:
    """Load cached HDBSCAN result if it exists"""
    cache_file = os.path.join(
        cache_dir, get_hdbscan_cache_filename(embeddings_hash, min_cluster_size)
    )

    if os.path.exists(cache_file):
        print(f"Loading cached HDBSCAN result (min_cluster_size={min_cluster_size})")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def save_hdbscan_cache(
    cache_dir: str,
    embeddings_hash: str,
    min_cluster_size: int,
    result_data: Dict[str, Any],
) -> None:
    """Save HDBSCAN result to cache"""
    cache_file = os.path.join(
        cache_dir, get_hdbscan_cache_filename(embeddings_hash, min_cluster_size)
    )

    print(f"Saving HDBSCAN result to cache (min_cluster_size={min_cluster_size})")
    with open(cache_file, "wb") as f:
        pickle.dump(result_data, f)


def relabel_clusters(labels: np.ndarray) -> np.ndarray:
    """Convert HDBSCAN labels to our format: noise (-1) -> 0, real clusters -> 1,2,3..."""
    new_labels = labels.copy()
    new_labels[labels == -1] = 0

    unique_clusters = np.unique(labels[labels != -1])
    for i, cluster in enumerate(sorted(unique_clusters)):
        new_labels[labels == cluster] = i + 1

    return new_labels


def get_or_compute_hdbscan(
    cache_dir: str,
    embeddings_hash: str,
    embeddings_50d: np.ndarray,
    min_cluster_size: int,
) -> Dict[str, Any]:
    """Get HDBSCAN result from cache or compute it"""
    cached_result = load_cached_hdbscan(cache_dir, embeddings_hash, min_cluster_size)
    if cached_result is not None:
        return cached_result

    print(f"Computing HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=1, gen_min_span_tree=True
    )

    hdbscan_labels = clusterer.fit_predict(embeddings_50d)
    labels = relabel_clusters(hdbscan_labels)

    unique_labels = set(labels)
    n_clusters = len(unique_labels)
    n_real_clusters = len(unique_labels) - (1 if 0 in unique_labels else 0)
    n_noise = np.sum(labels == 0)

    if n_real_clusters > 1:
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

    result_data = {
        "labels": labels,
        "hdbscan_labels": hdbscan_labels,
        "n_clusters": n_clusters,
        "n_real_clusters": n_real_clusters,
        "n_noise": n_noise,
        "dbcv_score": dbcv_score,
        "ch_score": ch_score,
        "clusterer": clusterer,
    }

    save_hdbscan_cache(cache_dir, embeddings_hash, min_cluster_size, result_data)
    return result_data


def calculate_test_parameters(
    n_samples: int, min_absolute_size: int = 50, min_percentage: float = 0.02
) -> List[int]:
    """Calculate min_cluster_size values to test"""
    min_cluster_size = max(min_absolute_size, int(n_samples * min_percentage))
    max_k = n_samples // min_cluster_size

    if max_k < 2:
        return []

    min_cluster_sizes = []
    for k in range(2, max_k + 1):
        min_size = n_samples // k
        if min_size >= min_cluster_size:
            min_cluster_sizes.append(min_size)

    return min_cluster_sizes


def process_single_embedding(
    embeddings: np.ndarray,
    texts: List[str],
    preds: List[Any],
    embed_key: str,
    cache_dir: str,
    output_dir: str,
    config: ClusteringConfig,
) -> Dict[str, Any]:
    """Process clustering for a single embedding type"""
    n_samples = len(embeddings)
    print(
        f"\nProcessing {embed_key}: {n_samples} samples with {embeddings.shape[1]}D embeddings"
    )

    # Generate hash for caching
    embeddings_hash = get_embeddings_hash(embeddings)
    print(f"Embeddings hash for {embed_key}: {embeddings_hash}")

    # Adjust UMAP parameters for small datasets
    min_neighbors_50d = max(2, min(config.umap_50d_neighbors, n_samples - 1))
    min_neighbors_2d = max(2, min(config.umap_2d_neighbors, n_samples - 1))

    # UMAP reductions
    embeddings_50d = get_or_compute_umap(
        embeddings,
        cache_dir,
        embeddings_hash,
        50,
        min_neighbors_50d,
        config.umap_min_dist_50d,
    )
    embeddings_2d = get_or_compute_umap(
        embeddings,
        cache_dir,
        embeddings_hash,
        2,
        min_neighbors_2d,
        config.umap_min_dist_2d,
    )

    # Calculate test parameters
    min_cluster_sizes = calculate_test_parameters(
        n_samples, config.min_absolute_size, config.min_percentage
    )

    if not min_cluster_sizes:
        print(f"Dataset too small for {embed_key} - forcing single cluster")
        return create_single_cluster_result(
            embeddings,
            texts,
            preds,
            embed_key,
            embeddings_2d,
            output_dir,
            n_samples,
            "Dataset too small",
        )

    # Test different clustering parameters
    print(
        f"\nTesting {len(min_cluster_sizes)} different min_cluster_size values for {embed_key}:"
    )
    best_result = find_best_clustering(
        min_cluster_sizes, cache_dir, embeddings_hash, embeddings_50d, embed_key
    )

    # Apply quality checks
    final_result = apply_quality_checks(
        best_result, n_samples, embed_key, config.max_clusters, config.dbcv_threshold
    )

    # Create output
    create_output_files(
        final_result,
        embeddings,
        embeddings_2d,
        texts,
        preds,
        embed_key,
        output_dir,
        config,
    )

    return {
        "embedding_type": embed_key,
        "n_samples": n_samples,
        "n_real_clusters": final_result["n_real_clusters"],
        "n_noise": final_result["n_noise"],
        "dbcv_score": final_result["dbcv_score"],
        "ch_score": final_result["ch_score"],
        "output_dir": output_dir,
    }


def create_single_cluster_result(
    embeddings: np.ndarray,
    texts: List[str],
    preds: List[Any],
    embed_key: str,
    embeddings_2d: np.ndarray,
    output_dir: str,
    n_samples: int,
    reason: str,
) -> Dict[str, Any]:
    """Create a single cluster result for edge cases"""
    labels = np.ones(n_samples, dtype=int)

    result = {
        "labels": labels,
        "n_real_clusters": 1,
        "n_noise": 0,
        "dbcv_score": f"N/A ({reason})",
        "ch_score": "N/A (single cluster)",
        "min_size": f"N/A ({reason})",
    }

    create_output_files(
        result,
        embeddings,
        embeddings_2d,
        texts,
        preds,
        embed_key,
        output_dir,
        ClusteringConfig(),
    )
    return result


def find_best_clustering(
    min_cluster_sizes: List[int],
    cache_dir: str,
    embeddings_hash: str,
    embeddings_50d: np.ndarray,
    embed_key: str,
) -> Dict[str, Any]:
    """Find the best clustering configuration"""
    best_score = -1
    best_result = None
    all_results = []

    print(
        f"\nRunning HDBSCAN with different min_cluster_size values for {embed_key}..."
    )

    for i, min_size in enumerate(min_cluster_sizes):
        n_samples = len(embeddings_50d)
        k = n_samples // min_size
        print(
            f"Testing {i + 1}/{len(min_cluster_sizes)}: k={k}, min_cluster_size={min_size}"
        )

        result = get_or_compute_hdbscan(
            cache_dir, embeddings_hash, embeddings_50d, min_size
        )

        # Calculate cluster size info for reporting
        cluster_size_info = ""
        if result["n_real_clusters"] > 1:
            unique_real_clusters = [c for c in set(result["labels"]) if c > 0]
            cluster_sizes = [
                np.sum(result["labels"] == c) for c in unique_real_clusters
            ]
            cluster_size_info = f" (sizes: {cluster_sizes})"

        all_results.append((min_size, result, cluster_size_info))
        print(
            f"  → {result['n_real_clusters']} real clusters + {result['n_noise']} noise points, "
            f"DBCV: {result['dbcv_score']:.4f}, CH: {result['ch_score']:.2f}{cluster_size_info}"
        )

        if result["dbcv_score"] > best_score and result["n_real_clusters"] > 1:
            best_score = result["dbcv_score"]
            best_result = result.copy()
            best_result["min_size"] = min_size
            best_result["all_results"] = all_results

    if best_result is None:
        print(f"\nWarning: No valid clustering found for {embed_key}! Using fallback.")
        # Use result with most real clusters as fallback
        fallback = max(all_results, key=lambda x: x[1]["n_real_clusters"])
        best_result = fallback[1].copy()
        best_result["min_size"] = fallback[0]
        best_result["all_results"] = all_results

    return best_result


def apply_quality_checks(
    result: Dict[str, Any],
    n_samples: int,
    embed_key: str,
    max_clusters: int,
    dbcv_threshold: float,
) -> Dict[str, Any]:
    """Apply quality checks and force single cluster if needed"""
    labels = result["labels"]
    n_real_clusters = result["n_real_clusters"]
    n_noise = result["n_noise"]
    dbcv_score = result["dbcv_score"]

    # Check excessive noise
    if n_real_clusters > 1 and n_noise > 0:
        real_cluster_ids = [c for c in set(labels) if c > 0]
        cluster_sizes = [np.sum(labels == c) for c in real_cluster_ids]
        smallest_cluster_size = min(cluster_sizes)

        if n_noise > smallest_cluster_size:
            print(
                f"\nNoise check for {embed_key}: Noise ({n_noise}) > smallest cluster ({smallest_cluster_size})"
            )
            return force_single_cluster(n_samples, "excessive noise")

    # Check too many clusters
    total_clusters = n_real_clusters + (1 if n_noise > 0 else 0)
    if total_clusters > max_clusters:
        print(
            f"\nMax clusters check for {embed_key}: {total_clusters} > {max_clusters}"
        )
        return force_single_cluster(n_samples, "too many clusters")

    # Check DBCV threshold
    if isinstance(dbcv_score, (int, float)) and dbcv_score < dbcv_threshold:
        print(f"\nDBCV check for {embed_key}: {dbcv_score:.4f} < {dbcv_threshold}")
        return force_single_cluster(n_samples, "poor quality")

    return result


def force_single_cluster(n_samples: int, reason: str) -> Dict[str, Any]:
    """Force all points into a single cluster"""
    return {
        "labels": np.ones(n_samples, dtype=int),
        "n_real_clusters": 1,
        "n_noise": 0,
        "dbcv_score": f"N/A (single cluster - {reason})",
        "ch_score": "N/A (single cluster)",
        "min_size": f"N/A (single cluster - {reason})",
    }


def create_output_files(
    result: Dict[str, Any],
    embeddings: np.ndarray,
    embeddings_2d: np.ndarray,
    texts: List[str],
    preds: List[Any],
    embed_key: str,
    base_output_dir: str,
    config: ClusteringConfig,
) -> None:
    """Create all output files for this embedding type"""
    output_dir = f"{base_output_dir}_{embed_key}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results for {embed_key} to {output_dir}/")

    # Save clustered data
    save_clustered_data(
        result["labels"], embeddings, texts, preds, embed_key, output_dir
    )

    # Create visualization
    create_visualization(result, embeddings_2d, embed_key, output_dir, config)

    # Save results summary
    save_results_summary(result, embed_key, output_dir, config)

    # Save cluster examples
    save_cluster_examples(result["labels"], texts, embed_key, output_dir)


def save_clustered_data(
    labels: np.ndarray,
    embeddings: np.ndarray,
    texts: List[str],
    preds: List[Any],
    embed_key: str,
    output_dir: str,
) -> None:
    """Save clustered data to pickle file"""
    print(f"Saving clustered data to pickle for {embed_key}...")
    clustered_data = []
    for i in range(len(texts)):
        clustered_data.append(
            {
                "text": texts[i],
                "preds": preds[i],
                "embedding": embeddings[i].tolist(),
                "embedding_type": embed_key,
                "cluster_id": int(labels[i]),
            }
        )

    pickle_output_path = f"{output_dir}/clustered_data.pkl"
    with open(pickle_output_path, "wb") as f:
        pickle.dump(clustered_data, f)
    print(f"Clustered data saved to {pickle_output_path}")


def create_visualization(
    result: Dict[str, Any],
    embeddings_2d: np.ndarray,
    embed_key: str,
    output_dir: str,
    config: ClusteringConfig,
) -> None:
    """Create UMAP visualization"""
    print(f"Creating UMAP visualization for {embed_key}...")
    plt.figure(figsize=(10, 8))

    colors = result["labels"].copy()
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap="tab10", alpha=0.6, s=5
    )

    # Generate title based on result type
    title = generate_plot_title(result, embed_key, config)

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"UMAP plot saved for {embed_key}")


def generate_plot_title(
    result: Dict[str, Any], embed_key: str, config: ClusteringConfig
) -> str:
    """Generate appropriate title for the plot"""
    n_real_clusters = result["n_real_clusters"]
    n_noise = result["n_noise"]
    dbcv_score = result["dbcv_score"]

    if n_real_clusters == 1:
        if isinstance(dbcv_score, str):
            if "too many clusters" in dbcv_score:
                return f"UMAP 2D with 1 cluster (forced due to > {config.max_clusters} clusters) - {embed_key}"
            elif "excessive noise" in dbcv_score:
                return f"UMAP 2D with 1 cluster (forced due to excessive noise) - {embed_key}"
            else:
                return f"UMAP 2D with 1 cluster (forced due to low DBCV < {config.dbcv_threshold}) - {embed_key}"
        else:
            return f"UMAP 2D with 1 cluster - {embed_key}"
    else:
        min_size = result.get("min_size", "unknown")
        return f"UMAP 2D with {n_real_clusters} clusters + {n_noise} noise (HDBSCAN min_size={min_size}, DBCV: {dbcv_score:.3f}) - {embed_key}"


def save_results_summary(
    result: Dict[str, Any], embed_key: str, output_dir: str, config: ClusteringConfig
) -> None:
    """Save results summary to text file"""
    print(f"Saving results summary for {embed_key}...")
    with open(f"{output_dir}/hdbscan_results.txt", "w") as f:
        f.write(f"Embedding type: {embed_key}\n")
        f.write(f"DBCV threshold: {config.dbcv_threshold}\n")
        f.write(f"Max clusters allowed: {config.max_clusters}\n")
        f.write(f"Clustering scheme: Noise = Cluster 0, Real clusters = 1, 2, 3, ...\n")

        if isinstance(result["dbcv_score"], str):
            f.write(
                f"Result: Single cluster (all samples), reason: {result['dbcv_score']}\n\n"
            )
        else:
            f.write(
                f"Best result: min_cluster_size={result['min_size']}, "
                f"{result['n_real_clusters']} real clusters, {result['n_noise']} noise points, "
                f"DBCV={result['dbcv_score']}, CH={result['ch_score']}\n\n"
            )

        # Write detailed results if available
        if "all_results" in result:
            f.write("All results (sorted by DBCV score):\n")
            f.write(
                "Rank | Min_Size | Real_Clusters | Noise_Points | DBCV Score | CH Score | Notes\n"
            )
            f.write("-" * 80 + "\n")

            sorted_results = sorted(
                result["all_results"], key=lambda x: x[1]["dbcv_score"], reverse=True
            )
            for i, (min_size, res, cluster_info) in enumerate(sorted_results):
                marker = " <-- BEST" if min_size == result.get("min_size") else ""
                f.write(
                    f"{i + 1:4d} | {min_size:8d} | {res['n_real_clusters']:12d} | "
                    f"{res['n_noise']:11d} | {res['dbcv_score']:10.4f} | "
                    f"{res['ch_score']:8.2f} |{marker}{cluster_info}\n"
                )


def save_cluster_examples(
    labels: np.ndarray, texts: List[str], embed_key: str, output_dir: str
) -> None:
    """Save text examples from each cluster"""
    print(f"Saving cluster examples for {embed_key}...")
    with open(f"{output_dir}/cluster_examples.txt", "w") as f:
        unique_clusters = sorted(set(labels))

        for cluster_id in unique_clusters:
            cluster_indices = np.where(labels == cluster_id)[0]

            if cluster_id == 0:
                f.write(f"\n=== CLUSTER {cluster_id} (NOISE) - {embed_key} ===\n")
            else:
                f.write(f"\n=== CLUSTER {cluster_id} - {embed_key} ===\n")

            f.write(f"Size: {len(cluster_indices)} samples\n\n")

            # Get up to 20 random examples
            sample_indices = np.random.choice(
                cluster_indices, min(20, len(cluster_indices)), replace=False
            )

            for i, idx in enumerate(sample_indices, 1):
                text_clean = texts[idx].replace("\n", "\\n").replace("\r", "\\r")
                f.write(f"{i}. {text_clean}\n")


def load_and_validate_data(
    pkl_file: str,
) -> Tuple[Dict[str, np.ndarray], List[str], List[Any]]:
    """Load and validate data from pickle file"""
    print("Loading data...")
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load {pkl_file}: {e}")

    # Extract and validate embeddings for each type
    embeddings_dict = {}
    texts = None
    preds = None

    for embed_key in EMBEDDING_TYPES:
        try:
            embeddings = np.array([row[embed_key] for row in data])
            embeddings_dict[embed_key] = embeddings

            # Extract texts and preds only once (they should be the same)
            if texts is None:
                texts = [row["text"] for row in data]
                preds = [row["preds"] for row in data]

        except KeyError:
            print(f"Warning: {embed_key} not found in data, skipping...")
        except Exception as e:
            print(f"Error extracting {embed_key}: {e}")

    if not embeddings_dict:
        raise RuntimeError("No valid embeddings found in data")

    return embeddings_dict, texts, preds


def process_file(
    pkl_file: str, cache_dir: str, config: ClusteringConfig
) -> Dict[str, Any]:
    """Process a single pickle file with multiple embedding types"""
    filename_without_ext = os.path.splitext(os.path.basename(pkl_file))[0]
    print(f"\n{'=' * 80}")
    print(f"Processing file: {pkl_file}")
    print(f"{'=' * 80}")

    # Load and validate data
    embeddings_dict, texts, preds = load_and_validate_data(pkl_file)

    print(f"Found embeddings: {list(embeddings_dict.keys())}")

    # Process each embedding type
    all_results = {}
    base_output_dir = f"clusters_concat/{filename_without_ext}"

    for embed_key, embeddings in embeddings_dict.items():
        try:
            print(f"\n{'-' * 60}")
            print(f"Processing embedding type: {embed_key}")
            print(f"{'-' * 60}")

            result = process_single_embedding(
                embeddings, texts, preds, embed_key, cache_dir, base_output_dir, config
            )
            all_results[embed_key] = result

        except Exception as e:
            print(f"Error processing {embed_key}: {e}")
            all_results[embed_key] = {"error": str(e)}

    return {
        "filename": pkl_file,
        "results": all_results,
        "total_embeddings_processed": len(
            [r for r in all_results.values() if "error" not in r]
        ),
    }


def main():
    """Main function to process multiple pickle files"""
    if len(sys.argv) < 2:
        print("Usage: python script.py pickle1.pkl pickle2.pkl pickle3.pkl ...")
        print("       python script.py *.pkl")
        sys.exit(1)

    pkl_files = sys.argv[1:]

    # Validate files
    valid_files = [f for f in pkl_files if os.path.exists(f)]
    if not valid_files:
        print("ERROR: No valid pickle files found!")
        sys.exit(1)

    print(f"Found {len(valid_files)} valid pickle files to process:")
    for f in valid_files:
        print(f"  - {f}")

    # Setup
    config = ClusteringConfig()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "umap_cache_multi")
    print(f"\nCache directory: {cache_dir}")

    # Process files
    successful_results = []
    failed_files = []

    for i, pkl_file in enumerate(valid_files, 1):
        print(f"\n{'#' * 80}")
        print(f"PROCESSING FILE {i}/{len(valid_files)}: {pkl_file}")
        print(f"{'#' * 80}")

        try:
            result = process_file(pkl_file, cache_dir, config)
            if result["total_embeddings_processed"] > 0:
                successful_results.append(result)
                print(f"✓ Successfully processed {pkl_file}")
            else:
                failed_files.append(pkl_file)
                print(f"✗ No embeddings processed for {pkl_file}")
        except Exception as e:
            failed_files.append(pkl_file)
            print(f"✗ Error processing {pkl_file}: {e}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files processed: {len(valid_files)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_files)}")

    if successful_results:
        print("\nSuccessful results:")
        print(
            f"{'File':<30} {'Embedding':<15} {'Samples':<8} {'Clusters':<8} {'Noise':<8} {'DBCV':<10}"
        )
        print("-" * 90)

        for result in successful_results:
            filename = os.path.basename(result["filename"])
            for embed_key, embed_result in result["results"].items():
                if "error" not in embed_result:
                    dbcv_str = (
                        f"{embed_result['dbcv_score']:.4f}"
                        if isinstance(embed_result["dbcv_score"], (int, float))
                        else str(embed_result["dbcv_score"])[:10]
                    )
                    print(
                        f"{filename:<30} {embed_key:<15} {embed_result['n_samples']:<8} "
                        f"{embed_result['n_real_clusters']:<8} {embed_result['n_noise']:<8} {dbcv_str:<10}"
                    )

    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

    # Print cache statistics
    if os.path.exists(cache_dir):
        umap_cache_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("umap_") and f.endswith(".pkl")
        ]
        hdbscan_cache_files = [
            f
            for f in os.listdir(cache_dir)
            if f.startswith("hdbscan_") and f.endswith(".pkl")
        ]
        print(f"\nCache statistics:")
        print(f"UMAP cached reductions: {len(umap_cache_files)}")
        print(f"HDBSCAN cached results: {len(hdbscan_cache_files)}")

    print(f"\nBatch processing complete!")


if __name__ == "__main__":
    main()
