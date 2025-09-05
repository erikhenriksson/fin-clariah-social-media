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


class ClusteringConfig:
    def __init__(
        self,
        max_clusters: int = 10,
        dbcv_threshold: float = 0.2,
        min_absolute_size: int = 100,
        min_percentage: float = 0.02,
        umap_50d_neighbors: int = 30,
        umap_2d_neighbors: int = 15,
        umap_min_dist_50d: float = 0.0,
        umap_min_dist_2d: float = 0.1,
        min_dataset_size: int = 1000,
        output_base_dir: str = "clusters_output",
        cache_dir: str = "clusters_cache",
    ):
        self.max_clusters = max_clusters
        self.dbcv_threshold = dbcv_threshold
        self.min_absolute_size = min_absolute_size
        self.min_percentage = min_percentage
        self.umap_50d_neighbors = umap_50d_neighbors
        self.umap_2d_neighbors = umap_2d_neighbors
        self.umap_min_dist_50d = umap_min_dist_50d
        self.umap_min_dist_2d = umap_min_dist_2d
        self.min_dataset_size = min_dataset_size
        self.output_base_dir = output_base_dir
        self.cache_dir = cache_dir


def get_embeddings_hash(embeddings: np.ndarray) -> str:
    """Create a hash of the embeddings to use as cache key"""
    return hashlib.md5(embeddings.tobytes()).hexdigest()[:16]


def get_umap_cache_path(
    cache_dir: str,
    embeddings_hash: str,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
) -> str:
    """Generate cache filepath for UMAP results"""
    filename = (
        f"umap_{n_components}d_nn{n_neighbors}_md{min_dist:.3f}_{embeddings_hash}.pkl"
    )
    return os.path.join(cache_dir, filename)


def get_hdbscan_cache_path(
    cache_dir: str, embeddings_hash: str, min_cluster_size: int
) -> str:
    """Generate cache filepath for HDBSCAN results"""
    filename = f"hdbscan_mcs{min_cluster_size}_{embeddings_hash}.pkl"
    return os.path.join(cache_dir, filename)


def load_or_compute_umap(
    embeddings: np.ndarray,
    cache_dir: str,
    embeddings_hash: str,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    """Load UMAP from cache or compute if not cached"""
    cache_path = get_umap_cache_path(
        cache_dir, embeddings_hash, n_components, n_neighbors, min_dist
    )

    if os.path.exists(cache_path):
        print(f"Loading cached {n_components}D UMAP from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(
        f"Computing {n_components}D UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})..."
    )
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist
    )
    result = reducer.fit_transform(embeddings)

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Saving {n_components}D UMAP to cache: {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result


def load_or_compute_hdbscan(
    embeddings_50d: np.ndarray,
    cache_dir: str,
    embeddings_hash: str,
    min_cluster_size: int,
) -> Dict[str, Any]:
    """Load HDBSCAN from cache or compute if not cached"""
    cache_path = get_hdbscan_cache_path(cache_dir, embeddings_hash, min_cluster_size)

    if os.path.exists(cache_path):
        print(f"Loading cached HDBSCAN result (min_cluster_size={min_cluster_size})")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Computing HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=1, gen_min_span_tree=True
    )
    hdbscan_labels = clusterer.fit_predict(embeddings_50d)

    # Convert labels: noise (-1) -> 0, clusters -> 1,2,3...
    labels = hdbscan_labels.copy()
    labels[hdbscan_labels == -1] = 0
    unique_clusters = np.unique(hdbscan_labels[hdbscan_labels != -1])
    for i, cluster in enumerate(sorted(unique_clusters)):
        labels[hdbscan_labels == cluster] = i + 1

    n_real_clusters = len(unique_clusters)
    n_noise = np.sum(labels == 0)

    # Calculate quality scores
    if n_real_clusters > 1:
        non_noise_mask = labels != 0
        if np.sum(non_noise_mask) > 0:
            dbcv_score = clusterer.relative_validity_
            ch_score = calinski_harabasz_score(
                embeddings_50d[non_noise_mask], labels[non_noise_mask]
            )
        else:
            dbcv_score = -1
            ch_score = -1
    else:
        dbcv_score = -1
        ch_score = -1

    result = {
        "labels": labels,
        "n_real_clusters": n_real_clusters,
        "n_noise": n_noise,
        "dbcv_score": dbcv_score,
        "ch_score": ch_score,
    }

    print(f"Saving HDBSCAN result to cache (min_cluster_size={min_cluster_size})")
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result


def calculate_min_cluster_sizes(
    n_samples: int, min_absolute_size: int, min_percentage: float
) -> List[int]:
    """Calculate min_cluster_size values to test"""
    min_cluster_size = max(min_absolute_size, int(n_samples * min_percentage))
    max_k = n_samples // min_cluster_size

    if max_k < 2:
        return []

    return [
        n_samples // k
        for k in range(2, max_k + 1)
        if n_samples // k >= min_cluster_size
    ]


def find_best_clustering(
    embeddings_50d: np.ndarray,
    cache_dir: str,
    embeddings_hash: str,
    min_cluster_sizes: List[int],
) -> Dict[str, Any]:
    """Find best clustering by testing different min_cluster_size values"""
    best_score = -1
    best_result = None
    all_results = []

    for min_size in min_cluster_sizes:
        k = len(embeddings_50d) // min_size
        print(f"Testing k={k}, min_cluster_size={min_size}")

        result = load_or_compute_hdbscan(
            embeddings_50d, cache_dir, embeddings_hash, min_size
        )
        result["min_size"] = min_size
        all_results.append(result)

        # Report cluster info
        if result["n_real_clusters"] > 1:
            unique_clusters = [c for c in set(result["labels"]) if c > 0]
            cluster_sizes = [np.sum(result["labels"] == c) for c in unique_clusters]
            size_info = f" (sizes: {cluster_sizes})"
        else:
            size_info = ""

        print(
            f"  → {result['n_real_clusters']} real clusters + {result['n_noise']} noise, "
            f"DBCV: {result['dbcv_score']:.4f}, CH: {result['ch_score']:.2f}{size_info}"
        )

        if result["dbcv_score"] > best_score and result["n_real_clusters"] > 1:
            best_score = result["dbcv_score"]
            best_result = result.copy()

    if best_result is None:
        # Use result with most real clusters as fallback
        best_result = max(all_results, key=lambda x: x["n_real_clusters"])

    best_result["all_results"] = all_results
    return best_result


def apply_quality_filters(
    result: Dict[str, Any], config: ClusteringConfig
) -> Dict[str, Any]:
    """Apply quality filters and force single cluster if needed"""
    n_real_clusters = result["n_real_clusters"]
    n_noise = result["n_noise"]
    dbcv_score = result["dbcv_score"]
    n_samples = len(result["labels"])

    # Check too many clusters
    total_clusters = n_real_clusters + (1 if n_noise > 0 else 0)
    if total_clusters > config.max_clusters:
        print(f"Too many clusters: {total_clusters} > {config.max_clusters}")
        return create_single_cluster_result(n_samples, "too many clusters")

    # Check DBCV threshold
    if isinstance(dbcv_score, (int, float)) and dbcv_score < config.dbcv_threshold:
        print(f"Poor quality: DBCV {dbcv_score:.4f} < {config.dbcv_threshold}")
        return create_single_cluster_result(n_samples, "poor quality")

    return result


def create_single_cluster_result(n_samples: int, reason: str) -> Dict[str, Any]:
    """Create single cluster result for edge cases"""
    return {
        "labels": np.ones(n_samples, dtype=int),
        "n_real_clusters": 1,
        "n_noise": 0,
        "dbcv_score": f"N/A (single cluster - {reason})",
        "ch_score": "N/A (single cluster)",
        "min_size": f"N/A (single cluster - {reason})",
    }


def save_clustered_data(
    labels: np.ndarray,
    embeddings: np.ndarray,
    texts: List[str],
    preds: List[Any],
    embed_type: str,
    output_dir: str,
) -> None:
    """Save clustered data to pickle file"""
    print(f"Saving clustered data to {output_dir}/clustered_data.pkl")

    clustered_data = []
    for i in range(len(texts)):
        clustered_data.append(
            {
                "text": texts[i],
                "preds": preds[i],
                "embedding": embeddings[i].tolist(),
                "embedding_type": embed_type,
                "cluster_id": int(labels[i]),
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/clustered_data.pkl", "wb") as f:
        pickle.dump(clustered_data, f)


def save_visualization(
    result: Dict[str, Any],
    embeddings_2d: np.ndarray,
    embed_type: str,
    output_dir: str,
    config: ClusteringConfig,
) -> None:
    """Create and save UMAP visualization"""
    print(f"Creating UMAP visualization for {embed_type}")

    plt.figure(figsize=(10, 8))
    colors = result["labels"].copy()
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap="tab10", alpha=0.6, s=5
    )

    # Generate title
    n_real_clusters = result["n_real_clusters"]
    n_noise = result["n_noise"]
    dbcv_score = result["dbcv_score"]

    if n_real_clusters == 1:
        if isinstance(dbcv_score, str):
            if "too many clusters" in dbcv_score:
                title = f"UMAP 2D with 1 cluster (forced due to > {config.max_clusters} clusters) - {embed_type}"
            else:
                title = f"UMAP 2D with 1 cluster (forced due to low DBCV < {config.dbcv_threshold}) - {embed_type}"
        else:
            title = f"UMAP 2D with 1 cluster - {embed_type}"
    else:
        min_size = result.get("min_size", "unknown")
        title = f"UMAP 2D with {n_real_clusters} clusters + {n_noise} noise (HDBSCAN min_size={min_size}, DBCV: {dbcv_score:.3f}) - {embed_type}"

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_results_summary(
    result: Dict[str, Any], embed_type: str, output_dir: str, config: ClusteringConfig
) -> None:
    """Save results summary to text file"""
    print(f"Saving results summary for {embed_type}")

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/hdbscan_results.txt", "w") as f:
        f.write(f"Embedding type: {embed_type}\n")
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
                result["all_results"], key=lambda x: x["dbcv_score"], reverse=True
            )
            for i, res in enumerate(sorted_results):
                marker = (
                    " <-- BEST" if res["min_size"] == result.get("min_size") else ""
                )
                f.write(
                    f"{i + 1:4d} | {res['min_size']:8d} | {res['n_real_clusters']:12d} | "
                    f"{res['n_noise']:11d} | {res['dbcv_score']:10.4f} | "
                    f"{res['ch_score']:8.2f} |{marker}\n"
                )


def save_cluster_examples(
    labels: np.ndarray, texts: List[str], embed_type: str, output_dir: str
) -> None:
    """Save text examples from each cluster"""
    print(f"Saving cluster examples for {embed_type}")

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/cluster_examples.txt", "w") as f:
        unique_clusters = sorted(set(labels))

        for cluster_id in unique_clusters:
            cluster_indices = np.where(labels == cluster_id)[0]

            if cluster_id == 0:
                f.write(f"\n=== CLUSTER {cluster_id} (NOISE) - {embed_type} ===\n")
            else:
                f.write(f"\n=== CLUSTER {cluster_id} - {embed_type} ===\n")

            f.write(f"Size: {len(cluster_indices)} samples\n\n")

            # Get up to 20 random examples
            sample_indices = np.random.choice(
                cluster_indices, min(20, len(cluster_indices)), replace=False
            )

            for i, idx in enumerate(sample_indices, 1):
                text_clean = texts[idx].replace("\n", "\\n").replace("\r", "\\r")
                f.write(f"{i}. {text_clean}\n")


def parse_filename(pkl_file: str) -> Tuple[str, str]:
    """Parse filename to extract language and register"""
    # Expect format like: something_sv_ID-NA-NB_something.pkl
    basename = os.path.splitext(os.path.basename(pkl_file))[0]
    parts = basename.split("_")

    # Find language (2-letter code) and register (contains hyphens)
    language = None
    register = None

    for part in parts:
        if len(part) == 2 and part.isalpha():
            language = part
        elif "-" in part:
            register = part

    if language is None or register is None:
        raise ValueError(
            f"Could not parse language and register from filename: {pkl_file}"
        )

    return language, register


def process_embedding_type(
    embeddings: np.ndarray,
    texts: List[str],
    preds: List[Any],
    embed_type: str,
    language: str,
    register: str,
    config: ClusteringConfig,
) -> Dict[str, Any]:
    """Process clustering for a single embedding type"""
    n_samples = len(embeddings)
    print(
        f"\nProcessing {embed_type}: {n_samples} samples with {embeddings.shape[1]}D embeddings"
    )

    # Check minimum dataset size
    if n_samples < config.min_dataset_size:
        print(f"Dataset too small ({n_samples} < {config.min_dataset_size}), skipping")
        return {"skipped": True, "reason": "too_small", "n_samples": n_samples}

    # Setup output directory: {output_base_dir}/{language}/{register}/{embed_type}
    output_dir = os.path.join(config.output_base_dir, language, register, embed_type)

    # Generate embeddings hash for caching
    embeddings_hash = get_embeddings_hash(embeddings)
    print(f"Embeddings hash: {embeddings_hash}")

    # Adjust UMAP parameters for dataset size
    umap_50d_neighbors = max(2, min(config.umap_50d_neighbors, n_samples - 1))
    umap_2d_neighbors = max(2, min(config.umap_2d_neighbors, n_samples - 1))

    # UMAP reductions
    embeddings_50d = load_or_compute_umap(
        embeddings,
        config.cache_dir,
        embeddings_hash,
        50,
        umap_50d_neighbors,
        config.umap_min_dist_50d,
    )
    embeddings_2d = load_or_compute_umap(
        embeddings,
        config.cache_dir,
        embeddings_hash,
        2,
        umap_2d_neighbors,
        config.umap_min_dist_2d,
    )

    # Calculate clustering parameters
    min_cluster_sizes = calculate_min_cluster_sizes(
        n_samples, config.min_absolute_size, config.min_percentage
    )

    if not min_cluster_sizes:
        print("Dataset too small for clustering - forcing single cluster")
        result = create_single_cluster_result(n_samples, "Dataset too small")
    else:
        # Find best clustering
        print(f"Testing {len(min_cluster_sizes)} different min_cluster_size values:")
        result = find_best_clustering(
            embeddings_50d, config.cache_dir, embeddings_hash, min_cluster_sizes
        )

        # Apply quality filters
        result = apply_quality_filters(result, config)

    # Save outputs
    save_clustered_data(
        result["labels"], embeddings, texts, preds, embed_type, output_dir
    )
    save_visualization(result, embeddings_2d, embed_type, output_dir, config)
    save_results_summary(result, embed_type, output_dir, config)
    save_cluster_examples(result["labels"], texts, embed_type, output_dir)

    return {
        "embedding_type": embed_type,
        "n_samples": n_samples,
        "n_real_clusters": result["n_real_clusters"],
        "n_noise": result["n_noise"],
        "dbcv_score": result["dbcv_score"],
        "output_dir": output_dir,
    }


def load_data(pkl_file: str) -> Tuple[Dict[str, np.ndarray], List[str], List[Any]]:
    """Load and validate data from pickle file"""
    print("Loading data...")

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # Extract embeddings for both types
    embeddings_dict = {}
    texts = [row["text"] for row in data]
    preds = [row["preds"] for row in data]

    for embed_type in ["embed_ref", "embed_last"]:
        if embed_type in data[0]:
            embeddings_dict[embed_type] = np.array([row[embed_type] for row in data])
        else:
            print(f"Warning: {embed_type} not found in data")

    if not embeddings_dict:
        raise RuntimeError("No valid embeddings found in data")

    return embeddings_dict, texts, preds


def process_file(pkl_file: str, config: ClusteringConfig) -> Dict[str, Any]:
    """Process a single pickle file"""
    print(f"\n{'=' * 80}")
    print(f"Processing file: {pkl_file}")
    print(f"{'=' * 80}")

    # Parse filename to get language and register
    try:
        language, register = parse_filename(pkl_file)
        print(f"Parsed: language={language}, register={register}")
    except ValueError as e:
        print(f"Error: {e}")
        return {"error": str(e)}

    # Load data
    try:
        embeddings_dict, texts, preds = load_data(pkl_file)
        print(f"Found embeddings: {list(embeddings_dict.keys())}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return {"error": str(e)}

    # Process each embedding type
    results = {}
    for embed_type, embeddings in embeddings_dict.items():
        embed_key = "last" if embed_type == "embed_last" else "ref"
        try:
            result = process_embedding_type(
                embeddings, texts, preds, embed_key, language, register, config
            )
            results[embed_key] = result
        except Exception as e:
            print(f"Error processing {embed_type}: {e}")
            results[embed_key] = {"error": str(e)}

    return {
        "filename": pkl_file,
        "language": language,
        "register": register,
        "results": results,
    }


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python script.py pickle1.pkl pickle2.pkl ...")
        sys.exit(1)

    pkl_files = [f for f in sys.argv[1:] if os.path.exists(f)]
    if not pkl_files:
        print("ERROR: No valid pickle files found!")
        sys.exit(1)

    print(f"Found {len(pkl_files)} valid pickle files to process")

    # Setup
    config = ClusteringConfig()

    # Process files
    successful = 0
    skipped = 0
    failed = 0

    for i, pkl_file in enumerate(pkl_files, 1):
        print(f"\n{'#' * 80}")
        print(f"PROCESSING FILE {i}/{len(pkl_files)}: {pkl_file}")
        print(f"{'#' * 80}")

        try:
            result = process_file(pkl_file, config)

            if "error" in result:
                failed += 1
                print(f"✗ Failed: {result['error']}")
            else:
                processed_any = False
                for embed_result in result["results"].values():
                    if "skipped" not in embed_result and "error" not in embed_result:
                        processed_any = True
                        break
                    elif "skipped" in embed_result:
                        skipped += 1

                if processed_any:
                    successful += 1
                    print(f"✓ Successfully processed {pkl_file}")
                else:
                    print(f"○ All embeddings skipped for {pkl_file}")

        except Exception as e:
            failed += 1
            print(f"✗ Error processing {pkl_file}: {e}")

    # Summary
    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files: {len(pkl_files)}")
    print(f"Successful: {successful}")
    print(f"Skipped (too small): {skipped}")
    print(f"Failed: {failed}")

    print("\nBatch processing complete!")


if __name__ == "__main__":
    main()
