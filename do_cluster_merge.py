import argparse
import glob
import hashlib
import math
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
    """Configuration class for clustering parameters"""

    def __init__(
        self,
        dbcv_threshold: float = 0.2,
        min_absolute_size: int = 100,
        min_percentage: float = 0.05,
        umap_min_dist_50d: float = 0.0,
        umap_min_dist_2d: float = 0.1,
        min_dataset_size: int = 1000,
        merge_min_samples: int = 100,  # New: minimum samples for merge mode
        output_base_dir: str = "clusters_output_final_3",
        cache_dir: str = "clusters_cache",
    ):
        self.dbcv_threshold = dbcv_threshold
        self.min_absolute_size = min_absolute_size
        self.min_percentage = min_percentage
        self.umap_min_dist_50d = umap_min_dist_50d
        self.umap_min_dist_2d = umap_min_dist_2d
        self.min_dataset_size = min_dataset_size
        self.merge_min_samples = merge_min_samples
        self.output_base_dir = output_base_dir
        self.cache_dir = cache_dir


def calculate_optimal_neighbors(n_samples: int, n_components: int) -> int:
    """
    Calculate optimal number of neighbors based on dataset size and target dimensions.

    Based on research:
    - McInnes et al. (2018): n_neighbors should scale with log(n_samples)
    - For manifold learning: neighbors should be 2-3x the intrinsic dimensionality
    - Empirical studies suggest sqrt(n_samples) as good starting point for large datasets

    Args:
        n_samples: Number of samples in dataset
        n_components: Target dimensionality

    Returns:
        Optimal number of neighbors
    """
    if n_samples <= 10:
        return max(2, n_samples - 1)

    # Base calculation using log scaling with square root component
    # This balances local vs global structure preservation
    log_component = max(5, int(3 * math.log10(n_samples)))
    sqrt_component = max(10, int(math.sqrt(n_samples) * 0.3))

    # Take geometric mean to balance both approaches
    base_neighbors = int(math.sqrt(log_component * sqrt_component))

    # Adjust for target dimensionality
    # Higher dimensions need more neighbors for stable embeddings
    dim_factor = 1.0 + (n_components / 100.0)  # Modest scaling with dimensions
    neighbors = int(base_neighbors * dim_factor)

    # Apply bounds
    min_neighbors = max(5, int(0.005 * n_samples))  # At least 0.5% of data
    max_neighbors = min(200, int(0.1 * n_samples))  # At most 10% of data

    neighbors = max(min_neighbors, min(neighbors, max_neighbors))
    neighbors = min(neighbors, n_samples - 1)  # Can't exceed sample size - 1

    return neighbors


def get_embeddings_hash(embeddings: np.ndarray) -> str:
    """Create a hash of the embeddings to use as cache key"""
    return hashlib.md5(embeddings.tobytes()).hexdigest()[:16]


def get_cache_path(
    cache_dir: str, embeddings_hash: str, operation: str, **params
) -> str:
    """Generate cache filepath for any operation"""
    param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    filename = f"{operation}_{param_str}_{embeddings_hash}.pkl"
    return os.path.join(cache_dir, filename)


def load_or_compute_umap(
    embeddings: np.ndarray,
    cache_dir: str,
    embeddings_hash: str,
    n_components: int,
    min_dist: float,
) -> np.ndarray:
    """Load UMAP from cache or compute with optimal parameters"""

    # Calculate optimal neighbors
    n_neighbors = calculate_optimal_neighbors(len(embeddings), n_components)

    cache_path = get_cache_path(
        cache_dir,
        embeddings_hash,
        "umap",
        nc=n_components,
        nn=n_neighbors,
        md=f"{min_dist:.3f}",
    )

    if os.path.exists(cache_path):
        print(f"Loading cached {n_components}D UMAP from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(
        f"Computing {n_components}D UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})..."
    )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,  # For reproducibility
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

    cache_path = get_cache_path(
        cache_dir, embeddings_hash, "hdbscan", mcs=min_cluster_size
    )

    if os.path.exists(cache_path):
        print(f"Loading cached HDBSCAN result (min_cluster_size={min_cluster_size})")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Computing HDBSCAN (min_cluster_size={min_cluster_size})...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_epsilon=0.0,
        algorithm="best",
        core_dist_n_jobs=1,
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
            dbcv_score = -1.0
            ch_score = -1.0
    else:
        dbcv_score = -1.0
        ch_score = -1.0

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
    """Calculate min_cluster_size values to test based on dataset size"""
    min_cluster_size = max(min_absolute_size, int(n_samples * min_percentage))
    max_reasonable_clusters = min(50, n_samples // min_cluster_size)

    if max_reasonable_clusters < 2:
        return []

    # Generate cluster sizes to test
    cluster_sizes = []
    for k in range(2, max_reasonable_clusters + 1):
        size = n_samples // k
        if size >= min_cluster_size:
            cluster_sizes.append(size)

    return sorted(set(cluster_sizes), reverse=True)  # Test larger clusters first


def find_best_clustering(
    embeddings_50d: np.ndarray,
    cache_dir: str,
    embeddings_hash: str,
    min_cluster_sizes: List[int],
) -> Dict[str, Any]:
    """Find best clustering by testing different min_cluster_size values"""
    best_score = -1.0
    best_result = None
    all_results = []

    for min_size in min_cluster_sizes:
        k = len(embeddings_50d) // min_size
        print(f"Testing k≈{k}, min_cluster_size={min_size}")

        result = load_or_compute_hdbscan(
            embeddings_50d, cache_dir, embeddings_hash, min_size
        )
        result["min_size"] = min_size
        all_results.append(result)

        # Report cluster info with size distribution
        if result["n_real_clusters"] > 1:
            unique_clusters = [c for c in set(result["labels"]) if c > 0]
            cluster_sizes = [np.sum(result["labels"] == c) for c in unique_clusters]
            size_stats = f"sizes: [{min(cluster_sizes)}-{max(cluster_sizes)}]"
        else:
            size_stats = "no clusters"

        print(
            f"  → {result['n_real_clusters']} clusters + {result['n_noise']} noise, "
            f"DBCV: {result['dbcv_score']:.4f}, CH: {result['ch_score']:.1f} ({size_stats})"
        )

        # Select best based on DBCV score
        if result["dbcv_score"] > best_score and result["n_real_clusters"] > 1:
            best_score = result["dbcv_score"]
            best_result = result.copy()

    # Fallback if no good clustering found
    if best_result is None and all_results:
        best_result = max(all_results, key=lambda x: x["n_real_clusters"])

    if best_result is not None:
        best_result["all_results"] = all_results

    return best_result


def apply_quality_filters(
    result: Dict[str, Any], config: ClusteringConfig
) -> Dict[str, Any]:
    """Apply quality filters and force single cluster if needed"""
    if result is None:
        n_samples = 0  # This shouldn't happen, but handle gracefully
        return create_single_cluster_result(n_samples, "no valid clustering found")

    dbcv_score = result.get("dbcv_score", -1.0)
    n_samples = len(result.get("labels", []))

    # Check DBCV threshold
    if isinstance(dbcv_score, (int, float)) and dbcv_score < config.dbcv_threshold:
        print(f"Quality too low: DBCV {dbcv_score:.4f} < {config.dbcv_threshold}")
        return create_single_cluster_result(n_samples, "quality below threshold")

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
    registers: List[str],
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
                "register": registers[i],
                "embedding": embeddings[i].tolist(),
                "embedding_type": embed_type,
                "cluster_id": int(labels[i]),
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/clustered_data.pkl", "wb") as f:
        pickle.dump(clustered_data, f)


def create_visualizations(
    result: Dict[str, Any],
    embeddings_2d: np.ndarray,
    registers: List[str],
    embed_type: str,
    output_dir: str,
    is_merged: bool = False,
) -> None:
    """Create cluster and register visualizations"""
    print(f"Creating visualizations for {embed_type}")
    os.makedirs(output_dir, exist_ok=True)

    # Generate title
    n_real_clusters = result["n_real_clusters"]
    n_noise = result["n_noise"]
    dbcv_score = result["dbcv_score"]

    if isinstance(dbcv_score, str):
        title = f"Single cluster ({dbcv_score.split(' - ')[-1]})"
    else:
        min_size = result.get("min_size", "unknown")
        title = f"{n_real_clusters} clusters + {n_noise} noise (min_size={min_size}, DBCV: {dbcv_score:.3f})"

    # Main cluster visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=result["labels"],
        cmap="tab20",
        alpha=0.6,
        s=20,
        edgecolors="black",
        linewidth=0.1,
    )

    plt.title(f"UMAP 2D: {title} - {embed_type}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Register visualization (if multiple registers)
    if is_merged and len(set(registers)) > 1:
        plt.figure(figsize=(12, 8))
        unique_registers = sorted(set(registers))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_registers)))

        for i, register in enumerate(unique_registers):
            reg_mask = np.array(registers) == register
            plt.scatter(
                embeddings_2d[reg_mask, 0],
                embeddings_2d[reg_mask, 1],
                c=[colors[i]],
                label=register,
                alpha=0.7,
                s=20,
                edgecolors="black",
                linewidth=0.1,
            )

        plt.title(f"UMAP 2D: Colored by Register - {embed_type}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/umap_registers.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_results_summary(
    result: Dict[str, Any],
    embed_type: str,
    output_dir: str,
    config: ClusteringConfig,
    registers: List[str] = None,
) -> None:
    """Save detailed results summary"""
    print(f"Saving results summary for {embed_type}")
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/clustering_results.txt", "w") as f:
        f.write(f"Clustering Results Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Embedding type: {embed_type}\n")
        f.write(f"DBCV threshold: {config.dbcv_threshold}\n")
        f.write(f"Min absolute cluster size: {config.min_absolute_size}\n")
        f.write(f"Min percentage cluster size: {config.min_percentage}\n\n")

        if registers:
            unique_registers = sorted(set(registers))
            f.write(f"Registers: {', '.join(unique_registers)}\n")
            f.write(f"Total samples: {len(registers)}\n\n")

        # Main results
        if isinstance(result["dbcv_score"], str):
            f.write(f"Result: {result['dbcv_score']}\n\n")
        else:
            f.write(f"Best clustering:\n")
            f.write(f"  Min cluster size: {result['min_size']}\n")
            f.write(f"  Real clusters: {result['n_real_clusters']}\n")
            f.write(f"  Noise points: {result['n_noise']}\n")
            f.write(f"  DBCV score: {result['dbcv_score']:.4f}\n")
            f.write(f"  Calinski-Harabasz score: {result['ch_score']:.2f}\n\n")

        # Detailed results table
        if "all_results" in result:
            f.write("All tested configurations:\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Rank':<4} {'MinSize':<8} {'Clusters':<8} {'Noise':<8} {'DBCV':<10} {'CH-Score':<10} {'Notes'}\n"
            )
            f.write("-" * 80 + "\n")

            sorted_results = sorted(
                result["all_results"],
                key=lambda x: x.get("dbcv_score", -1),
                reverse=True,
            )

            for i, res in enumerate(sorted_results):
                marker = " ★" if res["min_size"] == result.get("min_size") else ""
                f.write(
                    f"{i + 1:<4} {res['min_size']:<8} {res['n_real_clusters']:<8} "
                    f"{res['n_noise']:<8} {res['dbcv_score']:<10.4f} "
                    f"{res['ch_score']:<10.2f} {marker}\n"
                )


def save_cluster_examples(
    labels: np.ndarray,
    texts: List[str],
    embed_type: str,
    output_dir: str,
    registers: List[str] = None,
    max_examples: int = 15,
) -> None:
    """Save representative text examples from each cluster"""
    print(f"Saving cluster examples for {embed_type}")
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/cluster_examples.txt", "w", encoding="utf-8") as f:
        unique_clusters = sorted(set(labels))

        for cluster_id in unique_clusters:
            cluster_indices = np.where(labels == cluster_id)[0]

            if cluster_id == 0:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"CLUSTER {cluster_id} (NOISE) - {embed_type}\n")
                f.write(f"{'=' * 60}\n")
            else:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"CLUSTER {cluster_id} - {embed_type}\n")
                f.write(f"{'=' * 60}\n")

            f.write(f"Size: {len(cluster_indices)} samples\n")

            # Register distribution if available
            if registers:
                cluster_registers = [registers[i] for i in cluster_indices]
                reg_counts = {}
                for reg in cluster_registers:
                    reg_counts[reg] = reg_counts.get(reg, 0) + 1
                f.write(f"Register distribution: {reg_counts}\n")

            f.write("\nExamples:\n")
            f.write("-" * 40 + "\n")

            # Sample examples
            sample_size = min(max_examples, len(cluster_indices))
            sample_indices = np.random.choice(
                cluster_indices, sample_size, replace=False
            )

            for i, idx in enumerate(sample_indices, 1):
                text_clean = texts[idx].replace("\n", " ").replace("\r", " ").strip()
                if len(text_clean) > 200:
                    text_clean = text_clean[:197] + "..."

                reg_info = f" [{registers[idx]}]" if registers else ""
                f.write(f"{i:2d}. {text_clean}{reg_info}\n")


def parse_filename(pkl_file: str) -> Tuple[str, str]:
    """Parse filename to extract language and register"""
    basename = os.path.splitext(os.path.basename(pkl_file))[0]

    # Expected format: {language}_embeds_{registers}.pkl
    if "_embeds_" in basename:
        parts = basename.split("_embeds_")
        if len(parts) == 2:
            return parts[0], parts[1]

    # Fallback parsing
    parts = basename.split("_")
    language = register = None

    for part in parts:
        if part in ["sv", "en", "fi", "de", "fr", "es"]:  # Common language codes
            language = part
        elif part not in ["embeds", language] and part:
            register = part

    if not language or not register:
        raise ValueError(f"Cannot parse language and register from: {pkl_file}")

    return language, register


def find_pickle_files(data_dir: str) -> List[str]:
    """Find all pickle files in directory"""
    pattern = os.path.join(data_dir, "*.pkl")
    return sorted(glob.glob(pattern))


def group_files_by_language(pkl_files: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Group pickle files by language"""
    language_groups = {}

    for pkl_file in pkl_files:
        try:
            language, register = parse_filename(pkl_file)
            if language not in language_groups:
                language_groups[language] = []
            language_groups[language].append((pkl_file, register))
        except ValueError as e:
            print(f"Warning: Skipping {pkl_file}: {e}")

    return language_groups


def filter_files_by_register(
    files_and_registers: List[Tuple[str, str]], target_register: str, min_samples: int
) -> List[Tuple[str, str]]:
    """Filter files by target register and minimum sample size"""
    filtered = []

    for pkl_file, register in files_and_registers:
        # Check if target register is in the register string
        register_parts = register.split("-")
        if target_register not in register_parts:
            continue

        # Check file size (quick sample count check)
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                if len(data) >= min_samples:
                    filtered.append((pkl_file, register))
                else:
                    print(
                        f"Skipping {pkl_file}: only {len(data)} samples < {min_samples}"
                    )
        except Exception as e:
            print(f"Error checking {pkl_file}: {e}")

    return filtered


def load_and_merge_data(
    files_to_merge: List[Tuple[str, str]],
) -> Tuple[Dict[str, np.ndarray], List[str], List[Any], List[str]]:
    """Load and merge data from multiple pickle files"""
    all_embeddings = {"embed_ref": [], "embed_last": []}
    all_texts = []
    all_preds = []
    all_registers = []

    for pkl_file, original_register in files_to_merge:
        print(f"Loading {os.path.basename(pkl_file)} (register: {original_register})")

        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # Extract basic data
        texts = [row["text"] for row in data]
        preds = [row["preds"] for row in data]
        registers = [original_register] * len(texts)

        all_texts.extend(texts)
        all_preds.extend(preds)
        all_registers.extend(registers)

        # Extract embeddings
        for embed_type in ["embed_ref", "embed_last"]:
            if embed_type in data[0]:
                embeddings = np.array([row[embed_type] for row in data])
                all_embeddings[embed_type].extend(embeddings)

    # Convert to numpy arrays
    merged_embeddings = {}
    for embed_type, embeddings_list in all_embeddings.items():
        if embeddings_list:
            merged_embeddings[embed_type] = np.array(embeddings_list)

    return merged_embeddings, all_texts, all_preds, all_registers


def process_embedding_type(
    embeddings: np.ndarray,
    texts: List[str],
    preds: List[Any],
    registers: List[str],
    embed_type: str,
    language: str,
    register_group: str,
    config: ClusteringConfig,
    is_merged: bool = False,
) -> Dict[str, Any]:
    """Process clustering for a single embedding type"""
    n_samples = len(embeddings)
    print(
        f"\nProcessing {embed_type}: {n_samples} samples, {embeddings.shape[1]}D embeddings"
    )

    # Check minimum dataset size
    min_required = config.merge_min_samples if is_merged else config.min_dataset_size
    if n_samples < min_required:
        print(f"Dataset too small ({n_samples} < {min_required}), skipping")
        return {
            "skipped": True,
            "reason": "too_small",
            "n_samples": n_samples,
            "min_required": min_required,
        }

    # Setup output directory
    base_path = "merged" if is_merged else "individual"
    output_dir = os.path.join(
        config.output_base_dir, base_path, language, register_group, embed_type
    )

    # Generate embeddings hash for caching
    embeddings_hash = get_embeddings_hash(embeddings)
    print(f"Embeddings hash: {embeddings_hash}")

    # UMAP reductions with dynamic neighbor selection
    print(f"Optimal neighbors for 50D: {calculate_optimal_neighbors(n_samples, 50)}")
    print(f"Optimal neighbors for 2D: {calculate_optimal_neighbors(n_samples, 2)}")

    embeddings_50d = load_or_compute_umap(
        embeddings, config.cache_dir, embeddings_hash, 50, config.umap_min_dist_50d
    )
    embeddings_2d = load_or_compute_umap(
        embeddings, config.cache_dir, embeddings_hash, 2, config.umap_min_dist_2d
    )

    # Calculate clustering parameters
    min_cluster_sizes = calculate_min_cluster_sizes(
        n_samples, config.min_absolute_size, config.min_percentage
    )

    if not min_cluster_sizes:
        print("Dataset too small for meaningful clustering")
        result = create_single_cluster_result(
            n_samples, "dataset too small for clustering"
        )
    else:
        print(f"Testing {len(min_cluster_sizes)} different cluster sizes")
        result = find_best_clustering(
            embeddings_50d, config.cache_dir, embeddings_hash, min_cluster_sizes
        )
        result = apply_quality_filters(result, config)

    # Save all outputs
    save_clustered_data(
        result["labels"], embeddings, texts, preds, registers, embed_type, output_dir
    )
    create_visualizations(
        result, embeddings_2d, registers, embed_type, output_dir, is_merged
    )
    save_results_summary(result, embed_type, output_dir, config, registers)
    save_cluster_examples(result["labels"], texts, embed_type, output_dir, registers)

    return {
        "embedding_type": embed_type,
        "n_samples": n_samples,
        "n_real_clusters": result["n_real_clusters"],
        "n_noise": result["n_noise"],
        "dbcv_score": result["dbcv_score"],
        "output_dir": output_dir,
    }


def load_data(
    pkl_file: str,
) -> Tuple[Dict[str, np.ndarray], List[str], List[Any], List[str]]:
    """Load and validate data from pickle file"""
    print(f"Loading data from {os.path.basename(pkl_file)}...")

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    if not data:
        raise RuntimeError("Empty dataset")

    # Extract basic data
    texts = [row["text"] for row in data]
    preds = [row["preds"] for row in data]

    # Extract register information
    if "register" in data[0]:
        registers = [row["register"] for row in data]
    else:
        # Fallback to filename parsing
        try:
            _, register = parse_filename(pkl_file)
            registers = [register] * len(texts)
        except:
            registers = ["UNKNOWN"] * len(texts)

    # Extract embeddings
    embeddings_dict = {}
    for embed_type in ["embed_ref", "embed_last"]:
        if embed_type in data[0]:
            embeddings_dict[embed_type] = np.array([row[embed_type] for row in data])

    if not embeddings_dict:
        raise RuntimeError("No valid embeddings found in data")

    return embeddings_dict, texts, preds, registers


def process_single_file(pkl_file: str, config: ClusteringConfig) -> Dict[str, Any]:
    """Process a single pickle file"""
    print(f"\n{'=' * 80}")
    print(f"Processing file: {os.path.basename(pkl_file)}")
    print(f"{'=' * 80}")

    try:
        language, register = parse_filename(pkl_file)
        print(f"Parsed: language={language}, register={register}")
    except ValueError as e:
        return {"error": f"Filename parsing failed: {e}"}

    try:
        embeddings_dict, texts, preds, registers = load_data(pkl_file)
        print(f"Found embeddings: {list(embeddings_dict.keys())}")
    except Exception as e:
        return {"error": f"Data loading failed: {e}"}

    # Process each embedding type
    results = {}
    for embed_type, embeddings in embeddings_dict.items():
        embed_key = "last" if embed_type == "embed_last" else "ref"
        try:
            result = process_embedding_type(
                embeddings,
                texts,
                preds,
                registers,
                embed_key,
                language,
                register,
                config,
                is_merged=False,
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


def process_merged_dataset(
    language: str,
    target_register: str,
    files_to_merge: List[Tuple[str, str]],
    config: ClusteringConfig,
) -> Dict[str, Any]:
    """Process a merged dataset for a specific target register"""
    print(f"\n{'=' * 80}")
    print(f"Processing MERGED dataset: {language} - {target_register}")
    print(f"Files to merge: {[os.path.basename(f[0]) for f in files_to_merge]}")
    print(f"{'=' * 80}")

    try:
        embeddings_dict, texts, preds, registers = load_and_merge_data(files_to_merge)
        print(f"Merged dataset: {len(texts)} samples")
        print(f"Found embeddings: {list(embeddings_dict.keys())}")

        # Show register distribution
        from collections import Counter

        reg_counts = Counter(registers)
        print(f"Register distribution: {dict(reg_counts)}")

    except Exception as e:
        return {"error": f"Data merging failed: {e}"}

    # Process each embedding type
    results = {}
    for embed_type, embeddings in embeddings_dict.items():
        embed_key = "last" if embed_type == "embed_last" else "ref"
        try:
            result = process_embedding_type(
                embeddings,
                texts,
                preds,
                registers,
                embed_key,
                language,
                target_register,
                config,
                is_merged=True,
            )
            results[embed_key] = result
        except Exception as e:
            print(f"Error processing {embed_type}: {e}")
            results[embed_key] = {"error": str(e)}

    return {
        "language": language,
        "target_register": target_register,
        "files_merged": [f[0] for f in files_to_merge],
        "results": results,
    }


def run_merge_mode(data_dir: str, config: ClusteringConfig) -> None:
    """Run clustering in merge mode with strict filtering"""
    print("Running in MERGE mode")
    print(f"Data directory: {data_dir}")
    print(f"Minimum samples required: {config.merge_min_samples}")

    # Find all pickle files
    pkl_files = find_pickle_files(data_dir)
    if not pkl_files:
        print(f"No pickle files found in {data_dir}")
        return

    print(f"Found {len(pkl_files)} pickle files")

    # Group by language
    language_groups = group_files_by_language(pkl_files)
    print(f"Languages found: {list(language_groups.keys())}")

    # Target registers for merging
    target_registers = ["NB", "OB", "ID"]

    # Statistics
    successful = 0
    skipped = 0
    failed = 0

    for language, files_and_registers in language_groups.items():
        print(f"\n{'#' * 60}")
        print(f"Processing language: {language}")
        print(f"Available files: {len(files_and_registers)}")
        print(f"{'#' * 60}")

        for target_register in target_registers:
            print(f"\nProcessing register: {target_register}")

            # Filter files with minimum sample requirement
            files_to_merge = filter_files_by_register(
                files_and_registers, target_register, config.merge_min_samples
            )

            if not files_to_merge:
                print(
                    f"No valid files for {language}-{target_register} (min {config.merge_min_samples} samples)"
                )
                skipped += 1
                continue

            print(f"Found {len(files_to_merge)} files meeting requirements:")
            for pkl_file, register in files_to_merge:
                print(f"  ✓ {os.path.basename(pkl_file)} ({register})")

            try:
                result = process_merged_dataset(
                    language, target_register, files_to_merge, config
                )

                if "error" in result:
                    failed += 1
                    print(f"✗ Failed: {result['error']}")
                else:
                    # Check if any embedding type was successfully processed
                    processed_any = any(
                        "error" not in embed_result and "skipped" not in embed_result
                        for embed_result in result["results"].values()
                    )

                    if processed_any:
                        successful += 1
                        print(f"✓ Successfully processed {language}-{target_register}")
                    else:
                        skipped += 1
                        print(
                            f"○ All embeddings skipped for {language}-{target_register}"
                        )

            except Exception as e:
                failed += 1
                print(
                    f"✗ Unexpected error processing {language}-{target_register}: {e}"
                )

    # Final summary
    print(f"\n{'=' * 80}")
    print("MERGE MODE SUMMARY")
    print(f"{'=' * 80}")
    print(f"Successful: {successful}")
    print(f"Skipped (insufficient samples): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total attempted: {successful + skipped + failed}")


def run_individual_mode(pkl_files: List[str], config: ClusteringConfig) -> None:
    """Run clustering on individual files"""
    print("Running in INDIVIDUAL mode")
    print(f"Files to process: {len(pkl_files)}")

    successful = 0
    skipped = 0
    failed = 0

    for i, pkl_file in enumerate(pkl_files, 1):
        print(f"\n{'#' * 80}")
        print(f"PROCESSING FILE {i}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
        print(f"{'#' * 80}")

        try:
            result = process_single_file(pkl_file, config)

            if "error" in result:
                failed += 1
                print(f"✗ Failed: {result['error']}")
            else:
                # Check if any embedding type was successfully processed
                processed_any = any(
                    "error" not in embed_result and "skipped" not in embed_result
                    for embed_result in result["results"].values()
                )

                if processed_any:
                    successful += 1
                    print(f"✓ Successfully processed {os.path.basename(pkl_file)}")
                else:
                    skipped += 1
                    print(f"○ All embeddings skipped for {os.path.basename(pkl_file)}")

        except Exception as e:
            failed += 1
            print(f"✗ Unexpected error processing {os.path.basename(pkl_file)}: {e}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("INDIVIDUAL MODE SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files: {len(pkl_files)}")
    print(f"Successful: {successful}")
    print(f"Skipped (too small): {skipped}")
    print(f"Failed: {failed}")


def main():
    """Main function with improved argument handling"""
    parser = argparse.ArgumentParser(
        description="Advanced clustering pipeline for text embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process individual files
  python clustering.py file1.pkl file2.pkl file3.pkl
  
  # Run merge mode with default directory
  python clustering.py --merge
  
  # Run merge mode with custom directory
  python clustering.py --merge --data-dir /path/to/embeddings/
  
  # Adjust minimum samples for merge mode
  python clustering.py --merge --min-samples 200
        """,
    )

    parser.add_argument(
        "files", nargs="*", help="Pickle files to process (used in individual mode)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Run in merge mode (combines files by language and register)",
    )
    parser.add_argument(
        "--data-dir",
        default="../data/model_embeds/concat/xlm-r-reference/th-optimised/sm/",
        help="Data directory for merge mode (default: %(default)s)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples required for merge mode (default: %(default)s)",
    )
    parser.add_argument(
        "--dbcv-threshold",
        type=float,
        default=0.2,
        help="DBCV quality threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="clusters_output_final_3",
        help="Base output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        default="clusters_cache",
        help="Cache directory for intermediate results (default: %(default)s)",
    )

    args = parser.parse_args()

    # Create configuration
    config = ClusteringConfig(
        merge_min_samples=args.min_samples,
        dbcv_threshold=args.dbcv_threshold,
        output_base_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )

    print("Clustering Pipeline Configuration:")
    print(f"  DBCV threshold: {config.dbcv_threshold}")
    print(f"  Min dataset size: {config.min_dataset_size}")
    print(f"  Merge mode min samples: {config.merge_min_samples}")
    print(f"  Output directory: {config.output_base_dir}")
    print(f"  Cache directory: {config.cache_dir}")

    if args.merge:
        # Merge mode
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory not found: {args.data_dir}")
            sys.exit(1)

        run_merge_mode(args.data_dir, config)
    else:
        # Individual mode
        if not args.files:
            print("Error: No files specified for individual mode")
            print("Use --help for usage examples")
            sys.exit(1)

        # Validate files exist
        valid_files = [f for f in args.files if os.path.exists(f)]
        if not valid_files:
            print("Error: No valid pickle files found")
            sys.exit(1)

        if len(valid_files) != len(args.files):
            missing = set(args.files) - set(valid_files)
            print(f"Warning: {len(missing)} files not found: {missing}")

        run_individual_mode(valid_files, config)

    print("\n🎉 Processing complete!")


if __name__ == "__main__":
    main()
