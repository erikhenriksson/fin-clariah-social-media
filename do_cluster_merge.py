import argparse
import glob
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
        max_clusters: int = 100,
        dbcv_threshold: float = 0.5,
        min_absolute_size: int = 150,
        min_percentage: float = 0.05,
        umap_50d_neighbors: int = 30,
        umap_2d_neighbors: int = 15,
        umap_min_dist_50d: float = 0.0,
        umap_min_dist_2d: float = 0.1,
        min_dataset_size: int = 1000,
        output_base_dir: str = "clusters_output_final",
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
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
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
        min_cluster_size=min_cluster_size,
        min_samples=1,
        gen_min_span_tree=True,
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


def create_combined_visualization(
    result: Dict[str, Any],
    embeddings_2d: np.ndarray,
    registers: List[str],
    embed_type: str,
    output_dir: str,
    config: ClusteringConfig,
    is_merged: bool = False,
) -> None:
    """Create visualizations with both cluster and register information"""
    print(f"Creating visualizations for {embed_type}")

    os.makedirs(output_dir, exist_ok=True)

    # Get unique registers and clusters
    unique_registers = sorted(set(registers))
    unique_clusters = sorted(set(result["labels"]))

    # Create markers for registers
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x"]
    register_markers = {
        reg: markers[i % len(markers)] for i, reg in enumerate(unique_registers)
    }

    # Colors for clusters (using tab10 colormap)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_clusters))))
    cluster_colors = {
        cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)
    }

    # Title generation
    n_real_clusters = result["n_real_clusters"]
    n_noise = result["n_noise"]
    dbcv_score = result["dbcv_score"]

    if n_real_clusters == 1:
        if isinstance(dbcv_score, str):
            if "too many clusters" in dbcv_score:
                base_title = (
                    f"1 cluster (forced due to > {config.max_clusters} clusters)"
                )
            else:
                base_title = (
                    f"1 cluster (forced due to low DBCV < {config.dbcv_threshold})"
                )
        else:
            base_title = f"1 cluster"
    else:
        min_size = result.get("min_size", "unknown")
        base_title = f"{n_real_clusters} clusters + {n_noise} noise (min_size={min_size}, DBCV: {dbcv_score:.3f})"

    if is_merged:
        # Combined plot with markers for registers and colors for clusters
        plt.figure(figsize=(12, 8))

        for register in unique_registers:
            reg_mask = np.array(registers) == register
            reg_x = embeddings_2d[reg_mask, 0]
            reg_y = embeddings_2d[reg_mask, 1]
            reg_clusters = result["labels"][reg_mask]

            for cluster in unique_clusters:
                cluster_mask = reg_clusters == cluster
                if np.any(cluster_mask):
                    plt.scatter(
                        reg_x[cluster_mask],
                        reg_y[cluster_mask],
                        c=[cluster_colors[cluster]],
                        marker=register_markers[register],
                        s=30,
                        alpha=0.7,
                        label=f"Cluster {cluster}, {register}"
                        if len(unique_registers) > 1
                        else f"Cluster {cluster}",
                        edgecolors="black",
                        linewidth=0.3,
                    )

        plt.title(f"UMAP 2D: {base_title} - {embed_type}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

        # Create custom legend
        if len(unique_registers) > 1:
            # Legend for markers (registers)
            register_legend = [
                plt.Line2D(
                    [0],
                    [0],
                    marker=register_markers[reg],
                    color="gray",
                    linestyle="None",
                    markersize=8,
                    label=reg,
                )
                for reg in unique_registers
            ]
            # Legend for colors (clusters)
            cluster_legend = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=cluster_colors[cluster],
                    linestyle="None",
                    markersize=8,
                    label=f"Cluster {cluster}",
                )
                for cluster in unique_clusters
            ]

            # Two separate legends
            leg1 = plt.legend(
                handles=register_legend,
                title="Registers",
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
            )
            leg2 = plt.legend(
                handles=cluster_legend,
                title="Clusters",
                loc="upper left",
                bbox_to_anchor=(1.02, 0.7),
            )
            plt.gca().add_artist(leg1)
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/umap_clusters_combined.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Separate register plot
        plt.figure(figsize=(10, 8))
        for i, register in enumerate(unique_registers):
            reg_mask = np.array(registers) == register
            plt.scatter(
                embeddings_2d[reg_mask, 0],
                embeddings_2d[reg_mask, 1],
                c=colors[i % len(colors)],
                # marker=register_markers[register],
                s=30,
                alpha=0.7,
                label=register,
                edgecolors="black",
                linewidth=0.3,
            )

        plt.title(f"UMAP 2D: Colored by Register - {embed_type}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/umap_registers.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Standard cluster plot (always create this)
    plt.figure(figsize=(10, 8))
    cluster_labels = result["labels"].copy()
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.6,
        s=15,
    )

    plt.title(f"UMAP 2D: {base_title} - {embed_type}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_results_summary(
    result: Dict[str, Any],
    embed_type: str,
    output_dir: str,
    config: ClusteringConfig,
    registers: List[str] = None,
) -> None:
    """Save results summary to text file"""
    print(f"Saving results summary for {embed_type}")

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/hdbscan_results.txt", "w") as f:
        f.write(f"Embedding type: {embed_type}\n")
        f.write(f"DBCV threshold: {config.dbcv_threshold}\n")
        f.write(f"Max clusters allowed: {config.max_clusters}\n")
        f.write(f"Clustering scheme: Noise = Cluster 0, Real clusters = 1, 2, 3, ...\n")

        if registers:
            unique_registers = sorted(set(registers))
            f.write(f"Registers included: {', '.join(unique_registers)}\n")
            f.write(f"Total samples: {len(registers)}\n")

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
    labels: np.ndarray,
    texts: List[str],
    embed_type: str,
    output_dir: str,
    registers: List[str] = None,
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

            f.write(f"Size: {len(cluster_indices)} samples\n")

            if registers:
                cluster_registers = [registers[i] for i in cluster_indices]
                reg_counts = {
                    reg: cluster_registers.count(reg) for reg in set(cluster_registers)
                }
                f.write(f"Register distribution: {reg_counts}\n")

            f.write("\n")

            # Get up to 20 random examples
            sample_indices = np.random.choice(
                cluster_indices, min(20, len(cluster_indices)), replace=False
            )

            for i, idx in enumerate(sample_indices, 1):
                text_clean = texts[idx].replace("\n", "\\n").replace("\r", "\\r")
                reg_info = f" [{registers[idx]}]" if registers else ""
                f.write(f"{i}. {text_clean}{reg_info}\n")


def parse_filename(pkl_file: str) -> Tuple[str, str]:
    """Parse filename to extract language and register"""
    basename = os.path.splitext(os.path.basename(pkl_file))[0]

    # Expected format: {language}_embeds_{registers}.pkl
    if "_embeds_" in basename:
        parts = basename.split("_embeds_")
        if len(parts) == 2:
            language = parts[0]
            register = parts[1]
            return language, register

    # Fallback to original parsing logic
    parts = basename.split("_")
    language = None
    register = None

    for part in parts:
        if part in ["sv", "en", "fi"]:
            language = part
        elif part != "embeds":
            register = part

    if language is None or register is None:
        raise ValueError(
            f"Could not parse language and register from filename: {pkl_file}"
        )

    return language, register


def find_pickle_files(data_dir: str) -> List[str]:
    """Find all pickle files in the data directory"""
    pattern = os.path.join(data_dir, "*.pkl")
    return glob.glob(pattern)


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
            print(f"Warning: Skipping file {pkl_file}: {e}")

    return language_groups


def filter_files_by_target_register(
    files_and_registers: List[Tuple[str, str]], target_register: str
) -> List[Tuple[str, str]]:
    """Filter files that contain the target register"""
    filtered = []
    for pkl_file, register in files_and_registers:
        register_parts = register.split("-")
        if target_register in register_parts:
            filtered.append((pkl_file, register))
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
        print(f"Loading {pkl_file} (register: {original_register})")

        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # Extract data
        texts = [row["text"] for row in data]
        preds = [row["preds"] for row in data]

        # Add register information for each sample
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
        f"\nProcessing {embed_type}: {n_samples} samples with {embeddings.shape[1]}D embeddings"
    )

    # Check minimum dataset size
    if n_samples < config.min_dataset_size:
        print(f"Dataset too small ({n_samples} < {config.min_dataset_size}), skipping")
        return {"skipped": True, "reason": "too_small", "n_samples": n_samples}

    # Setup output directory
    if is_merged:
        output_dir = os.path.join(
            config.output_base_dir, "merged", language, register_group, embed_type
        )
    else:
        output_dir = os.path.join(
            config.output_base_dir, language, register_group, embed_type
        )

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
        result["labels"], embeddings, texts, preds, registers, embed_type, output_dir
    )
    create_combined_visualization(
        result, embeddings_2d, registers, embed_type, output_dir, config, is_merged
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
    print("Loading data...")

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # Extract embeddings for both types
    embeddings_dict = {}
    texts = [row["text"] for row in data]
    preds = [row["preds"] for row in data]

    # Try to get register from data, fallback to parsing filename
    if "register" in data[0]:
        registers = [row["register"] for row in data]
    else:
        # Fallback: parse from filename or use default
        try:
            _, register = parse_filename(pkl_file)
            registers = [register] * len(texts)
        except:
            registers = ["UNKNOWN"] * len(texts)

    for embed_type in ["embed_ref", "embed_last"]:
        if embed_type in data[0]:
            embeddings_dict[embed_type] = np.array([row[embed_type] for row in data])
        else:
            print(f"Warning: {embed_type} not found in data")

    if not embeddings_dict:
        raise RuntimeError("No valid embeddings found in data")

    return embeddings_dict, texts, preds, registers


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
        embeddings_dict, texts, preds, registers = load_data(pkl_file)
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
                embeddings,
                texts,
                preds,
                registers,
                embed_key,
                language,
                register,
                config,
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
    print(f"Files to merge: {[f[0] for f in files_to_merge]}")
    print(f"{'=' * 80}")

    try:
        # Load and merge data
        embeddings_dict, texts, preds, registers = load_and_merge_data(files_to_merge)
        print(f"Merged dataset: {len(texts)} samples")
        print(f"Found embeddings: {list(embeddings_dict.keys())}")

        # Print register distribution
        from collections import Counter

        reg_counts = Counter(registers)
        print(f"Register distribution: {dict(reg_counts)}")

    except Exception as e:
        print(f"Error loading merged data: {e}")
        return {"error": str(e)}

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
    """Run clustering in merge mode"""
    print("Running in MERGE mode")
    print(f"Data directory: {data_dir}")

    # Find all pickle files
    pkl_files = find_pickle_files(data_dir)
    if not pkl_files:
        print(f"No pickle files found in {data_dir}")
        return

    print(f"Found {len(pkl_files)} pickle files")

    # Group by language
    language_groups = group_files_by_language(pkl_files)
    print(f"Languages found: {list(language_groups.keys())}")

    # Target registers to create merged datasets for
    target_registers = ["NB", "OB", "ID"]

    successful = 0
    skipped = 0
    failed = 0

    for language, files_and_registers in language_groups.items():
        print(f"\n{'#' * 60}")
        print(f"Processing language: {language}")
        print(f"{'#' * 60}")

        for target_register in target_registers:
            print(f"\nLooking for files containing register: {target_register}")

            # Filter files that contain the target register
            files_to_merge = filter_files_by_target_register(
                files_and_registers, target_register
            )

            if not files_to_merge:
                print(f"No files found for {language}-{target_register}")
                skipped += 1
                continue

            print(f"Found {len(files_to_merge)} files to merge:")
            for pkl_file, register in files_to_merge:
                print(f"  - {os.path.basename(pkl_file)} ({register})")

            try:
                result = process_merged_dataset(
                    language, target_register, files_to_merge, config
                )

                if "error" in result:
                    failed += 1
                    print(f"✗ Failed: {result['error']}")
                else:
                    processed_any = False
                    for embed_result in result["results"].values():
                        if (
                            "skipped" not in embed_result
                            and "error" not in embed_result
                        ):
                            processed_any = True
                            break
                        elif "skipped" in embed_result:
                            skipped += 1

                    if processed_any:
                        successful += 1
                        print(f"✓ Successfully processed {language}-{target_register}")
                    else:
                        print(
                            f"○ All embeddings skipped for {language}-{target_register}"
                        )

            except Exception as e:
                failed += 1
                print(f"✗ Error processing {language}-{target_register}: {e}")

    # Summary
    print(f"\n{'=' * 80}")
    print("MERGE MODE PROCESSING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Successful: {successful}")
    print(f"Skipped (too small): {skipped}")
    print(f"Failed: {failed}")


def run_individual_mode(pkl_files: List[str], config: ClusteringConfig) -> None:
    """Run clustering in individual file mode"""
    print("Running in INDIVIDUAL mode")

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
    print("INDIVIDUAL MODE PROCESSING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files: {len(pkl_files)}")
    print(f"Successful: {successful}")
    print(f"Skipped (too small): {skipped}")
    print(f"Failed: {failed}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Clustering analysis tool")
    parser.add_argument(
        "files", nargs="*", help="Pickle files to process (ignored in merge mode)"
    )
    parser.add_argument("--merge", action="store_true", help="Run in merge mode")
    parser.add_argument(
        "--data-dir",
        default="../data/model_embeds/concat/xlm-r-reference/th-optimised/sm/",
        help="Data directory for merge mode",
    )

    args = parser.parse_args()

    # Setup configuration
    config = ClusteringConfig()

    if args.merge:
        # Merge mode: automatically find and process files
        run_merge_mode(args.data_dir, config)
    else:
        # Individual mode: process specified files
        if not args.files:
            print("Usage: python script.py pickle1.pkl pickle2.pkl ...")
            print("   or: python script.py --merge [--data-dir /path/to/data]")
            sys.exit(1)

        pkl_files = [f for f in args.files if os.path.exists(f)]
        if not pkl_files:
            print("ERROR: No valid pickle files found!")
            sys.exit(1)

        print(f"Found {len(pkl_files)} valid pickle files to process")
        run_individual_mode(pkl_files, config)

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
