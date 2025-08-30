import hashlib
import os
import pickle
import sys
import gc

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save memory
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


def process_file(pkl_file, cache_dir, dbcv_threshold=0.3):
    """Process a single pickle file and return results"""
    filename_without_ext = os.path.splitext(os.path.basename(pkl_file))[0]
    print(f"\n{'='*80}")
    print(f"Processing file: {pkl_file}")
    print(f"{'='*80}")

    # Initialize variables to None for proper cleanup
    data = None
    embeddings = None
    texts = None
    preds = None
    embeddings_50d = None
    embeddings_2d = None
    
    try:
        # Load data
        print("Loading data...")
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load {pkl_file}: {e}")
            return None

        # Extract data
        try:
            embeddings = np.array([row["embed_last"] for row in data])
            texts = [row["text"] for row in data]
            preds = [row["preds"] for row in data]
        except KeyError as e:
            print(f"ERROR: Missing required field in data: {e}")
            return None
        except Exception as e:
            print(f"ERROR: Failed to extract data: {e}")
            return None

        n_samples = len(embeddings)
        print(f"Loaded {n_samples} samples with {embeddings.shape[1]}D embeddings")

        # Clear original data to free memory
        del data
        gc.collect()

        if n_samples < 10:
            print(f"WARNING: Very small dataset ({n_samples} samples), results may not be meaningful")

        # Generate hash for embeddings to use as cache key
        embeddings_hash = get_embeddings_hash(embeddings)
        print(f"Embeddings hash: {embeddings_hash}")

        # UMAP reduction to 50D (with caching)
        embeddings_50d = get_or_compute_umap(
            embeddings,
            cache_dir,
            embeddings_hash,
            n_components=50,
            n_neighbors=min(30, n_samples - 1),  # Handle small datasets
            min_dist=0.0,
        )
        print("50D reduction complete")

        # UMAP reduction to 2D (with caching)
        embeddings_2d = get_or_compute_umap(
            embeddings, 
            cache_dir, 
            embeddings_hash, 
            n_components=2, 
            n_neighbors=min(15, n_samples - 1),  # Handle small datasets
            min_dist=0.1
        )
        print("2D reduction complete")

        # Clear original high-dimensional embeddings to free memory
        del embeddings
        gc.collect()

        # Define min_cluster_size values as percentages of dataset
        percentages = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        min_cluster_sizes = [max(2, int(n_samples * p / 100)) for p in percentages]

        # Filter out cluster sizes that would result in fewer than 100 examples per cluster
        # But relax this constraint for very small datasets
        min_samples_per_cluster = min(100, max(10, n_samples // 10))
        valid_params = []
        for p, size in zip(percentages, min_cluster_sizes):
            if size >= min_samples_per_cluster:
                valid_params.append((p, size))
            else:
                print(f"Skipping {p}% ({size} samples) - below {min_samples_per_cluster} sample threshold")

        if not valid_params:
            print(f"ERROR: No valid cluster sizes found! All percentages produce <{min_samples_per_cluster} samples.")
            return None

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

            # Check if all real clusters meet the minimum size requirement
            cluster_sizes = {}
            meets_size_requirement = True
            
            if n_real_clusters > 0:
                unique_labels = np.unique(labels)
                for cluster_id in unique_labels:
                    if cluster_id != 0:  # Skip noise cluster
                        cluster_size = np.sum(labels == cluster_id)
                        cluster_sizes[cluster_id] = cluster_size
                        if cluster_size < min_samples_per_cluster:
                            meets_size_requirement = False
            
            # Create size summary for logging
            if cluster_sizes:
                sizes_str = ", ".join([f"C{k}:{v}" for k, v in cluster_sizes.items()])
            else:
                sizes_str = "no real clusters"

            all_results.append(
                (
                    percentage,
                    min_size,
                    n_clusters,
                    n_real_clusters,
                    n_noise,
                    dbcv_score,
                    ch_score,
                    meets_size_requirement,  # Add this flag
                    cluster_sizes,
                )
            )
            
            status_marker = "✓" if meets_size_requirement else "✗"
            print(
                f"  → {n_real_clusters} real clusters + {n_noise} noise points, DBCV: {dbcv_score:.4f}, CH: {ch_score:.2f} [{sizes_str}] {status_marker}"
            )

            # Only consider this result if it meets size requirements AND has good DBCV
            if dbcv_score > best_score and n_real_clusters > 1 and meets_size_requirement:
                best_score = dbcv_score
                best_labels = labels.copy()  # Make a copy to avoid reference issues
                best_min_size = min_size
                best_n_real_clusters = n_real_clusters
                print(f"    → New best result!")

            # Force garbage collection after each HDBSCAN run
            gc.collect()

        if best_labels is None:
            print(f"\nWarning: No valid clustering found that meets size requirement (≥{min_samples_per_cluster} per cluster)!")
            print("Looking for fallback options...")
            
            # Try to find results with valid clustering but relaxed size requirements
            valid_clustering_results = [r for r in all_results if r[3] > 1]  # n_real_clusters > 1
            
            if valid_clustering_results:
                # Use the result with best DBCV among valid clusterings, even if size requirements aren't met
                best_result = max(valid_clustering_results, key=lambda x: x[5])  # x[5] is dbcv_score
                percentage, min_size, n_clusters, n_real_clusters, n_noise, dbcv_score, ch_score, meets_size_req, cluster_sizes = best_result
                print(f"Using fallback (ignoring size requirement): {percentage}% ({min_size} samples) with {n_real_clusters} real clusters")
                print(f"Cluster sizes: {cluster_sizes}")
                
                # Get the cached result for the fallback parameters
                fallback_result = get_or_compute_hdbscan(
                    cache_dir, embeddings_hash, embeddings_50d, min_size
                )
                best_labels = fallback_result["labels"].copy()
                best_min_size = min_size
                best_n_real_clusters = n_real_clusters
                best_score = dbcv_score
            else:
                # No valid clustering found at all - this was the original fallback
                best_result = max(all_results, key=lambda x: x[3])  # x[3] is n_real_clusters  
                percentage, min_size, n_clusters, n_real_clusters, n_noise, dbcv_score, ch_score, meets_size_req, cluster_sizes = best_result
                print(f"Using final fallback: {percentage}% ({min_size} samples) with {n_real_clusters} real clusters")

                # Get the cached result for the fallback parameters
                fallback_result = get_or_compute_hdbscan(
                    cache_dir, embeddings_hash, embeddings_50d, min_size
                )
                best_labels = fallback_result["labels"].copy()
                best_min_size = min_size
                best_n_real_clusters = n_real_clusters
                best_score = dbcv_score

        # Check if best DBCV score is below threshold
        if best_score < dbcv_threshold:
            print(
                f"\nDBCV threshold check: Best DBCV ({best_score:.4f}) < threshold ({dbcv_threshold})"
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
            best_result_data = next(
                result for result in all_results if result[1] == best_min_size
            )
            n_noise_best = best_result_data[4]  # n_noise is at index 4
            cluster_sizes_best = best_result_data[8]  # cluster_sizes is at index 8

            print(
                f"\nBest result: min_cluster_size={best_min_size}, {best_n_real_clusters} real clusters, {n_noise_best} noise points, DBCV={best_score:.4f}, CH={best_ch_score:.2f}"
            )
            print(f"Final cluster sizes: {cluster_sizes_best}")

        # Create output directory
        output_dir = f"clusters/{filename_without_ext}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to {output_dir}/")

        # Plot 2D UMAP with clusters
        print("Creating UMAP visualization...")
        
        # Create plot with explicit figure management
        fig, ax = plt.subplots(figsize=(10, 8))

        # Color clusters
        colors = best_labels.copy()
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap="tab10", alpha=0.6
        )

        # Handle title formatting for single cluster case
        if best_n_real_clusters == 1:
            if isinstance(best_score, str):  # Single cluster due to threshold
                title = f"UMAP 2D with {best_n_real_clusters} cluster (forced due to low DBCV < {dbcv_threshold})"
            else:
                title = f"UMAP 2D with {best_n_real_clusters} cluster"
        else:
            n_noise_display = n_noise_best if "n_noise_best" in locals() else 0
            title = f"UMAP 2D with {best_n_real_clusters} clusters + {n_noise_display} noise (HDBSCAN min_size={best_min_size}, DBCV: {best_score:.3f})"

        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/umap_clusters.png", dpi=300, bbox_inches="tight")
        
        # Explicitly close the figure and clear matplotlib cache
        plt.close(fig)
        plt.close('all')
        gc.collect()
        
        print("UMAP plot saved")

        # Save all results sorted by DBCV score
        print("Saving results summary...")
        with open(f"{output_dir}/hdbscan_results.txt", "w") as f:
            f.write(f"Dataset: {pkl_file}\n")
            f.write(f"Dataset size: {n_samples} samples\n")
            f.write(f"Embeddings hash: {embeddings_hash}\n")
            f.write(f"DBCV threshold: {dbcv_threshold}\n")
            f.write(f"Minimum samples per cluster requirement: {min_samples_per_cluster}\n")
            f.write(f"Clustering scheme: Noise = Cluster 0, Real clusters = 1, 2, 3, ...\n")
            if isinstance(best_score, str):
                f.write(f"Best result: Single cluster (all samples), reason: {best_score}\n\n")
            else:
                f.write(
                    f"Best result: min_cluster_size={best_min_size}, {best_n_real_clusters} real clusters, {n_noise_best} noise points, DBCV={best_score}, Calinski-Harabasz={best_ch_score}\n\n"
                )
            f.write("All results (sorted by DBCV score):\n")
            f.write(
                "Rank | Percentage | Min_Size | Total_Clusters | Real_Clusters | Noise_Points | DBCV Score | Calinski-Harabasz | Size_Check | Cluster_Sizes | Notes\n"
            )
            f.write("-" * 150 + "\n")
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
                meets_size_req,
                cluster_sizes,
            ) in enumerate(sorted_results):
                marker = " <-- BEST" if min_size == best_min_size else ""
                size_check = "PASS" if meets_size_req else "FAIL"
                sizes_str = ",".join([f"C{k}:{v}" for k, v in cluster_sizes.items()]) if cluster_sizes else "none"
                f.write(
                    f"{i + 1:4d} | {percentage:9.1f}% | {min_size:8d} | {n_clusters:13d} | {n_real_clusters:12d} | {n_noise:11d} | {dbcv_score:10.4f} | {ch_score:17.2f} | {size_check:10s} | {sizes_str:13s} |{marker}\n"
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

        # Return summary for batch processing
        result_summary = {
            'filename': pkl_file,
            'n_samples': n_samples,
            'n_real_clusters': best_n_real_clusters,
            'n_noise': n_noise_best if 'n_noise_best' in locals() else 0,
            'dbcv_score': best_score,
            'ch_score': best_ch_score if 'best_ch_score' in locals() else 'N/A',
            'output_dir': output_dir
        }

        print(f"Analysis complete for {pkl_file}! Results saved in {output_dir}/")
        if isinstance(best_score, str):  # Single cluster case
            print(f"Final result: 1 cluster with all {n_samples} samples")
        else:
            print(
                f"Final best: min_cluster_size={best_min_size}, {best_n_real_clusters} real clusters, {n_noise_best} noise points, DBCV={best_score:.4f}, CH={best_ch_score:.2f}"
            )
        
        return result_summary

    finally:
        # Aggressive cleanup of all variables
        print("Cleaning up memory...")
        
        # Delete all large variables
        variables_to_delete = [
            'data', 'embeddings', 'embeddings_50d', 'embeddings_2d', 
            'texts', 'preds', 'best_labels', 'all_results'
        ]
        
        for var_name in variables_to_delete:
            if var_name in locals():
                del locals()[var_name]
        
        # Force garbage collection
        gc.collect()
        
        # Clear matplotlib cache
        plt.close('all')
        if hasattr(plt, 'clf'):
            plt.clf()
        if hasattr(plt, 'cla'):
            plt.cla()
        
        print("Memory cleanup complete")n_noise': n_noise_best if 'n_noise_best' in locals() else 0,
        'dbcv_score': best_score,
        'ch_score': best_ch_score if 'best_ch_score' in locals() else 'N/A',
        'output_dir': output_dir
    }

    print(f"Analysis complete for {pkl_file}! Results saved in {output_dir}/")
    if isinstance(best_score, str):  # Single cluster case
        print(f"Final result: 1 cluster with all {n_samples} samples")
    else:
        print(
            f"Final best: min_cluster_size={best_min_size}, {best_n_real_clusters} real clusters, {n_noise_best} noise points, DBCV={best_score:.4f}, CH={best_ch_score:.2f}"
        )
    
    return result_summary


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py pickle1.pkl pickle2.pkl pickle3.pkl ...")
        print("       python script.py *.pkl")
        sys.exit(1)

    pkl_files = sys.argv[1:]
    
    # Validate files exist
    valid_files = []
    for pkl_file in pkl_files:
        if os.path.exists(pkl_file):
            valid_files.append(pkl_file)
        else:
            print(f"WARNING: File not found: {pkl_file}")
    
    if not valid_files:
        print("ERROR: No valid pickle files found!")
        sys.exit(1)

    print(f"Found {len(valid_files)} valid pickle files to process:")
    for f in valid_files:
        print(f"  - {f}")

    # Create cache directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "umap_cache")
    print(f"\nCache directory: {cache_dir}")

    # Process each file
    successful_results = []
    failed_files = []

            for i, pkl_file in enumerate(valid_files, 1):
        print(f"\n{'#'*80}")
        print(f"PROCESSING FILE {i}/{len(valid_files)}: {pkl_file}")
        print(f"{'#'*80}")
        
        try:
            result = process_file(pkl_file, cache_dir)
            if result:
                successful_results.append(result)
                print(f"✓ Successfully processed {pkl_file}")
            else:
                failed_files.append(pkl_file)
                print(f"✗ Failed to process {pkl_file}")
        except Exception as e:
            failed_files.append(pkl_file)
            print(f"✗ Error processing {pkl_file}: {e}")
        
        # Force garbage collection between files
        print("Running inter-file garbage collection...")
        gc.collect()
        
        # Clear any remaining matplotlib figures
        plt.close('all')

    # Print summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {len(valid_files)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_files)}")

    if successful_results:
        print(f"\nSuccessful results:")
        print(f"{'File':<40} {'Samples':<8} {'Clusters':<8} {'Noise':<8} {'DBCV':<10}")
        print("-" * 80)
        for result in successful_results:
            dbcv_str = f"{result['dbcv_score']:.4f}" if isinstance(result['dbcv_score'], (int, float)) else str(result['dbcv_score'])[:10]
            print(f"{os.path.basename(result['filename']):<40} {result['n_samples']:<8} {result['n_real_clusters']:<8} {result['n_noise']:<8} {dbcv_str:<10}")

    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")

    # Print cache statistics
    if os.path.exists(cache_dir):
        umap_cache_files = [
            f for f in os.listdir(cache_dir) if f.startswith("umap_") and f.endswith(".pkl")
        ]
        hdbscan_cache_files = [
            f for f in os.listdir(cache_dir) if f.startswith("hdbscan_") and f.endswith(".pkl")
        ]
        print(f"\nCache statistics:")
        print(f"UMAP cached reductions: {len(umap_cache_files)}")
        print(f"HDBSCAN cached results: {len(hdbscan_cache_files)}")

    print(f"\nBatch processing complete!")


if __name__ == "__main__":
    main()