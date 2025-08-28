import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import silhouette_samples, silhouette_score

from hdbscan import HDBSCAN

UMAP_COMPONENTS = 50
SIZE_LIMIT = 1000


def load_data(pickle_path):
    """Load embeddings and metadata from pickle file"""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    embeddings = np.array([row["embed_last"] for row in data])
    texts = [row["text"] for row in data]
    preds = [row["preds"] for row in data]

    print(f"Loaded {len(embeddings)} documents with {embeddings.shape[1]}D embeddings")

    if len(embeddings) < SIZE_LIMIT:
        raise ValueError(
            f"Dataset too small ({len(embeddings)} documents). Need {SIZE_LIMIT}+ for HDBSCAN."
        )

    return embeddings, texts, preds, data


def reduce_dimensions(embeddings):
    """Apply UMAP to reduce dimensions for clustering"""
    print("Reducing dimensions with UMAP...")

    n_neighbors = min(30, len(embeddings) // 30)

    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS, n_neighbors=n_neighbors, min_dist=0.0
    )

    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"Reduced to {reduced_embeddings.shape[1]}D")

    return reduced_embeddings


def cluster_documents(embeddings):
    """Run HDBSCAN clustering"""
    print("Clustering with HDBSCAN...")

    min_cluster_size = len(embeddings) // 10

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=1, cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(
        f"Found {n_clusters} clusters with {n_noise} noise points ({n_noise / len(labels) * 100:.1f}%)"
    )

    return labels, min_cluster_size


def assign_noise_to_clusters(embeddings, labels):
    """Assign noise points to nearest cluster centroids"""
    if -1 not in labels:
        return labels

    print("Assigning noise points to nearest clusters...")

    # Get cluster centroids
    unique_clusters = [c for c in set(labels) if c != -1]
    centroids = []

    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        centroid = np.mean(embeddings[cluster_mask], axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Assign noise points
    new_labels = labels.copy()
    noise_indices = np.where(labels == -1)[0]

    for idx in noise_indices:
        distances = np.linalg.norm(centroids - embeddings[idx], axis=1)
        nearest_cluster = unique_clusters[np.argmin(distances)]
        new_labels[idx] = nearest_cluster

    print(f"Assigned {len(noise_indices)} noise points to clusters")
    return new_labels


def calculate_cluster_silhouettes(embeddings, labels):
    """Calculate silhouette score for each individual cluster"""
    print("Calculating per-cluster silhouette scores...")

    unique_clusters = sorted([c for c in set(labels) if c != -1])
    cluster_silhouettes = {}

    # Get silhouette scores for all samples
    sample_silhouettes = silhouette_samples(embeddings, labels)

    # Calculate average silhouette for each cluster
    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        cluster_silhouette = np.mean(sample_silhouettes[cluster_mask])
        cluster_silhouettes[cluster_id] = cluster_silhouette
        print(f"Cluster {cluster_id}: silhouette = {cluster_silhouette:.3f}")

    return cluster_silhouettes


def merge_low_quality_clusters(embeddings, labels, cluster_silhouettes, threshold=0.5):
    """Merge clusters with silhouette scores <= threshold into a single cluster"""
    print(f"Merging clusters with silhouette scores <= {threshold}...")

    # Identify clusters to merge
    clusters_to_merge = [
        cluster_id
        for cluster_id, score in cluster_silhouettes.items()
        if score <= threshold
    ]

    clusters_to_keep = [
        cluster_id
        for cluster_id, score in cluster_silhouettes.items()
        if score > threshold
    ]

    if not clusters_to_merge:
        print("No clusters need merging - all have good silhouette scores!")
        return labels, cluster_silhouettes

    print(f"Merging {len(clusters_to_merge)} low-quality clusters: {clusters_to_merge}")
    print(f"Keeping {len(clusters_to_keep)} high-quality clusters: {clusters_to_keep}")

    # Create new labels
    new_labels = labels.copy()

    # Find the next available cluster ID for the merged cluster
    max_cluster_id = max(set(labels)) if set(labels) else -1
    merged_cluster_id = max_cluster_id + 1

    # Merge low-quality clusters
    for cluster_id in clusters_to_merge:
        cluster_mask = labels == cluster_id
        new_labels[cluster_mask] = merged_cluster_id

    # Recalculate silhouette for the new merged cluster
    updated_silhouettes = {}

    # Keep scores for unchanged clusters
    for cluster_id in clusters_to_keep:
        updated_silhouettes[cluster_id] = cluster_silhouettes[cluster_id]

    # Calculate score for merged cluster
    if clusters_to_merge:
        merged_mask = new_labels == merged_cluster_id
        merged_silhouette = np.mean(
            silhouette_samples(embeddings, new_labels)[merged_mask]
        )
        updated_silhouettes[merged_cluster_id] = merged_silhouette

        print(
            f"Merged cluster {merged_cluster_id}: silhouette = {merged_silhouette:.3f}"
        )

    n_final_clusters = len(set(new_labels))
    print(f"Final result: {n_final_clusters} clusters after merging")

    return new_labels, updated_silhouettes


def save_clustered_data(original_data, labels, pickle_path):
    """Save the original data with cluster labels added"""
    print("Saving clustered data...")

    # Add cluster labels to each document
    clustered_data = []
    for i, row in enumerate(original_data):
        new_row = row.copy()
        new_row["cluster_label"] = int(labels[i])
        clustered_data.append(new_row)

    # Create output filename with _clustered suffix
    input_path = Path(pickle_path)
    output_path = input_path.parent / (input_path.stem + "_clustered.pkl")

    # Save the clustered data
    with open(output_path, "wb") as f:
        pickle.dump(clustered_data, f)

    print(f"Clustered data saved to {output_path}")
    return output_path


def create_visualization(embeddings, labels, output_dir):
    """Create 2D visualization of clusters"""
    print("Creating visualization...")

    # Reduce to 2D for visualization
    reducer_2d = umap.UMAP(n_components=2)
    embeddings_2d = reducer_2d.fit_transform(embeddings)

    # Plot clusters
    plt.figure(figsize=(12, 8))
    unique_clusters = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f"Cluster {cluster_id} ({np.sum(mask)} docs)",
            alpha=0.7,
            s=10,
        )

    plt.title(f"HDBSCAN Clusters: {len(unique_clusters)}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plot_path = output_dir / f"clusters_{len(unique_clusters)}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to {plot_path}")


def analyze_clusters(labels, texts, preds, embeddings, output_dir, cluster_silhouettes):
    """Analyze cluster composition and save results"""
    print("Analyzing clusters...")

    unique_clusters = sorted(set(labels))
    n_clusters = len(unique_clusters)

    # Calculate overall silhouette score
    sil_score = silhouette_score(embeddings, labels)

    # Save cluster analysis
    analysis_file = output_dir / f"cluster_analysis_{n_clusters}.txt"

    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write(f"HDBSCAN CLUSTER ANALYSIS\n")
        f.write(f"={'=' * 40}\n\n")
        f.write(f"Total clusters: {n_clusters}\n")
        f.write(f"Overall silhouette score: {sil_score:.3f}\n\n")

        f.write(f"INDIVIDUAL CLUSTER SILHOUETTE SCORES:\n")
        f.write(f"{'-' * 40}\n")
        for cluster_id in unique_clusters:
            cluster_score = cluster_silhouettes.get(cluster_id, "N/A")
            cluster_size = np.sum(labels == cluster_id)
            f.write(
                f"Cluster {cluster_id}: {cluster_score:.3f} (size: {cluster_size})\n"
            )
        f.write(f"\n")

        for cluster_id in unique_clusters:
            members = np.where(labels == cluster_id)[0]
            cluster_score = cluster_silhouettes.get(cluster_id, "N/A")

            f.write(
                f"\n--- Cluster {cluster_id} ({len(members)} documents, silhouette: {cluster_score:.3f}) ---\n"
            )

            # Sample up to 20 documents from cluster
            sample_size = min(20, len(members))
            sample_indices = np.random.choice(members, sample_size, replace=False)

            for i, idx in enumerate(sample_indices):
                doc_text = texts[idx].replace("\n", " ")[:200] + "..."
                f.write(f"{i + 1}. [{idx}] {preds[idx]} {doc_text}\n")

            if len(members) > sample_size:
                f.write(f"... and {len(members) - sample_size} more documents\n")

    print(f"Analysis saved to {analysis_file}")
    print(f"Final results: {n_clusters} clusters, silhouette score: {sil_score:.3f}")


def process_file(pickle_path, results_dir="hdbscan_results"):
    """Process a single pickle file through the complete pipeline"""

    # Setup output directory
    filename = Path(pickle_path).stem
    output_dir = Path(results_dir) / filename

    print(f"\nProcessing {filename}...")

    try:
        # Load data - this will raise ValueError if too small
        embeddings, texts, preds, original_data = load_data(pickle_path)

        # Only create output directory AFTER confirming data is valid
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")

        # Reduce dimensions
        reduced_embeddings = reduce_dimensions(embeddings)

        # Cluster
        labels, min_cluster_size = cluster_documents(reduced_embeddings)

        # Handle noise
        labels_no_noise = assign_noise_to_clusters(reduced_embeddings, labels)

        # Calculate per-cluster silhouette scores
        cluster_silhouettes = calculate_cluster_silhouettes(
            reduced_embeddings, labels_no_noise
        )

        # Merge low-quality clusters
        final_labels, final_silhouettes = merge_low_quality_clusters(
            reduced_embeddings, labels_no_noise, cluster_silhouettes, threshold=0.5
        )

        # Save clustered data with labels
        save_clustered_data(original_data, final_labels, pickle_path)

        # Analyze and save results
        analyze_clusters(
            final_labels,
            texts,
            preds,
            reduced_embeddings,
            output_dir,
            final_silhouettes,
        )

        # Visualize
        create_visualization(embeddings, final_labels, output_dir)

        return True

    except ValueError as e:
        print(f"Skipping {filename}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False


def main():
    """Process a single specified pickle file"""
    parser = argparse.ArgumentParser(
        description="Process a single pickle file with HDBSCAN clustering"
    )
    parser.add_argument("pickle_file", help="Path to the pickle file to process")

    args = parser.parse_args()

    pickle_path = args.pickle_file

    if not os.path.exists(pickle_path):
        print(f"Error: File {pickle_path} does not exist")
        return

    if not pickle_path.endswith(".pkl"):
        print(f"Warning: File {pickle_path} does not have .pkl extension")

    print(f"Processing file: {pickle_path}")

    results_dir = f"hdbscan_results_c{UMAP_COMPONENTS}_one"
    Path(results_dir).mkdir(exist_ok=True)

    if process_file(pickle_path, results_dir):
        print(f"\nCompleted: File processed successfully")
        print(f"Results saved to: {results_dir}/")
    else:
        print(f"\nFailed to process file: {pickle_path}")


if __name__ == "__main__":
    main()
