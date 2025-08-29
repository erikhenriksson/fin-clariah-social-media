import glob
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

    return embeddings, texts, preds


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
    return labels


def assign_noise_to_clusters(embeddings, labels):
    """Assign noise points to nearest cluster centroids"""
    if -1 not in labels:
        return labels

    print("Assigning noise points to nearest clusters...")

    # Get cluster centroids
    unique_clusters = [c for c in set(labels) if c != -1]
    centroids = np.array(
        [
            np.mean(embeddings[labels == cluster_id], axis=0)
            for cluster_id in unique_clusters
        ]
    )

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
    """Calculate silhouette score for each cluster"""
    print("Calculating per-cluster silhouette scores...")

    unique_clusters = sorted([c for c in set(labels) if c != -1])
    sample_silhouettes = silhouette_samples(embeddings, labels)

    cluster_silhouettes = {}
    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        cluster_silhouette = np.mean(sample_silhouettes[cluster_mask])
        cluster_silhouettes[cluster_id] = cluster_silhouette
        print(f"Cluster {cluster_id}: silhouette = {cluster_silhouette:.3f}")

    return cluster_silhouettes


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
    sil_score = silhouette_score(embeddings, labels)

    # Save cluster analysis
    analysis_file = output_dir / f"cluster_analysis_{n_clusters}.txt"

    with open(analysis_file, "w", encoding="utf-8") as f:
        f.write(f"HDBSCAN CLUSTER ANALYSIS\n")
        f.write(f"{'=' * 40}\n\n")
        f.write(f"Total clusters: {n_clusters}\n")
        f.write(f"Overall silhouette score: {sil_score:.3f}\n\n")

        f.write(f"INDIVIDUAL CLUSTER SILHOUETTE SCORES:\n")
        f.write(f"{'-' * 40}\n")
        for cluster_id in unique_clusters:
            cluster_score = cluster_silhouettes[cluster_id]
            cluster_size = np.sum(labels == cluster_id)
            f.write(
                f"Cluster {cluster_id}: {cluster_score:.3f} (size: {cluster_size})\n"
            )
        f.write(f"\n")

        for cluster_id in unique_clusters:
            members = np.where(labels == cluster_id)[0]
            cluster_score = cluster_silhouettes[cluster_id]

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
    filename = Path(pickle_path).stem
    output_dir = Path(results_dir) / filename

    print(f"\nProcessing {filename}...")

    try:
        # Load data
        embeddings, texts, preds = load_data(pickle_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")

        # Process pipeline
        reduced_embeddings = reduce_dimensions(embeddings)
        labels = cluster_documents(reduced_embeddings)
        labels_no_noise = assign_noise_to_clusters(reduced_embeddings, labels)

        # Quality gate - check overall silhouette score
        overall_silhouette = silhouette_score(reduced_embeddings, labels_no_noise)
        print(f"Overall silhouette score: {overall_silhouette:.3f}")

        if overall_silhouette < 0.5:
            print(
                f"Skipping {filename}: Poor clustering quality (silhouette = {overall_silhouette:.3f} < 0.5)"
            )

            # Save skip report
            skip_report = output_dir / f"clustering_skipped_{filename}.txt"
            with open(skip_report, "w", encoding="utf-8") as f:
                f.write(f"CLUSTERING SKIPPED FOR {filename}\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(
                    f"Reason: Poor overall silhouette score ({overall_silhouette:.3f} < 0.5)\n"
                )
                f.write(
                    f"This indicates the data does not have clear cluster structure.\n\n"
                )
                f.write(f"Dataset details:\n")
                f.write(f"- Documents: {len(embeddings)}\n")
                f.write(f"- Embedding dimension: {embeddings.shape[1]}\n")
                f.write(f"- Reduced dimension: {reduced_embeddings.shape[1]}\n")
                f.write(f"- Attempted clusters: {len(set(labels_no_noise))}\n")

            return False

        # Calculate per-cluster scores and generate outputs
        cluster_silhouettes = calculate_cluster_silhouettes(
            reduced_embeddings, labels_no_noise
        )
        analyze_clusters(
            labels_no_noise,
            texts,
            preds,
            reduced_embeddings,
            output_dir,
            cluster_silhouettes,
        )
        create_visualization(embeddings, labels_no_noise, output_dir)

        return True

    except ValueError as e:
        print(f"Skipping {filename}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False


def main():
    """Process all pickle files in the directory"""
    pkl_directory = "../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/"
    pkl_files = glob.glob(os.path.join(pkl_directory, "*.pkl"))

    if not pkl_files:
        print(f"No .pkl files found in {pkl_directory}")
        return

    print(f"Found {len(pkl_files)} files to process")

    results_dir = f"hdbscan_results_c{UMAP_COMPONENTS}"
    Path(results_dir).mkdir(exist_ok=True)

    successful = 0
    skipped_quality = 0
    skipped_size = 0

    for pkl_file in pkl_files:
        result = process_file(pkl_file, results_dir)
        if result is True:
            successful += 1
        elif result is False:
            filename = Path(pkl_file).stem
            skip_report = (
                Path(results_dir) / filename / f"clustering_skipped_{filename}.txt"
            )
            if skip_report.exists():
                skipped_quality += 1
            else:
                skipped_size += 1

    print(f"\nCompleted: {successful}/{len(pkl_files)} files processed successfully")
    print(f"Skipped due to poor quality: {skipped_quality}")
    print(f"Skipped due to small size: {skipped_size}")
    print(f"Results saved to: {results_dir}/")


if __name__ == "__main__":
    main()
