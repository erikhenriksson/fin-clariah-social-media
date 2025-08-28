import glob
import os
import pickle
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from hdbscan import HDBSCAN

warnings.filterwarnings("ignore")


class HDBSCANSubregisterAnalyzer:
    def __init__(self, pickle_path, results_base_dir="hdbscan_results"):
        """Load and initialize the embedding data"""
        self.pickle_path = pickle_path

        print(f"Loading data from {pickle_path}...")

        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)

        # Extract embeddings and metadata
        self.embeddings = np.array([row["embed_last"] for row in self.data])
        self.texts = [row["text"] for row in self.data]
        self.preds = [row["preds"] for row in self.data]
        self.labels = [
            row["preds"][0] if row["preds"] else "unknown" for row in self.data
        ]

        print(f"Loaded {len(self.embeddings)} documents")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
        print(f"Register distribution: {Counter(self.labels)}")

        # Check minimum size requirement
        if len(self.embeddings) < 500:
            raise ValueError(
                f"Dataset too small ({len(self.embeddings)} documents). Minimum 500 required for HDBSCAN."
            )

        # Create output directories
        input_filename = Path(pickle_path).stem
        self.results_base_dir = Path(results_base_dir)
        self.results_base_dir.mkdir(exist_ok=True)
        self.output_dir = self.results_base_dir / input_filename
        self.output_dir.mkdir(exist_ok=True)

        print(f"Output will be saved to: {self.output_dir}")

        # Normalize embeddings once for cosine distance
        # self.embeddings_norm = normalize(self.embeddings, norm="l2").astype(np.float64)
        self.embeddings_norm = self.embeddings

    def reduce_dimensions_umap(self):
        """Apply UMAP for dimensionality reduction - create both clustering and visualization embeddings"""
        print("Reducing dimensions with UMAP...")

        # Parameters for clustering (moderate reduction)
        n_neighbors = min(15, len(self.embeddings) // 30)

        # UMAP for clustering (to ~15-30 dimensions)
        print(f"Creating clustering embeddings (15D, n_neighbors={n_neighbors})...")
        self.umap_clustering = umap.UMAP(
            n_components=15,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            # metric="cosine",
            random_state=42,
            low_memory=True,
        )

        self.embeddings_clustering = self.umap_clustering.fit_transform(
            self.embeddings_norm
        )
        print(f"Clustering embeddings shape: {self.embeddings_clustering.shape}")

        # UMAP for visualization (to 2D, using original embeddings)
        print(f"Creating visualization embeddings (2D, n_neighbors={n_neighbors})...")
        self.umap_visualization = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            # metric="cosine",
            random_state=42,
            low_memory=True,
        )

        self.embeddings_2d = self.umap_visualization.fit_transform(self.embeddings_norm)
        print(f"Visualization embeddings shape: {self.embeddings_2d.shape}")

        return self.embeddings_clustering

    def find_optimal_hdbscan_params(self):
        """Find optimal HDBSCAN parameters by testing different min_cluster_size values"""
        print("=" * 60)
        print("OPTIMIZING HDBSCAN PARAMETERS")
        print("=" * 60)

        # Test different min_cluster_size values - include smaller clusters
        min_sizes = [
            5,
            10,
            15,
            max(20, len(self.embeddings) // 200),  # 0.5% of data
            max(25, len(self.embeddings) // 100),  # 1% of data
            max(50, len(self.embeddings) // 50),  # 2% of data
        ]

        # Remove duplicates and ensure reasonable range
        min_sizes = sorted(
            list(set([s for s in min_sizes if s <= len(self.embeddings) // 4]))
        )

        results = {}

        for min_size in min_sizes:
            print(f"Testing min_cluster_size={min_size}...")

            # Use direct metric computation instead of precomputed distances
            clusterer = HDBSCAN(
                min_cluster_size=min_size,
                min_samples=max(3, min_size // 5),
                # metric="cosine",
                cluster_selection_epsilon=0.0,
                cluster_selection_method="eom",
                # random_state=42,
            )

            cluster_labels = clusterer.fit_predict(self.embeddings_clustering)

            # Count non-noise clusters and noise points
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            noise_pct = (n_noise / len(cluster_labels)) * 100

            # Only compute silhouette if we have valid clusters and not too much noise
            if n_clusters >= 2 and noise_pct < 70:
                # For silhouette calculation, exclude noise points
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 50:
                    sil_score = silhouette_score(
                        self.embeddings_clustering[non_noise_mask],
                        cluster_labels[non_noise_mask],
                        # metric="cosine",
                    )
                else:
                    sil_score = -1
            else:
                sil_score = -1

            results[min_size] = {
                "clusterer": clusterer,
                "labels": cluster_labels,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": noise_pct,
                "silhouette": sil_score,
            }

            print(
                f"  Clusters: {n_clusters}, Noise: {n_noise} ({noise_pct:.1f}%), Silhouette: {sil_score:.3f}"
            )

        # Find best parameters - balance clusters and noise
        valid_results = {k: v for k, v in results.items() if v["silhouette"] > 0}

        if not valid_results:
            raise ValueError(
                "No valid HDBSCAN clustering found. Dataset may not have clear cluster structure."
            )

        # Choose result with good balance: prefer more clusters but not excessive noise
        best_min_size = max(
            valid_results.keys(),
            key=lambda k: (
                valid_results[k]["silhouette"] - valid_results[k]["noise_pct"] / 100
            ),
        )

        best_result = results[best_min_size]

        print("\n" + "=" * 60)
        print("HDBSCAN PARAMETER OPTIMIZATION RESULTS")
        print("=" * 60)
        print("\nmin_cluster_size | Clusters | Noise% | Silhouette")
        print("-" * 50)

        for min_size in sorted(results.keys()):
            r = results[min_size]
            marker = " <- BEST" if min_size == best_min_size else ""
            print(
                f"      {min_size:3d}        |    {r['n_clusters']:2d}    | {r['noise_pct']:5.1f}% |   {r['silhouette']:6.3f}{marker}"
            )

        print(f"\n✓ OPTIMAL PARAMETERS:")
        print(f"  min_cluster_size: {best_min_size}")
        print(f"  Clusters found: {best_result['n_clusters']}")
        print(
            f"  Noise points: {best_result['n_noise']} ({best_result['noise_pct']:.1f}%)"
        )
        print(f"  Silhouette score: {best_result['silhouette']:.3f}")

        # Save optimization results
        optimization_file = self.output_dir / "hdbscan_optimization.txt"
        with open(optimization_file, "w", encoding="utf-8") as f:
            f.write("HDBSCAN PARAMETER OPTIMIZATION\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Optimal min_cluster_size: {best_min_size}\n")
            f.write(f"Clusters found: {best_result['n_clusters']}\n")
            f.write(
                f"Noise points: {best_result['n_noise']} ({best_result['noise_pct']:.1f}%)\n"
            )
            f.write(f"Silhouette score: {best_result['silhouette']:.3f}\n\n")
            f.write("All tested parameters:\n")
            f.write("min_cluster_size | Clusters | Noise% | Silhouette\n")
            f.write("-" * 50 + "\n")
            for min_size in sorted(results.keys()):
                r = results[min_size]
                marker = " <- OPTIMAL" if min_size == best_min_size else ""
                f.write(
                    f"      {min_size:3d}        |    {r['n_clusters']:2d}    | {r['noise_pct']:5.1f}% |   {r['silhouette']:6.3f}{marker}\n"
                )

        print(f"Optimization results saved to: {optimization_file}")

        return best_min_size, best_result, results

    def handle_noise_points(self, labels):
        """Assign noise points to nearest cluster using cluster centroids"""
        print("Assigning noise points to nearest clusters...")

        # Find noise points
        noise_mask = labels == -1
        n_noise = np.sum(noise_mask)

        if n_noise == 0:
            print("No noise points found.")
            return labels.copy()

        print(f"Found {n_noise} noise points ({n_noise / len(labels) * 100:.1f}%)")

        # Get cluster centers from non-noise points (using clustering embeddings)
        unique_clusters = sorted([c for c in set(labels) if c != -1])
        cluster_centers = []

        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            cluster_center = np.mean(self.embeddings_clustering[cluster_mask], axis=0)
            cluster_center = cluster_center / np.linalg.norm(cluster_center)
            cluster_centers.append(cluster_center)

        cluster_centers = np.array(cluster_centers)

        # Assign noise points to nearest cluster center using cosine similarity
        noise_embeddings = self.embeddings_clustering[noise_mask]
        # Normalize noise embeddings
        noise_embeddings = noise_embeddings / np.linalg.norm(
            noise_embeddings, axis=1, keepdims=True
        )

        similarities = np.dot(noise_embeddings, cluster_centers.T)
        nearest_clusters = np.argmax(similarities, axis=1)

        # Create new labels with noise points assigned
        new_labels = labels.copy()
        noise_indices = np.where(noise_mask)[0]

        for i, noise_idx in enumerate(noise_indices):
            new_labels[noise_idx] = unique_clusters[nearest_clusters[i]]

        print(f"Assigned all {n_noise} noise points to nearest clusters")
        return new_labels

    def analyze_communities(self, labels, top_n=20):
        """Analyze and sample documents from each cluster"""
        unique_clusters = sorted(set(labels))
        n_clusters = len(unique_clusters)

        # Compute silhouette score for final clustering
        sil_score = silhouette_score(
            self.embeddings_clustering,
            labels,  # metric="cosine"
        )

        analysis_text = f"\n=== HDBSCAN COMMUNITY ANALYSIS ({n_clusters} clusters, Silhouette={sil_score:.3f}) ===\n"
        print(analysis_text)

        # Save analysis to file
        analysis_file = (
            self.output_dir / f"community_analysis_{n_clusters}_clusters.txt"
        )

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)

            for cluster_id in unique_clusters:
                members = np.where(labels == cluster_id)[0]

                section = (
                    f"\n--- Community {cluster_id} ({len(members)} documents) ---\n"
                )
                print(section)
                f.write(section)

                # Sample representative documents
                sample_indices = np.random.choice(
                    members, min(top_n, len(members)), replace=False
                )

                for i, idx in enumerate(sample_indices):
                    doc_preds = self.preds[idx]
                    full_text = (
                        self.texts[idx].replace("\n", "\\n").replace("\r", "\\r")
                    )
                    line = f"{i + 1}. [{idx}] {doc_preds} {full_text}\n"

                    # For console output, show predictions and truncated text
                    truncated_text = (
                        self.texts[idx][:100] + "..."
                        if len(self.texts[idx]) > 100
                        else self.texts[idx]
                    )
                    print(f"{i + 1}. [{idx}] {doc_preds} {truncated_text}")
                    f.write(line)

                if len(members) > top_n:
                    remaining = f"... and {len(members) - top_n} more documents\n"
                    print(remaining.strip())
                    f.write(remaining)

        print(f"Community analysis saved to: {analysis_file}")
        return sil_score

    def compute_community_coherence(self, labels):
        """Compute coherence scores for each cluster"""
        unique_clusters = sorted(set(labels))
        n_clusters = len(unique_clusters)

        print(f"Computing coherence scores for {n_clusters} clusters...")

        coherence_scores = {}

        for cluster_id in unique_clusters:
            members = np.where(labels == cluster_id)[0]

            if len(members) < 2:
                coherence_scores[cluster_id] = 0.0
                continue

            # Sample for large clusters to avoid memory issues
            if len(members) > 1000:
                sample_size = min(500, len(members))
                sample_members = np.random.choice(members, sample_size, replace=False)
                cluster_embeddings = self.embeddings_clustering[sample_members]
            else:
                cluster_embeddings = self.embeddings_clustering[members]

            # Normalize for cosine similarity
            cluster_embeddings = cluster_embeddings / np.linalg.norm(
                cluster_embeddings, axis=1, keepdims=True
            )

            # Compute average pairwise cosine similarity within cluster
            similarities = np.dot(cluster_embeddings, cluster_embeddings.T)

            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(similarities, k=1)
            coherence = np.mean(upper_triangle[upper_triangle > 0])
            coherence_scores[cluster_id] = coherence

        # Print and save results
        print(f"\nCommunity Coherence Scores ({n_clusters} clusters):")
        coherence_text = f"Community Coherence Scores ({n_clusters} clusters):\n"

        for cluster_id, score in sorted(coherence_scores.items()):
            line = f"Community {cluster_id}: {score:.3f}"
            print(line)
            coherence_text += line + "\n"

        # Save coherence scores
        coherence_file = self.output_dir / f"coherence_scores_{n_clusters}_clusters.txt"
        with open(coherence_file, "w", encoding="utf-8") as f:
            f.write(coherence_text)
        print(f"Coherence scores saved to: {coherence_file}")

        return coherence_scores

    def visualize_communities(self, labels):
        """Create 2D visualization of clusters using pre-computed 2D embeddings"""
        try:
            unique_clusters = sorted(set(labels))
            n_clusters = len(unique_clusters)

            print(f"Creating 2D visualization for {n_clusters} clusters...")

            # Use the pre-computed 2D embeddings from original normalized embeddings
            embedding_2d = self.embeddings_2d

            # Create color map for clusters
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

            # Plot
            plt.figure(figsize=(14, 10))

            for i, cluster_id in enumerate(unique_clusters):
                mask = labels == cluster_id
                plt.scatter(
                    embedding_2d[mask, 0],
                    embedding_2d[mask, 1],
                    c=[colors[i]],
                    label=f"Community {cluster_id} ({np.sum(mask)} docs)",
                    alpha=0.7,
                    s=6,
                )

                # Add cluster label at centroid
                if np.sum(mask) > 0:
                    centroid_x = np.mean(embedding_2d[mask, 0])
                    centroid_y = np.mean(embedding_2d[mask, 1])

                    plt.annotate(
                        f"{cluster_id}",
                        (centroid_x, centroid_y),
                        fontsize=12,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor="black",
                            alpha=0.9,
                        ),
                    )

            plt.title(
                f"HDBSCAN Communities: {n_clusters} clusters\n(2D UMAP projection from original embeddings)"
            )
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Save plot
            plot_path = (
                self.output_dir / f"hdbscan_communities_{n_clusters}_clusters.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Visualization saved to: {plot_path}")
            return embedding_2d

        except Exception as e:
            print(f"Error creating visualization: {e}")
            plt.close("all")
            return None

    def run_full_analysis(self):
        """Run the complete HDBSCAN subregister discovery pipeline"""
        try:
            print("=" * 60)
            print("HDBSCAN SUBREGISTER DISCOVERY ANALYSIS")
            print("Strategy: UMAP reduction + HDBSCAN clustering + noise assignment")
            print("=" * 60)

            # Step 1: UMAP dimensionality reduction (both clustering and visualization)
            self.reduce_dimensions_umap()

            # Step 2: Find optimal HDBSCAN parameters
            optimal_min_size, optimal_result, all_results = (
                self.find_optimal_hdbscan_params()
            )

            # Step 3: Handle noise points by assigning to nearest clusters
            raw_labels = optimal_result["labels"]
            final_labels = self.handle_noise_points(raw_labels)

            # Recompute final metrics
            final_n_clusters = len(set(final_labels))
            final_sil_score = silhouette_score(
                self.embeddings_clustering, final_labels, metric="cosine"
            )

            # Save summary info
            summary_file = self.output_dir / "analysis_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("HDBSCAN SUBREGISTER DISCOVERY ANALYSIS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Input file: {self.pickle_path}\n")
                f.write(f"Number of documents: {len(self.embeddings)}\n")
                f.write(f"Original embedding dimension: {self.embeddings.shape[1]}\n")
                f.write(
                    f"Clustering embedding dimension: {self.embeddings_clustering.shape[1]}\n"
                )
                f.write(f"Register distribution: {dict(Counter(self.labels))}\n\n")
                f.write(f"METHOD: UMAP + HDBSCAN + Noise Assignment\n")
                f.write(
                    f"UMAP parameters: n_components=15, min_dist=0.0, metric=cosine\n"
                )
                f.write(f"Optimal min_cluster_size: {optimal_min_size}\n")
                f.write(f"Raw clusters found: {optimal_result['n_clusters']}\n")
                f.write(
                    f"Final clusters (after noise assignment): {final_n_clusters}\n"
                )
                f.write(f"Original noise points: {optimal_result['n_noise']}\n")
                f.write(f"Final silhouette score: {final_sil_score:.3f}\n\n")

            print(f"\n{'=' * 60}")
            print(f"FINAL CLUSTERING RESULTS:")
            print(f"Clusters: {final_n_clusters}")
            print(f"Silhouette Score: {final_sil_score:.3f}")
            print(f"All {len(final_labels)} documents assigned (no noise)")
            print(f"{'=' * 60}")

            # Step 4: Analyze final communities
            self.analyze_communities(final_labels)

            # Step 5: Compute coherence scores
            self.compute_community_coherence(final_labels)

            # Step 6: Create visualization using original embeddings → 2D
            self.visualize_communities(final_labels)

            print("\n" + "=" * 60)
            print("HDBSCAN ANALYSIS COMPLETE")
            print(f"Final clustering: {final_n_clusters} clusters")
            print(f"Final silhouette score: {final_sil_score:.3f}")
            print(f"All results saved to: {self.output_dir}")
            print("=" * 60)

            return final_labels, final_sil_score

        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback

            traceback.print_exc()
            return None, None
        finally:
            plt.close("all")


# Usage example
if __name__ == "__main__":
    # Find all pkl files in the directory
    pkl_directory = "../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/"
    pkl_files = glob.glob(os.path.join(pkl_directory, "*.pkl"))

    # Create common results directory
    results_dir = "hdbscan_results"
    Path(results_dir).mkdir(exist_ok=True)

    print(f"Found {len(pkl_files)} pkl files to process:")
    for i, file in enumerate(pkl_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    print(f"\nAll results will be saved to: {results_dir}/")
    print("Method: UMAP (15D) + HDBSCAN + noise assignment")
    print("Visualization: UMAP (2D) from original embeddings")

    print("\n" + "=" * 80)
    print("PROCESSING ALL PKL FILES")
    print("=" * 80)

    successful_analyses = 0
    failed_analyses = []
    skipped_files = []

    for i, pkl_file in enumerate(pkl_files, 1):
        try:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING FILE {i}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
            print(f"{'=' * 60}")

            # Initialize analyzer for this file
            analyzer = HDBSCANSubregisterAnalyzer(
                pkl_file, results_base_dir=results_dir
            )

            # Run analysis
            labels, sil_score = analyzer.run_full_analysis()

            if labels is not None:
                successful_analyses += 1
                print(
                    f"✓ Successfully completed analysis for {os.path.basename(pkl_file)}"
                )
            else:
                failed_analyses.append((pkl_file, "Analysis returned None"))

        except ValueError as e:
            if "too small" in str(e):
                print(f"⚠ Skipping {os.path.basename(pkl_file)}: {e}")
                skipped_files.append((pkl_file, str(e)))
            else:
                print(f"✗ Error processing {os.path.basename(pkl_file)}: {e}")
                failed_analyses.append((pkl_file, str(e)))
            continue

        except Exception as e:
            print(f"✗ Error processing {os.path.basename(pkl_file)}: {e}")
            failed_analyses.append((pkl_file, str(e)))
            import traceback

            traceback.print_exc()
            continue

        finally:
            plt.close("all")

    # Final summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {successful_analyses}/{len(pkl_files)} files")

    if skipped_files:
        print(f"\nSkipped files ({len(skipped_files)}):")
        for file, reason in skipped_files:
            print(f"  ⚠ {os.path.basename(file)}: {reason}")

    if failed_analyses:
        print(f"\nFailed analyses ({len(failed_analyses)}):")
        for file, error in failed_analyses:
            print(f"  ✗ {os.path.basename(file)}: {error}")

    if successful_analyses + len(skipped_files) == len(pkl_files):
        print("All eligible files processed successfully! ✓")

    print(f"\nResults organized in: {results_dir}/")

    # List created subdirectories
    if os.path.exists(results_dir):
        subdirs = [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ]
        if subdirs:
            print("Subdirectories created:")
            for subdir in sorted(subdirs):
                print(f"  - {results_dir}/{subdir}/")

    print("\nHDBScan analysis completed.")
