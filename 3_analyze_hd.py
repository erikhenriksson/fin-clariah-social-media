import glob
import os
import pickle
import warnings
from collections import Counter
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


class UMAPHDBSCANAnalyzer:
    def __init__(self, pickle_path, results_base_dir="umap_hdbscan_results"):
        """Load and initialize the embedding data"""
        self.pickle_path = pickle_path

        print(f"Loading data from {pickle_path}...")

        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)

        # Extract embeddings and metadata
        self.embeddings = np.array([row["embed_last"] for row in self.data])
        self.texts = [row["text"] for row in self.data]
        self.preds = [row["preds"] for row in self.data]

        print(f"Loaded {len(self.embeddings)} documents")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
        print(
            f"Register distribution: {Counter([p[0] if p else 'unknown' for p in self.preds])}"
        )

        # Check minimum size requirement
        if len(self.embeddings) < 1000:
            raise ValueError(
                f"Dataset too small ({len(self.embeddings)} documents). Minimum 1000 required."
            )

        # Setup output directory
        input_filename = Path(pickle_path).stem
        self.results_base_dir = Path(results_base_dir)
        self.results_base_dir.mkdir(exist_ok=True)
        self.output_dir = self.results_base_dir / input_filename
        self.output_dir.mkdir(exist_ok=True)

        print(f"Output will be saved to: {self.output_dir}")

        # Normalize embeddings
        self.embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    def apply_umap(self, n_components=20):
        """Apply UMAP for manifold learning"""
        print(f"Applying UMAP to reduce to {n_components} dimensions...")

        # UMAP for clustering
        self.umap_cluster = umap.UMAP(
            n_components=n_components,
            n_neighbors=30,
            min_dist=0.0,
            random_state=42,
        )

        self.embeddings_umap = self.umap_cluster.fit_transform(self.embeddings_norm)

        # UMAP for 2D visualization
        self.umap_viz = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        )

        self.embeddings_2d = self.umap_viz.fit_transform(self.embeddings_norm)

        print(f"UMAP embedding shape: {self.embeddings_umap.shape}")
        return self.embeddings_umap

    def apply_hdbscan_clustering(self):
        """Apply HDBSCAN with simple fixed parameters that favor large clusters"""
        print("\n" + "=" * 60)
        print("APPLYING HDBSCAN CLUSTERING")
        print("=" * 60)

        n_docs = len(self.embeddings_umap)

        # Simple fixed parameters
        min_cluster_size = max(50, n_docs // 10)  # 10% of dataset, minimum 50
        min_samples = 30  # Fixed high value - only dense regions become clusters

        print(f"Dataset size: {n_docs}")
        print(f"min_cluster_size: {min_cluster_size}")
        print(f"min_samples: {min_samples} (fixed - favors large dense clusters)")

        # Fit HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )

        cluster_labels = clusterer.fit_predict(self.embeddings_umap)

        # Report results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)

        print(f"\nClustering Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        print(
            f"  Clustered: {n_docs - n_noise} ({(n_docs - n_noise) / n_docs * 100:.1f}%)"
        )

        if n_clusters > 0:
            avg_stability = np.mean(clusterer.cluster_persistence_)
            print(f"  Average cluster stability: {avg_stability:.3f}")

        return clusterer, cluster_labels

    def analyze_clusters(self, cluster_labels, top_n=10):
        """Analyze cluster contents"""
        print("\n" + "=" * 60)
        print("CLUSTER ANALYSIS")
        print("=" * 60)

        unique_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])
        n_noise = np.sum(cluster_labels == -1)

        # Compute cluster statistics
        cluster_stats = {}
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_stats[cluster_id] = {
                "size": np.sum(mask),
                "coherence": self.compute_cluster_coherence(mask),
            }

        # Save analysis
        analysis_file = self.output_dir / "cluster_analysis.txt"

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(
                f"HDBSCAN CLUSTER ANALYSIS ({len(unique_clusters)} clusters, {n_noise} noise)\n"
            )
            f.write("=" * 60 + "\n\n")

            # Noise points if they exist
            if n_noise > 0:
                f.write(f"NOISE POINTS: {n_noise} documents\n")
                noise_indices = np.where(cluster_labels == -1)[0]
                sample_noise = np.random.choice(
                    noise_indices, min(5, len(noise_indices)), replace=False
                )

                for i, idx in enumerate(sample_noise):
                    doc_preds = self.preds[idx]
                    full_text = (
                        self.texts[idx].replace("\n", "\\n").replace("\r", "\\r")
                    )
                    f.write(f"{i + 1}. [{idx}] {doc_preds} {full_text}\n")
                f.write("\n")

            # Analyze each cluster
            for cluster_id in unique_clusters:
                stats = cluster_stats[cluster_id]
                mask = cluster_labels == cluster_id
                members = np.where(mask)[0]

                section = (
                    f"\nCLUSTER {cluster_id}: {stats['size']} documents "
                    f"(coherence: {stats['coherence']:.3f})\n"
                )
                section += "-" * 50 + "\n"

                print(
                    f"Cluster {cluster_id}: {stats['size']} documents, coherence: {stats['coherence']:.3f}"
                )
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
                    truncated_text = (
                        self.texts[idx][:100] + "..."
                        if len(self.texts[idx]) > 100
                        else self.texts[idx]
                    )

                    line = f"{i + 1}. [{idx}] {doc_preds} {full_text}\n"
                    print(f"  {i + 1}. [{idx}] {doc_preds} {truncated_text}")
                    f.write(line)

                if len(members) > top_n:
                    remaining = f"... and {len(members) - top_n} more documents\n"
                    f.write(remaining)

        print(f"\nCluster analysis saved to: {analysis_file}")
        return cluster_stats

    def compute_cluster_coherence(self, mask):
        """Compute intra-cluster coherence using cosine similarity"""
        if np.sum(mask) < 2:
            return 0.0

        cluster_embeddings = self.embeddings_umap[mask]

        # Sample for large clusters
        if len(cluster_embeddings) > 300:
            sample_indices = np.random.choice(
                len(cluster_embeddings), 300, replace=False
            )
            cluster_embeddings = cluster_embeddings[sample_indices]

        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(cluster_embeddings)
        upper_triangle = np.triu(similarity_matrix, k=1)

        return np.mean(upper_triangle[upper_triangle > 0])

    def visualize_clusters(self, cluster_labels):
        """Create 2D UMAP visualization"""
        print("Creating 2D UMAP visualization...")

        try:
            plt.figure(figsize=(14, 10))

            # Plot noise points first
            noise_mask = cluster_labels == -1
            if np.sum(noise_mask) > 0:
                plt.scatter(
                    self.embeddings_2d[noise_mask, 0],
                    self.embeddings_2d[noise_mask, 1],
                    c="lightgray",
                    alpha=0.4,
                    s=6,
                    label=f"Noise ({np.sum(noise_mask)} points)",
                )

            # Plot clustered points
            cluster_mask = cluster_labels != -1
            if np.sum(cluster_mask) > 0:
                unique_clusters = np.unique(cluster_labels[cluster_mask])
                scatter = plt.scatter(
                    self.embeddings_2d[cluster_mask, 0],
                    self.embeddings_2d[cluster_mask, 1],
                    c=cluster_labels[cluster_mask],
                    cmap="Set1",
                    alpha=0.8,
                    s=12,
                )

                # Add cluster labels at centroids
                for cluster_id in unique_clusters:
                    mask = cluster_labels == cluster_id
                    if np.sum(mask) > 0:
                        centroid_x = np.mean(self.embeddings_2d[mask, 0])
                        centroid_y = np.mean(self.embeddings_2d[mask, 1])

                        plt.annotate(
                            f"C{cluster_id}",
                            (centroid_x, centroid_y),
                            fontsize=12,
                            fontweight="bold",
                            ha="center",
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                edgecolor="black",
                                alpha=0.8,
                            ),
                        )

            n_clusters = len([c for c in np.unique(cluster_labels) if c != -1])
            n_noise = np.sum(cluster_labels == -1)
            clustered_pct = (len(cluster_labels) - n_noise) / len(cluster_labels) * 100

            plt.title(
                f"HDBSCAN Clustering Results\n{n_clusters} Clusters, {n_noise} Noise Points ({clustered_pct:.1f}% Clustered)"
            )
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")

            if noise_mask.any():
                plt.legend(loc="upper right")

            # Save plot
            plot_path = self.output_dir / "hdbscan_visualization.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Visualization saved to: {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error creating visualization: {e}")
            plt.close("all")
            return None

    def save_results(self, cluster_labels, clusterer):
        """Save comprehensive results"""
        results_file = self.output_dir / "clustering_results.txt"

        n_docs = len(cluster_labels)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        min_cluster_size = max(50, n_docs // 50)

        # Compute final metrics
        mask = cluster_labels != -1
        if np.sum(mask) > 10 and n_clusters > 1:
            ch_score = calinski_harabasz_score(
                self.embeddings_umap[mask], cluster_labels[mask]
            )
            sil_score = silhouette_score(
                self.embeddings_umap[mask], cluster_labels[mask]
            )
        else:
            ch_score = sil_score = 0.0

        with open(results_file, "w", encoding="utf-8") as f:
            f.write("HDBSCAN CLUSTERING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {os.path.basename(self.pickle_path)}\n")
            f.write(f"Total documents: {n_docs}\n")
            f.write(f"Original embedding dimensions: {self.embeddings.shape[1]}\n")
            f.write(f"UMAP dimensions: {self.embeddings_umap.shape[1]}\n\n")

            f.write("HDBSCAN Parameters:\n")
            f.write(f"  min_cluster_size: {min_cluster_size} (2% of dataset)\n")
            f.write(f"  min_samples: 30 (fixed - favors large dense clusters)\n\n")

            f.write("Clustering Results:\n")
            f.write(f"  Number of clusters: {n_clusters}\n")
            f.write(f"  Noise points: {n_noise}\n")
            f.write(f"  Clustered documents: {n_docs - n_noise}\n")
            f.write(f"  Clustering rate: {(n_docs - n_noise) / n_docs * 100:.1f}%\n\n")

            f.write("Quality Metrics:\n")
            f.write(f"  Calinski-Harabasz Index: {ch_score:.1f}\n")
            f.write(f"  Silhouette Score: {sil_score:.3f}\n")
            if n_clusters > 0:
                f.write(
                    f"  Average cluster stability: {np.mean(clusterer.cluster_persistence_):.3f}\n\n"
                )

            f.write("Method Summary:\n")
            f.write("- UMAP for manifold learning (cosine similarity)\n")
            f.write("- HDBSCAN with fixed conservative parameters\n")
            f.write("- Natural emergence of density-based clusters\n")

        print(f"Results saved to: {results_file}")

    def run_full_analysis(self):
        """Run the complete UMAP + HDBSCAN pipeline"""
        try:
            print("=" * 80)
            print("UMAP + HDBSCAN SUBREGISTER DISCOVERY")
            print("Pipeline: UMAP (20D) → HDBSCAN (stability-based) → Analysis")
            print("=" * 80)

            # Step 1: Apply UMAP
            self.apply_umap(n_components=15)

            # Step 2: Apply HDBSCAN clustering
            clusterer, cluster_labels = self.apply_hdbscan_clustering()

            # Step 3: Analyze clusters
            cluster_stats = self.analyze_clusters(cluster_labels)

            # Step 4: Visualize results
            self.visualize_clusters(cluster_labels)

            # Step 5: Save results
            self.save_results(cluster_labels, clusterer)

            # Final summary
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = np.sum(cluster_labels == -1)

            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE")
            print(f"Clusters found: {n_clusters}")
            print(f"Noise points: {n_noise}")
            print(
                f"Clustering rate: {(len(cluster_labels) - n_noise) / len(cluster_labels) * 100:.1f}%"
            )
            print(f"Cluster stability: {np.mean(clusterer.cluster_persistence_):.3f}")
            print(f"Results saved to: {self.output_dir}")
            print("=" * 80)

            return cluster_labels, cluster_stats

        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback

            traceback.print_exc()
        finally:
            plt.close("all")
            import gc

            gc.collect()


# Usage example
if __name__ == "__main__":
    # Find all pkl files
    pkl_directory = "../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/"
    pkl_files = glob.glob(os.path.join(pkl_directory, "*.pkl"))

    # Create results directory
    results_dir = "umap_hdbscan_results"
    Path(results_dir).mkdir(exist_ok=True)

    print(f"Found {len(pkl_files)} pkl files to process:")
    for i, file in enumerate(pkl_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    print(f"\nAll results will be saved to: {results_dir}/")
    print("Method: UMAP + HDBSCAN (principled parameter selection)")

    successful_analyses = 0
    failed_analyses = []
    skipped_files = []

    for i, pkl_file in enumerate(pkl_files, 1):
        try:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING FILE {i}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
            print(f"{'=' * 60}")

            analyzer = UMAPHDBSCANAnalyzer(pkl_file, results_base_dir=results_dir)
            cluster_labels, cluster_stats = analyzer.run_full_analysis()

            successful_analyses += 1
            print(f"✓ Successfully completed: {os.path.basename(pkl_file)}")

        except ValueError as e:
            if "too small" in str(e):
                print(f"⚠ Skipping {os.path.basename(pkl_file)}: {e}")
                skipped_files.append((pkl_file, str(e)))
            else:
                print(f"✗ Error: {os.path.basename(pkl_file)}: {e}")
                failed_analyses.append((pkl_file, str(e)))
            continue

        except Exception as e:
            print(f"✗ Error: {os.path.basename(pkl_file)}: {e}")
            failed_analyses.append((pkl_file, str(e)))
            continue

        finally:
            plt.close("all")
            import gc

            gc.collect()

    # Final summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {successful_analyses}/{len(pkl_files)} files")

    if skipped_files:
        print(f"\nSkipped: {len(skipped_files)} files")
        for file, reason in skipped_files:
            print(f"  ⚠ {os.path.basename(file)}: {reason}")

    if failed_analyses:
        print(f"\nFailed: {len(failed_analyses)} files")
        for file, error in failed_analyses:
            print(f"  ✗ {os.path.basename(file)}: {error}")

    print(f"\nResults in: {results_dir}/")
