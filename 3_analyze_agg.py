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
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


class UMAPSubregisterAnalyzer:
    def __init__(self, pickle_path, results_base_dir="umap_subregister_results"):
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

    def reduce_dimensions_pca(self, target_variance=0.80):
        """Apply PCA for noise reduction, automatically selecting components for target variance"""
        print("Applying PCA for dimensionality reduction...")

        # First, fit PCA with all components to get variance ratios
        pca_full = PCA(random_state=42)
        pca_full.fit(self.embeddings_norm)

        # Find minimum components needed for target variance
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= target_variance) + 1

        # Ensure we don't exceed the maximum possible components
        max_components = min(self.embeddings_norm.shape)
        n_components = min(n_components, max_components)

        print(
            f"Selected {n_components} components to achieve {target_variance:.1%} explained variance"
        )

        # Apply PCA with optimal number of components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.embeddings_pca = self.pca.fit_transform(self.embeddings_norm)

        # Normalize PCA embeddings
        self.embeddings_pca_norm = self.embeddings_pca / np.linalg.norm(
            self.embeddings_pca, axis=1, keepdims=True
        )

        actual_variance = self.pca.explained_variance_ratio_.sum()
        print(
            f"PCA total variance explained: {actual_variance:.3f} (using {n_components} components)"
        )

        return self.embeddings_pca_norm

    def apply_umap_clustering(self, n_components=10, n_neighbors=None, min_dist=0.0):
        """Apply UMAP for manifold learning with parameters optimized for clustering"""
        print(f"Applying UMAP to reduce to {n_components} dimensions for clustering...")

        # Auto-compute n_neighbors if not provided
        if n_neighbors is None:
            # Use larger n_neighbors for clustering as recommended
            n_neighbors = min(int(np.sqrt(len(self.embeddings_pca_norm)) * 2), 100)
            n_neighbors = max(30, n_neighbors)  # Minimum 30 for good manifold structure

        print(f"Using n_neighbors={n_neighbors}, min_dist={min_dist}")

        # UMAP for clustering (higher dimensions)
        self.umap_cluster = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
            low_memory=True,
        )

        self.embeddings_umap_cluster = self.umap_cluster.fit_transform(
            self.embeddings_pca_norm
        )

        print(f"UMAP clustering embedding shape: {self.embeddings_umap_cluster.shape}")
        return self.embeddings_umap_cluster

    def apply_umap_visualization(self, n_neighbors=None, min_dist=0.1):
        """Apply UMAP for 2D visualization with different parameters"""
        print("Applying UMAP for 2D visualization...")

        # Auto-compute n_neighbors for visualization (can be smaller)
        if n_neighbors is None:
            n_neighbors = min(int(np.sqrt(len(self.embeddings_pca_norm))), 50)
            n_neighbors = max(15, n_neighbors)

        print(f"Using n_neighbors={n_neighbors}, min_dist={min_dist} for visualization")

        # UMAP for visualization (2D)
        self.umap_viz = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
            low_memory=True,
        )

        self.embeddings_umap_2d = self.umap_viz.fit_transform(self.embeddings_pca_norm)

        print(f"UMAP 2D visualization embedding shape: {self.embeddings_umap_2d.shape}")
        return self.embeddings_umap_2d

    def optimize_hdbscan_parameters(self):
        """Optimize HDBSCAN parameters using grid search with Calinski-Harabasz Index"""
        print("\n" + "=" * 60)
        print("OPTIMIZING HDBSCAN PARAMETERS")
        print("=" * 60)

        # Parameter grid for HDBSCAN
        min_cluster_sizes = [
            max(10, len(self.embeddings_umap_cluster) // 200),
            max(20, len(self.embeddings_umap_cluster) // 100),
            max(50, len(self.embeddings_umap_cluster) // 50),
        ]

        min_samples_list = [5, 10, 15, 20]

        best_score = -1
        best_params = None
        best_labels = None
        optimization_results = []

        print(
            f"Testing {len(min_cluster_sizes)} × {len(min_samples_list)} parameter combinations..."
        )
        print("Parameters tested: min_cluster_size × min_samples")

        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_list:
                try:
                    # Fit HDBSCAN
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric="euclidean",
                        cluster_selection_method="eom",
                    )

                    cluster_labels = clusterer.fit_predict(self.embeddings_umap_cluster)

                    # Filter out noise points for scoring
                    mask = cluster_labels != -1
                    if np.sum(mask) < 2 or len(np.unique(cluster_labels[mask])) < 2:
                        score = -1
                        n_clusters = 0
                        n_noise = len(cluster_labels)
                    else:
                        score = calinski_harabasz_score(
                            self.embeddings_umap_cluster[mask], cluster_labels[mask]
                        )
                        n_clusters = len(np.unique(cluster_labels[mask]))
                        n_noise = np.sum(cluster_labels == -1)

                    optimization_results.append(
                        {
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            "calinski_harabasz": score,
                            "n_clusters": n_clusters,
                            "n_noise": n_noise,
                            "labels": cluster_labels,
                        }
                    )

                    print(
                        f"  {min_cluster_size:3d} × {min_samples:2d}: "
                        f"CH={score:6.1f}, clusters={n_clusters:2d}, noise={n_noise:4d}"
                    )

                    if score > best_score:
                        best_score = score
                        best_params = {
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                        }
                        best_labels = cluster_labels.copy()

                except Exception as e:
                    print(f"  {min_cluster_size:3d} × {min_samples:2d}: Error - {e}")
                    continue

        if best_params is None:
            raise ValueError("No valid HDBSCAN clustering found!")

        print(f"\n✓ OPTIMAL PARAMETERS:")
        print(f"  min_cluster_size: {best_params['min_cluster_size']}")
        print(f"  min_samples: {best_params['min_samples']}")
        print(f"  Calinski-Harabasz Score: {best_score:.3f}")
        print(f"  Number of clusters: {len(np.unique(best_labels[best_labels != -1]))}")
        print(f"  Noise points: {np.sum(best_labels == -1)}")

        # Save optimization results
        optimization_file = self.output_dir / "hdbscan_optimization.txt"
        with open(optimization_file, "w", encoding="utf-8") as f:
            f.write("HDBSCAN PARAMETER OPTIMIZATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Optimization Metric: Calinski-Harabasz Index\n")
            f.write(f"Dataset Size: {len(self.embeddings_umap_cluster)} documents\n")
            f.write(
                f"UMAP Clustering Dimensions: {self.embeddings_umap_cluster.shape[1]}\n\n"
            )
            f.write(f"OPTIMAL PARAMETERS:\n")
            f.write(f"  min_cluster_size: {best_params['min_cluster_size']}\n")
            f.write(f"  min_samples: {best_params['min_samples']}\n")
            f.write(f"  Calinski-Harabasz Score: {best_score:.3f}\n")
            f.write(
                f"  Number of clusters: {len(np.unique(best_labels[best_labels != -1]))}\n"
            )
            f.write(f"  Noise points: {np.sum(best_labels == -1)}\n\n")

            f.write("All Parameter Combinations Tested:\n")
            f.write("min_cluster_size | min_samples | CH_Score | Clusters | Noise\n")
            f.write("-" * 60 + "\n")
            for result in optimization_results:
                f.write(
                    f"      {result['min_cluster_size']:3d}        |     {result['min_samples']:2d}      | "
                    f"{result['calinski_harabasz']:6.1f}   |    {result['n_clusters']:2d}    | {result['n_noise']:4d}\n"
                )

        print(f"\nOptimization results saved to: {optimization_file}")
        return best_params, best_labels, best_score

    def compute_cluster_metrics(self, cluster_labels):
        """Compute comprehensive cluster quality metrics"""
        mask = cluster_labels != -1

        if np.sum(mask) < 2:
            return {}

        metrics = {}

        # Calinski-Harabasz Index (higher is better)
        metrics["calinski_harabasz"] = calinski_harabasz_score(
            self.embeddings_umap_cluster[mask], cluster_labels[mask]
        )

        # Silhouette Score (for comparison)
        if len(np.unique(cluster_labels[mask])) > 1:
            metrics["silhouette"] = silhouette_score(
                self.embeddings_umap_cluster[mask], cluster_labels[mask]
            )
        else:
            metrics["silhouette"] = 0.0

        # Cluster statistics
        metrics["n_clusters"] = len(np.unique(cluster_labels[mask]))
        metrics["n_noise"] = np.sum(cluster_labels == -1)
        metrics["noise_ratio"] = metrics["n_noise"] / len(cluster_labels)

        return metrics

    def analyze_clusters(self, cluster_labels, top_n=10):
        """Analyze and sample documents from each cluster"""
        print("\n" + "=" * 60)
        print("CLUSTER ANALYSIS")
        print("=" * 60)

        unique_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])

        analysis_text = f"\nCLUSTER ANALYSIS ({len(unique_clusters)} clusters)\n"
        analysis_text += "=" * 50 + "\n"

        # Compute coherence scores for clusters
        coherence_scores = self.compute_coherence_for_clusters(cluster_labels)

        analysis_file = self.output_dir / "cluster_analysis.txt"

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)

            # Analyze noise points first
            noise_count = np.sum(cluster_labels == -1)
            if noise_count > 0:
                noise_section = f"\n--- NOISE POINTS ({noise_count} documents) ---\n"
                print(noise_section.strip())
                f.write(noise_section)

                noise_indices = np.where(cluster_labels == -1)[0]
                sample_noise = np.random.choice(
                    noise_indices, min(5, len(noise_indices)), replace=False
                )

                for i, idx in enumerate(sample_noise):
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
                    print(f"{i + 1}. [{idx}] {doc_preds} {truncated_text}")
                    f.write(line)

            # Analyze each cluster
            for cluster_id in unique_clusters:
                members = np.where(cluster_labels == cluster_id)[0]
                coherence_score = coherence_scores.get(cluster_id, 0.0)

                section = f"\n--- Cluster {cluster_id} ({len(members)} documents, coherence: {coherence_score:.3f}) ---\n"
                print(section.strip())
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
                    print(f"{i + 1}. [{idx}] {doc_preds} {truncated_text}")
                    f.write(line)

                if len(members) > top_n:
                    remaining = f"... and {len(members) - top_n} more documents\n"
                    print(remaining.strip())
                    f.write(remaining)

        print(f"\nCluster analysis saved to: {analysis_file}")
        return coherence_scores

    def compute_coherence_for_clusters(self, cluster_labels):
        """Compute coherence scores for clusters (memory-efficient)"""
        coherence_scores = {}
        unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]

        for cluster_id in unique_clusters:
            members = np.where(cluster_labels == cluster_id)[0]

            if len(members) < 2:
                coherence_scores[cluster_id] = 0.0
                continue

            # Memory-efficient coherence computation
            if len(members) > 500:
                sample_size = min(300, len(members))
                sample_members = np.random.choice(members, sample_size, replace=False)
                cluster_embeddings = self.embeddings_umap_cluster[sample_members]
            else:
                cluster_embeddings = self.embeddings_umap_cluster[members]

            # Compute average pairwise cosine similarity within cluster
            similarity_matrix = cosine_similarity(cluster_embeddings)
            upper_triangle = np.triu(similarity_matrix, k=1)
            coherence = np.mean(upper_triangle[upper_triangle > 0])
            coherence_scores[cluster_id] = coherence

            # Cleanup
            del similarity_matrix, upper_triangle, cluster_embeddings

        return coherence_scores

    def visualize_clusters(self, cluster_labels):
        """Create 2D UMAP visualization with cluster labels"""
        print("Creating 2D UMAP visualization with cluster labels...")

        try:
            plt.figure(figsize=(16, 12))

            # Plot noise points first (gray)
            noise_mask = cluster_labels == -1
            if np.sum(noise_mask) > 0:
                plt.scatter(
                    self.embeddings_umap_2d[noise_mask, 0],
                    self.embeddings_umap_2d[noise_mask, 1],
                    c="lightgray",
                    alpha=0.3,
                    s=4,
                    label=f"Noise ({np.sum(noise_mask)} points)",
                )

            # Plot clustered points
            cluster_mask = cluster_labels != -1
            if np.sum(cluster_mask) > 0:
                unique_clusters = np.unique(cluster_labels[cluster_mask])
                scatter = plt.scatter(
                    self.embeddings_umap_2d[cluster_mask, 0],
                    self.embeddings_umap_2d[cluster_mask, 1],
                    c=cluster_labels[cluster_mask],
                    cmap="tab10",
                    alpha=0.7,
                    s=8,
                )
                plt.colorbar(scatter, label="Cluster ID")

                # Add cluster labels at centroids
                for cluster_id in unique_clusters:
                    mask = cluster_labels == cluster_id
                    if np.sum(mask) > 0:
                        centroid_x = np.mean(self.embeddings_umap_2d[mask, 0])
                        centroid_y = np.mean(self.embeddings_umap_2d[mask, 1])

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

            plt.title(
                f"UMAP 2D Visualization\n{n_clusters} Clusters, {n_noise} Noise Points"
            )
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")

            if noise_mask.any():
                plt.legend()

            # Save plot
            plot_path = self.output_dir / "umap_cluster_visualization.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Cluster visualization saved to: {plot_path}")

            return plot_path

        except Exception as e:
            print(f"Error creating visualization: {e}")
            plt.close("all")
            return None

    def save_cluster_metrics(self, cluster_labels, metrics, best_params):
        """Save comprehensive clustering metrics"""
        metrics_file = self.output_dir / "clustering_metrics.txt"

        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write("CLUSTERING PERFORMANCE METRICS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {os.path.basename(self.pickle_path)}\n")
            f.write(f"Total documents: {len(cluster_labels)}\n")
            f.write(
                f"UMAP clustering dimensions: {self.embeddings_umap_cluster.shape[1]}\n\n"
            )

            f.write("HDBSCAN Parameters:\n")
            f.write(f"  min_cluster_size: {best_params['min_cluster_size']}\n")
            f.write(f"  min_samples: {best_params['min_samples']}\n\n")

            f.write("Clustering Results:\n")
            f.write(f"  Number of clusters: {metrics['n_clusters']}\n")
            f.write(f"  Noise points: {metrics['n_noise']}\n")
            f.write(f"  Noise ratio: {metrics['noise_ratio']:.3f}\n\n")

            f.write("Quality Metrics:\n")
            f.write(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz']:.3f}\n")
            f.write(f"  Silhouette Score: {metrics['silhouette']:.3f}\n\n")

            f.write("Metric Interpretation:\n")
            f.write("  Calinski-Harabasz Index (higher is better):\n")
            f.write("    > 300: Excellent cluster separation\n")
            f.write("    > 100: Good cluster separation\n")
            f.write("    > 50: Moderate cluster separation\n")
            f.write("    < 50: Poor cluster separation\n\n")

        print(f"Clustering metrics saved to: {metrics_file}")

    def run_full_analysis(self):
        """Run the complete UMAP-based subregister discovery pipeline"""
        try:
            print("=" * 80)
            print("UMAP-BASED SUBREGISTER DISCOVERY ANALYSIS")
            print("Pipeline: PCA → UMAP (10D) → HDBSCAN → 2D Visualization")
            print("=" * 80)

            # Step 1: PCA for dimensionality reduction
            self.reduce_dimensions_pca(target_variance=0.80)

            # Step 2: UMAP for clustering (10D)
            self.apply_umap_clustering(n_components=10)

            # Step 3: UMAP for visualization (2D)
            self.apply_umap_visualization()

            # Step 4: Optimize HDBSCAN parameters
            best_params, best_labels, best_score = self.optimize_hdbscan_parameters()

            # Step 5: Compute comprehensive metrics
            metrics = self.compute_cluster_metrics(best_labels)

            # Step 6: Analyze clusters
            coherence_scores = self.analyze_clusters(best_labels)

            # Step 7: Save metrics
            self.save_cluster_metrics(best_labels, metrics, best_params)

            # Step 8: Create visualization
            self.visualize_clusters(best_labels)

            # Final summary
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE")
            print(f"Number of clusters found: {metrics['n_clusters']}")
            print(f"Noise points: {metrics['n_noise']} ({metrics['noise_ratio']:.1%})")
            print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.3f}")
            print(f"Silhouette Score: {metrics['silhouette']:.3f}")
            print(f"All results saved to: {self.output_dir}")
            print("=" * 80)

            return best_labels, metrics

        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Cleanup
            plt.close("all")
            import gc

            gc.collect()


# Usage example
if __name__ == "__main__":
    # Find all pkl files in the directory
    pkl_directory = "../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/"
    pkl_files = glob.glob(os.path.join(pkl_directory, "*.pkl"))

    # Create results directory
    results_dir = "umap_subregister_results"
    Path(results_dir).mkdir(exist_ok=True)

    print(f"Found {len(pkl_files)} pkl files to process:")
    for i, file in enumerate(pkl_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    print(f"\nAll results will be saved to: {results_dir}/")
    print("Method: UMAP-based clustering with HDBSCAN")

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
            analyzer = UMAPSubregisterAnalyzer(pkl_file, results_base_dir=results_dir)

            # Run analysis
            labels, metrics = analyzer.run_full_analysis()

            successful_analyses += 1
            print(f"✓ Successfully completed analysis for {os.path.basename(pkl_file)}")

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
            # Cleanup between files
            plt.close("all")
            import gc

            gc.collect()

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
    print("Analysis completed with UMAP-HDBSCAN pipeline.")
