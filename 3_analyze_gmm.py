import glob
import os
import pickle
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")


class UMAPGMMAnalyzer:
    def __init__(self, pickle_path, results_base_dir="umap_gmm_results"):
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

        # Normalize embeddings for cosine similarity
        self.embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    def apply_umap(self, n_components=20):
        """Apply UMAP for manifold learning"""
        print(f"Applying UMAP to reduce to {n_components} dimensions...")

        # Conservative n_neighbors for stable manifold structure
        n_neighbors = min(int(len(self.embeddings_norm) ** 0.4), 100)
        n_neighbors = max(50, n_neighbors)

        print(f"Using n_neighbors={n_neighbors}")

        # UMAP for clustering
        self.umap_cluster = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            low_memory=True,
        )

        self.embeddings_umap = self.umap_cluster.fit_transform(self.embeddings_norm)

        # UMAP for 2D visualization
        self.umap_viz = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            low_memory=True,
        )

        self.embeddings_2d = self.umap_viz.fit_transform(self.embeddings_norm)

        print(f"UMAP embedding shape: {self.embeddings_umap.shape}")
        print(f"2D visualization shape: {self.embeddings_2d.shape}")

        return self.embeddings_umap

    def optimize_gmm_clusters(self, max_clusters=6):
        """Find optimal number of GMM clusters using multiple criteria"""
        print("\n" + "=" * 60)
        print("OPTIMIZING GMM CLUSTER COUNT")
        print("=" * 60)

        cluster_range = range(2, max_clusters + 1)
        results = []

        print("Clusters | AIC    | BIC    | CH_Score | Silhouette")
        print("-" * 50)

        for n_clusters in cluster_range:
            # Fit GMM
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                random_state=42,
                max_iter=200,
                tol=1e-4,
            )

            labels = gmm.fit_predict(self.embeddings_umap)

            # Compute metrics
            aic = gmm.aic(self.embeddings_umap)
            bic = gmm.bic(self.embeddings_umap)
            ch_score = calinski_harabasz_score(self.embeddings_umap, labels)
            sil_score = silhouette_score(self.embeddings_umap, labels)

            results.append(
                {
                    "n_clusters": n_clusters,
                    "gmm": gmm,
                    "labels": labels,
                    "aic": aic,
                    "bic": bic,
                    "calinski_harabasz": ch_score,
                    "silhouette": sil_score,
                }
            )

            print(
                f"   {n_clusters:2d}    | {aic:6.0f} | {bic:6.0f} | {ch_score:7.1f}  |   {sil_score:.3f}"
            )

        # Select best model using BIC (lower is better, penalizes complexity)
        best_result = min(results, key=lambda x: x["bic"])

        print(f"\n✓ OPTIMAL: {best_result['n_clusters']} clusters (lowest BIC)")
        print(f"  BIC: {best_result['bic']:.0f}")
        print(f"  Calinski-Harabasz: {best_result['calinski_harabasz']:.1f}")
        print(f"  Silhouette: {best_result['silhouette']:.3f}")

        # Save optimization results
        optimization_file = self.output_dir / "gmm_optimization.txt"
        with open(optimization_file, "w", encoding="utf-8") as f:
            f.write("GMM CLUSTER OPTIMIZATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Optimization Criterion: BIC (Bayesian Information Criterion)\n")
            f.write(f"Dataset Size: {len(self.embeddings_umap)} documents\n")
            f.write(f"UMAP Dimensions: {self.embeddings_umap.shape[1]}\n\n")
            f.write(f"OPTIMAL PARAMETERS:\n")
            f.write(f"  Number of clusters: {best_result['n_clusters']}\n")
            f.write(f"  BIC Score: {best_result['bic']:.1f}\n")
            f.write(f"  Calinski-Harabasz: {best_result['calinski_harabasz']:.1f}\n")
            f.write(f"  Silhouette Score: {best_result['silhouette']:.3f}\n\n")

            f.write("All Cluster Counts Tested:\n")
            f.write("Clusters | AIC    | BIC    | CH_Score | Silhouette\n")
            f.write("-" * 50 + "\n")
            for result in results:
                marker = (
                    " <- OPTIMAL"
                    if result["n_clusters"] == best_result["n_clusters"]
                    else ""
                )
                f.write(
                    f"   {result['n_clusters']:2d}    | {result['aic']:6.0f} | {result['bic']:6.0f} | "
                    f"{result['calinski_harabasz']:7.1f}  |   {result['silhouette']:.3f}{marker}\n"
                )

        print(f"\nOptimization results saved to: {optimization_file}")

        return best_result

    def analyze_clusters(self, gmm_result, top_n=10):
        """Analyze cluster contents and compute statistics"""
        print("\n" + "=" * 60)
        print("CLUSTER ANALYSIS")
        print("=" * 60)

        labels = gmm_result["labels"]
        gmm = gmm_result["gmm"]
        n_clusters = gmm_result["n_clusters"]

        # Get cluster probabilities for uncertainty analysis
        probabilities = gmm.predict_proba(self.embeddings_umap)

        # Compute cluster statistics
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_stats[cluster_id] = {
                "size": np.sum(mask),
                "mean_confidence": np.mean(np.max(probabilities[mask], axis=1)),
                "coherence": self.compute_cluster_coherence(cluster_id, mask),
            }

        # Save analysis
        analysis_file = self.output_dir / "cluster_analysis.txt"

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(f"GMM CLUSTER ANALYSIS ({n_clusters} clusters)\n")
            f.write("=" * 50 + "\n\n")

            for cluster_id in range(n_clusters):
                stats = cluster_stats[cluster_id]
                mask = labels == cluster_id
                members = np.where(mask)[0]

                section = (
                    f"\n--- Cluster {cluster_id} "
                    f"({stats['size']} documents, "
                    f"confidence: {stats['mean_confidence']:.3f}, "
                    f"coherence: {stats['coherence']:.3f}) ---\n"
                )

                print(section.strip())
                f.write(section)

                # Sample representative documents
                sample_indices = np.random.choice(
                    members, min(top_n, len(members)), replace=False
                )

                for i, idx in enumerate(sample_indices):
                    doc_preds = self.preds[idx]
                    confidence = np.max(probabilities[idx])
                    full_text = (
                        self.texts[idx].replace("\n", "\\n").replace("\r", "\\r")
                    )
                    truncated_text = (
                        self.texts[idx][:100] + "..."
                        if len(self.texts[idx]) > 100
                        else self.texts[idx]
                    )

                    line = f"{i + 1}. [{idx}] conf={confidence:.3f} {doc_preds} {full_text}\n"
                    print(
                        f"{i + 1}. [{idx}] conf={confidence:.3f} {doc_preds} {truncated_text}"
                    )
                    f.write(line)

                if len(members) > top_n:
                    remaining = f"... and {len(members) - top_n} more documents\n"
                    print(remaining.strip())
                    f.write(remaining)

        print(f"\nCluster analysis saved to: {analysis_file}")
        return cluster_stats

    def compute_cluster_coherence(self, cluster_id, mask):
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

    def visualize_clusters(self, gmm_result):
        """Create 2D UMAP visualization with GMM cluster labels"""
        print("Creating 2D UMAP visualization...")

        labels = gmm_result["labels"]
        probabilities = gmm_result["gmm"].predict_proba(self.embeddings_umap)
        n_clusters = gmm_result["n_clusters"]

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Plot 1: Hard clustering
            scatter1 = ax1.scatter(
                self.embeddings_2d[:, 0],
                self.embeddings_2d[:, 1],
                c=labels,
                cmap="Set1",
                alpha=0.7,
                s=8,
            )

            # Add cluster centroids
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                if np.sum(mask) > 0:
                    centroid_x = np.mean(self.embeddings_2d[mask, 0])
                    centroid_y = np.mean(self.embeddings_2d[mask, 1])

                    ax1.annotate(
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

            ax1.set_title(f"GMM Hard Clustering\n{n_clusters} Clusters")
            ax1.set_xlabel("UMAP Dimension 1")
            ax1.set_ylabel("UMAP Dimension 2")

            # Plot 2: Uncertainty (confidence)
            max_probs = np.max(probabilities, axis=1)
            scatter2 = ax2.scatter(
                self.embeddings_2d[:, 0],
                self.embeddings_2d[:, 1],
                c=max_probs,
                cmap="viridis",
                alpha=0.7,
                s=8,
            )

            ax2.set_title("Cluster Assignment Confidence\n(Higher = More Certain)")
            ax2.set_xlabel("UMAP Dimension 1")
            ax2.set_ylabel("UMAP Dimension 2")
            plt.colorbar(scatter2, ax=ax2, label="Max Probability")

            plt.tight_layout()

            # Save plot
            plot_path = self.output_dir / "gmm_cluster_visualization.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Cluster visualization saved to: {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error creating visualization: {e}")
            plt.close("all")
            return None

    def save_results(self, gmm_result, cluster_stats):
        """Save comprehensive results summary"""
        results_file = self.output_dir / "clustering_results.txt"

        labels = gmm_result["labels"]
        n_clusters = gmm_result["n_clusters"]

        with open(results_file, "w", encoding="utf-8") as f:
            f.write("GMM CLUSTERING RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {os.path.basename(self.pickle_path)}\n")
            f.write(f"Total documents: {len(labels)}\n")
            f.write(f"Original embedding dimensions: {self.embeddings.shape[1]}\n")
            f.write(f"UMAP dimensions: {self.embeddings_umap.shape[1]}\n\n")

            f.write("Clustering Results:\n")
            f.write(f"  Number of clusters: {n_clusters}\n")
            f.write(f"  BIC Score: {gmm_result['bic']:.1f}\n")
            f.write(f"  Calinski-Harabasz: {gmm_result['calinski_harabasz']:.1f}\n")
            f.write(f"  Silhouette Score: {gmm_result['silhouette']:.3f}\n\n")

            f.write("Cluster Statistics:\n")
            f.write("Cluster | Size | Avg_Confidence | Coherence\n")
            f.write("-" * 45 + "\n")

            for cluster_id in range(n_clusters):
                stats = cluster_stats[cluster_id]
                f.write(
                    f"   {cluster_id:2d}   | {stats['size']:4d} |     {stats['mean_confidence']:.3f}      |   {stats['coherence']:.3f}\n"
                )

            f.write(f"\nTotal clustered: {len(labels)} documents (100%)\n")
            f.write("\nGMM Advantages:\n")
            f.write("- Probabilistic assignments (soft clustering)\n")
            f.write("- Confidence scores for each assignment\n")
            f.write("- Automatic handling of elliptical cluster shapes\n")
            f.write("- BIC-based model selection prevents overfitting\n")

        print(f"Results summary saved to: {results_file}")

    def run_full_analysis(self):
        """Run the complete UMAP + GMM pipeline"""
        try:
            print("=" * 80)
            print("UMAP + GMM SUBREGISTER DISCOVERY")
            print(
                "Pipeline: UMAP (20D) → GMM Optimization → Clustering → Visualization"
            )
            print("=" * 80)

            # Step 1: Apply UMAP
            self.apply_umap(n_components=20)

            # Step 2: Optimize GMM cluster count
            gmm_result = self.optimize_gmm_clusters(max_clusters=6)

            # Step 3: Analyze clusters
            cluster_stats = self.analyze_clusters(gmm_result)

            # Step 4: Visualize results
            self.visualize_clusters(gmm_result)

            # Step 5: Save comprehensive results
            self.save_results(gmm_result, cluster_stats)

            # Final summary
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE")
            print(f"Optimal clusters: {gmm_result['n_clusters']}")
            print(f"BIC Score: {gmm_result['bic']:.1f}")
            print(f"Calinski-Harabasz: {gmm_result['calinski_harabasz']:.1f}")
            print(f"Silhouette Score: {gmm_result['silhouette']:.3f}")
            print(f"All results saved to: {self.output_dir}")
            print("=" * 80)

            return gmm_result, cluster_stats

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
    # Find all pkl files
    pkl_directory = "../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/"
    pkl_files = glob.glob(os.path.join(pkl_directory, "*.pkl"))

    # Create results directory
    results_dir = "umap_gmm_results"
    Path(results_dir).mkdir(exist_ok=True)

    print(f"Found {len(pkl_files)} pkl files to process:")
    for i, file in enumerate(pkl_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    print(f"\nAll results will be saved to: {results_dir}/")
    print("Method: UMAP + Gaussian Mixture Models")

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

            # Initialize analyzer
            analyzer = UMAPGMMAnalyzer(pkl_file, results_base_dir=results_dir)

            # Run analysis
            gmm_result, cluster_stats = analyzer.run_full_analysis()

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
            continue

        finally:
            # Cleanup
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

    print(f"\nResults organized in: {results_dir}/")
    print("Analysis completed with UMAP + GMM pipeline.")
