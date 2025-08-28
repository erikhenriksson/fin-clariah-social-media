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

warnings.filterwarnings("ignore")


class SubregisterAnalyzer:
    def __init__(self, pickle_path, results_base_dir="subregister_results"):
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
        print(f"Register: {Counter(self.labels)}")

        # Check minimum size requirement
        if len(self.embeddings) < 1000:
            raise ValueError(
                f"Dataset too small ({len(self.embeddings)} documents). Minimum 1000 required."
            )

        # Create output directories
        input_filename = Path(pickle_path).stem
        self.results_base_dir = Path(results_base_dir)
        self.results_base_dir.mkdir(exist_ok=True)
        self.output_dir = self.results_base_dir / input_filename
        self.output_dir.mkdir(exist_ok=True)

        print(f"Output will be saved to: {self.output_dir}")

        # Normalize embeddings for spherical k-means
        self.embeddings_norm = normalize(self.embeddings, norm="l2")

    def reduce_dimensions_umap(self):
        """Apply UMAP for dimensionality reduction to 20 components"""
        print("Reducing dimensions with UMAP to 20 components...")

        # Use high n_neighbors and min_dist=0.0 for dimension reduction
        n_neighbors = min(100, len(self.embeddings) // 10)

        self.umap_reducer = umap.UMAP(
            n_components=20,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            low_memory=True,
        )

        self.embeddings_reduced = self.umap_reducer.fit_transform(self.embeddings_norm)

        # Normalize reduced embeddings for spherical k-means
        self.embeddings_reduced = normalize(self.embeddings_reduced, norm="l2")

        print(f"Reduced to {self.embeddings_reduced.shape[1]} dimensions")
        return self.embeddings_reduced

    def find_optimal_clusters(self, k_range=[2, 3, 4, 5, 6]):
        """Find optimal number of clusters using Silhouette score"""
        print("=" * 60)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("=" * 60)

        results = {}

        for k in k_range:
            print(f"Testing {k} clusters...")

            # Spherical K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)

            clusters = kmeans.fit_predict(self.embeddings_reduced)

            # Silhouette score (higher is better)
            sil_score = silhouette_score(
                self.embeddings_reduced, clusters, metric="cosine"
            )

            results[k] = {
                "clusters": clusters,
                "kmeans": kmeans,
                "silhouette": sil_score,
            }

            print(f"  {k} clusters: Silhouette = {sil_score:.3f}")

        # Find best clustering
        best_k = max(results.keys(), key=lambda k: results[k]["silhouette"])
        best_result = results[best_k]

        print("\n" + "=" * 60)
        print("CLUSTERING RESULTS")
        print("=" * 60)
        print("\nCluster Count | Silhouette Score")
        print("-" * 35)

        for k in sorted(results.keys()):
            marker = " <- BEST" if k == best_k else ""
            print(
                f"     {k:2d}       |      {results[k]['silhouette']:6.3f}    {marker}"
            )

        print(f"\n✓ OPTIMAL CLUSTERING: {best_k} clusters")
        print(f"  Silhouette Score: {best_result['silhouette']:.3f}")

        # Save optimization results
        optimization_file = self.output_dir / "resolution_optimization.txt"
        with open(optimization_file, "w", encoding="utf-8") as f:
            f.write("CLUSTER OPTIMIZATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"TARGET CLUSTER RANGE: 2-6 clusters\n")
            f.write(f"OPTIMIZATION CRITERION: Silhouette Score\n")
            f.write(f"Optimal Number of Clusters: {best_k}\n")
            f.write(f"Best Silhouette Score: {best_result['silhouette']:.3f}\n\n")
            f.write("All Tested Cluster Counts:\n")
            f.write("Clusters | Silhouette\n")
            f.write("-" * 25 + "\n")
            for k in sorted(results.keys()):
                marker = " <- OPTIMAL" if k == best_k else ""
                f.write(f"   {k:2d}    |    {results[k]['silhouette']:6.3f}{marker}\n")

        print(f"Optimization results saved to: {optimization_file}")

        return best_k, best_result, results

    def analyze_communities(self, k, result, top_n=20):
        """Analyze and sample documents from each cluster"""
        clusters = result["clusters"]
        sil_score = result["silhouette"]

        analysis_text = (
            f"\n=== COMMUNITY ANALYSIS ({k} clusters, Silhouette={sil_score:.3f}) ===\n"
        )
        print(analysis_text)

        # Save analysis to file
        analysis_file = self.output_dir / f"community_analysis_{k}_clusters.txt"

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)

            for cluster_id in sorted(set(clusters)):
                members = np.where(clusters == cluster_id)[0]

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
                    # Get full predictions for this document
                    doc_preds = self.preds[idx]
                    # Escape newlines and save full text
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

    def compute_community_coherence(self, k, result):
        """Compute and save coherence scores for clusters"""
        clusters = result["clusters"]
        sil_score = result["silhouette"]

        print(f"Computing coherence scores for {k} clusters...")

        coherence_scores = {}

        for cluster_id in sorted(set(clusters)):
            members = np.where(clusters == cluster_id)[0]

            if len(members) < 2:
                coherence_scores[cluster_id] = 0.0
                continue

            # Sample for large clusters to avoid memory issues
            if len(members) > 1000:
                sample_size = min(500, len(members))
                sample_members = np.random.choice(members, sample_size, replace=False)
                cluster_embeddings = self.embeddings_reduced[sample_members]
            else:
                cluster_embeddings = self.embeddings_reduced[members]

            # Compute average pairwise cosine similarity within cluster
            similarities = np.dot(cluster_embeddings, cluster_embeddings.T)

            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(similarities, k=1)
            coherence = np.mean(upper_triangle[upper_triangle > 0])
            coherence_scores[cluster_id] = coherence

        coherence_text = (
            f"Community Coherence Scores ({k} clusters, Silhouette={sil_score:.3f}):\n"
        )

        # Print and save results
        print(f"\nCommunity Coherence Scores ({k} clusters):")
        for cluster_id, score in sorted(coherence_scores.items()):
            line = f"Community {cluster_id}: {score:.3f}"
            print(line)
            coherence_text += line + "\n"

        # Save coherence scores
        coherence_file = self.output_dir / f"coherence_scores_{k}_clusters.txt"
        with open(coherence_file, "w", encoding="utf-8") as f:
            f.write(coherence_text)
        print(f"Coherence scores saved to: {coherence_file}")

        return coherence_scores

    def compute_silhouette_analysis(self, k, result):
        """Compute silhouette scores for cluster validation"""
        from sklearn.metrics import silhouette_samples

        clusters = result["clusters"]
        sil_score = result["silhouette"]

        print(f"Computing detailed silhouette analysis for {k} clusters...")

        try:
            # Overall silhouette score (already computed)
            overall_silhouette = sil_score

            # Individual silhouette scores for each document
            sample_silhouette_values = silhouette_samples(
                self.embeddings_reduced, clusters, metric="cosine"
            )

            # Compute average silhouette score per cluster
            cluster_silhouettes = {}
            for cluster_id in sorted(set(clusters)):
                mask = clusters == cluster_id
                if np.sum(mask) > 1:
                    cluster_avg = np.mean(sample_silhouette_values[mask])
                    cluster_silhouettes[cluster_id] = cluster_avg
                else:
                    cluster_silhouettes[cluster_id] = 0.0

            # Print results
            print(
                f"\nOverall Silhouette Score ({k} clusters): {overall_silhouette:.3f}"
            )
            print(f"Per-Community Silhouette Scores:")

            silhouette_text = (
                f"SILHOUETTE ANALYSIS ({k} clusters, Silhouette={sil_score:.3f})\n"
            )
            silhouette_text += f"==================\n\n"
            silhouette_text += f"Overall Silhouette Score: {overall_silhouette:.3f}\n"
            silhouette_text += f"Interpretation:\n"
            silhouette_text += f"  > 0.7: Strong cluster structure\n"
            silhouette_text += f"  > 0.5: Reasonable cluster structure\n"
            silhouette_text += f"  > 0.3: Weak but acceptable structure\n"
            silhouette_text += f"  < 0.3: Poor cluster structure\n"
            silhouette_text += f"  < 0.0: Documents may be in wrong clusters\n\n"
            silhouette_text += f"Per-Community Silhouette Scores:\n"

            for cluster_id, score in sorted(cluster_silhouettes.items()):
                line = f"Community {cluster_id}: {score:.3f}"
                print(line)
                silhouette_text += line + "\n"

            # Save silhouette scores
            silhouette_file = self.output_dir / f"silhouette_analysis_{k}_clusters.txt"
            with open(silhouette_file, "w", encoding="utf-8") as f:
                f.write(silhouette_text)
            print(f"Silhouette analysis saved to: {silhouette_file}")

            return overall_silhouette, cluster_silhouettes

        except Exception as e:
            print(f"Error computing silhouette scores: {e}")
            return None, {}

    def visualize_communities_umap(self, k, result):
        """Visualize clusters using UMAP 2D projection"""
        try:
            clusters = result["clusters"]
            sil_score = result["silhouette"]

            print(f"Creating UMAP visualization for {k} clusters...")

            # UMAP projection to 2D for visualization
            umap_vis = umap.UMAP(
                n_components=2,
                n_neighbors=min(30, len(self.embeddings_reduced) // 10),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
                low_memory=True,
            )

            embedding_2d = umap_vis.fit_transform(self.embeddings_reduced)

            # Plot and save
            plt.figure(figsize=(14, 10))
            scatter = plt.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=clusters,
                cmap="tab10",
                alpha=0.7,
                s=6,
            )
            plt.colorbar(scatter, label="Community ID")

            # Add cluster labels at centroids
            unique_clusters = sorted(set(clusters))
            for cluster_id in unique_clusters:
                # Find centroid of each cluster
                mask = clusters == cluster_id
                if np.sum(mask) > 0:
                    centroid_x = np.mean(embedding_2d[mask, 0])
                    centroid_y = np.mean(embedding_2d[mask, 1])

                    # Add text label with background
                    plt.annotate(
                        f"{cluster_id}",
                        (centroid_x, centroid_y),
                        fontsize=14,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.4",
                            facecolor="white",
                            edgecolor="black",
                            alpha=0.9,
                        ),
                    )

            plt.title(
                f"UMAP Visualization: {k} Communities (Silhouette={sil_score:.3f})\nNumbers show Community IDs"
            )
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")

            # Save plot
            plot_path = self.output_dir / f"umap_communities_{k}_clusters.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"UMAP plot saved to: {plot_path}")
            return embedding_2d, clusters

        except Exception as e:
            print(f"Error creating UMAP visualization: {e}")
            plt.close("all")
            return None, None

    def run_full_analysis(self):
        """Run the complete subregister discovery pipeline"""
        try:
            print("=" * 60)
            print("SUBREGISTER DISCOVERY ANALYSIS")
            print("Target cluster range: 2-6 clusters")
            print("=" * 60)

            # Step 1: UMAP dimensionality reduction
            self.reduce_dimensions_umap()

            # Step 2: Find optimal clustering
            optimal_k, optimal_result, all_results = self.find_optimal_clusters()

            # Save summary info
            summary_file = self.output_dir / "analysis_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("SUBREGISTER DISCOVERY ANALYSIS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Input file: {self.pickle_path}\n")
                f.write(f"Number of documents: {len(self.embeddings)}\n")
                f.write(f"Embedding dimension: {self.embeddings.shape[1]}\n")
                f.write(f"Register distribution: {dict(Counter(self.labels))}\n")
                f.write(f"Method: Spherical K-means\n")
                f.write(f"Target cluster range: 2-6 clusters\n")
                f.write(f"Optimization criterion: Calinski-Harabasz Score\n")
                f.write(f"Optimal number of clusters: {optimal_k}\n")
                f.write(f"Best silhouette: {optimal_result['silhouette']:.2f}\n\n")

            # Step 3: Full analysis at optimal clustering
            print(f"\n{'=' * 60}")
            print(f"ANALYZING OPTIMAL CLUSTERING: {optimal_k} clusters")
            print(f"Silhouette Score: {optimal_result['silhouette']:.3f}")
            print(f"{'=' * 60}")

            # Analyze communities
            self.analyze_communities(optimal_k, optimal_result)

            # Compute coherence scores
            self.compute_community_coherence(optimal_k, optimal_result)

            # Compute silhouette scores
            self.compute_silhouette_analysis(optimal_k, optimal_result)

            # Visualization
            try:
                self.visualize_communities_umap(optimal_k, optimal_result)
            except Exception as e:
                print(f"Skipping UMAP visualization due to error: {e}")

            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print(f"Optimal clustering: {optimal_k} clusters")
            print(f"Silhouette score: {optimal_result['silhouette']:.3f}")
            print(f"All results saved to: {self.output_dir}")
            print("=" * 60)

        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Cleanup
            plt.close("all")


# Usage example
if __name__ == "__main__":
    # Find all pkl files in the directory
    pkl_directory = "../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/"
    pkl_files = glob.glob(os.path.join(pkl_directory, "*.pkl"))

    # Create common results directory
    results_dir = "subregister_results"
    Path(results_dir).mkdir(exist_ok=True)

    print(f"Found {len(pkl_files)} pkl files to process:")
    for i, file in enumerate(pkl_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    print(f"\nAll results will be saved to: {results_dir}/")
    print("Target: 2-6 clusters per register")

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
            analyzer = SubregisterAnalyzer(pkl_file, results_base_dir=results_dir)

            # Run analysis
            analyzer.run_full_analysis()

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

    print("\nAnalysis completed with cleanup.")
