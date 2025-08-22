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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

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

        # Check minimum size requirement BEFORE creating directories
        if len(self.embeddings) < 1000:
            raise ValueError(
                f"Dataset too small ({len(self.embeddings)} documents). Minimum 1000 required."
            )

        # Only create directories if dataset is large enough
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

    def reduce_dimensions(self, n_components=40):
        """Apply PCA for noise reduction and speedup"""
        print(f"Applying PCA to reduce to {n_components} dimensions...")
        self.pca = PCA(n_components=n_components, random_state=42)
        self.embeddings_pca = self.pca.fit_transform(self.embeddings_norm)

        # Normalize PCA embeddings
        self.embeddings_pca_norm = self.embeddings_pca / np.linalg.norm(
            self.embeddings_pca, axis=1, keepdims=True
        )

        total_variance = self.pca.explained_variance_ratio_.sum()
        first_10_variance = self.pca.explained_variance_ratio_[:10].sum()
        print(
            f"PCA total variance explained: {total_variance:.3f} (all {n_components} components)"
        )
        print(f"First 10 components explain: {first_10_variance:.3f} of total variance")

        return self.embeddings_pca_norm

    def find_optimal_k(self, k_range=[2, 3, 4, 5, 6, 7, 8]):
        """Find optimal number of clusters using K-means and silhouette scores"""
        print("=" * 60)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS (K-MEANS)")
        print("=" * 60)

        k_scores = {}

        for k in k_range:
            print(f"\nTesting k={k} clusters...")
            try:
                # Run K-means clustering
                kmeans = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    n_init=10,
                    max_iter=300,
                    random_state=42,
                )
                communities = kmeans.fit_predict(self.embeddings_pca_norm)

                # Compute silhouette score
                print(f"  Computing silhouette score for {k} clusters...")
                silhouette = silhouette_score(
                    self.embeddings_pca_norm, communities, metric="cosine"
                )

                # Compute inertia (within-cluster sum of squares)
                inertia = kmeans.inertia_

                k_scores[k] = {
                    "silhouette": silhouette,
                    "inertia": inertia,
                    "communities": communities,
                    "kmeans_model": kmeans,
                }

                print(
                    f"  k={k}: SILHOUETTE = {silhouette:.3f}, inertia = {inertia:.0f}"
                )

            except Exception as e:
                print(f"  Error at k={k}: {e}")
                continue

        if not k_scores:
            raise ValueError(
                "No valid k found. Dataset may be too difficult to cluster."
            )

        # Find optimal k based on SILHOUETTE SCORE
        optimal_k = max(k_scores.keys(), key=lambda k: k_scores[k]["silhouette"])
        optimal_silhouette = k_scores[optimal_k]["silhouette"]
        optimal_communities = k_scores[optimal_k]["communities"]
        optimal_model = k_scores[optimal_k]["kmeans_model"]

        print(f"\n" + "=" * 60)
        print(f"OPTIMAL NUMBER OF CLUSTERS: {optimal_k}")
        print(f"OPTIMIZATION CRITERION: Silhouette Score")
        print(f"Best Silhouette Score: {optimal_silhouette:.3f}")
        print("=" * 60)

        # Save optimization results
        optimization_file = self.output_dir / "cluster_optimization.txt"
        with open(optimization_file, "w", encoding="utf-8") as f:
            f.write("CLUSTER OPTIMIZATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"METHOD: K-Means Clustering\n")
            f.write(f"OPTIMIZATION CRITERION: Silhouette Score\n")
            f.write(f"Optimal number of clusters: {optimal_k}\n")
            f.write(f"Best Silhouette Score: {optimal_silhouette:.3f}\n\n")
            f.write("All Tested Cluster Numbers:\n")
            f.write("Clusters | Silhouette | Inertia\n")
            f.write("-" * 35 + "\n")
            for k in sorted(k_scores.keys()):
                sil = k_scores[k]["silhouette"]
                inertia = k_scores[k]["inertia"]
                marker = " <- OPTIMAL" if k == optimal_k else ""
                f.write(f"   {k:2d}    |   {sil:.3f}    | {inertia:8.0f}{marker}\n")

        print(f"Cluster optimization results saved to: {optimization_file}")

        return optimal_k, optimal_communities, optimal_model, k_scores

    def compute_coherence_for_communities(self, communities):
        """Compute coherence scores for a set of communities (memory-efficient)"""
        coherence_scores = {}

        for community_id in sorted(set(communities)):
            members = np.where(communities == community_id)[0]

            if len(members) < 2:
                coherence_scores[community_id] = 0.0
                continue

            # Memory-efficient coherence computation for large communities
            if len(members) > 1000:
                # Sample for large communities to avoid memory issues
                sample_size = min(500, len(members))
                sample_members = np.random.choice(members, sample_size, replace=False)
                community_embeddings = self.embeddings_pca_norm[sample_members]
                print(
                    f"  Cluster {community_id}: Sampling {sample_size}/{len(members)} documents for coherence"
                )
            else:
                community_embeddings = self.embeddings_pca_norm[members]

            # Compute average pairwise cosine similarity within community
            similarity_matrix = cosine_similarity(community_embeddings)

            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(similarity_matrix, k=1)
            coherence = np.mean(upper_triangle[upper_triangle > 0])
            coherence_scores[community_id] = coherence

            # Clear memory immediately
            del similarity_matrix, upper_triangle, community_embeddings
            import gc

            gc.collect()

        return coherence_scores

    def analyze_communities(self, k, top_n=10):
        """Analyze and sample documents from each cluster"""
        communities = self.community_results[k]

        # Compute coherence scores for all communities
        print(f"Computing coherence scores for cluster analysis (k={k})...")
        coherence_scores = self.compute_coherence_for_communities(communities)

        analysis_text = f"\n=== CLUSTER ANALYSIS (k={k}) ===\n"
        print(analysis_text)

        # Save analysis to file
        analysis_file = self.output_dir / f"cluster_analysis_k_{k}.txt"

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)

            for cluster_id in sorted(set(communities)):
                members = np.where(communities == cluster_id)[0]
                coherence_score = coherence_scores.get(cluster_id, 0.0)

                section = f"\n--- Cluster {cluster_id} ({len(members)} documents, coherence: {coherence_score:.3f}) ---\n"
                print(section)
                f.write(section)

                # Sample representative documents
                if len(members) > 0:
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

        print(f"Cluster analysis saved to: {analysis_file}")

    def compute_cluster_coherence(self, k):
        """Compute intra-cluster coherence scores (memory-efficient)"""
        communities = self.community_results[k]

        print(f"Computing cluster coherence scores (k={k})...")
        coherence_scores = self.compute_coherence_for_communities(communities)

        coherence_text = f"Cluster Coherence Scores (k={k}):\n"

        # Print and save results
        print(f"\nCluster Coherence Scores (k={k}):")
        for cluster_id, score in sorted(coherence_scores.items()):
            line = f"Cluster {cluster_id}: {score:.3f}"
            print(line)
            coherence_text += line + "\n"

        # Save coherence scores
        coherence_file = self.output_dir / f"coherence_scores_k_{k}.txt"
        with open(coherence_file, "w", encoding="utf-8") as f:
            f.write(coherence_text)
        print(f"Coherence scores saved to: {coherence_file}")

        return coherence_scores

    def compute_silhouette_analysis(self, k):
        """Compute silhouette scores for cluster validation"""
        communities = self.community_results[k]

        print(f"Computing silhouette scores (k={k})...")

        try:
            # Overall silhouette score
            overall_silhouette = silhouette_score(
                self.embeddings_pca_norm, communities, metric="cosine"
            )

            # Individual silhouette scores for each document
            sample_silhouette_values = silhouette_samples(
                self.embeddings_pca_norm, communities, metric="cosine"
            )

            # Compute average silhouette score per cluster
            cluster_silhouettes = {}
            for cluster_id in sorted(set(communities)):
                mask = communities == cluster_id
                if np.sum(mask) > 1:  # Need at least 2 documents
                    cluster_avg = np.mean(sample_silhouette_values[mask])
                    cluster_silhouettes[cluster_id] = cluster_avg
                else:
                    cluster_silhouettes[cluster_id] = 0.0

            # Print results
            print(f"\nOverall Silhouette Score (k={k}): {overall_silhouette:.3f}")
            print(f"\nPer-Cluster Silhouette Scores (k={k}):")

            silhouette_text = f"SILHOUETTE ANALYSIS (k={k})\n"
            silhouette_text += f"==================\n\n"
            silhouette_text += f"Overall Silhouette Score: {overall_silhouette:.3f}\n"
            silhouette_text += f"Interpretation:\n"
            silhouette_text += f"  > 0.7: Strong cluster structure\n"
            silhouette_text += f"  > 0.5: Reasonable cluster structure\n"
            silhouette_text += f"  > 0.3: Weak but acceptable structure\n"
            silhouette_text += f"  < 0.3: Poor cluster structure\n"
            silhouette_text += f"  < 0.0: Documents may be in wrong clusters\n\n"
            silhouette_text += f"Per-Cluster Silhouette Scores:\n"

            for cluster_id, score in sorted(cluster_silhouettes.items()):
                line = f"Cluster {cluster_id}: {score:.3f}"
                print(line)
                silhouette_text += line + "\n"

            # Save silhouette scores
            silhouette_file = self.output_dir / f"silhouette_analysis_k_{k}.txt"
            with open(silhouette_file, "w", encoding="utf-8") as f:
                f.write(silhouette_text)
            print(f"\nSilhouette analysis saved to: {silhouette_file}")

            return overall_silhouette, cluster_silhouettes

        except Exception as e:
            print(f"Error computing silhouette scores: {e}")
            return None, {}

    def visualize_clusters_umap(self, k):
        """Visualize clusters using UMAP"""
        try:
            print(
                f"Creating UMAP visualization for all {len(self.embeddings)} documents (k={k})..."
            )

            # Use all data
            embeddings_vis = self.embeddings_pca_norm
            communities_vis = self.community_results[k]

            # UMAP projection
            umap_model = umap.UMAP(
                n_neighbors=min(30, len(embeddings_vis) // 10),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
                low_memory=True,
            )
            embedding_2d = umap_model.fit_transform(embeddings_vis)

            # Plot and save
            plt.figure(figsize=(14, 10))
            scatter = plt.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=communities_vis,
                cmap="tab10",
                alpha=0.7,
                s=6,
            )
            plt.colorbar(scatter, label="Cluster ID")

            # Add cluster labels at centroids
            unique_clusters = sorted(set(communities_vis))
            for cluster_id in unique_clusters:
                # Find centroid of each cluster
                mask = communities_vis == cluster_id
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

            plt.title(f"UMAP Visualization of {k} Clusters\nNumbers show Cluster IDs")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")

            # Save plot
            plot_path = self.output_dir / f"umap_clusters_k_{k}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            # Create a legend mapping colors to cluster IDs
            self._create_cluster_legend(unique_clusters, k)

            # Force garbage collection
            import gc

            gc.collect()

            print(f"UMAP plot saved to: {plot_path}")
            return embedding_2d, communities_vis

        except Exception as e:
            print(f"Error creating UMAP visualization: {e}")
            plt.close("all")
            return None, None

    def _create_cluster_legend(self, unique_clusters, k):
        """Create a separate legend figure showing cluster colors"""
        try:
            fig, ax = plt.subplots(figsize=(8, max(4, len(unique_clusters) * 0.5)))

            # Get colors from tab10 colormap (good for up to 10 clusters)
            cmap = plt.cm.tab10
            colors = [cmap(i / 10) for i in range(len(unique_clusters))]

            # Create legend entries
            for i, cluster_id in enumerate(unique_clusters):
                ax.scatter([], [], c=[colors[i]], s=120, label=f"Cluster {cluster_id}")

            ax.legend(loc="center", fontsize=12, ncol=max(1, len(unique_clusters) // 4))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_title(
                f"Cluster Color Legend (k={k})", fontsize=16, fontweight="bold"
            )

            # Save legend
            legend_path = self.output_dir / f"cluster_legend_k_{k}.png"
            plt.savefig(legend_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Cluster legend saved to: {legend_path}")

        except Exception as e:
            print(f"Error creating cluster legend: {e}")
            plt.close("all")

    def run_full_analysis(self):
        """Run the complete subregister discovery pipeline with K-means clustering"""
        try:
            print("=" * 60)
            print("SUBREGISTER DISCOVERY ANALYSIS")
            print("=" * 60)

            # Step 1: Dimensionality reduction
            self.reduce_dimensions()

            # Step 2: Find optimal number of clusters
            optimal_k, optimal_communities, optimal_model, all_scores = (
                self.find_optimal_k()
            )

            # Store results
            self.community_results = {optimal_k: optimal_communities}
            self.kmeans_model = optimal_model

            # Save summary info
            summary_file = self.output_dir / "analysis_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("SUBREGISTER DISCOVERY ANALYSIS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Input file: {self.pickle_path}\n")
                f.write(f"Number of documents: {len(self.embeddings)}\n")
                f.write(f"Embedding dimension: {self.embeddings.shape[1]}\n")
                f.write(f"Register distribution: {dict(Counter(self.labels))}\n")
                f.write(f"Clustering method: K-Means\n")
                f.write(f"Optimization criterion: Silhouette Score\n")
                f.write(f"Optimal number of clusters: {optimal_k}\n")
                f.write(
                    f"Best silhouette score: {all_scores[optimal_k]['silhouette']:.3f}\n\n"
                )

            # Step 3: Full analysis at optimal k
            print(f"\n{'=' * 60}")
            print(f"ANALYZING OPTIMAL CLUSTERING: k={optimal_k}")
            print(
                f"CHOSEN FOR BEST SILHOUETTE SCORE: {all_scores[optimal_k]['silhouette']:.3f}"
            )
            print(f"{'=' * 60}")

            # Analyze clusters
            self.analyze_communities(k=optimal_k)

            # Compute coherence scores
            self.compute_cluster_coherence(k=optimal_k)

            # Compute silhouette scores for detailed validation
            self.compute_silhouette_analysis(k=optimal_k)

            # Visualization
            try:
                self.visualize_clusters_umap(k=optimal_k)
            except Exception as e:
                print(f"Skipping UMAP visualization due to error: {e}")

            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print(
                f"Optimal number of clusters: {optimal_k} (chosen for best silhouette score)"
            )
            print(f"Best silhouette score: {all_scores[optimal_k]['silhouette']:.3f}")
            print(f"All results saved to: {self.output_dir}")
            print("=" * 60)

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

    # Create common results directory
    results_dir = "subregister_results"
    Path(results_dir).mkdir(exist_ok=True)

    print(f"Found {len(pkl_files)} pkl files to process:")
    for i, file in enumerate(pkl_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    print(f"\nAll results will be saved to: {results_dir}/")

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

            # Run analysis with K-MEANS clustering
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

            # Print traceback for debugging
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
