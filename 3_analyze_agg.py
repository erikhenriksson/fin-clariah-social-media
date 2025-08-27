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
from sklearn.metrics import silhouette_score
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

    def reduce_dimensions(self, target_variance=0.80):
        """Apply PCA for noise reduction and speedup, automatically selecting components for target variance"""

        # First, fit PCA with all components to get variance ratios
        print("Determining optimal number of components...")
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

        # Now apply PCA with the optimal number of components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.embeddings_pca = self.pca.fit_transform(self.embeddings_norm)

        # Normalize PCA embeddings
        self.embeddings_pca_norm = self.embeddings_pca / np.linalg.norm(
            self.embeddings_pca, axis=1, keepdims=True
        )

        # Report results
        actual_variance = self.pca.explained_variance_ratio_.sum()
        print(
            f"PCA total variance explained: {actual_variance:.3f} (using {n_components} components)"
        )

        return self.embeddings_pca_norm

    def cluster_with_hdbscan(self, min_cluster_size, min_samples=None, epsilon=None):
        """Perform HDBSCAN clustering"""
        if min_samples is None:
            min_samples = max(
                5, min_cluster_size // 10
            )  # Default: 10% of min_cluster_size

        print(
            f"Running HDBSCAN clustering (min_cluster_size={min_cluster_size}, min_samples={min_samples})..."
        )

        clusterer = hdbscan.HDBSCAN(
            metric="cosine",
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=epsilon,
            cluster_selection_method="eom",  # Excess of Mass
        )

        communities = clusterer.fit_predict(self.embeddings_pca_norm)
        n_clusters = len(set(communities)) - (
            1 if -1 in communities else 0
        )  # Exclude noise cluster (-1)
        n_noise = np.sum(communities == -1)

        print(f"  Found {n_clusters} clusters and {n_noise} noise points")

        return communities, n_clusters, clusterer

    def evaluate_clustering_quality(self, communities, n_clusters, clusterer):
        """Evaluate clustering quality using multiple metrics"""
        # Filter out noise points for evaluation
        mask = communities >= 0
        if np.sum(mask) < 2:
            return {
                "n_clusters": n_clusters,
                "communities": communities,
                "dbcv_score": -1.0,
                "silhouette": -1.0,
                "clusterer": clusterer,
                "n_noise": np.sum(communities == -1),
            }

        # DBCV Score (Density-Based Cluster Validation) - native to HDBSCAN
        try:
            dbcv_score = clusterer.relative_validity_
            if dbcv_score is None:
                dbcv_score = -1.0
        except:
            dbcv_score = -1.0

        # Silhouette Score (excluding noise points)
        try:
            if n_clusters > 1:
                sil_score = silhouette_score(
                    self.embeddings_pca_norm[mask], communities[mask], metric="cosine"
                )
            else:
                sil_score = -1.0
        except:
            sil_score = -1.0

        return {
            "n_clusters": n_clusters,
            "communities": communities,
            "dbcv_score": dbcv_score,
            "silhouette": sil_score,
            "clusterer": clusterer,
            "n_noise": np.sum(communities == -1),
        }

    def find_optimal_clusters(self, target_range=[2, 6]):
        """Find optimal clustering by testing different min_cluster_sizes"""
        print("=" * 60)
        print("FINDING OPTIMAL CLUSTERING WITH HDBSCAN")
        print("=" * 60)

        results = {}
        n_docs = len(self.embeddings)

        # Generate candidate min_cluster_sizes that might yield clusters in target range
        # Strategy: Start with larger min_cluster_size for fewer clusters, decrease for more clusters
        min_cluster_candidates = []

        # Conservative estimates based on dataset size and target clusters
        base_size = max(50, n_docs // (max(target_range) * 3))  # Start conservative
        for i in range(8):  # Test 8 different sizes
            size = max(25, int(base_size * (0.7**i)))  # Exponentially decrease
            min_cluster_candidates.append(size)

        # Remove duplicates and sort
        min_cluster_candidates = sorted(list(set(min_cluster_candidates)), reverse=True)

        print(f"Testing min_cluster_sizes: {min_cluster_candidates}")

        for min_cluster_size in min_cluster_candidates:
            print(f"\nTesting min_cluster_size={min_cluster_size}...")

            # Perform clustering
            communities, n_clusters, clusterer = self.cluster_with_hdbscan(
                min_cluster_size=min_cluster_size
            )

            # Skip if no clusters found or only one cluster
            if n_clusters < 2:
                print(f"  Skipping: only {n_clusters} clusters found")
                continue

            # Evaluate clustering quality
            result = self.evaluate_clustering_quality(
                communities, n_clusters, clusterer
            )
            result["min_cluster_size"] = min_cluster_size

            # Only keep results in our target range
            if target_range[0] <= n_clusters <= target_range[1]:
                results[f"{min_cluster_size}_{n_clusters}"] = result
                print(
                    f"  ✓ {n_clusters} clusters, DBCV: {result['dbcv_score']:.3f}, "
                    f"Silhouette: {result['silhouette']:.3f}, Noise: {result['n_noise']}"
                )
            else:
                print(
                    f"  ✗ {n_clusters} clusters (outside target range {target_range})"
                )

        if not results:
            # Fallback: try smaller min_cluster_sizes to get any reasonable clustering
            print(
                f"\nNo clustering in target range found. Trying smaller min_cluster_sizes..."
            )
            fallback_sizes = [20, 15, 10, 5]
            for min_cluster_size in fallback_sizes:
                print(f"\nFallback: testing min_cluster_size={min_cluster_size}...")
                communities, n_clusters, clusterer = self.cluster_with_hdbscan(
                    min_cluster_size=min_cluster_size
                )
                if n_clusters >= 2:
                    result = self.evaluate_clustering_quality(
                        communities, n_clusters, clusterer
                    )
                    result["min_cluster_size"] = min_cluster_size
                    results[f"{min_cluster_size}_{n_clusters}"] = result
                    print(
                        f"  ✓ {n_clusters} clusters, DBCV: {result['dbcv_score']:.3f}, "
                        f"Silhouette: {result['silhouette']:.3f}, Noise: {result['n_noise']}"
                    )
                    break

        return results

    def select_best_clustering(self, results):
        """Select the best clustering based on DBCV score (primary) and silhouette (secondary)"""
        print("\n" + "=" * 60)
        print("SELECTING BEST CLUSTERING")
        print("=" * 60)

        if not results:
            raise ValueError("No valid clustering results found!")

        print("\nMin_Cluster_Size | Clusters | DBCV Score | Silhouette | Noise Points")
        print("-" * 70)

        best_key = None
        best_dbcv = -2.0  # DBCV can be negative
        best_silhouette = -2.0

        for key in sorted(results.keys()):
            result = results[key]
            min_cluster_size = result["min_cluster_size"]
            n_clusters = result["n_clusters"]
            dbcv_score = result["dbcv_score"]
            sil_score = result["silhouette"]
            n_noise = result["n_noise"]

            marker = ""

            # Primary criterion: DBCV score
            if dbcv_score > best_dbcv:
                best_dbcv = dbcv_score
                best_silhouette = sil_score
                best_key = key
                marker = " <- BEST"
            elif dbcv_score == best_dbcv and sil_score > best_silhouette:
                # Tie-breaker: silhouette score
                best_silhouette = sil_score
                best_key = key
                marker = " <- BEST"

            print(
                f"      {min_cluster_size:3d}        |    {n_clusters:2d}    |   {dbcv_score:6.3f}   |   {sil_score:6.3f}   |     {n_noise:4d}    {marker}"
            )

        optimal_result = results[best_key]
        optimal_k = optimal_result["n_clusters"]

        print(f"\n✓ OPTIMAL CLUSTERING: {optimal_k} clusters")
        print(f"  Min Cluster Size: {optimal_result['min_cluster_size']}")
        print(f"  DBCV Score: {optimal_result['dbcv_score']:.3f}")
        print(f"  Silhouette Score: {optimal_result['silhouette']:.3f}")
        print(f"  Noise Points: {optimal_result['n_noise']}")

        # Save optimization results
        optimization_file = self.output_dir / "clustering_optimization.txt"
        with open(optimization_file, "w", encoding="utf-8") as f:
            f.write("CLUSTERING OPTIMIZATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"CLUSTERING METHOD: HDBSCAN (cosine distance)\n")
            f.write(
                f"OPTIMIZATION CRITERION: DBCV Score (primary), Silhouette (secondary)\n"
            )
            f.write(f"Optimal Number of Clusters: {optimal_k}\n")
            f.write(f"Optimal Min Cluster Size: {optimal_result['min_cluster_size']}\n")
            f.write(f"Best DBCV Score: {optimal_result['dbcv_score']:.3f}\n")
            f.write(
                f"Corresponding Silhouette Score: {optimal_result['silhouette']:.3f}\n"
            )
            f.write(f"Noise Points: {optimal_result['n_noise']}\n\n")
            f.write("All Tested Configurations:\n")
            f.write("Min_Cluster_Size | Clusters | DBCV Score | Silhouette | Noise\n")
            f.write("-" * 60 + "\n")
            for key in sorted(results.keys()):
                result = results[key]
                marker = " <- OPTIMAL" if key == best_key else ""
                f.write(
                    f"      {result['min_cluster_size']:3d}        |    {result['n_clusters']:2d}    |   {result['dbcv_score']:6.3f}   |   {result['silhouette']:6.3f}   |   {result['n_noise']:4d}  {marker}\n"
                )

        print(f"\nOptimization results saved to: {optimization_file}")
        return optimal_k, optimal_result

    def compute_coherence_for_communities(self, communities):
        """Compute coherence scores for a set of communities (memory-efficient)"""
        coherence_scores = {}

        for community_id in set(communities):
            if community_id == -1:  # Skip noise cluster
                continue

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

    def analyze_communities(self, k, optimal_result, top_n=10):
        """Analyze and sample documents from each community"""
        communities = optimal_result["communities"]
        dbcv_score = optimal_result["dbcv_score"]
        n_noise = optimal_result["n_noise"]

        # Compute coherence scores for all communities
        print(f"Computing coherence scores for {k} communities...")
        coherence_scores = self.compute_coherence_for_communities(communities)

        analysis_text = f"\n=== COMMUNITY ANALYSIS ({k} clusters, DBCV={dbcv_score:.3f}, {n_noise} noise points) ===\n"
        print(analysis_text)

        # Save analysis to file
        analysis_file = self.output_dir / f"community_analysis_{k}_clusters.txt"

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)

            # First, handle noise cluster if it exists
            if -1 in communities:
                noise_members = np.where(communities == -1)[0]
                section = f"\n--- NOISE CLUSTER ({len(noise_members)} documents) ---\n"
                print(section)
                f.write(section)

                # Sample a few noise points
                sample_indices = np.random.choice(
                    noise_members, min(5, len(noise_members)), replace=False
                )

                for i, idx in enumerate(sample_indices):
                    doc_preds = self.preds[idx]
                    full_text = (
                        self.texts[idx].replace("\n", "\\n").replace("\r", "\\r")
                    )
                    line = f"{i + 1}. [{idx}] {doc_preds} {full_text}\n"
                    truncated_text = (
                        self.texts[idx][:100] + "..."
                        if len(self.texts[idx]) > 100
                        else self.texts[idx]
                    )
                    print(f"{i + 1}. [{idx}] {doc_preds} {truncated_text}")
                    f.write(line)

            # Now handle regular clusters
            for community_id in sorted([c for c in set(communities) if c >= 0]):
                members = np.where(communities == community_id)[0]
                coherence_score = coherence_scores.get(community_id, 0.0)

                section = f"\n--- Community {community_id} ({len(members)} documents, coherence: {coherence_score:.3f}) ---\n"
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
        return coherence_scores

    def compute_community_coherence(self, k, optimal_result, coherence_scores):
        """Save community coherence scores"""
        dbcv_score = optimal_result["dbcv_score"]

        coherence_text = (
            f"Community Coherence Scores ({k} clusters, DBCV={dbcv_score:.3f}):\n"
        )

        # Print and save results
        print(f"\nCommunity Coherence Scores ({k} clusters):")
        for community_id, score in sorted(coherence_scores.items()):
            line = f"Community {community_id}: {score:.3f}"
            print(line)
            coherence_text += line + "\n"

        # Save coherence scores
        coherence_file = self.output_dir / f"coherence_scores_{k}_clusters.txt"
        with open(coherence_file, "w", encoding="utf-8") as f:
            f.write(coherence_text)
        print(f"Coherence scores saved to: {coherence_file}")

        return coherence_scores

    def compute_silhouette_analysis(self, k, optimal_result):
        """Compute silhouette scores for community validation"""
        communities = optimal_result["communities"]
        dbcv_score = optimal_result["dbcv_score"]
        n_noise = optimal_result["n_noise"]

        print(f"Computing detailed silhouette analysis for {k} clusters...")

        try:
            # Overall silhouette score (already computed, excluding noise)
            overall_silhouette = optimal_result["silhouette"]

            # Individual silhouette scores for each document (excluding noise)
            mask = communities >= 0
            if np.sum(mask) > 1:
                from sklearn.metrics import silhouette_samples

                sample_silhouette_values = silhouette_samples(
                    self.embeddings_pca_norm[mask], communities[mask], metric="cosine"
                )

                # Compute average silhouette score per community
                community_silhouettes = {}
                communities_filtered = communities[mask]

                for community_id in sorted([c for c in set(communities) if c >= 0]):
                    community_mask = communities_filtered == community_id
                    if np.sum(community_mask) > 1:  # Need at least 2 documents
                        community_avg = np.mean(
                            sample_silhouette_values[community_mask]
                        )
                        community_silhouettes[community_id] = community_avg
                    else:
                        community_silhouettes[community_id] = 0.0
            else:
                community_silhouettes = {}

            # Print results
            print(
                f"\nOverall Silhouette Score ({k} clusters): {overall_silhouette:.3f}"
            )
            print(f"Overall DBCV Score ({k} clusters): {dbcv_score:.3f}")
            print(f"Noise points: {n_noise}")
            print(f"\nPer-Community Silhouette Scores:")

            silhouette_text = (
                f"SILHOUETTE ANALYSIS ({k} clusters, DBCV={dbcv_score:.3f})\n"
            )
            silhouette_text += f"==================\n\n"
            silhouette_text += f"Overall Silhouette Score: {overall_silhouette:.3f}\n"
            silhouette_text += f"Overall DBCV Score: {dbcv_score:.3f}\n"
            silhouette_text += f"Noise Points: {n_noise}\n"
            silhouette_text += f"Interpretation:\n"
            silhouette_text += f"  Silhouette > 0.7: Strong cluster structure\n"
            silhouette_text += f"  Silhouette > 0.5: Reasonable cluster structure\n"
            silhouette_text += f"  Silhouette > 0.3: Weak but acceptable structure\n"
            silhouette_text += f"  Silhouette < 0.3: Poor cluster structure\n"
            silhouette_text += (
                f"  Silhouette < 0.0: Documents may be in wrong clusters\n"
            )
            silhouette_text += f"  DBCV > 0.0: Well-separated dense clusters\n"
            silhouette_text += f"  DBCV < 0.0: Poor cluster separation or density\n\n"
            silhouette_text += f"Per-Community Silhouette Scores:\n"

            for community_id, score in sorted(community_silhouettes.items()):
                line = f"Community {community_id}: {score:.3f}"
                print(line)
                silhouette_text += line + "\n"

            # Save silhouette scores
            silhouette_file = self.output_dir / f"silhouette_analysis_{k}_clusters.txt"
            with open(silhouette_file, "w", encoding="utf-8") as f:
                f.write(silhouette_text)
            print(f"\nSilhouette analysis saved to: {silhouette_file}")

            return overall_silhouette, community_silhouettes

        except Exception as e:
            print(f"Error computing silhouette scores: {e}")
            return None, {}

    def visualize_communities_umap(self, k, optimal_result):
        """Visualize communities using UMAP"""
        try:
            communities = optimal_result["communities"]
            dbcv_score = optimal_result["dbcv_score"]
            n_noise = optimal_result["n_noise"]

            print(f"Creating UMAP visualization for {k} clusters...")

            # Use all data
            embeddings_vis = self.embeddings_pca_norm
            communities_vis = communities

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

            # Handle noise points separately
            noise_mask = communities_vis == -1
            cluster_mask = communities_vis >= 0

            # Plot noise points in gray
            if np.any(noise_mask):
                plt.scatter(
                    embedding_2d[noise_mask, 0],
                    embedding_2d[noise_mask, 1],
                    c="lightgray",
                    alpha=0.3,
                    s=3,
                    label="Noise",
                )

            # Plot cluster points
            if np.any(cluster_mask):
                scatter = plt.scatter(
                    embedding_2d[cluster_mask, 0],
                    embedding_2d[cluster_mask, 1],
                    c=communities_vis[cluster_mask],
                    cmap="tab10",
                    alpha=0.7,
                    s=6,
                )
                plt.colorbar(scatter, label="Community ID")

                # Add community labels at centroids
                unique_communities = sorted([c for c in set(communities_vis) if c >= 0])
                for community_id in unique_communities:
                    # Find centroid of each community
                    mask = communities_vis == community_id
                    if np.sum(mask) > 0:
                        centroid_x = np.mean(embedding_2d[mask, 0])
                        centroid_y = np.mean(embedding_2d[mask, 1])

                        # Add text label with background
                        plt.annotate(
                            f"{community_id}",
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
                f"UMAP Visualization: {k} Communities (DBCV={dbcv_score:.3f}, {n_noise} noise)\nNumbers show Community IDs"
            )
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")

            if np.any(noise_mask):
                plt.legend()

            # Save plot
            plot_path = self.output_dir / f"umap_communities_{k}_clusters.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            # Force garbage collection
            import gc

            gc.collect()

            print(f"UMAP plot saved to: {plot_path}")
            return embedding_2d, communities_vis

        except Exception as e:
            print(f"Error creating UMAP visualization: {e}")
            plt.close("all")
            return None, None

    def run_full_analysis(self):
        """Run the complete subregister discovery pipeline with HDBSCAN"""
        try:
            print("=" * 60)
            print("SUBREGISTER DISCOVERY ANALYSIS")
            print("Method: HDBSCAN (cosine distance)")
            print("Target cluster range: 2-6 clusters")
            print("=" * 60)

            # Step 1: Dimensionality reduction
            self.reduce_dimensions()

            # Step 2: Find optimal clusters using DBCV score
            results = self.find_optimal_clusters()

            # Step 3: Select best clustering
            optimal_k, optimal_result = self.select_best_clustering(results)

            # Save summary info
            summary_file = self.output_dir / "analysis_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("SUBREGISTER DISCOVERY ANALYSIS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Input file: {self.pickle_path}\n")
                f.write(f"Number of documents: {len(self.embeddings)}\n")
                f.write(f"Embedding dimension: {self.embeddings.shape[1]}\n")
                f.write(f"Register distribution: {dict(Counter(self.labels))}\n")
                f.write(f"Clustering method: HDBSCAN (cosine distance)\n")
                f.write(f"Target cluster range: 2-6 clusters\n")
                f.write(
                    f"Optimization criterion: DBCV Score (primary), Silhouette (secondary)\n"
                )
                f.write(f"Optimal number of clusters: {optimal_k}\n")
                f.write(
                    f"Optimal min_cluster_size: {optimal_result['min_cluster_size']}\n"
                )
                f.write(f"Best DBCV score: {optimal_result['dbcv_score']:.3f}\n")
                f.write(
                    f"Corresponding silhouette score: {optimal_result['silhouette']:.3f}\n"
                )
                f.write(f"Noise points: {optimal_result['n_noise']}\n\n")

            # Step 4: Full analysis at optimal clustering
            print(f"\n{'=' * 60}")
            print(f"ANALYZING OPTIMAL CLUSTERING: {optimal_k} clusters")
            print(f"Min Cluster Size: {optimal_result['min_cluster_size']}")
            print(f"DBCV Score: {optimal_result['dbcv_score']:.3f}")
            print(f"Silhouette Score: {optimal_result['silhouette']:.3f}")
            print(f"Noise Points: {optimal_result['n_noise']}")
            print(f"{'=' * 60}")

            # Analyze communities
            coherence_scores = self.analyze_communities(optimal_k, optimal_result)

            # Compute coherence scores
            self.compute_community_coherence(
                optimal_k, optimal_result, coherence_scores
            )

            # Compute silhouette scores for detailed validation
            self.compute_silhouette_analysis(optimal_k, optimal_result)

            # Visualization
            try:
                self.visualize_communities_umap(optimal_k, optimal_result)
            except Exception as e:
                print(f"Skipping UMAP visualization due to error: {e}")

            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print(f"Optimal clustering: {optimal_k} clusters")
            print(f"Min cluster size: {optimal_result['min_cluster_size']}")
            print(f"DBCV score: {optimal_result['dbcv_score']:.3f}")
            print(f"Silhouette score: {optimal_result['silhouette']:.3f}")
            print(f"Noise points: {optimal_result['n_noise']}")
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

            # Run analysis with 2-6 cluster optimization
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
