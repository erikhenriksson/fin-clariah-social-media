import glob
import os
import pickle
import warnings
from collections import Counter
from pathlib import Path

import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

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

    def build_knn_graph(self, k=None):
        """Build k-nearest neighbor graph"""
        # Auto-compute k based on dataset size if not provided
        if k is None:
            # Use a more conservative formula for large datasets
            if len(self.embeddings) < 5000:
                k = int(np.sqrt(len(self.embeddings)))
            else:
                # For large datasets, use log-based scaling
                k = int(10 * np.log10(len(self.embeddings)))

            k = max(15, min(k, 50))  # Clamp between 15 and 50

        print(f"Building {k}-NN graph for {len(self.embeddings)} documents...")

        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
        nbrs.fit(self.embeddings_pca_norm)
        distances, indices = nbrs.kneighbors(self.embeddings_pca_norm)

        # Build graph (excluding self-connections)
        self.graph = nx.Graph()

        for i in range(len(self.embeddings)):
            for j in range(1, k + 1):  # Skip self (index 0)
                neighbor = indices[i][j]
                similarity = 1 - distances[i][j]  # Convert distance to similarity
                self.graph.add_edge(i, neighbor, weight=similarity)

        print(
            f"Graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )
        return self.graph

    def detect_communities_leiden(self, resolution=1.0):
        """Detect communities using Leiden algorithm"""
        # Convert NetworkX graph to igraph format
        edge_list = list(self.graph.edges(data=True))

        # Create igraph from edge list
        g = ig.Graph()
        g.add_vertices(self.graph.number_of_nodes())

        # Add edges with weights
        edges = [(u, v) for u, v, _ in edge_list]
        weights = [d["weight"] for u, v, d in edge_list]

        g.add_edges(edges)
        g.es["weight"] = weights

        # Run Leiden algorithm
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=42,
        )

        # Convert partition to array format
        communities = np.array(partition.membership)

        n_communities = len(set(communities))
        return communities, n_communities

    def find_resolutions_for_target_clusters(self, target_clusters=[1, 2, 3, 4, 5, 6]):
        """Find Leiden resolutions that produce exactly the target number of clusters"""
        print("=" * 60)
        print("FINDING RESOLUTIONS FOR TARGET CLUSTER COUNTS")
        print("=" * 60)

        resolution_mapping = {}

        for target_k in target_clusters:
            print(f"\nFinding resolution for {target_k} clusters...")

            # Binary search for resolution that gives target_k clusters
            # Note: In Leiden, LOWER resolution = FEWER clusters
            resolution_low = 0.01  # Very low resolution -> few clusters
            resolution_high = 2.0  # Higher resolution -> more clusters
            best_resolution = None
            best_diff = float("inf")
            max_iterations = 15

            for iteration in range(max_iterations):
                resolution_mid = (resolution_low + resolution_high) / 2

                communities, n_communities = self.detect_communities_leiden(
                    resolution=resolution_mid
                )

                print(
                    f"  Iteration {iteration + 1}: resolution={resolution_mid:.3f} -> {n_communities} clusters"
                )

                # Track the closest we've gotten to target
                diff = abs(n_communities - target_k)
                if diff < best_diff:
                    best_diff = diff
                    best_resolution = resolution_mid

                if n_communities == target_k:
                    # Found exact match
                    break
                elif n_communities > target_k:
                    # Too many clusters, need LOWER resolution
                    resolution_high = resolution_mid
                elif n_communities < target_k:
                    # Too few clusters, need HIGHER resolution
                    resolution_low = resolution_mid

                # Stop if range is too narrow
                if abs(resolution_high - resolution_low) < 0.001:
                    break

            if best_resolution is not None:
                # Get final communities with best resolution
                communities, n_communities = self.detect_communities_leiden(
                    resolution=best_resolution
                )

                # Only accept if we got reasonably close to target
                if abs(n_communities - target_k) <= 1:  # Allow ±1 cluster difference
                    # Compute silhouette score for this clustering
                    silhouette = silhouette_score(
                        self.embeddings_pca_norm, communities, metric="cosine"
                    )

                    resolution_mapping[target_k] = {
                        "resolution": best_resolution,
                        "communities": communities,
                        "n_communities": n_communities,
                        "silhouette": silhouette,
                    }

                    print(
                        f"  ✓ Found: resolution={best_resolution:.3f} gives {n_communities} clusters (silhouette={silhouette:.3f})"
                    )
                else:
                    print(
                        f"  ✗ Could not get close enough to {target_k} clusters (got {n_communities})"
                    )
            else:
                print(f"  ✗ Could not find resolution for {target_k} clusters")

        return resolution_mapping

    def select_best_clustering(self, resolution_mapping):
        """Select the best clustering based on silhouette scores"""
        print("\n" + "=" * 60)
        print("SELECTING BEST CLUSTERING")
        print("=" * 60)

        print("\nCluster Count | Resolution | Silhouette Score")
        print("-" * 45)

        best_k = None
        best_silhouette = -1

        for k in sorted(resolution_mapping.keys()):
            result = resolution_mapping[k]
            resolution = result["resolution"]
            silhouette = result["silhouette"]

            marker = ""
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
                marker = " <- BEST"

            print(
                f"     {k:2d}       |   {resolution:.3f}    |     {silhouette:.3f}    {marker}"
            )

        optimal_result = resolution_mapping[best_k]

        print(f"\n✓ OPTIMAL CLUSTERING: {best_k} clusters")
        print(f"  Resolution: {optimal_result['resolution']:.3f}")
        print(f"  Silhouette Score: {optimal_result['silhouette']:.3f}")

        # Save optimization results
        optimization_file = self.output_dir / "resolution_optimization.txt"
        with open(optimization_file, "w", encoding="utf-8") as f:
            f.write("RESOLUTION OPTIMIZATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"TARGET CLUSTER RANGE: 2-6 clusters\n")
            f.write(f"OPTIMIZATION CRITERION: Silhouette Score\n")
            f.write(f"Optimal Number of Clusters: {best_k}\n")
            f.write(f"Optimal Resolution: {optimal_result['resolution']:.3f}\n")
            f.write(f"Best Silhouette Score: {optimal_result['silhouette']:.3f}\n\n")
            f.write("All Tested Cluster Counts:\n")
            f.write("Clusters | Resolution | Silhouette\n")
            f.write("-" * 35 + "\n")
            for k in sorted(resolution_mapping.keys()):
                result = resolution_mapping[k]
                marker = " <- OPTIMAL" if k == best_k else ""
                f.write(
                    f"   {k:2d}    |   {result['resolution']:.3f}    |   {result['silhouette']:.3f}{marker}\n"
                )

        print(f"\nOptimization results saved to: {optimization_file}")

        return best_k, optimal_result

    def compute_coherence_for_communities(self, communities):
        """Compute coherence scores for a set of communities (memory-efficient)"""
        coherence_scores = {}

        for community_id in set(communities):
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
        resolution = optimal_result["resolution"]

        # Compute coherence scores for all communities
        print(f"Computing coherence scores for {k} communities...")
        coherence_scores = self.compute_coherence_for_communities(communities)

        analysis_text = f"\n=== COMMUNITY ANALYSIS ({k} clusters, resolution={resolution:.3f}) ===\n"
        print(analysis_text)

        # Save analysis to file
        analysis_file = self.output_dir / f"community_analysis_{k}_clusters.txt"

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)

            for community_id in sorted(set(communities)):
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
        resolution = optimal_result["resolution"]

        coherence_text = (
            f"Community Coherence Scores ({k} clusters, resolution={resolution:.3f}):\n"
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
        resolution = optimal_result["resolution"]

        print(f"Computing detailed silhouette analysis for {k} clusters...")

        try:
            # Overall silhouette score (already computed)
            overall_silhouette = optimal_result["silhouette"]

            # Individual silhouette scores for each document
            sample_silhouette_values = silhouette_samples(
                self.embeddings_pca_norm, communities, metric="cosine"
            )

            # Compute average silhouette score per community
            community_silhouettes = {}
            for community_id in sorted(set(communities)):
                mask = communities == community_id
                if np.sum(mask) > 1:  # Need at least 2 documents
                    community_avg = np.mean(sample_silhouette_values[mask])
                    community_silhouettes[community_id] = community_avg
                else:
                    community_silhouettes[community_id] = 0.0

            # Print results
            print(
                f"\nOverall Silhouette Score ({k} clusters): {overall_silhouette:.3f}"
            )
            print(f"\nPer-Community Silhouette Scores:")

            silhouette_text = (
                f"SILHOUETTE ANALYSIS ({k} clusters, resolution={resolution:.3f})\n"
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
            resolution = optimal_result["resolution"]

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
            scatter = plt.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=communities_vis,
                cmap="tab10",
                alpha=0.7,
                s=6,
            )
            plt.colorbar(scatter, label="Community ID")

            # Add community labels at centroids
            unique_communities = sorted(set(communities_vis))
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
                f"UMAP Visualization: {k} Communities (resolution={resolution:.3f})\nNumbers show Community IDs"
            )
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")

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
        """Run the complete subregister discovery pipeline with fixed cluster range"""
        try:
            print("=" * 60)
            print("SUBREGISTER DISCOVERY ANALYSIS")
            print("Target cluster range: 2-6 clusters")
            print("=" * 60)

            # Step 1: Dimensionality reduction
            self.reduce_dimensions()

            # Step 2: Build k-NN graph
            self.build_knn_graph()

            # Step 3: Find resolutions for target cluster counts
            resolution_mapping = self.find_resolutions_for_target_clusters()

            # Step 4: Select best clustering based on silhouette scores
            optimal_k, optimal_result = self.select_best_clustering(resolution_mapping)

            # Save summary info
            summary_file = self.output_dir / "analysis_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("SUBREGISTER DISCOVERY ANALYSIS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Input file: {self.pickle_path}\n")
                f.write(f"Number of documents: {len(self.embeddings)}\n")
                f.write(f"Embedding dimension: {self.embeddings.shape[1]}\n")
                f.write(f"Register distribution: {dict(Counter(self.labels))}\n")
                f.write(f"k-NN parameter: auto-computed\n")
                f.write(f"Target cluster range: 2-6 clusters\n")
                f.write(f"Optimization criterion: Silhouette Score\n")
                f.write(f"Optimal number of clusters: {optimal_k}\n")
                f.write(f"Optimal resolution: {optimal_result['resolution']:.3f}\n")
                f.write(
                    f"Best silhouette score: {optimal_result['silhouette']:.3f}\n\n"
                )

            # Step 5: Full analysis at optimal clustering
            print(f"\n{'=' * 60}")
            print(f"ANALYZING OPTIMAL CLUSTERING: {optimal_k} clusters")
            print(f"Resolution: {optimal_result['resolution']:.3f}")
            print(f"Silhouette Score: {optimal_result['silhouette']:.3f}")
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
            print(f"Resolution: {optimal_result['resolution']:.3f}")
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
