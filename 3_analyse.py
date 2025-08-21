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
import pandas as pd
import seaborn as sns
import umap
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


class SubregisterAnalyzer:
    def __init__(self, pickle_path, results_base_dir="subregister_results"):
        """Load and initialize the embedding data"""
        self.pickle_path = pickle_path

        # Create common results directory and subdirectory for this file
        input_filename = Path(pickle_path).stem  # removes extension
        self.results_base_dir = Path(results_base_dir)
        self.results_base_dir.mkdir(exist_ok=True)
        self.output_dir = self.results_base_dir / input_filename
        self.output_dir.mkdir(exist_ok=True)

        print(f"Loading data from {pickle_path}...")
        print(f"Output will be saved to: {self.output_dir}")

        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)

        # Extract embeddings and metadata
        self.embeddings = np.array([row["embed_last"] for row in self.data])
        self.texts = [row["text"] for row in self.data]
        self.preds = [row["preds"] for row in self.data]  # Keep full predictions list
        self.labels = [
            row["preds"][0] if row["preds"] else "unknown" for row in self.data
        ]

        print(f"Loaded {len(self.embeddings)} documents")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
        print(f"Register: {Counter(self.labels)}")

        # Normalize embeddings for cosine similarity
        self.embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    def reduce_dimensions(self, n_components=40):
        """Apply PCA for noise reduction and speedup"""
        # Adaptive n_components based on data size
        max_components = min(len(self.embeddings), self.embeddings.shape[1]) - 1
        n_components = min(n_components, max_components)

        if n_components < 2:
            print(f"Warning: Only {len(self.embeddings)} documents, skipping PCA")
            self.embeddings_pca_norm = self.embeddings_norm
            return self.embeddings_pca_norm

        print(f"Applying PCA to reduce to {n_components} dimensions...")
        self.pca = PCA(n_components=n_components, random_state=42)
        self.embeddings_pca = self.pca.fit_transform(self.embeddings_norm)

        # Normalize PCA embeddings
        self.embeddings_pca_norm = self.embeddings_pca / np.linalg.norm(
            self.embeddings_pca, axis=1, keepdims=True
        )

        total_variance = self.pca.explained_variance_ratio_.sum()
        first_10_variance = self.pca.explained_variance_ratio_[
            : min(10, n_components)
        ].sum()
        print(
            f"PCA total variance explained: {total_variance:.3f} (all {n_components} components)"
        )
        print(
            f"First {min(10, n_components)} components explain: {first_10_variance:.3f} of total variance"
        )
        return self.embeddings_pca_norm

    def build_knn_graph(self, k=None, use_pca=True):
        """Build k-nearest neighbor graph"""
        # Auto-compute k based on dataset size if not provided
        if k is None:
            k = int(np.sqrt(len(self.embeddings)))
            k = max(10, min(k, 100))  # Clamp between 10 and 100

        print(f"Building {k}-NN graph for {len(self.embeddings)} documents...")

        embeddings = self.embeddings_pca_norm if use_pca else self.embeddings_norm

        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

        # Build graph (excluding self-connections)
        self.graph = nx.Graph()
        self.similarities = {}

        for i in range(len(embeddings)):
            for j in range(1, k + 1):  # Skip self (index 0)
                neighbor = indices[i][j]
                similarity = 1 - distances[i][j]  # Convert distance to similarity
                self.graph.add_edge(i, neighbor, weight=similarity)
                self.similarities[(i, neighbor)] = similarity

        print(
            f"Graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )
        return self.graph

    def detect_communities_leiden(self, resolution=1.0):
        """Detect communities using Leiden algorithm"""
        print(f"Detecting communities with Leiden (resolution={resolution})...")

        # Convert NetworkX graph to igraph format (required for Leiden)
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
        print(f"Found {n_communities} communities")
        print(f"Modularity: {partition.modularity:.3f}")

        # Community sizes
        community_sizes = Counter(communities)
        print("Community sizes:", dict(sorted(community_sizes.items())))

        return communities

    def multi_resolution_analysis(self):
        """Run community detection at resolution=1.0 only"""
        print("Running community detection at resolution=1.0...")

        communities = self.detect_communities_leiden(resolution=1.0)
        self.community_results = {1.0: communities}

        return self.community_results

    def visualize_communities_umap(self):
        """Visualize communities using UMAP (all data, resolution=1.0)"""
        try:
            print(
                f"Creating UMAP visualization for all {len(self.embeddings)} documents..."
            )

            # Use all data
            embeddings_vis = self.embeddings_pca_norm
            communities_vis = self.community_results[1.0]

            # UMAP projection with memory-efficient settings
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
                cmap="tab20",
                alpha=0.7,
                s=8,
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

            plt.title(
                "UMAP Visualization of Communities (resolution=1.0)\nNumbers show Community IDs"
            )
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")

            # Save plot
            plot_path = self.output_dir / "umap_communities.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Create a legend mapping colors to community IDs
            self._create_community_legend(unique_communities)

            # Force garbage collection
            import gc

            gc.collect()

            print(f"UMAP plot saved to: {plot_path}")
            return embedding_2d, communities_vis

        except Exception as e:
            print(f"Error creating UMAP visualization: {e}")
            plt.close("all")
            return None, None

    def _create_community_legend(self, unique_communities):
        """Create a separate legend figure showing community colors"""
        try:
            fig, ax = plt.subplots(figsize=(8, max(6, len(unique_communities) * 0.4)))

            # Get colors from tab20 colormap
            cmap = plt.cm.tab20
            colors = [cmap(i / 20) for i in range(len(unique_communities))]

            # Create legend entries
            for i, community_id in enumerate(unique_communities):
                ax.scatter(
                    [], [], c=[colors[i]], s=100, label=f"Community {community_id}"
                )

            ax.legend(
                loc="center", fontsize=10, ncol=max(1, len(unique_communities) // 15)
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_title("Community Color Legend", fontsize=14, fontweight="bold")

            # Save legend
            legend_path = self.output_dir / "community_legend.png"
            plt.savefig(legend_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Community legend saved to: {legend_path}")

        except Exception as e:
            print(f"Error creating community legend: {e}")
            plt.close("all")

    def analyze_communities(self, top_n=10):
        """Analyze and sample documents from each community"""
        communities = self.community_results[1.0]

        # First compute coherence scores for all communities
        print("Computing coherence scores for community analysis...")
        coherence_scores = {}

        for community_id in set(communities):
            members = np.where(communities == community_id)[0]

            if len(members) < 2:
                coherence_scores[community_id] = 0.0
                continue

            # Compute average pairwise cosine similarity within community
            community_embeddings = self.embeddings_pca_norm[members]
            similarity_matrix = cosine_similarity(community_embeddings)

            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(similarity_matrix, k=1)
            coherence = np.mean(upper_triangle[upper_triangle > 0])
            coherence_scores[community_id] = coherence

        analysis_text = f"\n=== COMMUNITY ANALYSIS (resolution=1.0) ===\n"
        print(analysis_text)

        # Save analysis to file
        analysis_file = self.output_dir / "community_analysis.txt"

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

    def compute_community_coherence(self):
        """Compute intra-community coherence scores"""
        communities = self.community_results[1.0]
        coherence_scores = {}

        print(f"Computing community coherence scores...")

        coherence_text = "Community Coherence Scores:\n"

        for community_id in set(communities):
            members = np.where(communities == community_id)[0]

            if len(members) < 2:
                coherence_scores[community_id] = 0.0
                continue

            # Compute average pairwise cosine similarity within community
            community_embeddings = self.embeddings_pca_norm[members]
            similarity_matrix = cosine_similarity(community_embeddings)

            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(similarity_matrix, k=1)
            coherence = np.mean(upper_triangle[upper_triangle > 0])
            coherence_scores[community_id] = coherence

        # Print and save results
        print("\nCommunity Coherence Scores:")
        for community_id, score in sorted(coherence_scores.items()):
            line = f"Community {community_id}: {score:.3f}"
            print(line)
            coherence_text += line + "\n"

        # Save coherence scores
        coherence_file = self.output_dir / "coherence_scores.txt"
        with open(coherence_file, "w", encoding="utf-8") as f:
            f.write(coherence_text)
        print(f"Coherence scores saved to: {coherence_file}")

    def compute_silhouette_analysis(self):
        """Compute silhouette scores for community validation"""
        communities = self.community_results[1.0]

        print("Computing silhouette scores...")

        # Overall silhouette score
        try:
            # Use cosine distance for silhouette computation
            overall_silhouette = silhouette_score(
                self.embeddings_pca_norm, communities, metric="cosine"
            )

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
            print(f"\nOverall Silhouette Score: {overall_silhouette:.3f}")
            print("\nPer-Community Silhouette Scores:")

            silhouette_text = f"SILHOUETTE ANALYSIS\n"
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
            silhouette_file = self.output_dir / "silhouette_analysis.txt"
            with open(silhouette_file, "w", encoding="utf-8") as f:
                f.write(silhouette_text)
            print(f"\nSilhouette analysis saved to: {silhouette_file}")

            # Create silhouette plot
            self._plot_silhouette_analysis(
                communities, sample_silhouette_values, overall_silhouette
            )

            return overall_silhouette, community_silhouettes, sample_silhouette_values

        except Exception as e:
            print(f"Error computing silhouette scores: {e}")
            return None, {}, None

    def _plot_silhouette_analysis(
        self, communities, sample_silhouette_values, overall_silhouette
    ):
        """Create silhouette plot visualization"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            unique_communities = sorted(set(communities))
            n_communities = len(unique_communities)

            y_lower = 10

            # Color map for communities
            colors = plt.cm.tab20(np.linspace(0, 1, n_communities))

            for i, community_id in enumerate(unique_communities):
                # Get silhouette scores for this community
                community_mask = communities == community_id
                community_silhouette_values = sample_silhouette_values[community_mask]
                community_silhouette_values.sort()

                size_cluster = community_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster

                color = colors[i]
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    community_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax.text(-0.05, y_lower + 0.5 * size_cluster, str(community_id))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10

            ax.set_xlabel("Silhouette Coefficient Values")
            ax.set_ylabel("Community ID")
            ax.set_title(
                f"Silhouette Plot for Communities\nOverall Score: {overall_silhouette:.3f}"
            )

            # Vertical line for average silhouette score
            ax.axvline(
                x=overall_silhouette,
                color="red",
                linestyle="--",
                label=f"Overall Score: {overall_silhouette:.3f}",
            )
            ax.legend()

            # Save plot
            plot_path = self.output_dir / "silhouette_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Silhouette plot saved to: {plot_path}")

        except Exception as e:
            print(f"Error creating silhouette plot: {e}")
            plt.close("all")

    def hierarchical_clustering_comparison(self):
        """Compare with hierarchical clustering for validation"""
        print("Running hierarchical clustering comparison...")

        # Get number of communities found by Leiden for fair comparison
        leiden_communities = self.community_results[1.0]
        n_leiden_communities = len(set(leiden_communities))

        # Test hierarchical clustering with same number of clusters as Leiden
        n_clusters_range = [
            n_leiden_communities - 2,
            n_leiden_communities,
            n_leiden_communities + 2,
        ]
        n_clusters_range = [
            max(2, n) for n in n_clusters_range
        ]  # Ensure at least 2 clusters

        self.hierarchical_results = {}

        comparison_text = "HIERARCHICAL CLUSTERING COMPARISON\n"
        comparison_text += "==================================\n\n"
        comparison_text += f"Leiden found {n_leiden_communities} communities\n"
        comparison_text += (
            f"Testing hierarchical clustering with different cluster numbers\n\n"
        )

        print(f"Leiden found {n_leiden_communities} communities")
        print(f"Testing hierarchical clustering with: {n_clusters_range} clusters")

        for n_clusters in n_clusters_range:
            print(f"\nHierarchical clustering with {n_clusters} clusters:")

            # Run hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="cosine", linkage="average"
            )
            clusters = clustering.fit_predict(self.embeddings_pca_norm)
            self.hierarchical_results[n_clusters] = clusters

            # Compute silhouette score for this clustering
            try:
                h_silhouette = silhouette_score(
                    self.embeddings_pca_norm, clusters, metric="cosine"
                )
                print(f"  Silhouette score: {h_silhouette:.3f}")
                comparison_text += f"Hierarchical with {n_clusters} clusters:\n"
                comparison_text += f"  Silhouette score: {h_silhouette:.3f}\n"
            except Exception as e:
                print(f"  Could not compute silhouette: {e}")
                comparison_text += f"Hierarchical with {n_clusters} clusters:\n"
                comparison_text += f"  Silhouette computation failed\n"

            # Cluster sizes
            cluster_sizes = Counter(clusters)
            sizes_str = f"  Cluster sizes: {dict(sorted(cluster_sizes.items()))}"
            print(sizes_str)
            comparison_text += sizes_str + "\n\n"

        # Add comparison with Leiden results
        try:
            leiden_silhouette = silhouette_score(
                self.embeddings_pca_norm, leiden_communities, metric="cosine"
            )
            comparison_text += f"COMPARISON SUMMARY:\n"
            comparison_text += f"Leiden communities ({n_leiden_communities}): {leiden_silhouette:.3f}\n"

            # Find best hierarchical result
            best_h_score = -1
            best_h_clusters = None
            for n_clusters in n_clusters_range:
                try:
                    clusters = self.hierarchical_results[n_clusters]
                    score = silhouette_score(
                        self.embeddings_pca_norm, clusters, metric="cosine"
                    )
                    comparison_text += (
                        f"Hierarchical ({n_clusters} clusters): {score:.3f}\n"
                    )
                    if score > best_h_score:
                        best_h_score = score
                        best_h_clusters = n_clusters
                except:
                    continue

            if best_h_clusters:
                comparison_text += f"\nBest method: "
                if leiden_silhouette > best_h_score:
                    comparison_text += f"Leiden (score: {leiden_silhouette:.3f})\n"
                    print(
                        f"\n✓ Leiden communities perform better (silhouette: {leiden_silhouette:.3f} vs {best_h_score:.3f})"
                    )
                else:
                    comparison_text += f"Hierarchical with {best_h_clusters} clusters (score: {best_h_score:.3f})\n"
                    print(
                        f"\n✓ Hierarchical clustering performs better ({best_h_clusters} clusters, silhouette: {best_h_score:.3f} vs {leiden_silhouette:.3f})"
                    )

        except Exception as e:
            print(f"Could not compare silhouette scores: {e}")

        # Create comparison visualization
        self._plot_clustering_comparison()

        # Save comparison results
        comparison_file = self.output_dir / "hierarchical_comparison.txt"
        with open(comparison_file, "w", encoding="utf-8") as f:
            f.write(comparison_text)
        print(f"Hierarchical clustering comparison saved to: {comparison_file}")

    def _plot_clustering_comparison(self):
        """Create comparison plot of Leiden vs Hierarchical clustering"""
        try:
            leiden_communities = self.community_results[1.0]
            n_leiden = len(set(leiden_communities))

            # Use the hierarchical result with same number of clusters as Leiden
            if n_leiden in self.hierarchical_results:
                hierarchical_clusters = self.hierarchical_results[n_leiden]
            else:
                # Use closest available
                available_ns = list(self.hierarchical_results.keys())
                closest_n = min(available_ns, key=lambda x: abs(x - n_leiden))
                hierarchical_clusters = self.hierarchical_results[closest_n]

            # Create side-by-side comparison using UMAP projection
            # Reuse UMAP projection if available, otherwise create simple 2D projection
            if hasattr(self, "umap_projection"):
                embedding_2d = self.umap_projection
            else:
                # Simple PCA projection for comparison plot
                from sklearn.decomposition import PCA

                pca_2d = PCA(n_components=2, random_state=42)
                embedding_2d = pca_2d.fit_transform(self.embeddings_pca_norm)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

            # Leiden communities plot
            scatter1 = ax1.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=leiden_communities,
                cmap="tab20",
                alpha=0.7,
                s=8,
            )
            ax1.set_title(f"Leiden Communities ({n_leiden} communities)")
            ax1.set_xlabel("Dimension 1")
            ax1.set_ylabel("Dimension 2")

            # Hierarchical clustering plot
            scatter2 = ax2.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=hierarchical_clusters,
                cmap="tab20",
                alpha=0.7,
                s=8,
            )
            ax2.set_title(
                f"Hierarchical Clustering ({len(set(hierarchical_clusters))} clusters)"
            )
            ax2.set_xlabel("Dimension 1")
            ax2.set_ylabel("Dimension 2")

            plt.tight_layout()

            # Save comparison plot
            plot_path = self.output_dir / "clustering_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Clustering comparison plot saved to: {plot_path}")

        except Exception as e:
            print(f"Error creating clustering comparison plot: {e}")
            plt.close("all")

    def run_full_analysis(self):
        """Run the complete subregister discovery pipeline"""
        try:
            print("=" * 60)
            print("SUBREGISTER DISCOVERY ANALYSIS")
            print("=" * 60)

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
                f.write(f"Resolution: 1.0\n\n")

            # Step 1: Dimensionality reduction
            self.reduce_dimensions()

            # Step 2: Build k-NN graph (auto-compute k)
            self.build_knn_graph()

            # Step 3: Community detection at resolution=1.0
            self.multi_resolution_analysis()

            # Step 4: Analyze communities
            self.analyze_communities()

            # Step 5: Compute coherence scores
            self.compute_community_coherence()

            # Step 6: Compute silhouette scores
            self.compute_silhouette_analysis()

            # Step 6: Compute silhouette scores
            self.compute_silhouette_analysis()

            # Step 7: Visualization (with error handling)
            try:
                self.visualize_communities_umap()
            except Exception as e:
                print(f"Skipping UMAP visualization due to error: {e}")

            # Step 8: Compare with hierarchical clustering
            self.hierarchical_clustering_comparison()

            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
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
    import glob

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

    for i, pkl_file in enumerate(pkl_files, 1):
        try:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING FILE {i}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
            print(f"{'=' * 60}")

            # Initialize analyzer for this file (with common results dir)
            analyzer = SubregisterAnalyzer(pkl_file, results_base_dir=results_dir)

            # Run analysis
            analyzer.run_full_analysis()

            successful_analyses += 1
            print(f"✓ Successfully completed analysis for {os.path.basename(pkl_file)}")

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

    if failed_analyses:
        print(f"\nFailed analyses ({len(failed_analyses)}):")
        for file, error in failed_analyses:
            print(f"  ✗ {os.path.basename(file)}: {error}")
    else:
        print("All files processed successfully! ✓")

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
