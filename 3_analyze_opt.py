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

        # Create common results directory and subdirectory for this file
        input_filename = Path(pickle_path).stem
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

        # Normalize embeddings for cosine similarity
        self.embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    def reduce_dimensions(self, n_components=50):
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

    def build_knn_graph(self, k=None):
        """Build k-nearest neighbor graph"""
        # Auto-compute k based on dataset size if not provided
        if k is None:
            k = int(np.sqrt(len(self.embeddings)))
            k = max(10, min(k, 100))  # Clamp between 10 and 100

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
        print(f"Detecting communities with Leiden (resolution={resolution})...")

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
        print(f"Found {n_communities} communities")
        print(f"Modularity: {partition.modularity:.3f}")

        # Community sizes
        community_sizes = Counter(communities)
        print("Community sizes:", dict(sorted(community_sizes.items())))

        return communities

    def find_optimal_resolution(
        self, resolutions=[0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0]
    ):
        """Find optimal resolution based on silhouette scores"""
        print("=" * 50)
        print("FINDING OPTIMAL RESOLUTION")
        print("=" * 50)

        resolution_scores = {}

        for resolution in resolutions:
            print(f"\\nTesting resolution {resolution}...")
            try:
                # Detect communities at this resolution
                communities = self.detect_communities_leiden(resolution=resolution)

                # Skip if too few or too many communities
                n_communities = len(set(communities))
                if n_communities < 2:
                    print(f"  Skipping: Only {n_communities} community found")
                    continue
                if (
                    n_communities > len(self.embeddings) // 5
                ):  # Avoid too many tiny communities
                    print(f"  Skipping: Too many communities ({n_communities})")
                    continue

                # Compute silhouette score
                silhouette = silhouette_score(
                    self.embeddings_pca_norm, communities, metric="cosine"
                )

                resolution_scores[resolution] = {
                    "silhouette": silhouette,
                    "n_communities": n_communities,
                    "communities": communities,
                }

                print(
                    f"  Resolution {resolution}: {silhouette:.3f} silhouette, {n_communities} communities"
                )

            except Exception as e:
                print(f"  Error at resolution {resolution}: {e}")
                continue

        if not resolution_scores:
            raise ValueError(
                "No valid resolution found. Dataset may be too difficult to cluster."
            )

        # Find optimal resolution
        optimal_resolution = max(
            resolution_scores.keys(), key=lambda r: resolution_scores[r]["silhouette"]
        )
        optimal_score = resolution_scores[optimal_resolution]["silhouette"]
        optimal_communities = resolution_scores[optimal_resolution]["communities"]

        print(f"\\n" + "=" * 50)
        print(f"OPTIMAL RESOLUTION FOUND: {optimal_resolution}")
        print(f"Silhouette Score: {optimal_score:.3f}")
        print(
            f"Number of Communities: {resolution_scores[optimal_resolution]['n_communities']}"
        )
        print("=" * 50)

        # Save optimization results
        optimization_file = self.output_dir / "resolution_optimization.txt"
        with open(optimization_file, "w", encoding="utf-8") as f:
            f.write("RESOLUTION OPTIMIZATION RESULTS\\n")
            f.write("=" * 40 + "\\n\\n")
            f.write(f"Optimal Resolution: {optimal_resolution}\\n")
            f.write(f"Optimal Silhouette Score: {optimal_score:.3f}\\n")
            f.write(
                f"Number of Communities: {resolution_scores[optimal_resolution]['n_communities']}\\n\\n"
            )
            f.write("All Tested Resolutions:\\n")
            for res in sorted(resolution_scores.keys()):
                score = resolution_scores[res]["silhouette"]
                n_comm = resolution_scores[res]["n_communities"]
                marker = " <- OPTIMAL" if res == optimal_resolution else ""
                f.write(
                    f"  {res}: {score:.3f} silhouette, {n_comm} communities{marker}\\n"
                )

        print(f"Resolution optimization results saved to: {optimization_file}")

        return optimal_resolution, optimal_communities, resolution_scores

    def analyze_communities(self, resolution, top_n=10):
        """Analyze and sample documents from each community"""
        communities = self.community_results[resolution]

        # Compute coherence scores for all communities
        print(
            f"Computing coherence scores for community analysis (resolution={resolution})..."
        )
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

        analysis_text = f"\\n=== COMMUNITY ANALYSIS (resolution={resolution}) ===\\n"
        print(analysis_text)

        # Save analysis to file
        analysis_file = self.output_dir / f"community_analysis_res_{resolution}.txt"

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis_text)

            for community_id in sorted(set(communities)):
                members = np.where(communities == community_id)[0]
                coherence_score = coherence_scores.get(community_id, 0.0)

                section = f"\\n--- Community {community_id} ({len(members)} documents, coherence: {coherence_score:.3f}) ---\\n"
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
                        self.texts[idx].replace("\\n", "\\\\n").replace("\\r", "\\\\r")
                    )
                    line = f"{i + 1}. [{idx}] {doc_preds} {full_text}\\n"
                    # For console output, show predictions and truncated text
                    truncated_text = (
                        self.texts[idx][:100] + "..."
                        if len(self.texts[idx]) > 100
                        else self.texts[idx]
                    )
                    print(f"{i + 1}. [{idx}] {doc_preds} {truncated_text}")
                    f.write(line)

                if len(members) > top_n:
                    remaining = f"... and {len(members) - top_n} more documents\\n"
                    print(remaining.strip())
                    f.write(remaining)

        print(f"Community analysis saved to: {analysis_file}")

    def compute_community_coherence(self, resolution):
        """Compute intra-community coherence scores"""
        communities = self.community_results[resolution]
        coherence_scores = {}

        print(f"Computing community coherence scores (resolution={resolution})...")

        coherence_text = f"Community Coherence Scores (resolution={resolution}):\\n"

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
        print(f"\\nCommunity Coherence Scores (resolution={resolution}):")
        for community_id, score in sorted(coherence_scores.items()):
            line = f"Community {community_id}: {score:.3f}"
            print(line)
            coherence_text += line + "\\n"

        # Save coherence scores
        coherence_file = self.output_dir / f"coherence_scores_res_{resolution}.txt"
        with open(coherence_file, "w", encoding="utf-8") as f:
            f.write(coherence_text)
        print(f"Coherence scores saved to: {coherence_file}")

        return coherence_scores

    def compute_silhouette_analysis(self, resolution):
        """Compute silhouette scores for community validation"""
        communities = self.community_results[resolution]

        print(f"Computing silhouette scores (resolution={resolution})...")

        try:
            # Overall silhouette score
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
            print(
                f"\\nOverall Silhouette Score (resolution={resolution}): {overall_silhouette:.3f}"
            )
            print(f"\\nPer-Community Silhouette Scores (resolution={resolution}):")

            silhouette_text = f"SILHOUETTE ANALYSIS (resolution={resolution})\\n"
            silhouette_text += f"==================\\n\\n"
            silhouette_text += f"Overall Silhouette Score: {overall_silhouette:.3f}\\n"
            silhouette_text += f"Interpretation:\\n"
            silhouette_text += f"  > 0.7: Strong cluster structure\\n"
            silhouette_text += f"  > 0.5: Reasonable cluster structure\\n"
            silhouette_text += f"  > 0.3: Weak but acceptable structure\\n"
            silhouette_text += f"  < 0.3: Poor cluster structure\\n"
            silhouette_text += f"  < 0.0: Documents may be in wrong clusters\\n\\n"
            silhouette_text += f"Per-Community Silhouette Scores:\\n"

            for community_id, score in sorted(community_silhouettes.items()):
                line = f"Community {community_id}: {score:.3f}"
                print(line)
                silhouette_text += line + "\\n"

            # Save silhouette scores
            silhouette_file = (
                self.output_dir / f"silhouette_analysis_res_{resolution}.txt"
            )
            with open(silhouette_file, "w", encoding="utf-8") as f:
                f.write(silhouette_text)
            print(f"\\nSilhouette analysis saved to: {silhouette_file}")

            return overall_silhouette, community_silhouettes

        except Exception as e:
            print(f"Error computing silhouette scores: {e}")
            return None, {}

    def visualize_communities_umap(self, resolution):
        """Visualize communities using UMAP"""
        try:
            print(
                f"Creating UMAP visualization for all {len(self.embeddings)} documents (resolution={resolution})..."
            )

            # Use all data
            embeddings_vis = self.embeddings_pca_norm
            communities_vis = self.community_results[resolution]

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
                f"UMAP Visualization of Communities (resolution={resolution})\\nNumbers show Community IDs"
            )
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")

            # Save plot
            plot_path = self.output_dir / f"umap_communities_res_{resolution}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            # Create a legend mapping colors to community IDs
            self._create_community_legend(unique_communities, resolution)

            # Force garbage collection
            import gc

            gc.collect()

            print(f"UMAP plot saved to: {plot_path}")
            return embedding_2d, communities_vis

        except Exception as e:
            print(f"Error creating UMAP visualization: {e}")
            plt.close("all")
            return None, None

    def _create_community_legend(self, unique_communities, resolution):
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
            ax.set_title(
                f"Community Color Legend (resolution={resolution})",
                fontsize=14,
                fontweight="bold",
            )

            # Save legend
            legend_path = self.output_dir / f"community_legend_res_{resolution}.png"
            plt.savefig(legend_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Community legend saved to: {legend_path}")

        except Exception as e:
            print(f"Error creating community legend: {e}")
            plt.close("all")

    def run_full_analysis(self):
        """Run the complete subregister discovery pipeline with automatic resolution optimization"""
        try:
            print("=" * 60)
            print("SUBREGISTER DISCOVERY ANALYSIS")
            print("=" * 60)

            # Step 1: Dimensionality reduction
            self.reduce_dimensions()

            # Step 2: Build k-NN graph
            self.build_knn_graph()

            # Step 3: Find optimal resolution
            optimal_resolution, optimal_communities, all_scores = (
                self.find_optimal_resolution()
            )

            # Store the optimal result
            self.community_results = {optimal_resolution: optimal_communities}

            # Save summary info
            summary_file = self.output_dir / "analysis_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("SUBREGISTER DISCOVERY ANALYSIS SUMMARY\\n")
                f.write("=" * 60 + "\\n\\n")
                f.write(f"Input file: {self.pickle_path}\\n")
                f.write(f"Number of documents: {len(self.embeddings)}\\n")
                f.write(f"Embedding dimension: {self.embeddings.shape[1]}\\n")
                f.write(f"Register distribution: {dict(Counter(self.labels))}\\n")
                f.write(f"k-NN parameter: auto-computed\\n")
                f.write(f"Optimal resolution: {optimal_resolution}\\n")
                f.write(
                    f"Optimal silhouette score: {all_scores[optimal_resolution]['silhouette']:.3f}\\n"
                )
                f.write(
                    f"Number of communities: {all_scores[optimal_resolution]['n_communities']}\\n\\n"
                )

            # Step 4: Full analysis at optimal resolution
            print(f"\\n{'=' * 50}")
            print(f"ANALYZING OPTIMAL RESOLUTION: {optimal_resolution}")
            print(f"{'=' * 50}")

            # Analyze communities
            self.analyze_communities(resolution=optimal_resolution)

            # Compute coherence scores
            self.compute_community_coherence(resolution=optimal_resolution)

            # Compute silhouette scores (already computed, but save detailed analysis)
            self.compute_silhouette_analysis(resolution=optimal_resolution)

            # Visualization
            try:
                self.visualize_communities_umap(resolution=optimal_resolution)
            except Exception as e:
                print(f"Skipping UMAP visualization due to error: {e}")

            print("\\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print(f"Optimal resolution: {optimal_resolution}")
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

    print(f"\\nAll results will be saved to: {results_dir}/")

    print("\\n" + "=" * 80)
    print("PROCESSING ALL PKL FILES")
    print("=" * 80)

    successful_analyses = 0
    failed_analyses = []
    skipped_files = []

    for i, pkl_file in enumerate(pkl_files, 1):
        try:
            print(f"\\n{'=' * 60}")
            print(f"PROCESSING FILE {i}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
            print(f"{'=' * 60}")

            # Initialize analyzer for this file
            analyzer = SubregisterAnalyzer(pkl_file, results_base_dir=results_dir)

            # Run analysis with automatic resolution optimization
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
    print("\\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {successful_analyses}/{len(pkl_files)} files")

    if skipped_files:
        print(f"\\nSkipped files ({len(skipped_files)}):")
        for file, reason in skipped_files:
            print(f"  ⚠ {os.path.basename(file)}: {reason}")

    if failed_analyses:
        print(f"\\nFailed analyses ({len(failed_analyses)}):")
        for file, error in failed_analyses:
            print(f"  ✗ {os.path.basename(file)}: {error}")

    if successful_analyses + len(skipped_files) == len(pkl_files):
        print("All eligible files processed successfully! ✓")

    print(f"\\nResults organized in: {results_dir}/")

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

    print("\\nAnalysis completed with cleanup.")
