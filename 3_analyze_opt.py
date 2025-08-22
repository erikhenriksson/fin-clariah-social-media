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
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
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

    def find_optimal_k(self, k_range=[2, 3, 4, 5, 6]):
        """Find optimal number of clusters using multiple criteria"""
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

                # Compute metrics
                silhouette = silhouette_score(
                    self.embeddings_pca_norm, communities, metric="cosine"
                )
                ch_score = calinski_harabasz_score(
                    self.embeddings_pca_norm, communities
                )
                db_score = davies_bouldin_score(self.embeddings_pca_norm, communities)
                inertia = kmeans.inertia_

                k_scores[k] = {
                    "silhouette": silhouette,
                    "calinski_harabasz": ch_score,
                    "davies_bouldin": db_score,
                    "inertia": inertia,
                    "communities": communities,
                    "kmeans_model": kmeans,
                }

                print(
                    f"  k={k}: Silhouette = {silhouette:.3f}, CH = {ch_score:.0f}, DB = {db_score:.3f}"
                )

            except Exception as e:
                print(f"  Error at k={k}: {e}")
                continue

        if not k_scores:
            raise ValueError(
                "No valid k found. Dataset may be too difficult to cluster."
            )

        # Use Calinski-Harabasz for optimization (best for finding true k)
        optimal_k = max(k_scores.keys(), key=lambda k: k_scores[k]["calinski_harabasz"])
        optimal_communities = k_scores[optimal_k]["communities"]
        optimal_model = k_scores[optimal_k]["kmeans_model"]

        print(f"\nOPTIMAL NUMBER OF CLUSTERS: {optimal_k}")
        print(
            f"Chosen based on Calinski-Harabasz score: {k_scores[optimal_k]['calinski_harabasz']:.0f}"
        )

        return optimal_k, optimal_communities, optimal_model, k_scores

    def compute_coherence_for_communities(self, communities):
        """Compute coherence scores for communities (memory-efficient)"""
        coherence_scores = {}

        for community_id in sorted(set(communities)):
            members = np.where(communities == community_id)[0]

            if len(members) < 2:
                coherence_scores[community_id] = 0.0
                continue

            # Memory-efficient coherence computation
            if len(members) > 1000:
                sample_size = min(500, len(members))
                sample_members = np.random.choice(members, sample_size, replace=False)
                community_embeddings = self.embeddings_pca_norm[sample_members]
            else:
                community_embeddings = self.embeddings_pca_norm[members]

            similarity_matrix = cosine_similarity(community_embeddings)
            upper_triangle = np.triu(similarity_matrix, k=1)
            coherence = np.mean(upper_triangle[upper_triangle > 0])
            coherence_scores[community_id] = coherence

            # Clear memory
            del similarity_matrix, upper_triangle, community_embeddings
            import gc

            gc.collect()

        return coherence_scores

    def analyze_communities(self, k, top_n=10):
        """Analyze and sample documents from each cluster"""
        communities = self.community_results[k]

        print(f"Analyzing clusters (k={k})...")
        coherence_scores = self.compute_coherence_for_communities(communities)

        analysis_file = self.output_dir / f"cluster_analysis_k_{k}.txt"

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(f"=== CLUSTER ANALYSIS (k={k}) ===\n\n")

            for cluster_id in sorted(set(communities)):
                members = np.where(communities == cluster_id)[0]
                coherence_score = coherence_scores.get(cluster_id, 0.0)

                f.write(
                    f"--- Cluster {cluster_id} ({len(members)} documents, coherence: {coherence_score:.3f}) ---\n"
                )

                if len(members) > 0:
                    sample_indices = np.random.choice(
                        members, min(top_n, len(members)), replace=False
                    )

                    for i, idx in enumerate(sample_indices):
                        doc_preds = self.preds[idx]
                        full_text = (
                            self.texts[idx].replace("\n", "\\n").replace("\r", "\\r")
                        )
                        f.write(f"{i + 1}. [{idx}] {doc_preds} {full_text}\n")

                    if len(members) > top_n:
                        f.write(f"... and {len(members) - top_n} more documents\n")

                f.write("\n")

        print(f"Cluster analysis saved to: {analysis_file}")

    def visualize_clusters_umap(self, k):
        """Visualize clusters using UMAP"""
        try:
            print(f"Creating UMAP visualization (k={k})...")

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

            # Plot
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
            for cluster_id in sorted(set(communities_vis)):
                mask = communities_vis == cluster_id
                if np.sum(mask) > 0:
                    centroid_x = np.mean(embedding_2d[mask, 0])
                    centroid_y = np.mean(embedding_2d[mask, 1])

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

            plt.title(f"UMAP Visualization of {k} Clusters")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")

            plot_path = self.output_dir / f"umap_clusters_k_{k}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"UMAP plot saved to: {plot_path}")

        except Exception as e:
            print(f"Error creating UMAP visualization: {e}")
            plt.close("all")

    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            print("=" * 60)
            print("SUBREGISTER DISCOVERY ANALYSIS")
            print("=" * 60)

            # Step 1: Dimensionality reduction
            self.reduce_dimensions()

            # Step 2: Find optimal k
            optimal_k, optimal_communities, optimal_model, all_scores = (
                self.find_optimal_k()
            )

            # Store results
            self.community_results = {optimal_k: optimal_communities}

            # Step 3: Analysis and visualization
            self.analyze_communities(k=optimal_k)
            self.visualize_clusters_umap(k=optimal_k)

            print(f"\nAnalysis complete! Results saved to: {self.output_dir}")

        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback

            traceback.print_exc()
        finally:
            plt.close("all")
            import gc

            gc.collect()


class MultiLanguageAnalyzer:
    def __init__(self, pickle_paths, results_base_dir="subregister_results"):
        """Combine multiple language files for joint analysis"""
        self.pickle_paths = pickle_paths

        # Extract language codes and register info
        self.languages = []
        self.register_info = None

        for path in pickle_paths:
            filename = Path(path).stem
            lang_code = filename.split("_embeds_")[0]
            self.languages.append(lang_code)

            if self.register_info is None:
                self.register_info = filename.split("_embeds_")[1]

        # Create combined filename
        combined_name = (
            "_".join(sorted(self.languages)) + "_embeds_" + self.register_info
        )

        print(f"Combining languages: {', '.join(self.languages)}")
        print(f"Register: {self.register_info}")

        # Load and combine data
        all_embeddings = []
        all_texts = []
        all_preds = []
        all_labels = []
        self.language_labels = []

        for i, pickle_path in enumerate(pickle_paths):
            print(f"Loading {self.languages[i]} data...")

            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            embeddings = np.array([row["embed_last"] for row in data])
            texts = [row["text"] for row in data]
            preds = [row["preds"] for row in data]
            labels = [row["preds"][0] if row["preds"] else "unknown" for row in data]

            print(f"  Loaded {len(embeddings)} {self.languages[i]} documents")

            if len(embeddings) < 500:
                print(
                    f"  Warning: {self.languages[i]} has only {len(embeddings)} documents"
                )

            all_embeddings.append(embeddings)
            all_texts.extend(texts)
            all_preds.extend(preds)
            all_labels.extend(labels)

            # Track language for each document
            self.language_labels.extend([self.languages[i]] * len(embeddings))

        # Combine embeddings
        self.embeddings = np.vstack(all_embeddings)
        self.texts = all_texts
        self.preds = all_preds
        self.labels = all_labels

        print(f"Total combined documents: {len(self.embeddings)}")
        print(f"Language distribution: {Counter(self.language_labels)}")

        # Check minimum size
        if len(self.embeddings) < 1000:
            raise ValueError(
                f"Combined dataset too small ({len(self.embeddings)} documents)."
            )

        # Create output directory
        self.results_base_dir = Path(results_base_dir)
        self.results_base_dir.mkdir(exist_ok=True)
        self.output_dir = self.results_base_dir / combined_name
        self.output_dir.mkdir(exist_ok=True)

        print(f"Output will be saved to: {self.output_dir}")

        # Normalize embeddings
        self.embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

    def reduce_dimensions(self, n_components=40):
        """Apply PCA for noise reduction"""
        print(f"Applying PCA to reduce to {n_components} dimensions...")
        self.pca = PCA(n_components=n_components, random_state=42)
        self.embeddings_pca = self.pca.fit_transform(self.embeddings_norm)
        self.embeddings_pca_norm = self.embeddings_pca / np.linalg.norm(
            self.embeddings_pca, axis=1, keepdims=True
        )

        total_variance = self.pca.explained_variance_ratio_.sum()
        print(f"PCA total variance explained: {total_variance:.3f}")

        return self.embeddings_pca_norm

    def find_optimal_k(self, k_range=[2, 3, 4, 5, 6, 7, 8]):
        """Find optimal k using Calinski-Harabasz"""
        print("Finding optimal number of clusters...")

        k_scores = {}

        for k in k_range:
            try:
                kmeans = KMeans(
                    n_clusters=k, init="k-means++", n_init=10, random_state=42
                )
                communities = kmeans.fit_predict(self.embeddings_pca_norm)

                ch_score = calinski_harabasz_score(
                    self.embeddings_pca_norm, communities
                )

                k_scores[k] = {
                    "calinski_harabasz": ch_score,
                    "communities": communities,
                    "kmeans_model": kmeans,
                }

                print(f"  k={k}: CH = {ch_score:.0f}")

            except Exception as e:
                print(f"  Error at k={k}: {e}")
                continue

        # Choose best k
        optimal_k = max(k_scores.keys(), key=lambda k: k_scores[k]["calinski_harabasz"])
        optimal_communities = k_scores[optimal_k]["communities"]
        optimal_model = k_scores[optimal_k]["kmeans_model"]

        print(f"Optimal k: {optimal_k}")

        return optimal_k, optimal_communities, optimal_model, k_scores

    def analyze_multilingual_clusters(self, k, top_n=10):
        """Analyze clusters with language information"""
        communities = self.community_results[k]
        languages = np.array(self.language_labels)

        analysis_file = self.output_dir / f"multilingual_analysis_k_{k}.txt"

        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(f"=== MULTILINGUAL CLUSTER ANALYSIS (k={k}) ===\n")
            f.write(f"Languages: {', '.join(self.languages)}\n\n")

            for cluster_id in sorted(set(communities)):
                members = np.where(communities == cluster_id)[0]
                cluster_languages = languages[members]
                lang_distribution = Counter(cluster_languages)

                f.write(f"--- Cluster {cluster_id} ({len(members)} documents) ---\n")
                f.write(f"Language distribution: ")
                f.write(
                    ", ".join(
                        [
                            f"{lang}: {count} ({count / len(members) * 100:.1f}%)"
                            for lang, count in sorted(lang_distribution.items())
                        ]
                    )
                )
                f.write("\n\n")

                # Sample from each language
                samples_per_lang = max(1, top_n // len(lang_distribution))

                for lang in sorted(lang_distribution.keys()):
                    lang_members = members[cluster_languages == lang]
                    if len(lang_members) > 0:
                        n_samples = min(samples_per_lang, len(lang_members))
                        if n_samples > 0:
                            sample_indices = np.random.choice(
                                lang_members, n_samples, replace=False
                            )

                            f.write(f"{lang.upper()} examples:\n")

                            for i, idx in enumerate(sample_indices):
                                doc_preds = self.preds[idx]
                                full_text = (
                                    self.texts[idx]
                                    .replace("\n", "\\n")
                                    .replace("\r", "\\r")
                                )
                                f.write(f"  {i + 1}. [{idx}] {doc_preds} {full_text}\n")

                            f.write("\n")

                f.write("\n")

        print(f"Multilingual analysis saved to: {analysis_file}")

    def visualize_multilingual_clusters(self, k):
        """Create UMAP with different markers for languages"""
        try:
            print(f"Creating multilingual UMAP visualization (k={k})...")

            embeddings_vis = self.embeddings_pca_norm
            communities_vis = self.community_results[k]
            languages_vis = np.array(self.language_labels)

            # UMAP projection
            umap_model = umap.UMAP(
                n_neighbors=min(30, len(embeddings_vis) // 10),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
                low_memory=True,
            )
            embedding_2d = umap_model.fit_transform(embeddings_vis)

            # Plot with different markers for languages
            plt.figure(figsize=(16, 12))

            # Define markers for each language
            markers = ["o", "s", "^", "D", "v", "<", ">"]
            language_markers = {
                lang: markers[i % len(markers)]
                for i, lang in enumerate(sorted(set(languages_vis)))
            }

            # Plot each language separately
            for lang in sorted(set(languages_vis)):
                lang_mask = languages_vis == lang
                lang_communities = communities_vis[lang_mask]
                lang_coords = embedding_2d[lang_mask]

                scatter = plt.scatter(
                    lang_coords[:, 0],
                    lang_coords[:, 1],
                    c=lang_communities,
                    cmap="tab10",
                    marker=language_markers[lang],
                    alpha=0.7,
                    s=8,
                    label=f"{lang.upper()}",
                    edgecolors="black",
                    linewidth=0.1,
                )

            # Add colorbar and legend
            plt.colorbar(scatter, label="Cluster ID")
            plt.legend(title="Languages", bbox_to_anchor=(1.15, 1))

            # Add cluster labels
            for cluster_id in sorted(set(communities_vis)):
                mask = communities_vis == cluster_id
                if np.sum(mask) > 0:
                    centroid_x = np.mean(embedding_2d[mask, 0])
                    centroid_y = np.mean(embedding_2d[mask, 1])

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
                f"Multilingual UMAP: {k} Clusters\nColors=Clusters, Markers=Languages"
            )
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")

            plot_path = self.output_dir / f"multilingual_umap_k_{k}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Multilingual UMAP saved to: {plot_path}")

        except Exception as e:
            print(f"Error creating multilingual UMAP: {e}")
            plt.close("all")

    def run_full_analysis(self):
        """Run complete multilingual analysis"""
        try:
            print("=" * 60)
            print("MULTILINGUAL SUBREGISTER ANALYSIS")
            print("=" * 60)

            # Steps
            self.reduce_dimensions()
            optimal_k, optimal_communities, optimal_model, all_scores = (
                self.find_optimal_k()
            )

            self.community_results = {optimal_k: optimal_communities}

            self.analyze_multilingual_clusters(k=optimal_k)
            self.visualize_multilingual_clusters(k=optimal_k)

            print(
                f"\nMultilingual analysis complete! Results saved to: {self.output_dir}"
            )

        except Exception as e:
            print(f"Error during multilingual analysis: {e}")
            import traceback

            traceback.print_exc()
        finally:
            plt.close("all")
            import gc

            gc.collect()


def find_language_groups(pkl_files):
    """Group files by register"""
    register_groups = {}

    for pkl_file in pkl_files:
        filename = Path(pkl_file).stem
        if "_embeds_" in filename:
            register_part = filename.split("_embeds_")[1]

            if register_part not in register_groups:
                register_groups[register_part] = []
            register_groups[register_part].append(pkl_file)

    return register_groups


# Main execution
if __name__ == "__main__":
    pkl_directory = "../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/"
    pkl_files = glob.glob(os.path.join(pkl_directory, "*.pkl"))

    results_dir = "subregister_results"
    Path(results_dir).mkdir(exist_ok=True)

    print(f"Found {len(pkl_files)} pkl files")

    # Group by register
    register_groups = find_language_groups(pkl_files)
    print(f"Found {len(register_groups)} register groups")

    # Process individual files
    print("\n=== INDIVIDUAL LANGUAGE ANALYSES ===")
    for pkl_file in pkl_files:
        try:
            print(f"\nProcessing: {os.path.basename(pkl_file)}")
            analyzer = SubregisterAnalyzer(pkl_file, results_base_dir=results_dir)
            analyzer.run_full_analysis()
            print("✓ Success")
        except ValueError as e:
            if "too small" in str(e):
                print(f"⚠ Skipped: {e}")
            else:
                print(f"✗ Error: {e}")
        except Exception as e:
            print(f"✗ Error: {e}")
        finally:
            plt.close("all")
            import gc

            gc.collect()

    # Process multilingual combinations
    print("\n=== MULTILINGUAL COMBINATIONS ===")
    for register, files in register_groups.items():
        if len(files) >= 2:
            try:
                languages = [Path(f).stem.split("_embeds_")[0] for f in files]
                print(f"\nCombining {register}: {', '.join(languages)}")

                ml_analyzer = MultiLanguageAnalyzer(files, results_base_dir=results_dir)
                ml_analyzer.run_full_analysis()
                print("✓ Success")
            except Exception as e:
                print(f"✗ Error: {e}")
            finally:
                plt.close("all")
                import gc

                gc.collect()

    print("\n=== ALL ANALYSES COMPLETE ===")
