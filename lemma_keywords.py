import argparse
import math
import pickle
import re
import sys
from collections import Counter, defaultdict

from tqdm import tqdm


def load_lemmas(pkl_file):
    """Load lemmatized data from pickle file."""
    print("Loading data...")
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File '{pkl_file}' not found.")
        sys.exit(1)
    except pickle.UnpicklingError:
        print(f"Error: '{pkl_file}' is not a valid pickle file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print(f"Error: Expected list, got {type(data)}")
        sys.exit(1)

    print(f"Loaded {len(data)} lemmatized tokens")
    return data


def is_valid_word(word):
    """Check if word is valid for keyword analysis."""
    if not word or len(word) < 3:
        return False
    # Only alphabetic characters (including accented)
    return bool(re.match(r"^[a-zA-ZäöåÄÖÅéèêëíìîïóòôöúùûüýÿñçšžđþæø]+$", word))


def validate_item(item):
    """Validate that item has required keys."""
    if not isinstance(item, dict):
        return False
    return "cluster_id" in item and "lemma" in item


def group_by_clusters(data, min_count=2):
    """Group lemmas by cluster_id - optimized for large datasets."""
    print("Grouping by clusters...")
    clusters = defaultdict(Counter)  # Use Counter directly for efficiency
    skipped_items = 0

    for item in tqdm(data, desc="Processing tokens"):
        if not validate_item(item):
            skipped_items += 1
            continue

        cluster_id = item.get("cluster_id", "unknown")
        lemma = item.get("lemma", "").strip().lower()

        if is_valid_word(lemma):
            clusters[cluster_id][lemma] += 1  # Count directly

    if skipped_items > 0:
        print(f"  Skipped {skipped_items} invalid items")

    # Apply minimum count threshold and convert to regular dict
    print(f"  Applying minimum count threshold: {min_count}")
    filtered_clusters = {}

    for cluster_id, word_counts in clusters.items():
        # Filter out words below threshold
        filtered_counts = {
            word: count for word, count in word_counts.items() if count >= min_count
        }

        if filtered_counts:  # Only keep clusters with remaining words
            filtered_clusters[cluster_id] = filtered_counts
            total_words = sum(filtered_counts.values())
            unique_words = len(filtered_counts)
            print(f"  Cluster {cluster_id}: {total_words} words, {unique_words} unique")
        else:
            print(f"  Cluster {cluster_id}: skipped (no words above threshold)")

    return filtered_clusters


def calculate_tfidf(clusters):
    """Calculate TF-IDF scores - optimized for large datasets."""
    print("Calculating TF-IDF scores...")

    if not clusters:
        print("  No clusters to process")
        return {}

    # Get document frequency efficiently
    print("  Computing document frequencies...")
    doc_freq = defaultdict(int)
    total_clusters = len(clusters)

    for word_counts in clusters.values():
        for word in word_counts.keys():
            doc_freq[word] += 1

    print(f"  Found {len(doc_freq)} unique words across all clusters")

    # Calculate TF-IDF for each cluster
    cluster_scores = {}

    for cluster_id, word_counts in tqdm(clusters.items(), desc="Computing TF-IDF"):
        total_words = sum(word_counts.values())

        # Skip clusters with no words (safety check)
        if total_words == 0:
            print(f"    Warning: Cluster {cluster_id} has no words, skipping")
            continue

        scores = {}

        for word, count in word_counts.items():
            # Term frequency
            tf = count / total_words

            # Inverse document frequency
            idf = math.log(total_clusters / doc_freq[word])

            # TF-IDF score
            scores[word] = tf * idf

        # Get top 20 words by TF-IDF score
        top_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
        cluster_scores[cluster_id] = top_words

        if top_words:
            print(f"    Cluster {cluster_id}: top score = {top_words[0][1]:.4f}")
        else:
            print(f"    Cluster {cluster_id}: no keywords generated")

    return cluster_scores


def print_results(cluster_scores, pkl_file):
    """Print keyword analysis results."""
    print(f"\n{'=' * 60}")
    print(f"KEYWORD ANALYSIS: {pkl_file}")
    print(f"{'=' * 60}")

    if not cluster_scores:
        print("No results to display.")
        return

    for cluster_id, keywords in cluster_scores.items():
        print(f"\nCluster: {cluster_id}")
        print("-" * 30)

        if not keywords:
            print("  No keywords found")
            continue

        for i, (word, score) in enumerate(keywords, 1):
            print(f"  {i:2d}. {word:<15} ({score:.4f})")


def save_results(cluster_scores, pkl_file):
    """Save results to text file."""
    output_file = pkl_file.replace(".pkl", "_keywords.txt")

    print(f"Saving results to {output_file}...")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"KEYWORD ANALYSIS: {pkl_file}\n")
            f.write("=" * 60 + "\n\n")

            if not cluster_scores:
                f.write("No results to save.\n")
                return

            for cluster_id, keywords in cluster_scores.items():
                f.write(f"Cluster: {cluster_id}\n")
                f.write("-" * 30 + "\n")

                if not keywords:
                    f.write("  No keywords found\n\n")
                    continue

                for i, (word, score) in enumerate(keywords, 1):
                    f.write(f"  {i:2d}. {word:<15} ({score:.4f})\n")
                f.write("\n")

        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze keywords in lemmatized data")
    parser.add_argument("pkl_file", help="Path to lemmas.pkl file")
    parser.add_argument("--no-save", action="store_true", help="Don't save to file")
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum word count threshold (default: 2)",
    )

    args = parser.parse_args()

    if args.min_count < 1:
        print("Error: min-count must be at least 1")
        sys.exit(1)

    # Load data
    data = load_lemmas(args.pkl_file)

    if not data:
        print("No data to process.")
        sys.exit(1)

    # Group by clusters (with progress bar and minimum count filtering)
    clusters = group_by_clusters(data, min_count=args.min_count)
    print(f"Found {len(clusters)} clusters with valid keywords")

    if not clusters:
        print("No clusters found. Try lowering --min-count threshold.")
        sys.exit(1)

    # Calculate TF-IDF scores (with progress tracking)
    cluster_scores = calculate_tfidf(clusters)

    # Show results
    print_results(cluster_scores, args.pkl_file)

    # Save results
    if not args.no_save:
        save_results(cluster_scores, args.pkl_file)


if __name__ == "__main__":
    main()
