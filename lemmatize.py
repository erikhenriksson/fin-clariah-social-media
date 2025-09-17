import glob
import math
import os
import pickle
from typing import Any, Dict, List

import spacy
from spacy.lang.en import English
from spacy.lang.fi import Finnish
from spacy.lang.sv import Swedish
from tqdm import tqdm

# Language model configurations - using fastest models available
LANGUAGE_MODELS = {
    "english": "en_core_web_sm",  # Fast English model
    "swedish": "sv_core_news_sm",  # Fast Swedish model
    "finnish": "fi_core_news_sm",  # Fast Finnish model
}

# Fallback to rule-based if models not available
LANGUAGE_FALLBACKS = {"english": English, "swedish": Swedish, "finnish": Finnish}

# Chunk size for output files
CHUNK_SIZE = 1000


def find_clustered_data_files() -> List[str]:
    """Find all files that end with 'last/clustered_data.pkl'."""
    pattern = "**/last/clustered_data.pkl"
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} files ending with 'last/clustered_data.pkl'")
    return files


def detect_language_from_path(pickle_file: str) -> str:
    """Detect language from file path."""
    path_lower = pickle_file.lower()

    if "/sv/" in path_lower or "_sv_" in path_lower or "-sv-" in path_lower:
        return "swedish"
    elif "/en/" in path_lower or "_en_" in path_lower or "-en-" in path_lower:
        return "english"
    elif "/fi/" in path_lower or "_fi_" in path_lower or "-fi-" in path_lower:
        return "finnish"
    else:
        print(
            f"Warning: Could not detect language from path {pickle_file}, defaulting to English"
        )
        return "english"


def check_lemmatization_status(pickle_file: str) -> str:
    """
    Check if lemmatization is complete by looking for chunk files.

    Returns:
        'complete' - lemmatization done, skip
        'missing' - no lemma files, start fresh
        'partial' - some chunks exist, continue from where left off
        'error' - corrupted files, restart
    """
    base_dir = os.path.dirname(pickle_file)

    # Look for any lemma chunk files
    lemma_pattern = os.path.join(base_dir, "lemmas_chunk_*.pkl")
    existing_chunks = glob.glob(lemma_pattern)

    if not existing_chunks:
        return "missing"

    # Check if we have a completion marker
    completion_file = os.path.join(base_dir, "lemmas_complete.txt")
    if os.path.exists(completion_file):
        return "complete"

    # Check if existing chunks are valid
    try:
        for chunk_file in existing_chunks:
            with open(chunk_file, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, list) or len(data) == 0:
                return "error"
        return "partial"
    except Exception as e:
        print(f"Error checking lemma chunks in {base_dir}: {e}")
        return "error"


def get_existing_chunk_count(pickle_file: str) -> int:
    """Get the number of existing chunk files."""
    base_dir = os.path.dirname(pickle_file)
    lemma_pattern = os.path.join(base_dir, "lemmas_chunk_*.pkl")
    existing_chunks = glob.glob(lemma_pattern)
    return len(existing_chunks)


def save_lemma_chunk(lemmas: List[Dict], base_dir: str, chunk_num: int):
    """Save a chunk of lemmas to file."""
    chunk_file = os.path.join(base_dir, f"lemmas_chunk_{chunk_num:04d}.pkl")
    with open(chunk_file, "wb") as f:
        pickle.dump(lemmas, f)
    return chunk_file


def mark_completion(base_dir: str, total_chunks: int, total_lemmas: int):
    """Create a completion marker file with metadata."""
    completion_file = os.path.join(base_dir, "lemmas_complete.txt")
    with open(completion_file, "w") as f:
        f.write(f"Lemmatization completed\n")
        f.write(f"Total chunks: {total_chunks}\n")
        f.write(f"Total lemmas: {total_lemmas}\n")
        f.write(f"Chunk size: {CHUNK_SIZE}\n")


def clean_partial_files(base_dir: str):
    """Remove partial lemmatization files to start fresh."""
    lemma_pattern = os.path.join(base_dir, "lemmas_chunk_*.pkl")
    completion_file = os.path.join(base_dir, "lemmas_complete.txt")

    for file in glob.glob(lemma_pattern):
        try:
            os.remove(file)
        except:
            pass

    try:
        os.remove(completion_file)
    except:
        pass


def load_spacy_model(language: str):
    """Load the fastest available spaCy model for the language."""
    model_name = LANGUAGE_MODELS.get(language)

    try:
        # Try to load the full model first
        nlp = spacy.load(model_name)
        print(f"✓ Loaded {model_name} for {language}")

        # Disable unnecessary components for speed
        nlp.disable_pipes(["parser", "ner", "attribute_ruler"])

        return nlp

    except OSError:
        print(
            f"⚠ Model {model_name} not found, using rule-based lemmatizer for {language}"
        )

        # Fallback to rule-based
        fallback_class = LANGUAGE_FALLBACKS.get(language, English)
        nlp = fallback_class()

        # Add only tokenizer and lemmatizer
        if not nlp.has_pipe("lemmatizer"):
            nlp.add_pipe("lemmatizer")

        return nlp


def lemmatize_text(text: str, nlp) -> List[Dict[str, str]]:
    """Fast lemmatization of text using spaCy."""
    doc = nlp(str(text))

    lemmas = []
    for token in doc:
        # Skip whitespace and empty tokens
        if token.is_space or not token.text.strip():
            continue

        lemmas.append({"original": token.text, "lemma": token.lemma_})

    return lemmas


def has_multiple_clusters(data) -> bool:
    """Check if data has more than one unique cluster_id."""
    if isinstance(data, dict):
        return False
    elif isinstance(data, list):
        cluster_ids = set()
        for item in data:
            if isinstance(item, dict):
                cluster_id = item.get("cluster_id", "")
                if cluster_id:
                    cluster_ids.add(cluster_id)
        return len(cluster_ids) > 1
    return False


def lemmatize_pickle_file(pickle_file: str, nlp) -> bool:
    """Lemmatize a single pickle file with chunked output."""

    base_dir = os.path.dirname(pickle_file)

    # Check if already processed
    status = check_lemmatization_status(pickle_file)

    if status == "complete":
        print(f"✓ {pickle_file} already lemmatized, skipping")
        return True
    elif status == "error":
        print(f"⚠ {pickle_file} has corrupted lemma files, restarting")
        clean_partial_files(base_dir)
        start_from_chunk = 0
    elif status == "partial":
        start_from_chunk = get_existing_chunk_count(pickle_file)
        print(f"→ Resuming {pickle_file} from chunk {start_from_chunk}")
    else:  # missing
        start_from_chunk = 0

    print(f"→ Lemmatizing {pickle_file}")

    try:
        # Load data
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)

        # Check cluster diversity
        if not has_multiple_clusters(data):
            print(f"⚠ {pickle_file} has only one unique cluster, skipping")
            return True

        # Handle different data structures
        if isinstance(data, dict):
            # Single dictionary - process as single chunk
            text_content = data.get("text", "")
            if not text_content:
                print(f"Warning: No text in {pickle_file}")
                return False

            lemmas = lemmatize_text(text_content, nlp)

            chunk_lemmas = []
            for lemma_info in lemmas:
                lemma_row = {
                    "register": data.get("register", ""),
                    "cluster_id": data.get("cluster_id", ""),
                    "original": lemma_info["original"],
                    "lemma": lemma_info["lemma"],
                }
                chunk_lemmas.append(lemma_row)

            if chunk_lemmas:
                save_lemma_chunk(chunk_lemmas, base_dir, 0)
                mark_completion(base_dir, 1, len(chunk_lemmas))
                print(f"Saved {len(chunk_lemmas)} lemmas in 1 chunk")
                return True

        elif isinstance(data, list):
            # List of dictionaries - process in chunks
            total_items = len(data)
            total_chunks = math.ceil(total_items / CHUNK_SIZE)
            total_lemmas_processed = 0

            print(
                f"Processing {total_items} items in {total_chunks} chunks of {CHUNK_SIZE}"
            )

            # Skip already processed chunks
            items_to_skip = start_from_chunk * CHUNK_SIZE

            for chunk_num in range(start_from_chunk, total_chunks):
                start_idx = chunk_num * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, total_items)
                chunk_data = data[start_idx:end_idx]

                chunk_lemmas = []

                for i, item in tqdm(
                    enumerate(chunk_data, start=start_idx),
                    total=len(chunk_data),
                    desc=f"Chunk {chunk_num + 1}/{total_chunks}",
                    leave=False,
                ):
                    if not isinstance(item, dict):
                        continue

                    text_content = item.get("text", "")
                    if not text_content:
                        continue

                    lemmas = lemmatize_text(text_content, nlp)

                    for lemma_info in lemmas:
                        lemma_row = {
                            "item_index": i,
                            "register": item.get("register", ""),
                            "cluster_id": item.get("cluster_id", ""),
                            "original": lemma_info["original"],
                            "lemma": lemma_info["lemma"],
                        }
                        chunk_lemmas.append(lemma_row)

                # Save chunk
                if chunk_lemmas:
                    chunk_file = save_lemma_chunk(chunk_lemmas, base_dir, chunk_num)
                    total_lemmas_processed += len(chunk_lemmas)
                    print(
                        f"Saved chunk {chunk_num + 1}/{total_chunks}: {len(chunk_lemmas)} lemmas to {os.path.basename(chunk_file)}"
                    )

            # Mark completion
            mark_completion(base_dir, total_chunks, total_lemmas_processed)
            print(
                f"Completed! Total: {total_lemmas_processed} lemmas in {total_chunks} chunks"
            )
            return True

        else:
            print(f"Error: Unexpected data type {type(data)} in {pickle_file}")
            return False

    except Exception as e:
        print(f"Error processing {pickle_file}: {e}")
        return False


def lemmatize_all_files():
    """Lemmatize all clustered_data.pkl files."""

    # Find files
    pickle_files = find_clustered_data_files()
    if not pickle_files:
        print("No clustered_data.pkl files found")
        return

    # Group by language
    files_by_language = {}
    for pickle_file in pickle_files:
        lang = detect_language_from_path(pickle_file)
        if lang not in files_by_language:
            files_by_language[lang] = []
        files_by_language[lang].append(pickle_file)

    print(
        f"Files by language: {dict((k, len(v)) for k, v in files_by_language.items())}"
    )

    # Process each language
    for language, lang_files in files_by_language.items():
        print(f"\n{'=' * 50}")
        print(f"Processing {len(lang_files)} {language} files")
        print(f"{'=' * 50}")

        # Load spaCy model for this language
        nlp = load_spacy_model(language)

        # Process files
        successful = 0
        failed = 0

        for pickle_file in tqdm(lang_files, desc=f"Lemmatizing {language}"):
            try:
                if lemmatize_pickle_file(pickle_file, nlp):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error with {pickle_file}: {e}")
                failed += 1

        print(f"{language} completed - Success: {successful}, Failed: {failed}")


def main():
    """Main function."""
    print("Fast Chunked Lemmatizer for Swedish, English, Finnish")
    print("=" * 50)
    print(f"Chunk size: {CHUNK_SIZE} documents per output file")
    print("Required spaCy models (install if missing):")
    print("  python -m spacy download en_core_web_sm")
    print("  python -m spacy download sv_core_news_sm")
    print("  python -m spacy download fi_core_news_sm")
    print("=" * 50)

    lemmatize_all_files()


if __name__ == "__main__":
    main()
