import glob
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
    Check if lemmatization is complete.

    Returns:
        'complete' - lemmatization done, skip
        'missing' - no lemma file, start fresh
        'error' - corrupted file, restart
    """
    lemma_file = os.path.join(os.path.dirname(pickle_file), "lemmas.pkl")

    if not os.path.exists(lemma_file):
        return "missing"

    try:
        with open(lemma_file, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, list) and len(data) > 0:
            return "complete"
        else:
            return "error"

    except Exception as e:
        print(f"Error checking lemma file {lemma_file}: {e}")
        return "error"


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
    """Lemmatize a single pickle file."""

    # Check if already processed
    status = check_lemmatization_status(pickle_file)

    if status == "complete":
        print(f"✓ {pickle_file} already lemmatized, skipping")
        return True
    elif status == "error":
        print(f"⚠ {pickle_file} has corrupted lemma file, restarting")
        lemma_file = os.path.join(os.path.dirname(pickle_file), "lemmas.pkl")
        try:
            os.remove(lemma_file)
        except:
            pass

    print(f"→ Lemmatizing {pickle_file}")

    try:
        # Load data
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)

        # Check cluster diversity
        if not has_multiple_clusters(data):
            print(f"⚠ {pickle_file} has only one unique cluster, skipping")
            return True

        all_lemmas = []

        # Handle different data structures
        if isinstance(data, dict):
            # Single dictionary
            text_content = data.get("text", "")
            if not text_content:
                print(f"Warning: No text in {pickle_file}")
                return False

            lemmas = lemmatize_text(text_content, nlp)

            for lemma_info in lemmas:
                lemma_row = {
                    "register": data.get("register", ""),
                    "cluster_id": data.get("cluster_id", ""),
                    "original": lemma_info["original"],
                    "lemma": lemma_info["lemma"],
                }
                all_lemmas.append(lemma_row)

        elif isinstance(data, list):
            # List of dictionaries
            for i, item in tqdm(
                enumerate(data),
                total=len(data),
                desc=f"Processing {os.path.basename(pickle_file)}",
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
                    all_lemmas.append(lemma_row)

        else:
            print(f"Error: Unexpected data type {type(data)} in {pickle_file}")
            return False

        # Save results
        if all_lemmas:
            output_file = os.path.join(os.path.dirname(pickle_file), "lemmas.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(all_lemmas, f)
            print(f"Saved {len(all_lemmas)} lemmas to {output_file}")
            return True
        else:
            print(f"No lemmas generated for {pickle_file}")
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
    print("Fast Lemmatizer for Swedish, English, Finnish")
    print("=" * 50)
    print("Required spaCy models (install if missing):")
    print("  python -m spacy download en_core_web_sm")
    print("  python -m spacy download sv_core_news_sm")
    print("  python -m spacy download fi_core_news_sm")
    print("=" * 50)

    lemmatize_all_files()


if __name__ == "__main__":
    main()
