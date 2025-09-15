import glob
import os
import pickle

import pandas as pd
import trankit
from tqdm import tqdm


def find_clustered_data_files():
    """Find all files that end with 'last/clustered_data.pkl'."""
    pattern = "**/last/clustered_data.pkl"
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} files ending with 'last/clustered_data.pkl'")
    return files


def check_parsing_status(pickle_file: str) -> str:
    """
    Check if parsing is complete by checking if parsed.pkl exists and is valid.

    Returns:
        'complete' - parsing is done, skip this file
        'missing' - no parsed file exists, start fresh
        'error' - parsed file exists but is corrupted, restart
    """
    parsed_file = os.path.join(os.path.dirname(pickle_file), "parsed.pkl")

    if not os.path.exists(parsed_file):
        return "missing"

    try:
        # Try to load the parsed file to verify it's valid
        with open(parsed_file, "rb") as f:
            parsed_data = pickle.load(f)

        # Basic validation - check if it's a list/dataframe with expected structure
        if isinstance(parsed_data, (list, pd.DataFrame)) and len(parsed_data) > 0:
            return "complete"
        else:
            return "error"

    except Exception as e:
        print(f"Error checking parsed file {parsed_file}: {e}")
        return "error"


def parse_pickle_file(pickle_file: str, trankit_pipeline):
    """Parse a single pickle file and save results."""

    # Check if already parsed
    status = check_parsing_status(pickle_file)

    if status == "complete":
        print(f"✓ {pickle_file} already parsed, skipping")
        return True
    elif status == "error":
        print(f"⚠ {pickle_file} has corrupted parsed file, restarting")
        parsed_file = os.path.join(os.path.dirname(pickle_file), "parsed.pkl")
        try:
            os.remove(parsed_file)
        except:
            pass

    print(f"→ Processing {pickle_file}")

    try:
        # Load the pickle file
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)

        # Extract required fields
        text_content = data.get("text", "")
        register_info = data.get("register", "")
        cluster_id = data.get("cluster_id", "")

        if not text_content:
            print(f"Warning: No text content found in {pickle_file}")
            return False

        # Parse with Trankit
        try:
            parsed = trankit_pipeline(str(text_content))
        except Exception as e:
            print(f"Error parsing text in {pickle_file}: {e}")
            return False

        # Extract tokens from all sentences
        parsed_rows = []

        for sent_idx, sentence in enumerate(parsed["sentences"]):
            for token in sentence["tokens"]:
                # Define expected CoNLL-U columns in order
                expected_columns = [
                    "id",
                    "text",
                    "lemma",
                    "upos",
                    "xpos",
                    "feats",
                    "head",
                    "deprel",
                    "deps",
                    "misc",
                ]

                # Start with metadata
                parsed_row = {
                    "register": register_info,
                    "cluster_id": cluster_id,
                    "sentence_id": sent_idx,
                }

                # Add all expected token fields with defaults for missing keys
                for col in expected_columns:
                    parsed_row[col] = token.get(col, "")

                parsed_rows.append(parsed_row)

        # Save parsed data as pickle
        if parsed_rows:
            output_file = os.path.join(os.path.dirname(pickle_file), "parsed.pkl")

            with open(output_file, "wb") as f:
                pickle.dump(parsed_rows, f)

            print(f"Saved {len(parsed_rows)} tokens to {output_file}")
            return True
        else:
            print(f"No tokens parsed for {pickle_file}")
            return False

    except Exception as e:
        print(f"Error processing {pickle_file}: {e}")
        return False


def parse_all_pickle_files(language_code: str = "english"):
    """Parse all clustered_data.pkl files using Trankit."""

    # Map language codes to Trankit language names
    trankit_lang_map = {
        "en": "english",
        "fr": "french",
        "sv": "swedish",
        "de": "german",
        "es": "spanish",
        "it": "italian",
    }

    trankit_lang = trankit_lang_map.get(language_code, language_code)

    # Find all pickle files
    pickle_files = find_clustered_data_files()

    if not pickle_files:
        print("No clustered_data.pkl files found")
        return

    # Initialize Trankit pipeline
    print(f"Initializing Trankit pipeline for {trankit_lang}...")
    try:
        pipeline = trankit.Pipeline(trankit_lang, gpu=True)
    except Exception as e:
        print(f"Error initializing Trankit for '{trankit_lang}': {e}")
        print("Trying without GPU...")
        try:
            pipeline = trankit.Pipeline(trankit_lang, gpu=False)
        except Exception as e2:
            print(f"Error initializing Trankit without GPU: {e2}")
            print(
                "Available languages can be checked with: import trankit; print(trankit.supported_langs)"
            )
            return

    # Process each pickle file
    successful_parses = 0
    failed_parses = 0

    for pickle_file in tqdm(pickle_files, desc="Processing pickle files"):
        try:
            if parse_pickle_file(pickle_file, pipeline):
                successful_parses += 1
            else:
                failed_parses += 1
        except Exception as e:
            print(f"Unexpected error with {pickle_file}: {e}")
            failed_parses += 1

            # Reinitialize pipeline on error (as in original code)
            try:
                pipeline = trankit.Pipeline(trankit_lang, gpu=True)
            except:
                try:
                    pipeline = trankit.Pipeline(trankit_lang, gpu=False)
                except:
                    print("Failed to reinitialize pipeline, stopping")
                    break

    print(f"\n{'=' * 60}")
    print(f"Processing completed:")
    print(f"  Successful: {successful_parses}")
    print(f"  Failed: {failed_parses}")
    print(f"  Total: {len(pickle_files)}")
    print(f"{'=' * 60}")


def main():
    """Main function to process all pickle files."""

    # You can change the language here
    language = "english"  # Change to "fr", "sv", etc. as needed

    print(f"{'=' * 60}")
    print(f"Processing all clustered_data.pkl files with {language} parser...")
    print(f"{'=' * 60}")

    parse_all_pickle_files(language)


if __name__ == "__main__":
    main()
