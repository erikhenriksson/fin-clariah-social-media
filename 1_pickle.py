import ast
import csv
import os
import pickle
from collections import defaultdict

# Increase field size limit for large embeddings
csv.field_size_limit(10**7)  # 10MB limit

# Create output directory
os.makedirs("../data/model_embeds/concat/bge-m3-fold-6/th-optimised/sm", exist_ok=True)

# Process each file
files = ["en_embeds.tsv", "fi_embeds.tsv", "sv_embeds.tsv"]
target_values = {"ID", "NB", "OB"}

for filename in files:
    print(f"\n=== Processing {filename} ===")
    input_path = f"../data/model_embeds/concat/bge-m3-fold-6/th-optimised/{filename}"
    lang = filename.split("_")[0]  # Extract language code

    # Dictionary to accumulate rows for each preds combination
    # We'll write in batches to avoid memory issues
    preds_buffers = defaultdict(list)
    preds_counts = defaultdict(int)
    total_rows = 0
    matching_rows = 0
    batch_size = 5000  # Smaller batch size for better memory management

    with open(input_path, "r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile, delimiter="\t")

        # Get header and find column indices
        header = next(reader)
        text_index = header.index("text")
        embed_last_index = header.index("embed_last")
        embed_ref_index = header.index("embed_ref")
        preds_index = header.index("preds")

        print("Processing rows and grouping by preds...")

        # Process each row in single pass
        for row in reader:
            total_rows += 1
            if total_rows % 50000 == 0:
                print(
                    f"  Processed {total_rows:,} rows, found {matching_rows:,} matching..."
                )
                print(f"    Current preds groups: {dict(preds_counts)}")

            try:
                preds_str = row[preds_index]
                preds_list = ast.literal_eval(preds_str)

                # Check if any target value is in the preds list
                if any(pred in target_values for pred in preds_list):
                    matching_rows += 1

                    # Extract the three columns we want
                    text = row[text_index]
                    embed_last_str = row[embed_last_index]
                    embed_ref_str = row[embed_ref_index]

                    # Parse embedding and take index [0] to unwrap nested list
                    embed_last_nested = ast.literal_eval(embed_last_str)
                    embed_last = embed_last_nested[0]  # Get the actual embedding

                    embed_ref_nested = ast.literal_eval(embed_ref_str)
                    embed_ref = embed_ref_nested[0]  # Get the actual embedding

                    # Create filename suffix from preds (sorted for consistency)
                    preds_suffix = "-".join(sorted(preds_list))

                    # Store the row data
                    row_data = {
                        "text": text,
                        "embed_last": embed_last,
                        "embed_ref": embed_ref,
                        "preds": preds_list,
                    }

                    preds_buffers[preds_suffix].append(row_data)
                    preds_counts[preds_suffix] += 1

                    # Write batch if buffer is full
                    if len(preds_buffers[preds_suffix]) >= batch_size:
                        output_path = f"../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/{lang}_embeds_{preds_suffix}.pkl"

                        # Load existing data if file exists
                        existing_data = []
                        if os.path.exists(output_path):
                            try:
                                with open(output_path, "rb") as f:
                                    existing_data = pickle.load(f)
                            except:
                                existing_data = []

                        # Combine and save
                        existing_data.extend(preds_buffers[preds_suffix])
                        with open(output_path, "wb") as f:
                            pickle.dump(existing_data, f)

                        print(
                            f"    Saved batch for '{preds_suffix}' ({len(preds_buffers[preds_suffix])} rows, total: {preds_counts[preds_suffix]})"
                        )
                        preds_buffers[preds_suffix] = []  # Clear buffer

            except Exception as e:
                print(f"  Error processing row {total_rows}: {e}")
                continue

    # Save any remaining data in buffers
    print("\nSaving final batches...")
    for preds_suffix, buffer in preds_buffers.items():
        if buffer:
            output_path = f"../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/{lang}_embeds_{preds_suffix}.pkl"

            # Load existing data if file exists
            existing_data = []
            if os.path.exists(output_path):
                try:
                    with open(output_path, "rb") as f:
                        existing_data = pickle.load(f)
                except:
                    existing_data = []

            # Combine and save
            existing_data.extend(buffer)
            with open(output_path, "wb") as f:
                pickle.dump(existing_data, f)

            print(f"  Final batch for '{preds_suffix}': {len(buffer)} rows")

    print(f"\n✓ Completed {filename}:")
    print(f"  Total rows processed: {total_rows:,}")
    print(f"  Matching rows: {matching_rows:,}")
    print(f"  Final counts by preds: {dict(preds_counts)}")

print("\n🎉 All files processed!")
