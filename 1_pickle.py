import ast
import csv
import os
import pickle
from collections import defaultdict

# Increase field size limit for large embeddings
csv.field_size_limit(10**7)  # 10MB limit

# Create output directory
os.makedirs("../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm", exist_ok=True)

# Process each file
files = ["en_embeds.tsv", "fi_embeds.tsv", "sv_embeds.tsv"]
target_values = {"ID", "NB", "OB"}

for filename in files:
    print(f"\n=== Processing {filename} ===")
    input_path = f"../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/{filename}"
    lang = filename.split("_")[0]  # Extract language code

    print(f"First pass: scanning for unique preds combinations...")
    # First pass: collect all unique preds combinations
    preds_combinations = set()
    total_rows = 0
    matching_rows = 0

    with open(input_path, "r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile, delimiter="\t")
        header = next(reader)
        preds_index = header.index("preds")

        for row in reader:
            total_rows += 1
            if total_rows % 100000 == 0:
                print(
                    f"  Scanned {total_rows:,} rows, found {matching_rows:,} matching..."
                )

            preds_str = row[preds_index]
            preds_list = ast.literal_eval(preds_str)

            if any(pred in target_values for pred in preds_list):
                matching_rows += 1
                preds_suffix = "-".join(sorted(preds_list))
                preds_combinations.add(preds_suffix)

    print(
        f"First pass complete: {total_rows:,} total rows, {matching_rows:,} matching rows"
    )
    print(
        f"Found {len(preds_combinations)} unique preds combinations: {sorted(preds_combinations)}"
    )

    # Second pass: process each preds combination separately
    for i, preds_suffix in enumerate(sorted(preds_combinations), 1):
        print(
            f"\nSecond pass {i}/{len(preds_combinations)}: Processing preds='{preds_suffix}'..."
        )
        output_path = f"../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/{lang}_embeds_{preds_suffix}.pkl"
        rows_for_this_preds = []
        count = 0

        with open(input_path, "r", encoding="utf-8", newline="") as infile:
            reader = csv.reader(infile, delimiter="\t")

            # Get header and find column indices
            header = next(reader)
            text_index = header.index("text")
            embed_last_index = header.index("embed_last")
            preds_index = header.index("preds")

            # Process each row
            for row_num, row in enumerate(reader, 1):
                if row_num % 100000 == 0:
                    print(
                        f"    Processed {row_num:,} rows, found {count} matches for '{preds_suffix}'..."
                    )

                preds_str = row[preds_index]
                preds_list = ast.literal_eval(preds_str)

                # Check if this row matches the current preds combination
                current_preds_suffix = "-".join(sorted(preds_list))
                if current_preds_suffix == preds_suffix:
                    # Extract the three columns we want
                    text = row[text_index]
                    embed_last_str = row[embed_last_index]

                    # Parse embedding and take index [0] to unwrap nested list
                    embed_last_nested = ast.literal_eval(embed_last_str)
                    embed_last = embed_last_nested[0]  # Get the actual embedding

                    # Store the row data
                    row_data = {
                        "text": text,
                        "embed_last": embed_last,
                        "preds": preds_list,
                    }
                    rows_for_this_preds.append(row_data)
                    count += 1

                    # Save in batches to avoid memory issues
                    if len(rows_for_this_preds) >= 10000:
                        print(f"    Saving batch of {len(rows_for_this_preds)} rows...")
                        if count == len(rows_for_this_preds):  # First batch
                            with open(output_path, "wb") as outfile:
                                pickle.dump(rows_for_this_preds, outfile)
                        else:  # Append to existing
                            with open(output_path, "rb") as infile:
                                existing = pickle.load(infile)
                            existing.extend(rows_for_this_preds)
                            with open(output_path, "wb") as outfile:
                                pickle.dump(existing, outfile)
                        rows_for_this_preds = []

        # Save any remaining rows
        if rows_for_this_preds:
            print(f"    Saving final batch of {len(rows_for_this_preds)} rows...")
            if count == len(rows_for_this_preds):  # Only batch
                with open(output_path, "wb") as outfile:
                    pickle.dump(rows_for_this_preds, outfile)
            else:  # Final batch
                with open(output_path, "rb") as infile:
                    existing = pickle.load(infile)
                existing.extend(rows_for_this_preds)
                with open(output_path, "wb") as outfile:
                    pickle.dump(existing, outfile)

        print(
            f"✓ Completed '{preds_suffix}': saved {count:,} rows to {lang}_embeds_{preds_suffix}.pkl"
        )

print("\n🎉 Processing complete!")
