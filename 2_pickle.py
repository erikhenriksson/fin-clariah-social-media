import ast
import csv
import os
import pickle
from collections import defaultdict

# Increase field size limit for large embeddings
csv.field_size_limit(10**7)

# Process each file
files = ["en_embeds_sm.tsv", "fi_embeds_sm.tsv", "sv_embeds_sm.tsv"]

for filename in files:
    input_path = (
        f"../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/{filename}"
    )
    lang_prefix = filename.replace("_embeds_sm.tsv", "")

    # Dictionary to group rows by preds value
    preds_groups = defaultdict(list)

    with open(input_path, "r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile, delimiter="\t")

        # Get header and find column indices
        header = next(reader)
        column_indices = [header.index(col) for col in columns_to_keep]

        # Process each row
        for row in reader:
            preds_str = row[header.index("preds")]
            preds_list = ast.literal_eval(preds_str)

            # Create filename suffix from preds list
            preds_key = "-".join(sorted(preds_list))

            # Get the selected columns and process embed_last
            text = row[header.index("text")]
            embed_last_str = row[header.index("embed_last")]
            embed_last = ast.literal_eval(embed_last_str)[
                0
            ]  # Take index [0] from nested list

            # Store as dictionary for pickle
            row_data = {"text": text, "embed_last": embed_last, "preds": preds_list}

            preds_groups[preds_key].append(row_data)

    # Write each group to its own pickle file
    for preds_key, rows in preds_groups.items():
        output_path = f"../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/{lang_prefix}_embeds_{preds_key}.pkl"

        with open(output_path, "wb") as outfile:
            pickle.dump(rows, outfile)

print("Splitting complete!")
