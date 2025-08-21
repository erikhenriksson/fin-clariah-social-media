import ast
import csv
import os
from collections import defaultdict

# Increase field size limit for large embeddings
csv.field_size_limit(10**7)  # 10MB limit

# Create output directory
os.makedirs("../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm", exist_ok=True)

# Process each file
files = ["en_embeds.tsv", "fi_embeds.tsv", "sv_embeds.tsv"]
target_values = {"ID", "NB", "OB"}

for filename in files:
    input_path = f"../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/{filename}"
    lang_prefix = filename.replace("_embeds.tsv", "")

    # Dictionary to store rows by preds value
    preds_data = defaultdict(list)

    with open(input_path, "r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile, delimiter="\t")

        # Get header and find column indices
        header = next(reader)
        text_index = header.index("text")
        embed_last_index = header.index("embed_last")
        preds_index = header.index("preds")

        # Process each row
        for row in reader:
            preds_str = row[preds_index]
            preds_list = ast.literal_eval(preds_str)

            # Check if any target value is in the preds list
            if any(pred in target_values for pred in preds_list):
                # Create filename suffix from preds list
                preds_suffix = "-".join(sorted(preds_list))

                # Store the selected columns
                selected_row = [
                    row[text_index],
                    row[embed_last_index],
                    row[preds_index],
                ]
                preds_data[preds_suffix].append(selected_row)

    # Write each preds group to its own file
    for preds_suffix, rows in preds_data.items():
        output_path = f"../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/{lang_prefix}_embeds_{preds_suffix}.tsv"

        with open(output_path, "w", encoding="utf-8", newline="") as outfile:
            # Write header
            outfile.write("text\tembed_last\tpreds\n")

            # Write data rows
            for row in rows:
                outfile.write("\t".join(row) + "\n")

print("Filtering complete!")
