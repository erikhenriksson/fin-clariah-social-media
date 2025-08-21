import ast
import csv
import os

# Create output directory
os.makedirs("../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm", exist_ok=True)

# Process each file
files = ["en_embeds.tsv", "fi_embeds.tsv", "sv_embeds.tsv"]
target_values = {"ID", "NB", "OB"}

for filename in files:
    input_path = f"../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/{filename}"
    output_path = f"../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/{filename.replace('.tsv', '_sm.tsv')}"

    with (
        open(input_path, "r", encoding="utf-8", newline="") as infile,
        open(output_path, "w", encoding="utf-8", newline="") as outfile,
    ):
        reader = csv.reader(infile, delimiter="\t")

        # Copy header and find preds column index
        header = next(reader)
        outfile.write("\t".join(header) + "\n")
        preds_index = header.index("preds")

        # Process each row
        for row in reader:
            preds_str = row[preds_index]

            # Parse the string as a Python list
            preds_list = ast.literal_eval(preds_str)

            # Check if any target value is in the preds list
            if any(pred in target_values for pred in preds_list):
                outfile.write("\t".join(row) + "\n")

print("Filtering complete!")
