import glob
import os
import pickle
from collections import defaultdict

# The full label hierarchy
labels_structure = {
    "MT": [],
    "LY": [],
    "SP": ["IT"],
    "ID": [],
    "NA": ["NE", "SR", "NB"],
    "HI": ["RE"],
    "IN": ["EN", "RA", "DTP", "FI", "LT"],
    "OP": ["RV", "OB", "RS", "AV"],
    "IP": ["DS", "ED"],
}

# Create reverse mapping: child -> parent
child_to_parent = {}
for parent, children in labels_structure.items():
    for child in children:
        child_to_parent[child] = parent


def expand_labels(preds_list):
    """Expand a list of labels to include all necessary parent labels"""
    expanded = set(preds_list)

    # For each label, add its parent if it has one
    for label in preds_list:
        if label in child_to_parent:
            parent = child_to_parent[label]
            expanded.add(parent)

    return sorted(list(expanded))


def get_canonical_suffix(preds_list):
    """Get the canonical filename suffix for a preds list"""
    expanded_preds = expand_labels(preds_list)
    return "-".join(expanded_preds)


# Target values we're interested in
target_values = {"ID", "NB", "OB"}

# Process each language
languages = ["en", "fi", "sv"]
base_path = "../data/model_embeds/concat/bge-m3-fold-6/th-optimised/sm"

for lang in languages:
    print(f"\n=== Processing {lang} files ===")

    # Find all pickle files for this language
    pattern = f"{base_path}/{lang}_embeds_*.pkl"
    files = glob.glob(pattern)

    print(f"Found {len(files)} files for {lang}")

    # Dictionary to group files by their canonical representation
    canonical_groups = defaultdict(list)

    # Process each file
    for file_path in files:
        filename = os.path.basename(file_path)
        # Extract the preds part from filename like "en_embeds_ID-NB.pkl"
        preds_part = filename.replace(f"{lang}_embeds_", "").replace(".pkl", "")
        original_preds = preds_part.split("-")

        print(f"  Processing {filename} with preds: {original_preds}")

        # Load the data
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            print(f"    Loaded {len(data)} rows")

            # Group rows by their canonical preds representation
            for row in data:
                original_row_preds = row["preds"]
                canonical_preds = expand_labels(original_row_preds)
                canonical_suffix = "-".join(canonical_preds)

                # Update the row's preds to the canonical form
                row["preds"] = canonical_preds
                canonical_groups[canonical_suffix].append(row)

            print(
                f"    Added to canonical group: {get_canonical_suffix(original_preds)}"
            )

        except Exception as e:
            print(f"    Error loading {filename}: {e}")
            continue

    # Save merged canonical files
    print(f"\nSaving {len(canonical_groups)} canonical files for {lang}:")

    for canonical_suffix, rows in canonical_groups.items():
        # Only save if it contains our target values
        canonical_preds = canonical_suffix.split("-")
        if any(pred in target_values for pred in canonical_preds):
            output_path = f"{base_path}/{lang}_embeds_{canonical_suffix}_canonical.pkl"

            with open(output_path, "wb") as f:
                pickle.dump(rows, f)

            print(
                f"  ✓ {canonical_suffix}: {len(rows):,} rows -> {lang}_embeds_{canonical_suffix}_canonical.pkl"
            )
        else:
            print(
                f"  ✗ {canonical_suffix}: {len(rows):,} rows (skipped - no target labels)"
            )

    # Optionally, remove original files and rename canonical ones
    print(f"\nCleaning up original files for {lang}...")
    for file_path in files:
        try:
            os.remove(file_path)
            print(f"  Removed {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  Error removing {file_path}: {e}")

    # Rename canonical files to remove '_canonical' suffix
    canonical_files = glob.glob(f"{base_path}/{lang}_embeds_*_canonical.pkl")
    for file_path in canonical_files:
        new_path = file_path.replace("_canonical.pkl", ".pkl")
        try:
            os.rename(file_path, new_path)
            print(
                f"  Renamed {os.path.basename(file_path)} -> {os.path.basename(new_path)}"
            )
        except Exception as e:
            print(f"  Error renaming {file_path}: {e}")

print("\n🎉 Hierarchy fixing complete!")

# Print summary of what we should have now
print("\nExpected final files based on hierarchy:")
print("Files containing ID: *_ID.pkl")
print("Files containing NB: *_NA-NB.pkl (since NB requires NA)")
print("Files containing OB: *_OP-OB.pkl (since OB requires OP)")
print("Files containing combinations like ID+NB: *_ID-NA-NB.pkl")
