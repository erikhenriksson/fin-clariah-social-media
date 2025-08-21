import csv

# Let's examine the raw file to see what's happening
filename = "../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/en_embeds_sm.tsv"

print("=== RAW FILE INSPECTION ===")
with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines in file: {len(lines)}")
print(f"Header line: {repr(lines[0])}")
print(f"First data line: {repr(lines[1])}")
print(f"Second data line: {repr(lines[2])}")

# Count tabs in header vs first few data lines
header_tabs = lines[0].count("\t")
print(f"\nTabs in header: {header_tabs}")
print(f"Tabs in line 1: {lines[1].count('\t')}")
print(f"Tabs in line 2: {lines[2].count('\t')}")

# Check for quotes
print(f"\nQuotes in header: {lines[0].count('\"')}")
print(f"Quotes in line 1: {lines[1].count('\"')}")
print(f"Quotes in line 2: {lines[2].count('\"')}")

print("\n=== CSV PARSER TEST ===")
# Test CSV parsing
csv.field_size_limit(10**7)
with open(filename, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    print(f"Header parsed: {len(header)} columns")

    row1 = next(reader)
    print(f"Row 1 parsed: {len(row1)} columns")
    print(f"Row 1 content: {row1[:4]}...")  # First 4 columns
