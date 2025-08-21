import csv

# Let's examine the raw file to see what's happening
filename = "../data/model_embeds/cleaned/bge-m3-fold-6/th-optimised/sm/en_embeds_sm.tsv"

print("=== RAW FILE INSPECTION ===")
with open(filename, "r", encoding="utf-8") as f:
    header = f.readline()
    line1 = f.readline()
    line2 = f.readline()

print(f"Header line: {repr(header)}")
print(f"First data line: {repr(line1)}")
print(f"Second data line: {repr(line2)}")

# Count tabs in header vs first few data lines
tab_char = "\t"
header_tabs = header.count(tab_char)
print(f"\nTabs in header: {header_tabs}")
print(f"Tabs in line 1: {line1.count(tab_char)}")
print(f"Tabs in line 2: {line2.count(tab_char)}")

# Check for quotes
quote_char = '"'
print(f"\nQuotes in header: {header.count(quote_char)}")
print(f"Quotes in line 1: {line1.count(quote_char)}")
print(f"Quotes in line 2: {line2.count(quote_char)}")

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
