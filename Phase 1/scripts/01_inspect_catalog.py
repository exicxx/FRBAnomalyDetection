"""
Step 1 inspection: look at the raw CHIME/FRB Catalog 1 table before touching it.
Goal is understanding, not processing. Prints every column with its meaning,
the table size, and where values are missing.

We read the VOTable (an XML format) directly with Python's standard library so
there is no hidden library behaviour: every value is read in plain sight.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Path relative to this script. Script now lives at Phase 1/scripts/, so two parents
# up reaches the project root where data/ lives. parents[0] = scripts/, parents[1] = Phase 1/,
# parents[2] = project root.
RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "chime_cat1_table2.vot"

# VOTable tags carry an XML namespace prefix; this strips it so "FIELD" matches.
def local(tag):
    return tag.rsplit("}", 1)[-1]

root = ET.parse(RAW).getroot()

# FIELD elements describe each column: name, data type, unit, plain-English description.
fields = []
for el in root.iter():
    if local(el.tag) == "FIELD":
        desc = ""
        for child in el:
            if local(child.tag) == "DESCRIPTION":
                desc = (child.text or "").strip()
        fields.append(
            {
                "name": el.attrib.get("name") or el.attrib.get("ID"),
                "datatype": el.attrib.get("datatype", ""),
                "unit": el.attrib.get("unit", ""),
                "description": desc,
            }
        )

# TR = table row, TD = table cell. Empty cell -> None -> becomes missing later.
rows = []
for tr in root.iter():
    if local(tr.tag) == "TR":
        rows.append([td.text for td in tr if local(td.tag) == "TD"])

colnames = [f["name"] for f in fields]
df = pd.DataFrame(rows, columns=colnames)

# Convert numeric columns from text to numbers based on the declared VOTable type.
numeric_types = {"float", "double", "int", "long", "short", "unsignedByte"}
for f in fields:
    if f["datatype"] in numeric_types:
        df[f["name"]] = pd.to_numeric(df[f["name"]], errors="coerce")

print("=" * 95)
print("COLUMNS:  name | type | unit | meaning")
print("=" * 95)
for f in fields:
    unit = f["unit"] if f["unit"] else "-"
    print(f"  {f['name']:12s} | {f['datatype']:8s} | {unit:9s} | {f['description']}")

print("\n" + "=" * 95)
print(f"TABLE SIZE: {df.shape[0]} rows (bursts), {df.shape[1]} columns")
print("=" * 95)

print("\nMISSING VALUES (columns with any gaps):")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) == 0:
    print("  none")
else:
    for col, n in missing.items():
        print(f"  {col:12s} : {n:4d} missing  ({100 * n / len(df):.0f}%)")
