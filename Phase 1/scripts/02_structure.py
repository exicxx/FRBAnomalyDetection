"""Step 2: the structural numbers we need before choosing features.

How many bursts are repeaters vs one-offs, how many CHIME excluded, and how many
brightness/scattering values are limits rather than real measurements.
"""

import sys
from pathlib import Path

# Script lives at Phase 1/scripts/, so two parents up reaches the project root where src/ lives.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from frb_anomaly import data

fields, df = data.load_catalog1()
n = len(df)

# RpName = the repeating source's name, or the sentinel "-9999" for a one-off burst.
is_repeater = df["RpName"].astype(str).str.strip() != "-9999"
n_rep_bursts = int(is_repeater.sum())
n_rep_sources = int(df.loc[is_repeater, "RpName"].nunique())

print("REPEATER STRUCTURE")
print(f"  total sub-burst rows         : {n}")
print(f"  one-off (non-repeating) bursts: {n - n_rep_bursts}")
print(f"  bursts from repeaters         : {n_rep_bursts}  (from {n_rep_sources} repeating sources)")

# The most prolific repeaters: this is the double-counting risk made concrete.
print("\n  most prolific repeaters (burst count):")
top = df.loc[is_repeater, "RpName"].value_counts().head(5)
for name, cnt in top.items():
    print(f"    {name:18s}: {int(cnt)} bursts")

# Flag == 1 means CHIME excluded this burst from their own analysis (quality reasons).
print("\nQUALITY FLAG")
print(f"  bursts flagged excluded (Flag==1): {int((df['Flag'] == 1).sum())} of {n}")

# l_<col> carries a symbol ('<' or '>') when the value is a limit, else it is empty.
print("\nLIMIT FLAGS (values that are limits, not measurements)")
for col in ["l_Flux", "l_Fluence", "l_Scat", "l_Widthfitb"]:
    print(f"  {col:12s}: {int(df[col].notna().sum()):3d} of {n} rows are limits")

print("\nNote: per the column descriptions, Flux and Fluence are reported as LOWER")
print("limits for all bursts (beam-position uncertainty), even where l_ is unset.")
