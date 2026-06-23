"""Loading the CHIME/FRB Catalog 2 burst table.

Catalog 2 (The CHIME/FRB Collaboration 2026, ApJS 283) is downloaded from the
CANFAR archive as a plain CSV. pandas reads it directly; nothing is hidden.

The VOTable utilities (read_votable, _local) are kept as general-purpose tools
because the VOTable format is used across CHIME/FRB and IVOA-compliant archives
and may be needed in later phases. See data/raw/SOURCE.txt for data provenance.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# Project root is two levels up from this file: <project>/src/frb_anomaly/data.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
CATALOG2_CSV = DATA_RAW / "chimefrbcat2.csv"

# VOTable declares a type per column; these are the ones we read as numbers.
_NUMERIC_VOTABLE_TYPES = {"float", "double", "int", "long", "short", "unsignedByte"}


def _local(tag):
    """Strip the XML namespace so e.g. '{ns}FIELD' matches 'FIELD'."""
    return tag.rsplit("}", 1)[-1]


def read_votable(path):
    """Read a VOTable file into (fields, df).

    fields: list of dicts (name, datatype, unit, description) describing each column.
    df: pandas DataFrame with numeric columns converted from text to numbers and
        empty cells left as missing (NaN).
    """
    root = ET.parse(path).getroot()

    fields = []
    for el in root.iter():
        if _local(el.tag) == "FIELD":
            desc = ""
            for child in el:
                if _local(child.tag) == "DESCRIPTION":
                    desc = (child.text or "").strip()
            fields.append(
                {
                    "name": el.attrib.get("name") or el.attrib.get("ID"),
                    "datatype": el.attrib.get("datatype", ""),
                    "unit": el.attrib.get("unit", ""),
                    "description": desc,
                }
            )

    rows = [
        [td.text for td in tr if _local(td.tag) == "TD"]
        for tr in root.iter()
        if _local(tr.tag) == "TR"
    ]

    df = pd.DataFrame(rows, columns=[f["name"] for f in fields])
    for f in fields:
        if f["datatype"] in _NUMERIC_VOTABLE_TYPES:
            df[f["name"]] = pd.to_numeric(df[f["name"]], errors="coerce")
    return fields, df


def load_catalog2():
    """Load the Catalog 2 burst table as a DataFrame.

    Catalog 2 ships as a plain CSV, so there is no VOTable to hand-parse: pandas
    reads it directly. Unlike load_catalog1 this returns just the DataFrame (a CSV
    carries no per-column field metadata to return alongside it).

    Two things to know before using the result, both handled in the cleaning step,
    not here, so the read stays lossless:
      - One row per burst sub-component (see the `sub_num` column), so the file has
        more rows (~5045) than the catalogue's headline burst count (4539).
      - Quality flags such as `excluded_flag` (e.g. commissioning-period events the
        catalogue itself flags out of population studies), `sidelobe_flag` and
        `intrachan_flag` are kept as-is; filtering on them is the cleaning step's job.

    Missing values are written as the string 'nan' in the file, which pandas reads
    as NaN by default.
    """
    return pd.read_csv(CATALOG2_CSV)
