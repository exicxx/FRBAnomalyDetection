"""Shared pytest fixtures for the frb_anomaly test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest

# pytest 6.x does not support the pythonpath option in pyproject.toml (that
# arrived in 7.0), so we insert src/ explicitly so frb_anomaly is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# A minimal but real VOTable using the actual IVOA namespace.
# xml.etree.ElementTree parses namespaced tags as {ns}TAGNAME, so every
# read_votable test here implicitly verifies that _local() strips namespaces
# correctly -- without it, no FIELD, TR, or TD would ever be found.
_FIXTURE_VOTABLE = """\
<?xml version="1.0" encoding="UTF-8"?>
<VOTABLE version="1.2" xmlns="http://www.ivoa.net/xml/VOTable/v1.2">
  <RESOURCE>
    <TABLE>
      <FIELD name="dm" datatype="float" unit="pc cm-3">
        <DESCRIPTION>Dispersion measure</DESCRIPTION>
      </FIELD>
      <FIELD name="width" datatype="double" unit="ms"/>
      <FIELD name="source" datatype="char"/>
      <DATA>
        <TABLEDATA>
          <TR><TD>123.45</TD><TD>4.56</TD><TD>FRB20121102A</TD></TR>
          <TR><TD></TD><TD>7.89</TD><TD>FRB20180916B</TD></TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
"""

_DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture
def votable_path(tmp_path):
    """Write the minimal fixture VOTable to a temp file and return its path."""
    path = tmp_path / "fixture.vot"
    path.write_text(_FIXTURE_VOTABLE, encoding="utf-8")
    return path


@pytest.fixture
def catalog2_path():
    """Return the real Catalog 2 path, or skip the test if the file is absent."""
    path = _DATA_RAW / "chimefrbcat2.csv"
    if not path.exists():
        pytest.skip("chimefrbcat2.csv not present in data/raw/")
    return path


@pytest.fixture
def rng():
    """A fresh seeded random generator, so generator tests are deterministic."""
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_features():
    """A Gaussian blob of normal points plus a few obvious global outliers.

    The outliers are the LAST n_outliers rows, placed far from the blob, so any working
    distance- or density-based scorer must rank them at the top. Returns (X, n_outliers).
    """
    gen = np.random.default_rng(0)
    normal = gen.normal(0.0, 1.0, size=(200, 6))
    n_outliers = 5
    outliers = gen.normal(0.0, 1.0, size=(n_outliers, 6)) + 15.0
    return np.vstack([normal, outliers]), n_outliers


@pytest.fixture
def calibration():
    """A Calibration built from a small synthetic feature table with the real column names."""
    from frb_anomaly.injection import from_features

    gen = np.random.default_rng(0)
    cols = ["dm_fitb", "width_fitb", "flux", "fluence", "sp_idx", "sp_run", "peak_freq", "bandwidth"]
    X = gen.normal(0.0, 1.0, size=(300, 8))
    return from_features(X, cols)
