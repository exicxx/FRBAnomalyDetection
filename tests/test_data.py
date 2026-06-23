"""Tests for src/frb_anomaly/data.py."""

import pandas as pd
import pytest

from frb_anomaly.data import _local, load_catalog2, read_votable


# ---------------------------------------------------------------------------
# _local()
# ---------------------------------------------------------------------------

def test_local_strips_namespace():
    assert _local("{http://www.ivoa.net/xml/VOTable/v1.2}FIELD") == "FIELD"


def test_local_no_namespace_passthrough():
    assert _local("FIELD") == "FIELD"


# ---------------------------------------------------------------------------
# read_votable() -- all tests use the votable_path fixture (no real data file)
# ---------------------------------------------------------------------------

def test_read_votable_field_count(votable_path):
    fields, _ = read_votable(votable_path)
    assert len(fields) == 3


def test_read_votable_field_names(votable_path):
    fields, _ = read_votable(votable_path)
    assert [f["name"] for f in fields] == ["dm", "width", "source"]


def test_read_votable_field_datatypes(votable_path):
    fields, _ = read_votable(votable_path)
    assert [f["datatype"] for f in fields] == ["float", "double", "char"]


def test_read_votable_field_unit(votable_path):
    fields, _ = read_votable(votable_path)
    assert fields[0]["unit"] == "pc cm-3"


def test_read_votable_field_description(votable_path):
    fields, _ = read_votable(votable_path)
    assert fields[0]["description"] == "Dispersion measure"


def test_read_votable_row_count(votable_path):
    _, df = read_votable(votable_path)
    assert len(df) == 2


def test_read_votable_numeric_columns_are_float(votable_path):
    _, df = read_votable(votable_path)
    assert pd.api.types.is_float_dtype(df["dm"])
    assert pd.api.types.is_float_dtype(df["width"])


def test_read_votable_string_column_stays_string(votable_path):
    _, df = read_votable(votable_path)
    assert pd.api.types.is_object_dtype(df["source"])


def test_read_votable_empty_cell_becomes_nan(votable_path):
    _, df = read_votable(votable_path)
    assert pd.isna(df["dm"].iloc[1])


# ---------------------------------------------------------------------------
# load_catalog2() -- integration tests against the real data file.
# The catalog2_path fixture skips these automatically if the file is absent.
# ---------------------------------------------------------------------------

def test_load_catalog2_columns_present(catalog2_path):
    df = load_catalog2()
    for col in ("tns_name", "sub_num", "dm_fitb", "excluded_flag"):
        assert col in df.columns


def test_load_catalog2_row_count(catalog2_path):
    df = load_catalog2()
    assert 5000 <= len(df) <= 5100


def test_load_catalog2_dm_is_numeric(catalog2_path):
    df = load_catalog2()
    assert pd.api.types.is_float_dtype(df["dm_fitb"])
