"""Tests for writer modules -- DuckDB round-trip write/read verification."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from readers.from_duckdb import dataframes_from_duckdb, list_tables
from writers.to_duckdb import (
    _encode_multiindex_column,
    _flatten_dataframe,
    _get_columns_info,
    _get_index_info,
    dataframes_to_duckdb,
)

# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------

class TestGetIndexInfo:
    def test_single_index(self, entity_df):
        idx_type, idx_count = _get_index_info(entity_df)
        assert idx_type == "single"
        assert idx_count == 1

    def test_datetime_index(self, timeseries_df):
        idx_type, idx_count = _get_index_info(timeseries_df)
        assert idx_type == "datetime"
        assert idx_count == 1

    def test_multi_index(self, multiindex_entity_df):
        idx_type, idx_count = _get_index_info(multiindex_entity_df)
        assert idx_type == "multi"
        assert idx_count == 2


class TestGetColumnsInfo:
    def test_regular_columns(self, entity_df):
        is_multi, levels = _get_columns_info(entity_df)
        assert is_multi is False
        assert levels is None

    def test_multiindex_columns(self, multiindex_column_df):
        is_multi, levels = _get_columns_info(multiindex_column_df)
        assert is_multi is True
        assert levels == ["type", "area", "commodity"]


class TestEncodeMultiindexColumn:
    def test_basic_encoding(self):
        assert _encode_multiindex_column(("a", "b", "c")) == "a::b::c"

    def test_single_level(self):
        assert _encode_multiindex_column(("only",)) == "only"

    def test_numeric_levels(self):
        assert _encode_multiindex_column((1, 2)) == "1::2"


class TestFlattenDataframe:
    def test_single_index_flattened(self, entity_df):
        flat = _flatten_dataframe(entity_df)
        # Index should become a column
        assert "unit" in flat.columns
        assert len(flat) == len(entity_df)

    def test_datetime_index_flattened(self, timeseries_df):
        flat = _flatten_dataframe(timeseries_df)
        assert "datetime" in flat.columns

    def test_multiindex_columns_flattened(self, multiindex_column_df):
        flat = _flatten_dataframe(multiindex_column_df)
        # MultiIndex columns should be encoded with ::
        for col in flat.columns:
            if col != "datetime":
                assert "::" in col


# ---------------------------------------------------------------------------
# DuckDB round-trip tests
# ---------------------------------------------------------------------------

class TestDuckdbRoundTrip:
    def test_entity_df_roundtrip(self, entity_df, tmp_duckdb_path):
        """Write and read back an entity DataFrame, verify equality."""
        dataframes_to_duckdb({"unit": entity_df}, tmp_duckdb_path)
        result = dataframes_from_duckdb(tmp_duckdb_path)

        assert "unit" in result
        pd.testing.assert_frame_equal(result["unit"], entity_df)

    def test_timeseries_df_roundtrip(self, timeseries_df, tmp_duckdb_path):
        """Write and read back a time series DataFrame."""
        dataframes_to_duckdb({"ts": timeseries_df}, tmp_duckdb_path)
        result = dataframes_from_duckdb(tmp_duckdb_path)

        assert "ts" in result
        df = result["ts"]
        # Index should be DatetimeIndex
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "datetime"
        # Values should match
        np.testing.assert_array_almost_equal(df.values, timeseries_df.values)
        assert list(df.columns) == list(timeseries_df.columns)

    def test_multiindex_entity_df_roundtrip(self, multiindex_entity_df, tmp_duckdb_path):
        """Write and read back a MultiIndex entity DataFrame."""
        dataframes_to_duckdb({"mi": multiindex_entity_df}, tmp_duckdb_path)
        result = dataframes_from_duckdb(tmp_duckdb_path)

        assert "mi" in result
        df = result["mi"]
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.nlevels == 2
        pd.testing.assert_frame_equal(df, multiindex_entity_df)

    def test_multiindex_column_roundtrip(self, multiindex_column_df, tmp_duckdb_path):
        """Write and read back a DataFrame with MultiIndex columns."""
        dataframes_to_duckdb({"mic": multiindex_column_df}, tmp_duckdb_path)
        result = dataframes_from_duckdb(tmp_duckdb_path)

        assert "mic" in result
        df = result["mic"]
        assert isinstance(df.columns, pd.MultiIndex)
        assert df.columns.nlevels == 3
        np.testing.assert_array_almost_equal(df.values, multiindex_column_df.values)

    def test_multiple_dataframes_roundtrip(self, sample_dataframes, tmp_duckdb_path):
        """Write and read multiple DataFrames at once."""
        dataframes_to_duckdb(sample_dataframes, tmp_duckdb_path)
        result = dataframes_from_duckdb(tmp_duckdb_path)

        assert set(result.keys()) == set(sample_dataframes.keys())
        for key in sample_dataframes:
            assert result[key].shape == sample_dataframes[key].shape

    def test_list_tables_after_write(self, sample_dataframes, tmp_duckdb_path):
        """Verify list_tables returns the correct table names."""
        dataframes_to_duckdb(sample_dataframes, tmp_duckdb_path)
        tables = list_tables(tmp_duckdb_path)

        assert set(tables) == set(sample_dataframes.keys())

    def test_overwrite_mode(self, entity_df, tmp_duckdb_path):
        """When overwrite=True, old data should be replaced."""
        dataframes_to_duckdb({"unit": entity_df}, tmp_duckdb_path)
        # Write different data with overwrite
        new_df = pd.DataFrame(
            {"x": [1, 2]}, index=pd.Index(["a", "b"], name="name")
        )
        dataframes_to_duckdb({"other": new_df}, tmp_duckdb_path, overwrite=True)
        result = dataframes_from_duckdb(tmp_duckdb_path)
        assert "other" in result
        assert "unit" not in result

    def test_no_overwrite_adds_tables(self, entity_df, tmp_duckdb_path):
        """When overwrite=False, new tables should be added alongside existing."""
        dataframes_to_duckdb({"unit": entity_df}, tmp_duckdb_path)
        new_df = pd.DataFrame(
            {"x": [1, 2]}, index=pd.Index(["a", "b"], name="name")
        )
        dataframes_to_duckdb({"other": new_df}, tmp_duckdb_path, overwrite=False)
        result = dataframes_from_duckdb(tmp_duckdb_path)
        assert "unit" in result
        assert "other" in result

    def test_empty_dataframe_roundtrip(self, tmp_duckdb_path):
        """An empty DataFrame should survive the round-trip."""
        empty_df = pd.DataFrame(columns=["a", "b"])
        empty_df.index.name = "idx"
        dataframes_to_duckdb({"empty": empty_df}, tmp_duckdb_path)
        result = dataframes_from_duckdb(tmp_duckdb_path)
        assert "empty" in result
        assert len(result["empty"]) == 0

    def test_selective_table_read(self, sample_dataframes, tmp_duckdb_path):
        """Reading specific tables should only return those tables."""
        dataframes_to_duckdb(sample_dataframes, tmp_duckdb_path)
        result = dataframes_from_duckdb(tmp_duckdb_path, tables=["unit"])
        assert "unit" in result
        assert len(result) == 1
