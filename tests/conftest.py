"""Shared fixtures for oes-spec tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --- Path constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
MODEL_DIR = PROJECT_ROOT / "model"

CESM_SAMPLE_YAML = DATA_SAMPLES_DIR / "cesm-sample.yaml"
CESM_SCHEMA_YAML = MODEL_DIR / "cesm_v0.1.0.yaml"


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def entity_df():
    """Create a sample entity DataFrame (e.g., units with scalar parameters)."""
    data = {
        "efficiency": [38.0, 58.0, 34.0],
        "discount_rate": [6.0, 6.0, 8.0],
        "payback_time": [25, 30, 40],
    }
    df = pd.DataFrame(data, index=pd.Index(["ocgt", "ccgt", "nuclear"], name="unit"))
    return df


@pytest.fixture
def timeseries_df():
    """Create a sample time series DataFrame (entities in columns, datetime index)."""
    dates = pd.date_range("2023-01-01", periods=5, freq="h")
    data = {
        "ocgt": [100.0, 110.0, 105.0, 120.0, 115.0],
        "ccgt": [200.0, 210.0, 215.0, 220.0, 230.0],
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "datetime"
    return df


@pytest.fixture
def multiindex_entity_df():
    """Create a sample DataFrame with MultiIndex rows."""
    idx = pd.MultiIndex.from_tuples(
        [("ocgt", "west"), ("ccgt", "east"), ("nuclear", "west")],
        names=["unit", "node"],
    )
    data = {
        "capacity": [50.0, 500.0, 800.0],
        "investment_cost": [500.0, 1200.0, 5500.0],
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def multiindex_column_df():
    """Create a sample DataFrame with MultiIndex columns."""
    dates = pd.date_range("2023-01-01", periods=3, freq="h")
    col_tuples = [("region", "north", "heat"), ("region", "south", "elec")]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["type", "area", "commodity"])
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    df = pd.DataFrame(data, index=dates, columns=columns)
    df.index.name = "datetime"
    return df


@pytest.fixture
def tmp_duckdb_path(tmp_path):
    """Return a temporary path for a DuckDB file."""
    return str(tmp_path / "test_output.duckdb")


@pytest.fixture
def sample_dataframes(entity_df, timeseries_df, multiindex_entity_df):
    """Return a dictionary of sample DataFrames, simulating a typical CESM dataset."""
    return {
        "unit": entity_df,
        "unit.ts.flow_profile": timeseries_df,
        "unit_to_node": multiindex_entity_df,
    }
