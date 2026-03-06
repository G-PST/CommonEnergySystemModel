"""Tests for reader modules."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tests.conftest import CESM_SAMPLE_YAML, CESM_SCHEMA_YAML


# ---------------------------------------------------------------------------
# YAML sample file tests (using raw pyyaml since linkml SchemaView may
# have compatibility issues with the current schema)
# ---------------------------------------------------------------------------

class TestYamlSampleLoading:
    """Test that the CESM sample YAML file can be loaded and has expected structure."""

    def test_sample_yaml_exists(self):
        assert CESM_SAMPLE_YAML.exists(), f"Sample YAML not found at {CESM_SAMPLE_YAML}"

    def test_sample_yaml_loads_without_error(self):
        with open(CESM_SAMPLE_YAML, "r") as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert isinstance(data, dict)

    def test_sample_yaml_has_expected_top_level_keys(self):
        with open(CESM_SAMPLE_YAML, "r") as f:
            data = yaml.safe_load(f)
        expected_keys = {"balance", "unit", "commodity", "timeline"}
        assert expected_keys.issubset(set(data.keys())), (
            f"Missing keys: {expected_keys - set(data.keys())}"
        )

    def test_sample_yaml_has_storage(self):
        with open(CESM_SAMPLE_YAML, "r") as f:
            data = yaml.safe_load(f)
        assert "storage" in data

    def test_sample_yaml_has_links(self):
        with open(CESM_SAMPLE_YAML, "r") as f:
            data = yaml.safe_load(f)
        assert "link" in data

    def test_sample_yaml_has_ports(self):
        """node_to_unit and unit_to_node represent port connections."""
        with open(CESM_SAMPLE_YAML, "r") as f:
            data = yaml.safe_load(f)
        assert "node_to_unit" in data or "unit_to_node" in data

    def test_balance_entities_have_names(self):
        with open(CESM_SAMPLE_YAML, "r") as f:
            data = yaml.safe_load(f)
        for entity in data["balance"]:
            assert "name" in entity, "Every balance entity should have a name"

    def test_unit_entities_have_names(self):
        with open(CESM_SAMPLE_YAML, "r") as f:
            data = yaml.safe_load(f)
        for entity in data["unit"]:
            assert "name" in entity, "Every unit entity should have a name"

    def test_timeline_is_list_of_timestamps(self):
        with open(CESM_SAMPLE_YAML, "r") as f:
            data = yaml.safe_load(f)
        timeline = data["timeline"]
        assert isinstance(timeline, list)
        assert len(timeline) > 0
        # Should be parseable as datetime
        parsed = pd.to_datetime(timeline)
        assert len(parsed) == len(timeline)

    def test_balance_flow_profile_matches_timeline_length(self):
        with open(CESM_SAMPLE_YAML, "r") as f:
            data = yaml.safe_load(f)
        timeline_len = len(data["timeline"])
        for entity in data["balance"]:
            if "flow_profile" in entity:
                assert len(entity["flow_profile"]) == timeline_len, (
                    f"Balance '{entity['name']}' flow_profile length "
                    f"{len(entity['flow_profile'])} != timeline length {timeline_len}"
                )


# ---------------------------------------------------------------------------
# LinkML YAML reader tests (may be skipped if SchemaView fails)
# ---------------------------------------------------------------------------

class TestLinkmlYamlReader:
    """Test the linkml_to_dataframes.yaml_to_df function."""

    @pytest.fixture(autouse=True)
    def _check_schema_compat(self):
        """Skip these tests if SchemaView cannot load the schema."""
        try:
            from linkml_runtime.utils.schemaview import SchemaView
            SchemaView(str(CESM_SCHEMA_YAML))
        except Exception:
            pytest.skip("SchemaView cannot load cesm.yaml (linkml compatibility issue)")

    def test_yaml_to_df_returns_dict(self):
        from linkml_runtime.loaders import yaml_loader
        from core.linkml_to_dataframes import yaml_to_df
        from generated.cesm_pydantic import Dataset

        dataset = yaml_loader.load(str(CESM_SAMPLE_YAML), target_class=Dataset)
        result = yaml_to_df(dataset, schema_path=str(CESM_SCHEMA_YAML))
        assert isinstance(result, dict)

    def test_yaml_to_df_contains_expected_keys(self):
        from linkml_runtime.loaders import yaml_loader
        from core.linkml_to_dataframes import yaml_to_df
        from generated.cesm_pydantic import Dataset

        dataset = yaml_loader.load(str(CESM_SAMPLE_YAML), target_class=Dataset)
        result = yaml_to_df(dataset, schema_path=str(CESM_SCHEMA_YAML))
        # Should contain entity class dataframes
        assert "balance" in result or "unit" in result


# ---------------------------------------------------------------------------
# DuckDB reader tests (round-trip tested more thoroughly in test_writers.py)
# ---------------------------------------------------------------------------

class TestDuckdbReader:
    def test_read_nonexistent_raises(self):
        from readers.from_duckdb import dataframes_from_duckdb
        with pytest.raises(FileNotFoundError):
            dataframes_from_duckdb("/nonexistent/path.duckdb")

    def test_read_invalid_db_raises(self, tmp_path):
        """A DuckDB file without metadata should raise ValueError."""
        from readers.from_duckdb import dataframes_from_duckdb
        import duckdb

        db_path = tmp_path / "empty.duckdb"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE TABLE test (a INTEGER)")
        conn.close()

        with pytest.raises(ValueError, match="metadata"):
            dataframes_from_duckdb(str(db_path))

    def test_list_tables_nonexistent_raises(self):
        from readers.from_duckdb import list_tables
        with pytest.raises(FileNotFoundError):
            list_tables("/nonexistent/path.duckdb")
