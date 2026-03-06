"""Unit tests for src/core/transform_parameters.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path so we can import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.transform_parameters import (
    _filter_entities_by_parameters,
    get_operation_type,
    index_to_names,
    is_timeseries,
    list_of_lists_to_index,
    load_config,
    parse_spec,
)


# ---------------------------------------------------------------------------
# list_of_lists_to_index
# ---------------------------------------------------------------------------

class TestListOfListsToIndex:
    def test_empty_input(self):
        result = list_of_lists_to_index([])
        assert isinstance(result, pd.Index)
        assert len(result) == 0

    def test_single_dimension(self):
        data = [["a"], ["b"], ["c"]]
        result = list_of_lists_to_index(data)
        assert isinstance(result, pd.Index)
        assert not isinstance(result, pd.MultiIndex)
        assert list(result) == ["a", "b", "c"]

    def test_multi_dimension(self):
        data = [["a", "x"], ["b", "y"], ["c", "z"]]
        result = list_of_lists_to_index(data)
        assert isinstance(result, pd.MultiIndex)
        assert len(result) == 3
        assert result[0] == ("a", "x")
        assert result[1] == ("b", "y")
        assert result[2] == ("c", "z")

    def test_three_dimensions(self):
        data = [["a", "x", "1"], ["b", "y", "2"]]
        result = list_of_lists_to_index(data)
        assert isinstance(result, pd.MultiIndex)
        assert result.nlevels == 3
        assert result[0] == ("a", "x", "1")

    def test_single_element(self):
        data = [["only"]]
        result = list_of_lists_to_index(data)
        assert isinstance(result, pd.Index)
        assert list(result) == ["only"]

    def test_numeric_values(self):
        data = [[1], [2], [3]]
        result = list_of_lists_to_index(data)
        assert list(result) == [1, 2, 3]


# ---------------------------------------------------------------------------
# index_to_names
# ---------------------------------------------------------------------------

class TestIndexToNames:
    def test_simple_index(self):
        idx = pd.Index(["a", "b", "c"])
        result = index_to_names(idx)
        assert result == [["a"], ["b"], ["c"]]

    def test_multi_index(self):
        idx = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")])
        result = index_to_names(idx)
        assert result == [["a", "x"], ["b", "y"]]

    def test_empty_index(self):
        idx = pd.Index([])
        result = index_to_names(idx)
        assert result == []

    def test_roundtrip_single(self):
        """list_of_lists_to_index and index_to_names should round-trip."""
        original = [["a"], ["b"], ["c"]]
        idx = list_of_lists_to_index(original)
        roundtripped = index_to_names(idx)
        assert roundtripped == original

    def test_roundtrip_multi(self):
        """list_of_lists_to_index and index_to_names should round-trip."""
        original = [["a", "x"], ["b", "y"]]
        idx = list_of_lists_to_index(original)
        roundtripped = index_to_names(idx)
        assert roundtripped == original


# ---------------------------------------------------------------------------
# parse_spec
# ---------------------------------------------------------------------------

class TestParseSpec:
    def test_string_spec(self):
        result = parse_spec("unit")
        assert result == [{"class": "unit", "attribute": None}]

    def test_dict_spec_with_attribute(self):
        result = parse_spec({"unit": "efficiency"})
        assert result == [{"class": "unit", "attribute": "efficiency"}]

    def test_dict_spec_with_rule(self):
        result = parse_spec({"unit": {"if_parameter": "efficiency"}})
        assert len(result) == 1
        assert result[0]["class"] == "unit"
        assert result[0]["attribute"] is None
        assert result[0]["rule"] == {"if_parameter": "efficiency"}

    def test_list_spec(self):
        result = parse_spec(["unit", "storage"])
        assert len(result) == 2
        assert result[0] == {"class": "unit", "attribute": None}
        assert result[1] == {"class": "storage", "attribute": None}

    def test_nested_list_spec(self):
        result = parse_spec([{"unit": "efficiency"}, {"storage": "capacity"}])
        assert len(result) == 2
        assert result[0] == {"class": "unit", "attribute": "efficiency"}
        assert result[1] == {"class": "storage", "attribute": "capacity"}

    def test_empty_list(self):
        result = parse_spec([])
        assert result == []


# ---------------------------------------------------------------------------
# get_operation_type
# ---------------------------------------------------------------------------

class TestGetOperationType:
    def test_copy_entities(self):
        source = [{"class": "unit", "attribute": None}]
        target = [{"class": "process", "attribute": None}]
        assert get_operation_type(source, target, []) == "copy_entities"

    def test_create_parameter(self):
        source = [{"class": "unit", "attribute": None}]
        target = [{"class": "process", "attribute": "type"}]
        operations = [{"value": "conversion"}]
        assert get_operation_type(source, target, operations) == "create_parameter"

    def test_transform_parameter(self):
        source = [{"class": "unit", "attribute": "efficiency"}]
        target = [{"class": "process", "attribute": "eff"}]
        assert get_operation_type(source, target, []) == "transform_parameter"

    def test_unknown(self):
        source = [{"class": "unit", "attribute": None}]
        target = [{"class": "process", "attribute": "type"}]
        assert get_operation_type(source, target, []) == "unknown"


# ---------------------------------------------------------------------------
# is_timeseries
# ---------------------------------------------------------------------------

class TestIsTimeseries:
    def test_datetime_index_name(self):
        df = pd.DataFrame({"a": [1, 2]}, index=pd.Index([0, 1], name="datetime"))
        assert is_timeseries(df) is True

    def test_regular_index(self):
        df = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="unit"))
        assert is_timeseries(df) is False

    def test_multiindex_with_datetime(self):
        idx = pd.MultiIndex.from_tuples(
            [(1, "a"), (2, "b")], names=["datetime", "entity"]
        )
        df = pd.DataFrame({"val": [10, 20]}, index=idx)
        assert is_timeseries(df) is True

    def test_multiindex_without_datetime(self):
        idx = pd.MultiIndex.from_tuples(
            [(1, "a"), (2, "b")], names=["time", "entity"]
        )
        df = pd.DataFrame({"val": [10, 20]}, index=idx)
        assert is_timeseries(df) is False


# ---------------------------------------------------------------------------
# _filter_entities_by_parameters
# ---------------------------------------------------------------------------

class TestFilterEntitiesByParameters:
    def test_no_filters_returns_all(self):
        """When no if_params or if_not_params given, all entities are returned."""
        names = [["a"], ["b"], ["c"]]
        result = _filter_entities_by_parameters(
            entity_names=names,
            source_dfs={},
            base_class="unit",
        )
        assert result == names

    def test_if_parameter_filters_to_matching(self):
        """Only entities that have the parameter should be kept."""
        base_df = pd.DataFrame(
            {"efficiency": [38.0, None, 34.0]},
            index=pd.Index(["ocgt", "wind", "nuclear"], name="unit"),
        )
        names = [["ocgt"], ["wind"], ["nuclear"]]
        result = _filter_entities_by_parameters(
            entity_names=names,
            source_dfs={"unit": base_df},
            base_class="unit",
            base_df=base_df,
            if_params=["efficiency"],
        )
        # "wind" has NaN efficiency, so should be excluded
        assert result == [["ocgt"], ["nuclear"]]

    def test_if_not_parameter_excludes_matching(self):
        """Entities that have the parameter should be excluded."""
        base_df = pd.DataFrame(
            {"startup_method": ["linear", None, None]},
            index=pd.Index(["ocgt", "ccgt", "nuclear"], name="unit"),
        )
        names = [["ocgt"], ["ccgt"], ["nuclear"]]
        result = _filter_entities_by_parameters(
            entity_names=names,
            source_dfs={"unit": base_df},
            base_class="unit",
            base_df=base_df,
            if_not_params=["startup_method"],
        )
        # "ocgt" has startup_method, so should be excluded
        assert result == [["ccgt"], ["nuclear"]]

    def test_if_parameter_with_pivoted_data(self):
        """Test filtering based on presence in a pivoted (time series) dataframe."""
        # Entity "ocgt" appears as a column in the ts dataframe
        ts_df = pd.DataFrame(
            {"ocgt": [1.0, 2.0], "nuclear": [3.0, 4.0]},
            index=pd.DatetimeIndex(["2023-01-01", "2023-01-02"]),
        )
        ts_df.index.name = "datetime"
        source_dfs = {
            "unit": pd.DataFrame(index=pd.Index(["ocgt", "ccgt", "nuclear"], name="unit")),
            "unit.ts.flow_profile": ts_df,
        }
        names = [["ocgt"], ["ccgt"], ["nuclear"]]
        result = _filter_entities_by_parameters(
            entity_names=names,
            source_dfs=source_dfs,
            base_class="unit",
            if_params=["flow_profile"],
        )
        # Only ocgt and nuclear are in the ts columns
        assert result == [["ocgt"], ["nuclear"]]

    def test_string_entity_names(self):
        """Test with simple string entity names instead of lists."""
        base_df = pd.DataFrame(
            {"efficiency": [38.0, None]},
            index=pd.Index(["ocgt", "wind"], name="unit"),
        )
        names = ["ocgt", "wind"]
        result = _filter_entities_by_parameters(
            entity_names=names,
            source_dfs={"unit": base_df},
            base_class="unit",
            base_df=base_df,
            if_params=["efficiency"],
        )
        assert result == ["ocgt"]

    def test_combined_if_and_if_not(self):
        """Test with both if_params and if_not_params."""
        base_df = pd.DataFrame(
            {
                "efficiency": [38.0, 58.0, None],
                "startup_method": ["linear", None, None],
            },
            index=pd.Index(["ocgt", "ccgt", "nuclear"], name="unit"),
        )
        names = [["ocgt"], ["ccgt"], ["nuclear"]]
        result = _filter_entities_by_parameters(
            entity_names=names,
            source_dfs={"unit": base_df},
            base_class="unit",
            base_df=base_df,
            if_params=["efficiency"],
            if_not_params=["startup_method"],
        )
        # Only ccgt: has efficiency but no startup_method
        assert result == [["ccgt"]]


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("key1: value1\nkey2:\n  - item1\n  - item2\n")
        result = load_config(str(config_file))
        assert result == {"key1": "value1", "key2": ["item1", "item2"]}

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# transform_data (integration-level test with a minimal config)
# ---------------------------------------------------------------------------

class TestTransformData:
    def test_simple_copy_entities(self, tmp_path):
        """Test transform_data with a minimal copy_entities config."""
        from core.transform_parameters import transform_data

        # Create source dataframes
        source_dfs = {
            "unit": pd.DataFrame(
                {"efficiency": [38.0, 58.0]},
                index=pd.Index(["ocgt", "ccgt"], name="unit"),
            ),
        }

        # Create a minimal YAML config for copy_entities
        config_content = (
            "copy units:\n"
            "  - unit\n"
            "  - process\n"
        )
        config_path = tmp_path / "test_transform.yaml"
        config_path.write_text(config_content)

        result = transform_data(source_dfs, str(config_path))

        assert "process" in result
        assert len(result["process"]) == 2
        assert "ocgt" in result["process"].index
        assert "ccgt" in result["process"].index

    def test_simple_create_parameter(self, tmp_path):
        """Test transform_data with a create_parameter config."""
        from core.transform_parameters import transform_data

        source_dfs = {
            "unit": pd.DataFrame(
                {"efficiency": [38.0, 58.0]},
                index=pd.Index(["ocgt", "ccgt"], name="unit"),
            ),
        }

        config_content = (
            "create type:\n"
            "  - unit\n"
            "  - {process: type}\n"
            "  - {value: conversion}\n"
        )
        config_path = tmp_path / "test_create.yaml"
        config_path.write_text(config_content)

        result = transform_data(source_dfs, str(config_path))

        assert "process" in result
        assert "type" in result["process"].columns
        # The value "conversion" should be assigned to all entities
        assert (result["process"]["type"] == "conversion").all()
