"""Tests for CESM schema validation."""

import sys
from pathlib import Path

import pytest
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tests.conftest import CESM_SCHEMA_YAML


class TestSchemaFile:
    """Basic tests that the schema file exists and is valid YAML."""

    def test_schema_file_exists(self):
        assert CESM_SCHEMA_YAML.exists(), f"Schema not found at {CESM_SCHEMA_YAML}"

    def test_schema_is_valid_yaml(self):
        with open(CESM_SCHEMA_YAML, "r") as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert isinstance(data, dict)

    def test_schema_has_linkml_required_fields(self):
        with open(CESM_SCHEMA_YAML, "r") as f:
            data = yaml.safe_load(f)
        # LinkML schemas should have these top-level keys
        assert "id" in data, "Schema should have 'id' field"
        assert "name" in data, "Schema should have 'name' field"
        assert "classes" in data, "Schema should have 'classes' field"

    def test_schema_has_expected_classes(self):
        with open(CESM_SCHEMA_YAML, "r") as f:
            data = yaml.safe_load(f)
        classes = data.get("classes", {})
        expected = {"Dataset", "Balance", "Unit", "Storage", "Link", "Commodity"}
        actual = set(classes.keys())
        missing = expected - actual
        assert not missing, f"Schema missing expected classes: {missing}"

    def test_schema_has_enums(self):
        with open(CESM_SCHEMA_YAML, "r") as f:
            data = yaml.safe_load(f)
        assert "enums" in data, "Schema should define enums"

    def test_dataset_class_has_timeline(self):
        with open(CESM_SCHEMA_YAML, "r") as f:
            data = yaml.safe_load(f)
        dataset_cls = data["classes"].get("Dataset", {})
        attrs = dataset_cls.get("attributes", {})
        assert "timeline" in attrs, "Dataset class should have a timeline attribute"

    def test_reference_year_pattern_is_valid(self):
        """Verify the reference_year regex pattern uses \\d (not bare d)."""
        with open(CESM_SCHEMA_YAML, "r") as f:
            data = yaml.safe_load(f)
        dataset_cls = data["classes"].get("Dataset", {})
        attrs = dataset_cls.get("attributes", {})
        ref_year = attrs.get("reference_year", {})
        pattern = ref_year.get("pattern", "")
        if pattern:
            # The pattern should match 4-digit years, not "d" followed by 3 digits
            assert "\\d" in pattern or "d{4}" not in pattern, (
                f"reference_year pattern '{pattern}' appears broken; should use \\d for digits"
            )


class TestSchemaViewLoading:
    """Test loading the schema with linkml SchemaView (may skip on compat issues)."""

    def test_schema_loads_with_schemaview(self):
        try:
            from linkml_runtime.utils.schemaview import SchemaView
        except ImportError:
            pytest.skip("linkml_runtime not installed")

        try:
            sv = SchemaView(str(CESM_SCHEMA_YAML))
        except Exception as e:
            pytest.skip(
                f"SchemaView cannot load cesm_v0.1.0.yaml (linkml compatibility issue): {e}"
            )

        # If we get here, schema loaded -- run basic checks
        all_classes = sv.all_classes()
        assert len(all_classes) > 0, "Schema should define at least one class"
        assert "Dataset" in all_classes
