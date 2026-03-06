"""Validate the CESM LinkML schema file.

Usage:
    python scripts/validation/validate_linkml_schema.py [path/to/schema.yaml]

If no path is given, defaults to model/cesm.yaml relative to the project root.
Exits with code 0 on success, 1 on validation failure.
"""

import sys
from pathlib import Path

import yaml


def validate_schema(schema_path: Path) -> bool:
    """Validate a LinkML schema file.

    Checks:
    1. File exists and is valid YAML.
    2. Required top-level LinkML keys are present (id, name, classes).
    3. Schema loads successfully with linkml SchemaView.

    Returns True if all checks pass, False otherwise.
    """
    errors = []

    # Check file exists
    if not schema_path.exists():
        print(f"ERROR: Schema file not found: {schema_path}")
        return False

    # Check valid YAML
    try:
        with open(schema_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in {schema_path}: {e}")
        return False

    if not isinstance(data, dict):
        print(f"ERROR: Schema root must be a mapping, got {type(data).__name__}")
        return False

    # Check required LinkML fields
    for field in ("id", "name", "classes"):
        if field not in data:
            errors.append(f"Missing required top-level field: '{field}'")

    # Check classes are non-empty
    classes = data.get("classes", {})
    if not classes:
        errors.append("Schema defines no classes")

    # Check expected core classes exist
    expected_classes = {"Dataset", "Balance", "Unit", "Storage", "Link", "Commodity"}
    actual_classes = set(classes.keys()) if isinstance(classes, dict) else set()
    missing_classes = expected_classes - actual_classes
    if missing_classes:
        errors.append(f"Missing expected classes: {missing_classes}")

    # Check enums exist
    if "enums" not in data:
        errors.append("Schema defines no enums")

    # Try loading with SchemaView for deeper validation
    try:
        from linkml_runtime.utils.schemaview import SchemaView

        sv = SchemaView(str(schema_path))
        all_classes = sv.all_classes()
        if len(all_classes) == 0:
            errors.append("SchemaView loaded but found no classes")
        else:
            print(f"SchemaView loaded successfully: {len(all_classes)} classes found")
    except ImportError:
        print("WARNING: linkml_runtime not installed, skipping SchemaView validation")
    except Exception as e:
        errors.append(f"SchemaView failed to load schema: {e}")

    # Report results
    if errors:
        print(f"VALIDATION FAILED for {schema_path}:")
        for err in errors:
            print(f"  - {err}")
        return False

    print(f"VALIDATION PASSED: {schema_path}")
    return True


def main() -> int:
    if len(sys.argv) > 1:
        schema_path = Path(sys.argv[1])
    else:
        # Default to model/cesm.yaml relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        schema_path = project_root / "model" / "cesm.yaml"

    success = validate_schema(schema_path)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
