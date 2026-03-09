"""
Read CESM data from DuckDB and write to FlexTool Spine database.

This script reads CESM DataFrames from a DuckDB file, transforms them
to FlexTool format, and writes to a Spine database.
"""

import argparse
import importlib.util
from pathlib import Path

from readers.from_duckdb import dataframes_from_duckdb
from writers.to_spine_db import dataframes_to_spine


def path_to_sqlite_url(path_or_url: str) -> str:
    """Convert file path to SQLite URL if needed.

    Passes through URLs that are already in a valid format (sqlite:, http:, https:).
    Converts plain file paths to sqlite:/// URLs.
    """
    if path_or_url.startswith(("sqlite:", "http:", "https:")):
        return path_or_url
    return f"sqlite:///{Path(path_or_url).resolve()}"


def load_transformer_module(cesm_version: str, flextool_version: str):
    """
    Dynamically load the Python transformer module for the specified versions.

    Args:
        cesm_version: CESM version (e.g., 'cesm_v0.1.0')
        flextool_version: FlexTool version (e.g., 'v3.14.0')

    Returns:
        The loaded module containing transform_to_flextool function
    """
    module_path = Path(f"src/transformers/irena_flextool/{cesm_version}/{flextool_version}/to_flextool.py")

    if not module_path.exists():
        raise FileNotFoundError(
            f"Transformer module not found: {module_path}\n"
            f"Ensure the path exists for versions {cesm_version}/{flextool_version}"
        )

    spec = importlib.util.spec_from_file_location("to_flextool", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Main function to convert CESM DuckDB data to FlexTool Spine database."""
    parser = argparse.ArgumentParser(
        description="Convert CESM DuckDB data to FlexTool Spine database format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input DuckDB file path"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output Spine database file path or URL (e.g., output.sqlite or sqlite:///output.sqlite)"
    )
    parser.add_argument(
        "--cesm-version",
        "-c",
        type=str,
        default="cesm_v0.1.0",
        help="CESM version (default: cesm_v0.1.0)"
    )
    parser.add_argument(
        "--flextool-version",
        "-f",
        type=str,
        default="v3.14.0",
        help="FlexTool version (default: v3.14.0)"
    )

    args = parser.parse_args()

    # Load CESM data from DuckDB
    print(f"Loading CESM data from {args.input}...")
    cesm = dataframes_from_duckdb(args.input)

    # Determine transformer YAML config path
    transformer_config = (
        f"src/transformers/irena_flextool/{args.cesm_version}/{args.flextool_version}/to_flextool.yaml"
    )

    # Load and apply Python transformer
    print(f"Transforming to FlexTool format...")
    print(f"  Transformer versions: {args.cesm_version} / {args.flextool_version}")
    try:
        transformer_module = load_transformer_module(args.cesm_version, args.flextool_version)
        flextool = transformer_module.transform_to_flextool(cesm, transformer_config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error in transformer: {e}")
        raise SystemExit(1)

    # Write FlexTool dataset to Spine DB (FlexTool format)
    output_url = path_to_sqlite_url(args.output)
    print(f"Writing to Spine database: {output_url}...")
    dataframes_to_spine(flextool, output_url)

    print(f"Data successfully written to {args.output}")


if __name__ == "__main__":
    main()
