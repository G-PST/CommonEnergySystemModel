"""
Read CESM YAML data and write to DuckDB.

This script loads a CESM YAML file, converts it to DataFrames,
and writes them to a DuckDB database file.
"""

import argparse
from generated.cesm_pydantic import Dataset
from linkml_runtime.loaders import yaml_loader
from core.linkml_to_dataframes import yaml_to_df
from writers.to_duckdb import dataframes_to_duckdb


def main():
    """Main function to convert CESM YAML data to DuckDB."""
    parser = argparse.ArgumentParser(
        description="Convert CESM YAML data to DuckDB format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input YAML file path"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output DuckDB file path"
    )
    parser.add_argument(
        "--schema",
        "-s",
        type=str,
        default="model/cesm.yaml",
        help="CESM schema path (default: model/cesm.yaml)"
    )
    parser.add_argument(
        "--clear-target-db",
        action="store_true",
        default=False,
        help="If set, clear the target database before writing. "
             "Otherwise, add/replace tables from the YAML file (default: False). "
             "When not clearing, data checks are relaxed (e.g., timeline not required "
             "if already in database)."
    )

    args = parser.parse_args()

    # Load YAML data
    print(f"Loading YAML from {args.input}...")
    dataset = yaml_loader.load(args.input, target_class=Dataset)

    # Extract all DataFrames
    # When not clearing database, use relaxed validation (strict=False)
    print("Converting to DataFrames...")
    strict_validation = args.clear_target_db  # Only strict when clearing DB
    cesm = yaml_to_df(dataset, schema_path=args.schema, strict=strict_validation)

    # Write to DuckDB
    print(f"Writing to DuckDB: {args.output}...")
    dataframes_to_duckdb(cesm, args.output, overwrite=args.clear_target_db)

    print(f"Data successfully written to {args.output}")


if __name__ == "__main__":
    main()
