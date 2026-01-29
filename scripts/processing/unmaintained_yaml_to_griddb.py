"""
Read CESM YAML data and write to GridDB SQLite format.

This script loads a CESM YAML file, converts it to DataFrames,
transforms to GridDB format, and writes to a SQLite database.

Usage:
    python read_yaml_to_griddb.py <input_yaml> <output_db>
    python read_yaml_to_griddb.py <input_yaml> <output_db> --clear-target-db
"""

import argparse
import pandas as pd
from generated.cesm_pydantic import Dataset
from linkml_runtime.loaders import yaml_loader
from core.linkml_to_dataframes import yaml_to_df
from transformers.griddb import to_griddb
from transformers.griddb.to_sqlite import write_to_sqlite
from os import path
import sys
sys.path.append(path.abspath('scripts'))
from helpers.print_dataframe_structures import print_dataframes_structure


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CESM YAML data to GridDB SQLite format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python read_yaml_to_griddb.py data/samples/cesm-sample.yaml griddb.sqlite
    python read_yaml_to_griddb.py data/samples/cesm-sample.yaml griddb.sqlite --clear-target-db
        """
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input YAML file path"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output SQLite database file path"
    )
    parser.add_argument(
        "--schema-path",
        "-s",
        type=str,
        default="model/cesm.yaml",
        help="CESM schema path (default: model/cesm.yaml)"
    )
    parser.add_argument(
        "--griddb-schema",
        "-g",
        type=str,
        default="src/transformers/griddb/cesm_v0.1.0/v0.2.0/schema.sql",
        help="GridDB SQL schema path (default: src/transformers/griddb/cesm_v0.1.0/v0.2.0/schema.sql)"
    )
    parser.add_argument(
        "--clear-target-db",
        action="store_true",
        default=False,
        help="If set, clear the target database before writing. "
             "Otherwise, add/replace data from the YAML file (default: False). "
             "When not clearing, data checks are relaxed (e.g., timeline not required "
             "if already in database)."
    )

    args = parser.parse_args()

    # Validate input file exists
    if not path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    return args


def main():
    args = parse_args()

    # Load YAML data
    print(f"Loading YAML from {args.input}...")
    dataset = yaml_loader.load(args.input, target_class=Dataset)

    # Extract all DataFrames
    print("Converting to DataFrames...")
    cesm_dfs = yaml_to_df(dataset, schema_path=args.schema_path)

    # Process the transformation
    # When not clearing database, use relaxed validation (strict=False)
    print("Transforming to GridDB format...")
    strict_validation = args.clear_target_db  # Only strict when clearing DB
    griddb_dfs = to_griddb("cesm_v0.1.0", "v0.2.0", cesm_dfs, strict=strict_validation)

    # Optionally print structure for debugging
    # print_dataframes_structure(griddb_dfs, output_file='data_structure_griddb.txt')

    # Write to SQLite (GridDB format)
    print(f"Writing to SQLite: {args.output}...")
    write_to_sqlite(
        args.griddb_schema,
        griddb_dfs,
        args.output,
        clear_existing=args.clear_target_db
    )

    print("Done")


if __name__ == "__main__":
    main()
