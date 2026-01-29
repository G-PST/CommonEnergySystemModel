"""
Script to read GridDB SQLite database and convert it to CESM format DataFrames.

This is the reverse operation of read_yaml_to_griddb.py.
It reads a GridDB SQLite database and produces CESM-format DataFrames
matching the structure produced by linkml_to_dataframes.yaml_to_df().

Usage:
    python griddb_to_cesm.py <input_griddb> <output_duckdb>

Example:
    python griddb_to_cesm.py data/griddb.sqlite artifacts/cesm_from_griddb.duckdb
"""

import argparse
import sys
from os import path

# Add project paths
sys.path.insert(0, path.abspath('src'))
sys.path.insert(0, path.abspath('scripts'))

from transformers.griddb import to_cesm
from helpers.print_dataframe_structures import print_dataframes_structure
from writers.to_duckdb import dataframes_to_duckdb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert GridDB SQLite database to CESM format DataFrames and save as DuckDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python griddb_to_cesm.py data/griddb.sqlite artifacts/cesm_from_griddb.duckdb
    python griddb_to_cesm.py ../data/rts_psy5.sqlite output/cesm.duckdb
        """
    )
    parser.add_argument(
        "input_griddb",
        help="Path to the input GridDB SQLite database file"
    )
    parser.add_argument(
        "output_duckdb",
        help="Path for the output DuckDB file"
    )
    parser.add_argument(
        "--clear-target-db",
        action="store_true",
        default=False,
        help="If set, clear the target database before writing. "
             "Otherwise, add/replace tables from the input (default: False)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not path.isfile(args.input_griddb):
        parser.error(f"Input file not found: {args.input_griddb}")

    return args


def main():
    args = parse_args()

    # Transform GridDB to CESM DataFrames
    print(f"Reading GridDB from: {args.input_griddb}")
    cesm_dfs = to_cesm("cesm_v0.1.0", "v0.2.0", args.input_griddb)

    # Write to DuckDB
    print(f"\nWriting DataFrames to DuckDB: {args.output_duckdb}")
    dataframes_to_duckdb(cesm_dfs, args.output_duckdb, overwrite=args.clear_target_db)

    # Print summary
    print("\n" + "=" * 80)
    print("CESM DATAFRAMES SUMMARY")
    print("=" * 80)
    for name, df in sorted(cesm_dfs.items()):
        if hasattr(df, 'shape'):
            print(f"  {name}: {df.shape}")
        else:
            print(f"  {name}: {type(df)}")

    print("\nDone reading")


if __name__ == "__main__":
    main()
