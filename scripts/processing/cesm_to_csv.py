"""
Read CESM data from DuckDB and write to CSV files.

This script reads CESM DataFrames from a DuckDB file and writes them
as a directory of CSV files, one per table.
"""

import argparse

from readers.from_duckdb import dataframes_from_duckdb
from writers.to_csv import dataframes_to_csv


def main():
    """Main function to convert CESM DuckDB data to CSV files."""
    parser = argparse.ArgumentParser(
        description="Convert CESM DuckDB data to CSV format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input DuckDB file path"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output directory path for CSV files"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        default=False,
        help="If set, add/replace CSV files in existing directory "
             "instead of clearing it first (default: overwrite)"
    )

    args = parser.parse_args()

    # Load CESM data from DuckDB
    print(f"Loading CESM data from {args.input}...")
    cesm = dataframes_from_duckdb(args.input)

    # Write to CSV
    overwrite = not args.no_overwrite
    print(f"Writing CSV files to {args.output}...")
    dataframes_to_csv(cesm, args.output, overwrite=overwrite)

    print(f"Data successfully written to {args.output}")


if __name__ == "__main__":
    main()
