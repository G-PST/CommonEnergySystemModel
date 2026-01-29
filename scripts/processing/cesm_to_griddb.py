"""
Read CESM data from DuckDB and write to GridDB SQLite database.

This script reads CESM DataFrames from a DuckDB file, transforms them
to GridDB format, and writes to a SQLite database.
"""

import argparse
from readers.from_duckdb import dataframes_from_duckdb
from transformers.griddb import to_griddb
from transformers.griddb.to_sqlite import write_to_sqlite


def main():
    """Main function to convert CESM DuckDB data to GridDB SQLite database."""
    parser = argparse.ArgumentParser(
        description="Convert CESM DuckDB data to GridDB SQLite format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input DuckDB file path"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output SQLite file path"
    )
    parser.add_argument(
        "--cesm-version",
        "-c",
        type=str,
        default="cesm_v0.1.0",
        help="CESM version (default: cesm_v0.1.0)"
    )
    parser.add_argument(
        "--griddb-version",
        "-g",
        type=str,
        default="v0.2.0",
        help="GridDB version (default: v0.2.0)"
    )
    parser.add_argument(
        "--schema",
        "-s",
        type=str,
        default=None,
        help="GridDB SQL schema path (default: auto-detected from versions)"
    )

    args = parser.parse_args()

    # Determine schema path
    if args.schema:
        schema_path = args.schema
    else:
        schema_path = f"src/transformers/griddb/{args.cesm_version}/{args.griddb_version}/schema.sql"

    # Load CESM data from DuckDB
    print(f"Loading CESM data from {args.input}...")
    cesm_dfs = dataframes_from_duckdb(args.input)

    # Process the transformation
    print(f"Transforming to GridDB format ({args.cesm_version} -> {args.griddb_version})...")
    griddb_dfs = to_griddb(args.cesm_version, args.griddb_version, cesm_dfs)

    # Write to SQLite (GridDB format)
    print(f"Writing to SQLite: {args.output}...")
    write_to_sqlite(schema_path, griddb_dfs, args.output)

    print(f"Data successfully written to {args.output}")


if __name__ == "__main__":
    main()
