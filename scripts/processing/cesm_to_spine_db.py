"""
Read CESM data from DuckDB and write to Spine database.

This script reads CESM DataFrames from a DuckDB file and writes them
directly to a Spine database.
"""

import argparse
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


def main():
    """Main function to convert CESM DuckDB data to Spine database."""
    parser = argparse.ArgumentParser(
        description="Convert CESM DuckDB data to Spine database format"
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

    args = parser.parse_args()

    # Load CESM data from DuckDB
    print(f"Loading CESM data from {args.input}...")
    cesm = dataframes_from_duckdb(args.input)

    # Write CESM dataset to Spine DB
    output_url = path_to_sqlite_url(args.output)
    print(f"Writing to Spine database: {output_url}...")
    dataframes_to_spine(cesm, output_url)

    print(f"Data successfully written to {args.output}")


if __name__ == "__main__":
    main()
