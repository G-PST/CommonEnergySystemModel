"""
DuckDB writer for CESM DataFrames.

Writes a dictionary of pandas DataFrames to a DuckDB database file,
preserving structure metadata so the DataFrames can be reconstructed
exactly by the corresponding reader (readers/from_duckdb.py).

Metadata schema (simplified):
- index_type: 'single', 'datetime', or 'multi'
- index_count: number of index columns (stored first in table)
- columns_multiindex: boolean indicating if columns are MultiIndex
- columns_levels: JSON array of level names if MultiIndex columns

MultiIndex columns use :: separator: ("region", "north", "heat") -> "region::north::heat"

Usage:
    from writers.to_duckdb import dataframes_to_duckdb
    dataframes_to_duckdb(dataframes_dict, "output.duckdb")
"""

import duckdb
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def _get_index_info(df: pd.DataFrame) -> Tuple[str, int]:
    """
    Extract index type and count from DataFrame.

    Returns:
        Tuple of (index_type, index_count) where:
        - index_type: 'single', 'datetime', or 'multi'
        - index_count: number of index columns
    """
    index = df.index

    if isinstance(index, pd.MultiIndex):
        return ('multi', index.nlevels)
    elif isinstance(index, pd.DatetimeIndex):
        return ('datetime', 1)
    else:
        return ('single', 1)


def _get_columns_info(df: pd.DataFrame) -> Tuple[bool, Optional[List[str]]]:
    """
    Extract column structure info from DataFrame.

    Returns:
        Tuple of (is_multiindex, level_names) where:
        - is_multiindex: True if columns are MultiIndex
        - level_names: list of level names if MultiIndex, None otherwise
    """
    columns = df.columns

    if isinstance(columns, pd.MultiIndex):
        return (True, list(columns.names))
    else:
        return (False, None)


def _normalize_datetime_index_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DatetimeIndex to UTC (timezone-naive) for consistent storage.

    CESM timestamps should always be in Zulu/UTC time. This function:
    - Converts timezone-aware DatetimeIndex to UTC and removes timezone info
    - Leaves timezone-naive DatetimeIndex unchanged (assumed to be UTC)
    """
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        # Convert to UTC and remove timezone info
        df = df.copy()
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df


def _encode_multiindex_column(col_tuple: tuple) -> str:
    """
    Encode a MultiIndex column tuple using :: separator.

    ("region", "north", "heat") -> "region::north::heat"

    The :: separator is used because entity names may contain dots
    (e.g., "source.sink" naming convention).
    """
    return '::'.join(str(level) for level in col_tuple)


def _flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a DataFrame for storage in DuckDB.

    - Normalizes DatetimeIndex to UTC (timezone-naive)
    - Resets index to columns (index columns become first columns)
    - Flattens MultiIndex columns to single-level with dot convention
    """
    # Work with a copy
    flat_df = df.copy()

    # Normalize DatetimeIndex to UTC before flattening
    flat_df = _normalize_datetime_index_to_utc(flat_df)

    # Handle MultiIndex columns first - use dot convention
    if isinstance(flat_df.columns, pd.MultiIndex):
        flat_df.columns = [_encode_multiindex_column(t) for t in flat_df.columns.tolist()]

    # Reset index to convert it to columns (index columns become first)
    flat_df = flat_df.reset_index()

    return flat_df


def dataframes_to_duckdb(
    dataframes: Dict[str, pd.DataFrame],
    db_path: str,
    overwrite: bool = True
) -> None:
    """
    Write a dictionary of DataFrames to a DuckDB database.

    Args:
        dataframes: Dictionary mapping table names to DataFrames
        db_path: Path to the output DuckDB file
        overwrite: If True, delete existing file before writing

    The function stores metadata about each DataFrame's structure in a
    special '_dataframe_metadata' table, allowing exact reconstruction
    when reading back with from_duckdb.dataframes_from_duckdb().
    """
    db_path = Path(db_path)

    # Handle overwrite
    if overwrite and db_path.exists():
        db_path.unlink()
        # Also remove WAL file if present
        wal_path = db_path.with_suffix(db_path.suffix + '.wal')
        if wal_path.exists():
            wal_path.unlink()

    # Connect to DuckDB
    conn = duckdb.connect(str(db_path))

    try:
        # Load existing metadata if not overwriting and database exists
        existing_metadata: Dict[str, Dict[str, Any]] = {}
        if not overwrite:
            try:
                existing_df = conn.execute(
                    'SELECT * FROM "_dataframe_metadata"'
                ).fetchdf()
                # Index by table_name for easy lookup/update
                for _, row in existing_df.iterrows():
                    existing_metadata[row['table_name']] = row.to_dict()
            except duckdb.CatalogException:
                pass  # No existing metadata table

        # Collect metadata for all DataFrames
        all_metadata: List[Dict[str, Any]] = []

        for table_name, df in dataframes.items():
            print(f"  Writing table: {table_name} ({df.shape})")

            # Extract simplified metadata
            index_type, index_count = _get_index_info(df)
            columns_multiindex, columns_levels = _get_columns_info(df)

            # Sanitize table name for SQL (replace dots with underscores for actual table)
            sql_table_name = table_name.replace('.', '_').replace('-', '_')

            metadata_entry = {
                'table_name': table_name,
                'sql_table_name': sql_table_name,
                'index_type': index_type,
                'index_count': index_count,
                'columns_multiindex': columns_multiindex,
                'columns_levels': json.dumps(columns_levels) if columns_levels else None
            }
            all_metadata.append(metadata_entry)

            # Flatten the DataFrame (index columns become first, MultiIndex cols use dot convention)
            flat_df = _flatten_dataframe(df)

            # Write to DuckDB
            conn.execute(f'DROP TABLE IF EXISTS "{sql_table_name}"')
            conn.execute(f'CREATE TABLE "{sql_table_name}" AS SELECT * FROM flat_df')

        # Merge with existing metadata (new entries override existing ones)
        if not overwrite and existing_metadata:
            # Update existing metadata with new entries
            for entry in all_metadata:
                existing_metadata[entry['table_name']] = entry
            # Convert back to list
            all_metadata = list(existing_metadata.values())

        # Write metadata table
        metadata_df = pd.DataFrame(all_metadata)
        conn.execute('DROP TABLE IF EXISTS "_dataframe_metadata"')
        conn.execute('CREATE TABLE "_dataframe_metadata" AS SELECT * FROM metadata_df')

        print(f"\nSuccessfully wrote {len(dataframes)} tables to {db_path}")

    finally:
        conn.close()
