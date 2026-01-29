"""
DuckDB writer for CESM DataFrames.

Writes a dictionary of pandas DataFrames to a DuckDB database file,
preserving structure metadata so the DataFrames can be reconstructed
exactly by the corresponding reader (readers/from_duckdb.py).

Usage:
    from writers.to_duckdb import dataframes_to_duckdb
    dataframes_to_duckdb(dataframes_dict, "output.duckdb")
"""

import duckdb
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def _get_index_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract metadata about the DataFrame's index structure."""
    index = df.index

    if isinstance(index, pd.MultiIndex):
        return {
            'type': 'multiindex',
            'names': list(index.names),
            'nlevels': index.nlevels
        }
    elif isinstance(index, pd.DatetimeIndex):
        return {
            'type': 'datetime',
            'names': [index.name],
            'nlevels': 1
        }
    else:
        return {
            'type': 'single',
            'names': [index.name],
            'nlevels': 1
        }


def _get_columns_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract metadata about the DataFrame's column structure."""
    columns = df.columns

    if isinstance(columns, pd.MultiIndex):
        return {
            'type': 'multiindex',
            'names': list(columns.names),
            'nlevels': columns.nlevels,
            'tuples': [list(t) for t in columns.tolist()]
        }
    else:
        return {
            'type': 'single',
            'names': None,
            'nlevels': 1,
            'columns': list(columns)
        }


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


def _flatten_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten a DataFrame for storage in DuckDB.

    - Normalizes DatetimeIndex to UTC (timezone-naive)
    - Resets index to columns
    - Flattens MultiIndex columns to single-level with JSON-encoded tuples
    """
    # Work with a copy
    flat_df = df.copy()

    # Normalize DatetimeIndex to UTC before flattening
    flat_df = _normalize_datetime_index_to_utc(flat_df)

    # Handle MultiIndex columns first
    if isinstance(flat_df.columns, pd.MultiIndex):
        # Encode each tuple as JSON string for column name
        flat_df.columns = [json.dumps(list(t)) for t in flat_df.columns.tolist()]

    # Reset index to convert it to columns
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

            # Extract metadata
            index_meta = _get_index_metadata(df)
            columns_meta = _get_columns_metadata(df)

            metadata_entry = {
                'table_name': table_name,
                'index_metadata': json.dumps(index_meta),
                'columns_metadata': json.dumps(columns_meta),
                'original_shape_rows': df.shape[0],
                'original_shape_cols': df.shape[1]
            }
            all_metadata.append(metadata_entry)

            # Flatten the DataFrame
            flat_df = _flatten_dataframe(df, {
                'index': index_meta,
                'columns': columns_meta
            })

            # Sanitize table name for SQL (replace dots with underscores for actual table)
            safe_table_name = table_name.replace('.', '_').replace('-', '_')

            # Store mapping in metadata if name changed
            metadata_entry['sql_table_name'] = safe_table_name

            # Write to DuckDB
            conn.execute(f'DROP TABLE IF EXISTS "{safe_table_name}"')
            conn.execute(f'CREATE TABLE "{safe_table_name}" AS SELECT * FROM flat_df')

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
