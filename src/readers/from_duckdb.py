"""
DuckDB reader for CESM DataFrames.

Reads a DuckDB database file created by writers/to_duckdb.py and
reconstructs the original DataFrame structures including:
- Single and MultiIndex row indexes
- DatetimeIndex for time series
- MultiIndex columns

Usage:
    from readers.from_duckdb import dataframes_from_duckdb
    dataframes = dataframes_from_duckdb("input.duckdb")
"""

import duckdb
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional


def _reconstruct_index(df: pd.DataFrame, index_meta: Dict[str, Any]) -> pd.DataFrame:
    """Reconstruct the original index structure from metadata."""
    index_type = index_meta['type']
    index_names = index_meta['names']
    nlevels = index_meta['nlevels']

    if index_type == 'multiindex':
        # Set multiple columns as index
        df = df.set_index(index_names)
    elif index_type == 'datetime':
        # Convert column to DatetimeIndex
        index_col = index_names[0] if index_names[0] else df.columns[0]
        df[index_col] = pd.to_datetime(df[index_col])
        df = df.set_index(index_col)
        df.index.name = index_names[0]
    elif index_type == 'single':
        index_col = index_names[0]
        if index_col and index_col in df.columns:
            df = df.set_index(index_col)
        elif index_col is None and 'index' in df.columns:
            # Handle unnamed index stored as 'index'
            df = df.set_index('index')
            df.index.name = None

    return df


def _reconstruct_columns(df: pd.DataFrame, columns_meta: Dict[str, Any]) -> pd.DataFrame:
    """Reconstruct the original column structure from metadata."""
    columns_type = columns_meta['type']

    if columns_type == 'multiindex':
        # Decode JSON-encoded column tuples
        tuples = columns_meta['tuples']
        names = columns_meta['names']

        # Current columns are JSON-encoded strings
        # Map them back to the original tuples
        current_cols = df.columns.tolist()
        new_columns = []

        for col in current_cols:
            try:
                # Try to decode as JSON tuple
                decoded = json.loads(col)
                new_columns.append(tuple(decoded))
            except (json.JSONDecodeError, TypeError):
                # Not a JSON-encoded tuple, keep as-is (likely an index column)
                # This shouldn't happen after index reconstruction
                new_columns.append((col,) * len(names))

        df.columns = pd.MultiIndex.from_tuples(new_columns, names=names)

    return df


def dataframes_from_duckdb(
    db_path: str,
    tables: Optional[list] = None
) -> Dict[str, pd.DataFrame]:
    """
    Read DataFrames from a DuckDB database.

    Args:
        db_path: Path to the DuckDB database file
        tables: Optional list of table names to read. If None, reads all tables.

    Returns:
        Dictionary mapping original table names to reconstructed DataFrames

    The function uses metadata stored by to_duckdb.dataframes_to_duckdb()
    to exactly reconstruct the original DataFrame structures.
    """
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)

    try:
        # Read metadata table
        try:
            metadata_df = conn.execute(
                'SELECT * FROM "_dataframe_metadata"'
            ).fetchdf()
        except duckdb.CatalogException:
            raise ValueError(
                f"Database {db_path} does not contain metadata table. "
                "Was it created with to_duckdb.dataframes_to_duckdb()?"
            )

        dataframes: Dict[str, pd.DataFrame] = {}

        for _, row in metadata_df.iterrows():
            table_name = row['table_name']
            sql_table_name = row['sql_table_name']

            # Filter tables if specified
            if tables is not None and table_name not in tables:
                continue

            print(f"  Reading table: {table_name}")

            # Parse metadata
            index_meta = json.loads(row['index_metadata'])
            columns_meta = json.loads(row['columns_metadata'])

            # Read the table
            df = conn.execute(f'SELECT * FROM "{sql_table_name}"').fetchdf()

            # Reconstruct index
            df = _reconstruct_index(df, index_meta)

            # Reconstruct columns (for MultiIndex columns)
            if columns_meta['type'] == 'multiindex':
                df = _reconstruct_columns(df, columns_meta)

            dataframes[table_name] = df

        print(f"\nSuccessfully read {len(dataframes)} tables from {db_path}")

        return dataframes

    finally:
        conn.close()


def list_tables(db_path: str) -> list:
    """
    List all DataFrame tables in a DuckDB database.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        List of original table names (not SQL-safe names)
    """
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)

    try:
        metadata_df = conn.execute(
            'SELECT table_name FROM "_dataframe_metadata"'
        ).fetchdf()
        return metadata_df['table_name'].tolist()
    finally:
        conn.close()
