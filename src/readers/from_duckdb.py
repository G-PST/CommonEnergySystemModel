"""
DuckDB reader for CESM DataFrames.

Reads a DuckDB database file created by writers/to_duckdb.py and
reconstructs the original DataFrame structures including:
- Single and MultiIndex row indexes
- DatetimeIndex for time series
- MultiIndex columns (encoded with :: separator: "region::north::heat")

Metadata schema (current):
- index_type: 'single', 'datetime', or 'multi'
- index_count: number of index columns (first N columns in table)
- columns_multiindex: boolean indicating if columns are MultiIndex
- columns_levels: JSON array of level names if MultiIndex columns

Legacy metadata schema (auto-detected and converted):
- index_metadata: JSON with 'type', 'names', 'nlevels'
- columns_metadata: JSON with 'type', 'names', 'nlevels', optionally 'tuples'

Usage:
    from readers.from_duckdb import dataframes_from_duckdb
    dataframes = dataframes_from_duckdb("input.duckdb")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd


def _decode_multiindex_column(col_name: str) -> tuple:
    """
    Decode a ::-separated column name back to a tuple.

    "region::north::heat" -> ("region", "north", "heat")
    """
    return tuple(col_name.split('::'))


def _convert_legacy_metadata(row: pd.Series) -> Tuple[str, int, bool, Optional[str]]:
    """
    Convert legacy metadata format to current format.

    Legacy format uses JSON strings for 'index_metadata' and 'columns_metadata'.
    Current format uses flat columns: index_type, index_count, columns_multiindex,
    columns_levels.

    Args:
        row: A metadata row with legacy columns

    Returns:
        Tuple of (index_type, index_count, columns_multiindex, columns_levels_json)
    """
    # Parse index metadata
    index_meta = json.loads(row['index_metadata'])
    legacy_type = index_meta.get('type', 'single')
    # Legacy used 'multiindex' instead of 'multi'
    if legacy_type == 'multiindex':
        index_type = 'multi'
    else:
        index_type = legacy_type
    index_count = index_meta.get('nlevels', 1)

    # Parse columns metadata
    columns_meta = json.loads(row['columns_metadata'])
    columns_type = columns_meta.get('type', 'single')
    if columns_type == 'multiindex':
        columns_multiindex = True
        columns_levels = json.dumps(columns_meta.get('names'))
    else:
        columns_multiindex = False
        columns_levels = None

    return index_type, index_count, columns_multiindex, columns_levels


def _is_legacy_metadata(metadata_df: pd.DataFrame) -> bool:
    """Check if the metadata table uses the legacy schema."""
    return 'index_metadata' in metadata_df.columns and 'index_type' not in metadata_df.columns


def _reconstruct_index(df: pd.DataFrame, index_type: str, index_count: int) -> pd.DataFrame:
    """
    Reconstruct the original index structure from metadata.

    Index columns are stored as the first N columns in the table.

    Args:
        df: DataFrame with index columns as first columns
        index_type: 'single', 'datetime', or 'multi'
        index_count: number of index columns

    Returns:
        DataFrame with proper index set
    """
    # Get the first index_count columns as index columns
    index_cols = df.columns[:index_count].tolist()

    if index_type == 'multi':
        # Set multiple columns as MultiIndex
        df = df.set_index(index_cols)
    elif index_type == 'datetime':
        # Convert column to DatetimeIndex
        index_col = index_cols[0]
        df[index_col] = pd.to_datetime(df[index_col])
        df = df.set_index(index_col)
    elif index_type == 'single':
        index_col = index_cols[0]
        if index_col == 'index':
            # Handle unnamed index stored as 'index'
            df = df.set_index(index_col)
            df.index.name = None
        else:
            df = df.set_index(index_col)

    return df


def _decode_legacy_multiindex_column(col_name: str) -> tuple:
    """
    Decode a legacy JSON-array encoded column name back to a tuple.

    Legacy format stored MultiIndex columns as JSON arrays:
    '["region", "north", "heat"]' -> ("region", "north", "heat")

    Falls back to ::-separator decoding if not valid JSON.
    """
    try:
        parsed = json.loads(col_name)
        if isinstance(parsed, list):
            return tuple(parsed)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback to current :: separator format
    return _decode_multiindex_column(col_name)


def _reconstruct_columns(
    df: pd.DataFrame,
    columns_levels: Optional[List[str]],
    legacy: bool = False,
) -> pd.DataFrame:
    """
    Reconstruct MultiIndex columns from encoded column names.

    Supports both current (:: separator) and legacy (JSON array) encodings.

    Args:
        df: DataFrame with encoded column names
        columns_levels: list of level names for the MultiIndex
        legacy: if True, try JSON-array decoding first

    Returns:
        DataFrame with MultiIndex columns
    """
    current_cols = df.columns.tolist()
    if legacy:
        new_columns = [_decode_legacy_multiindex_column(col) for col in current_cols]
    else:
        new_columns = [_decode_multiindex_column(col) for col in current_cols]

    df.columns = pd.MultiIndex.from_tuples(new_columns, names=columns_levels)

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
    to exactly reconstruct the original DataFrame structures. Both the
    current and legacy metadata formats are supported.
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

        legacy = _is_legacy_metadata(metadata_df)
        if legacy:
            print("  (detected legacy metadata format, converting automatically)")

        dataframes: Dict[str, pd.DataFrame] = {}

        for _, row in metadata_df.iterrows():
            table_name = row['table_name']
            sql_table_name = row['sql_table_name']

            # Filter tables if specified
            if tables is not None and table_name not in tables:
                continue

            print(f"  Reading table: {table_name}")

            # Extract metadata - handle both current and legacy formats
            if legacy:
                index_type, index_count, columns_multiindex, columns_levels_raw = (
                    _convert_legacy_metadata(row)
                )
            else:
                index_type = row['index_type']
                index_count = row['index_count']
                columns_multiindex = row['columns_multiindex']
                columns_levels_raw = row['columns_levels']

            columns_levels = (
                json.loads(columns_levels_raw)
                if pd.notna(columns_levels_raw) and columns_levels_raw
                else None
            )

            # Read the table
            df = conn.execute(f'SELECT * FROM "{sql_table_name}"').fetchdf()

            # Reconstruct index (first N columns become index)
            df = _reconstruct_index(df, index_type, index_count)

            # Reconstruct MultiIndex columns if needed
            if columns_multiindex:
                df = _reconstruct_columns(df, columns_levels, legacy=legacy)

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
