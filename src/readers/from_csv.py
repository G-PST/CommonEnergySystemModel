"""
CSV reader for CESM DataFrames.

Reads a directory of CSV files and reconstructs a dictionary of DataFrames.
Each CSV file corresponds to one entity table, where the filename (without .csv)
is the table name.

Metadata is stored in a special _metadata.json file in the directory, following
the same schema as the DuckDB metadata table:
- index_type: 'single', 'datetime', or 'multi'
- index_count: number of index columns (first N columns in CSV)
- columns_multiindex: boolean indicating if columns are MultiIndex
- columns_levels: list of level names if MultiIndex columns

MultiIndex columns use :: separator (same as DuckDB): ("region", "north", "heat")
-> "region::north::heat"

Usage:
    from readers.from_csv import dataframes_from_csv
    dataframes = dataframes_from_csv("input_directory/")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _decode_multiindex_column(col_name: str) -> tuple:
    """
    Decode a ::-separated column name back to a tuple.

    "region::north::heat" -> ("region", "north", "heat")
    """
    return tuple(col_name.split('::'))


def _reconstruct_index(df: pd.DataFrame, index_type: str, index_count: int) -> pd.DataFrame:
    """
    Reconstruct the original index structure from metadata.

    Index columns are stored as the first N columns in the CSV.

    Args:
        df: DataFrame with index columns as first columns
        index_type: 'single', 'datetime', or 'multi'
        index_count: number of index columns

    Returns:
        DataFrame with proper index set
    """
    index_cols = df.columns[:index_count].tolist()

    if index_type == 'multi':
        df = df.set_index(index_cols)
    elif index_type == 'datetime':
        index_col = index_cols[0]
        df[index_col] = pd.to_datetime(df[index_col])
        df = df.set_index(index_col)
    elif index_type == 'single':
        index_col = index_cols[0]
        if index_col == 'index':
            df = df.set_index(index_col)
            df.index.name = None
        else:
            df = df.set_index(index_col)

    return df


def _reconstruct_columns(df: pd.DataFrame, columns_levels: Optional[List[str]]) -> pd.DataFrame:
    """
    Reconstruct MultiIndex columns from ::-encoded column names.

    Args:
        df: DataFrame with ::-encoded column names
        columns_levels: list of level names for the MultiIndex

    Returns:
        DataFrame with MultiIndex columns
    """
    current_cols = df.columns.tolist()
    new_columns = [_decode_multiindex_column(col) for col in current_cols]
    df.columns = pd.MultiIndex.from_tuples(new_columns, names=columns_levels)
    return df


def _load_metadata(csv_dir: Path) -> Optional[dict]:
    """
    Load metadata from _metadata.json in the CSV directory.

    Args:
        csv_dir: Path to the CSV directory

    Returns:
        Dictionary mapping table names to metadata dicts, or None if no metadata file
    """
    metadata_path = csv_dir / '_metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def dataframes_from_csv(
    csv_dir: str,
    tables: Optional[list] = None
) -> Dict[str, pd.DataFrame]:
    """
    Read DataFrames from a directory of CSV files.

    Args:
        csv_dir: Path to the directory containing CSV files
        tables: Optional list of table names to read. If None, reads all CSV files.

    Returns:
        Dictionary mapping table names to reconstructed DataFrames

    The function uses metadata stored in _metadata.json (written by
    to_csv.dataframes_to_csv()) to reconstruct the original DataFrame structures.
    If no metadata file exists, DataFrames are read with default pandas behavior.
    """
    csv_dir = Path(csv_dir)

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    if not csv_dir.is_dir():
        raise ValueError(f"Path is not a directory: {csv_dir}")

    # Load metadata if available
    metadata = _load_metadata(csv_dir)

    # Find all CSV files
    csv_files = sorted(csv_dir.glob('*.csv'))
    if not csv_files:
        print(f"Warning: No CSV files found in {csv_dir}")
        return {}

    dataframes: Dict[str, pd.DataFrame] = {}

    for csv_path in csv_files:
        table_name = csv_path.stem

        # Skip if not in requested tables
        if tables is not None and table_name not in tables:
            continue

        print(f"  Reading table: {table_name}")

        # Read CSV file
        df = pd.read_csv(csv_path)

        if metadata and table_name in metadata:
            # Use metadata to reconstruct structure
            table_meta = metadata[table_name]
            index_type = table_meta['index_type']
            index_count = table_meta['index_count']
            columns_multiindex = table_meta['columns_multiindex']
            columns_levels = table_meta.get('columns_levels')

            # Reconstruct index
            df = _reconstruct_index(df, index_type, index_count)

            # Reconstruct MultiIndex columns if needed
            if columns_multiindex:
                df = _reconstruct_columns(df, columns_levels)
        else:
            # No metadata - use heuristic: first column as index
            if len(df.columns) > 0:
                first_col = df.columns[0]
                # Try to detect datetime index
                try:
                    parsed = pd.to_datetime(df[first_col], format='ISO8601')
                    df[first_col] = parsed
                    df = df.set_index(first_col)
                except (ValueError, TypeError):
                    df = df.set_index(first_col)

        dataframes[table_name] = df

    print(f"\nSuccessfully read {len(dataframes)} tables from {csv_dir}")

    return dataframes


def list_tables(csv_dir: str) -> list:
    """
    List all table names available in a CSV directory.

    Args:
        csv_dir: Path to the directory containing CSV files

    Returns:
        List of table names (CSV filenames without extension)
    """
    csv_dir = Path(csv_dir)

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    return sorted(p.stem for p in csv_dir.glob('*.csv'))
