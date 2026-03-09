"""
CSV writer for CESM DataFrames.

Writes a dictionary of pandas DataFrames to a directory of CSV files,
preserving structure metadata so the DataFrames can be reconstructed
exactly by the corresponding reader (readers/from_csv.py).

Each DataFrame is written as a separate CSV file named after its key.
Metadata about index types and column structures is stored in a
_metadata.json file in the same directory.

Metadata schema (same as DuckDB writer):
- index_type: 'single', 'datetime', or 'multi'
- index_count: number of index columns (stored first in CSV)
- columns_multiindex: boolean indicating if columns are MultiIndex
- columns_levels: list of level names if MultiIndex columns

MultiIndex columns use :: separator: ("region", "north", "heat") -> "region::north::heat"

Usage:
    from writers.to_csv import dataframes_to_csv
    dataframes_to_csv(dataframes_dict, "output_directory/")
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


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
        df = df.copy()
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df


def _encode_multiindex_column(col_tuple: tuple) -> str:
    """
    Encode a MultiIndex column tuple using :: separator.

    ("region", "north", "heat") -> "region::north::heat"
    """
    return '::'.join(str(level) for level in col_tuple)


def _flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a DataFrame for storage as CSV.

    - Normalizes DatetimeIndex to UTC (timezone-naive)
    - Resets index to columns (index columns become first columns)
    - Flattens MultiIndex columns to single-level with :: convention
    """
    flat_df = df.copy()

    # Normalize DatetimeIndex to UTC before flattening
    flat_df = _normalize_datetime_index_to_utc(flat_df)

    # Handle MultiIndex columns - use :: convention
    if isinstance(flat_df.columns, pd.MultiIndex):
        flat_df.columns = [_encode_multiindex_column(t) for t in flat_df.columns.tolist()]

    # Reset index to convert it to columns (index columns become first)
    flat_df = flat_df.reset_index()

    return flat_df


def dataframes_to_csv(
    dataframes: Dict[str, pd.DataFrame],
    output_dir: str,
    overwrite: bool = True
) -> None:
    """
    Write a dictionary of DataFrames to a directory of CSV files.

    Args:
        dataframes: Dictionary mapping table names to DataFrames
        output_dir: Path to the output directory
        overwrite: If True, delete existing directory before writing.
                   If False, add/replace CSV files in the existing directory.

    The function stores metadata about each DataFrame's structure in a
    _metadata.json file, allowing exact reconstruction when reading back
    with from_csv.dataframes_from_csv().
    """
    output_dir = Path(output_dir)

    # Handle overwrite
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata if not overwriting
    existing_metadata: dict = {}
    if not overwrite:
        metadata_path = output_dir / '_metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                existing_metadata = json.load(f)

    # Collect metadata for all DataFrames
    all_metadata: dict = dict(existing_metadata)

    for table_name, df in dataframes.items():
        print(f"  Writing table: {table_name} ({df.shape})")

        # Extract metadata
        index_type, index_count = _get_index_info(df)
        columns_multiindex, columns_levels = _get_columns_info(df)

        metadata_entry = {
            'index_type': index_type,
            'index_count': index_count,
            'columns_multiindex': columns_multiindex,
            'columns_levels': columns_levels,
        }
        all_metadata[table_name] = metadata_entry

        # Flatten the DataFrame
        flat_df = _flatten_dataframe(df)

        # Write CSV file
        csv_path = output_dir / f"{table_name}.csv"
        flat_df.to_csv(csv_path, index=False)

    # Write metadata file
    metadata_path = output_dir / '_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nSuccessfully wrote {len(dataframes)} tables to {output_dir}")
