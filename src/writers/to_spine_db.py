"""
Convert dataframes to Spine Toolbox database.

This module provides functions to write pandas DataFrames to a Spine database,
handling entity classes, entities, parameters, and time series data.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from spinedb_api import DatabaseMapping, parameter_value
from spinedb_api.exception import NothingToCommit, SpineDBAPIError
from spinedb_api.parameter_value import to_database

logger = logging.getLogger(__name__)


def _format_datetime_index(index: pd.Index) -> List[str]:
    """
    Format an index as strings, with consistent datetime formatting.

    For DatetimeIndex:
    - Converts timezone-aware timestamps to UTC
    - Uses ISO 8601 format with 'T' separator: '2023-01-01T00:00:00'

    For other index types:
    - Uses default string conversion

    This ensures all timestamps in the output are UTC and consistently formatted.
    """
    if isinstance(index, pd.DatetimeIndex):
        # Convert to UTC if timezone-aware
        if index.tz is not None:
            index = index.tz_convert('UTC').tz_localize(None)
        # Format with 'T' separator, no 'Z' suffix (FlexTool compatibility)
        return index.strftime('%Y-%m-%dT%H:%M:%S').tolist()
    else:
        # Default string conversion for non-datetime indexes
        return index.astype(str).tolist()


def _class_name_to_db(class_name: str) -> str:
    """
    Convert a dot-separated class name to Spine DB double-underscore format.

    Args:
        class_name: Class name, possibly containing dots (e.g., 'unit.node').

    Returns:
        DB class name with '__' separators (e.g., 'unit__node').
    """
    if '.' in class_name:
        return '__'.join(class_name.split('.'))
    return class_name


def _get_entity_names_from_columns(
    df: pd.DataFrame,
) -> List[Tuple[str, ...]]:
    """
    Extract entity names from DataFrame columns.

    For MultiIndex columns, drops the first level (entity name) and returns
    remaining levels as entity_byname tuples.
    For regular columns, wraps each name in a single-element tuple.

    Args:
        df: DataFrame with entity names in columns.

    Returns:
        List of entity_byname tuples.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-dimensional columns: drop name dimension for entity_byname
        return [col[1:] for col in df.columns]
    return [(str(col),) for col in df.columns]


def _parse_df_name(
    df_name: str,
    separator: str,
) -> Optional[Tuple[str, str, str]]:
    """
    Parse a typed dataframe name into (class_name, db_class_name, param_name).

    Args:
        df_name: Name like 'class_name.ts.parameter_name'.
        separator: The type separator (e.g., '.ts.', '.str.', '.array.').

    Returns:
        Tuple of (class_name, db_class_name, param_name), or None if invalid format.
    """
    parts = df_name.split(separator)
    if len(parts) != 2:
        logger.warning("Invalid dataframe name format: %s (expected '%s' separator)", df_name, separator)
        return None
    class_name = parts[0]
    param_name = parts[1]
    db_class_name = _class_name_to_db(class_name)
    return class_name, db_class_name, param_name


def _add_param_definition(
    db_map: DatabaseMapping,
    db_class_name: str,
    param_name: str,
) -> None:
    """
    Add a parameter definition, logging on failure.

    Args:
        db_map: Open Spine database mapping.
        db_class_name: Entity class name in DB format.
        param_name: Parameter name.
    """
    try:
        db_map.add_parameter_definition(
            entity_class_name=db_class_name,
            name=param_name
        )
    except (RuntimeError, KeyError, SpineDBAPIError) as e:
        logger.debug("Parameter definition %s.%s already exists or error: %s", db_class_name, param_name, e)


def dataframes_to_spine(
    dataframes: Dict[str, pd.DataFrame],
    db_url: str,
    import_datetime: Optional[str] = None,
    purge_before_import: bool = True,
) -> None:
    """
    Write dataframes to Spine database.

    Args:
        dataframes: Dict mapping dataframe names to DataFrames
                   Entity names are in index (or MultiIndex for multi-dimensional)
                   For time series, entity names are in columns
        db_url: Database URL (e.g., "sqlite:///path/to/database.sqlite")
        import_datetime: Datetime string for alternative name (format: yyyy-mm-dd_hh-mm)
                        If None, uses current datetime
        purge_before_import: If True, purge parameter values, entities, and alternatives
                            before import (default: True)
    """
    from datetime import datetime

    # Generate alternative name with datetime
    if import_datetime is None:
        import_datetime = datetime.now().strftime('%Y_%m_%d-%H_%M')
    alternative_name = f'cesm-{import_datetime}'

    with DatabaseMapping(db_url, create=True) as db_map:
        # Phase -1: Purge if requested
        if purge_before_import:
            logger.info("Phase -1: Purging database...")
            db_map.purge_items('parameter_value')
            db_map.purge_items('entity')
            db_map.purge_items('alternative')
            db_map.purge_items('scenario')
            db_map.refresh_session()
            db_map.commit_session("Purged parameter values, entities and alternatives")
            logger.info("Purged parameter values, entities, and alternatives")

        # Separate dataframes by type
        entity_dfs: Dict[str, pd.DataFrame] = {}
        ts_dfs: Dict[str, pd.DataFrame] = {}
        str_dfs: Dict[str, pd.DataFrame] = {}
        array_dfs: Dict[str, pd.DataFrame] = {}

        for name, df in dataframes.items():
            if '.ts.' in name:
                ts_dfs[name] = df
            elif '.str.' in name:
                str_dfs[name] = df
            elif '.array.' in name:
                array_dfs[name] = df
            else:
                entity_dfs[name] = df

        # Phase 0: Add alternative
        logger.info("Phase 0: Adding a scenario and an alternative '%s'...", alternative_name)
        try:
            db_map.add_alternative(name=alternative_name)
            db_map.add_scenario(name='base')
            db_map.add_scenario_alternative(scenario_name='base',
                                            alternative_name=alternative_name,
                                            rank=0)
            db_map.commit_session(f"Added alternative {alternative_name}")
            logger.info("Added alternative: %s", alternative_name)
        except (RuntimeError, KeyError, SpineDBAPIError) as e:
            logger.warning("Alternative %s already exists or error: %s", alternative_name, e)

       # Phase 1: Add entity classes and entities
        logger.info("Phase 1: Adding entity classes and entities...")
        _add_entity_classes_and_entities(db_map, entity_dfs, alternative_name)
        try:
            db_map.commit_session("Added entity classes and entities")
        except NothingToCommit:
            logger.info("No entities to commit")

        # Phase 2: Add parameter definitions and values
        logger.info("Phase 2: Adding parameter definitions and values...")
        _add_parameters(db_map, entity_dfs, alternative_name)
        try:
            db_map.commit_session("Added parameter definitions and values")
        except NothingToCommit:
            logger.info("No parameters (constants) to commit")

        # Phase 3: Add time series parameters
        if ts_dfs:
            logger.info("Phase 3: Adding time series parameters...")
            _add_time_series(db_map, ts_dfs, dataframes.get("timeline"), alternative_name)
            try:
                db_map.commit_session("Added time series parameters")
            except NothingToCommit:
                logger.info("No time series parameters to commit")

        # Phase 4: Add str (map) and array parameters
        if str_dfs:
            logger.info("Phase 4: Adding str (map) parameters...")
            _add_indexed_values(db_map, str_dfs, alternative_name, value_type='map')
            try:
                db_map.commit_session("Added str parameters")
            except NothingToCommit:
                logger.info("No str (map) parameters to commit")

        # Phase 5: Add array parameters
        if array_dfs:
            logger.info("Phase 5: Adding array parameters...")
            _add_indexed_values(db_map, array_dfs, alternative_name, value_type='array')
            try:
                db_map.commit_session("Added array parameters")
            except NothingToCommit:
                logger.info("No array parameters to commit")

        logger.info("Done!")


def _get_entity_names_from_index(idx: Union[pd.Index, pd.MultiIndex]) -> List[str]:
    """
    Extract entity names from index, converting MultiIndex tuples to strings with '__'.

    For single Index, returns list of names as-is.
    For MultiIndex, joins levels with '__'.

    Args:
        idx: A pandas Index or MultiIndex.

    Returns:
        List of entity name strings.
    """
    if isinstance(idx, pd.MultiIndex):
        # Join multi-dimensional names with '__'
        return ['__'.join(map(str, t)) for t in idx]
    else:
        return [str(name) for name in idx]


def _add_entity_classes_and_entities(
    db_map: DatabaseMapping,
    entity_dfs: Dict[str, pd.DataFrame],
    alternative_name: str,
) -> None:
    """Add entity classes and their entities."""

    # List of entity_classes that require entity_alternative to be true
    ent_alt_classes = ['unit', 'node', 'connection', 'reserve__upDown__unit__node', 'reserve__upDown__connection__node']

    # Sort: single-dimensional classes first (no dots), then multi-dimensional
    sorted_classes = sorted(entity_dfs.keys(), key=lambda x: ('.' in x, x))

    for class_name in sorted_classes:
        df = entity_dfs[class_name]

        # Determine if multi-dimensional
        if '.' in class_name:
            dimensions = class_name.split('.')

            # Get dimension names from index (skip first 'name' level)
            if isinstance(df.index, pd.MultiIndex):
                dimension_name_list = tuple(df.index.names[1:])
            else:
                # Fallback if names not set
                dimension_name_list = tuple(dimensions)

            class_name = '__'.join(dimensions)
        else:
            dimension_name_list = None

        # Add entity class
        try:
            db_map.add_entity_class(
                name=class_name,
                dimension_name_list=dimension_name_list
            )
            logger.info("Added entity class: %s", class_name)
        except (RuntimeError, KeyError, SpineDBAPIError) as e:
            logger.debug("Entity class %s already exists or error: %s", class_name, e)

        # Add entities
        if dimension_name_list:
            # Multi-dimensional: index levels are dimensions
            if isinstance(df.index, pd.MultiIndex):
                for element_tuple in df.index.unique():
                    element_name_list = tuple(str(elem) for elem in element_tuple[1:])
                    try:
                        db_map.add_entity(
                            entity_class_name=class_name,
                            element_name_list=element_name_list,
                            name=element_tuple[0]
                        )
                    except (RuntimeError, KeyError, SpineDBAPIError) as e:
                        logger.debug("Entity already exists or error: %s", e)

                    if class_name in ent_alt_classes:
                        try:
                            db_map.add_entity_alternative(
                                entity_class_name=class_name,
                                element_name_list=element_name_list,
                                alternative_name=alternative_name
                            )
                        except (RuntimeError, KeyError, SpineDBAPIError) as e:
                            logger.debug("Entity alternative already exists or error: %s", e)
            else:
                # Single index but class name has dots - treat as single entity per row
                for entity_name in df.index.unique():
                    try:
                        db_map.add_entity(
                            entity_class_name=class_name,
                            element_name_list=(str(entity_name),)
                        )
                    except (RuntimeError, KeyError, SpineDBAPIError) as e:
                        logger.debug("Entity already exists or error: %s", e)

        else:
            # Single-dimensional: index contains entity names
            for entity_name in df.index.unique():
                try:
                    db_map.add_entity(
                        entity_class_name=class_name,
                        name=str(entity_name)
                    )
                except (RuntimeError, KeyError, SpineDBAPIError) as e:
                    logger.debug("Entity already exists or error: %s", e)

                if class_name in ent_alt_classes:
                    try:
                        db_map.add_entity_alternative(
                            entity_class_name=class_name,
                            entity_byname=(str(entity_name),),
                            alternative_name=alternative_name,
                            active=True
                        )
                    except (RuntimeError, KeyError, SpineDBAPIError) as e:
                        logger.debug("Entity alternative already exists or error: %s", e)


def _add_parameters(
    db_map: DatabaseMapping,
    entity_dfs: Dict[str, pd.DataFrame],
    alternative_name: str,
) -> None:
    """Add parameter definitions and constant values."""

    for class_name, df in entity_dfs.items():
        db_class_name = _class_name_to_db(class_name)

        # Get parameter columns (all columns that aren't part of the structure)
        param_cols = df.columns.tolist()

        # Add parameter definitions
        for param_name in param_cols:
            _add_param_definition(db_map, db_class_name, param_name)

        # Add parameter values
        for param_name in param_cols:
            for idx, value in df[param_name].items():
                # Skip if value is None or NaN
                if pd.isna(value):
                    continue

                # Build entity_byname tuple
                if isinstance(idx, tuple):
                    # MultiIndex - use tuple of strings
                    entity_byname = tuple(str(elem) for elem in idx[1:])
                else:
                    # Single index
                    entity_byname = (str(idx),)

                # Parse value
                if isinstance(value, (int, float)):
                    parsed_value: Any = float(value)
                elif isinstance(value, pd.Timedelta):
                    # Convert Timedelta to hours
                    parsed_value = value.total_seconds() / 3600
                elif isinstance(value, str):
                    parsed_value = value
                elif isinstance(value, list):
                    if isinstance(value[0], (int, float)):  # Assume there is only one type in the array
                        parsed_value = parameter_value.Array(value, float, 'index')
                    elif isinstance(value[0], str):
                        parsed_value = parameter_value.Array(value, str, 'index')
                    else:
                        parsed_value = value
                else:
                    parsed_value = value

                try:
                    db_map.add_parameter_value(
                        entity_class_name=db_class_name,
                        parameter_definition_name=param_name,
                        entity_byname=entity_byname,
                        alternative_name=alternative_name,
                        parsed_value=parsed_value
                    )
                except (RuntimeError, KeyError, ValueError, SpineDBAPIError) as e:
                    logger.warning("Could not add value for %s.%s: %s", db_class_name, param_name, e)


def _add_time_series(
    db_map: DatabaseMapping,
    ts_dfs: Dict[str, pd.DataFrame],
    timeline_df: Optional[pd.DataFrame],
    alternative_name: str,
) -> None:
    """Add time series parameter values."""
    # Extract start time from timeline (formatted consistently as UTC)
    if timeline_df is not None and timeline_df.index.name == 'datetime':
        start_time: Optional[str] = _format_datetime_index(timeline_df.index[:1])[0]
    elif timeline_df is not None and 'datetime' in timeline_df.columns:
        start_time = _format_datetime_index(pd.DatetimeIndex([timeline_df['datetime'].iloc[0]]))[0]
    else:
        start_time = None

    for ts_name, df in ts_dfs.items():
        parsed = _parse_df_name(ts_name, '.ts.')
        if parsed is None:
            continue
        _class_name, db_class_name, param_name = parsed

        # Add parameter definition if needed
        _add_param_definition(db_map, db_class_name, param_name)

        # Entity names are in columns (for time series, index is datetime, columns are entities)
        entity_names = _get_entity_names_from_columns(df)

        for i, entity_name in enumerate(entity_names):
            values = df.iloc[:, i].tolist()

            # Build time series in Spine format
            if start_time and df.index.name == 'datetime':
                ts_value: Dict[str, Any] = {
                    "type": "time_series",
                    "data": values,
                    "index": {
                        "start": start_time,
                        "resolution": "1h"
                    }
                }
            else:
                # Use datetime index if available
                if df.index.name == 'datetime':
                    timestamps = _format_datetime_index(df.index)
                    ts_value = {
                        "type": "time_series",
                        "data": [[ts, val] for ts, val in zip(timestamps, values)]
                    }
                else:
                    # Fallback to array without timestamps
                    ts_value = {
                        "type": "time_series",
                        "data": values
                    }

            # Convert to database format
            db_value, value_type = to_database(ts_value)

            try:
                db_map.add_parameter_value(
                    entity_class_name=db_class_name,
                    parameter_definition_name=param_name,
                    entity_byname=entity_name,
                    alternative_name=alternative_name,
                    value=db_value,
                    type=value_type
                )
                logger.debug("Added time series: %s.%s for %s", db_class_name, param_name, entity_name)
            except (RuntimeError, KeyError, ValueError, SpineDBAPIError) as e:
                logger.warning("Could not add time series for %s: %s", entity_name, e)


def _add_indexed_values(
    db_map: DatabaseMapping,
    indexed_dfs: Dict[str, pd.DataFrame],
    alternative_name: str,
    value_type: str,
) -> None:
    """
    Add map (str) or array parameter values to the database.

    This consolidates the common pattern shared by _add_strs and _add_arrays.
    Both follow the same flow: parse name, add definition, extract entity names,
    format index, create typed value object, and write to DB.

    Args:
        db_map: Open Spine database mapping.
        indexed_dfs: Dict mapping dataframe names to DataFrames.
        alternative_name: Alternative name for parameter values.
        value_type: Either 'map' or 'array'.
    """
    from spinedb_api.parameter_value import Array, Map

    separator = '.str.' if value_type == 'map' else '.array.'

    for df_name, df in indexed_dfs.items():
        parsed = _parse_df_name(df_name, separator)
        if parsed is None:
            continue
        _class_name, db_class_name, param_name = parsed

        # Add parameter definition if needed
        _add_param_definition(db_map, db_class_name, param_name)

        # Entity names are in columns
        entity_names = _get_entity_names_from_columns(df)

        # Format index consistently (handles datetime conversion to UTC)
        df.index = _format_datetime_index(df.index)
        index_name = df.index.name

        for i, entity_name in enumerate(entity_names):
            # Extract values for this entity, filtering out NaN values
            col_data = df.iloc[:, i].dropna()

            if value_type == 'map':
                col_values = col_data.tolist()
                col_indexes = col_data.index.tolist()
            else:
                # For arrays, just get the list of values
                col_values = col_data.tolist()

            # Skip if no valid values
            if not col_values:
                continue

            # Create the appropriate Spine value object
            if value_type == 'map':
                spine_value = Map(
                    indexes=col_indexes,
                    values=col_values,
                    index_name=index_name
                )
            else:
                spine_value = Array(
                    values=col_values,
                    index_name=index_name
                )

            # Convert to database format
            db_value, db_value_type = to_database(spine_value)

            try:
                db_map.add_parameter_value(
                    entity_class_name=db_class_name,
                    parameter_definition_name=param_name,
                    entity_byname=entity_name,
                    alternative_name=alternative_name,
                    value=db_value,
                    type=db_value_type
                )
            except (RuntimeError, KeyError, ValueError, SpineDBAPIError) as e:
                logger.warning("Could not add %s for %s: %s", value_type, entity_name, e)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: Create sample dataframes with new index structure
    sample_dfs = {
        'node': pd.DataFrame({
            'annual_flow': [100000.0, 80000.0, None],
            'penalty_up': [10000.0, 10000.0, 10000.0]
        }, index=pd.Index(['west', 'east', 'heat'], name='node')),

        'connection': pd.DataFrame({
            'efficiency': [0.90, 0.98],
            'capacity': [750.0, 500.0]
        }, index=pd.Index(['charger', 'pony1'], name='connection')),

        'unit.outputNode': pd.DataFrame({
            'capacity': [100.0, 50.0],
            'efficiency': [0.9, 0.95]
        }, index=pd.MultiIndex.from_tuples(
            [('coal_plant', 'west'), ('gas_plant', 'east')],
            names=['unit', 'outputNode']
        )),

        'node.str.inflow': pd.DataFrame({
            'west': [-1002.1, -980.7, -968, -969.1, -971.9, -957.8, -975.2, -975.1, -973.2, -800],
            'east': [-1002.1, -980.7, -968, -969.1, -971.9, -957.8, -975.2, -975.1, -973.2, -800],
            'heat': [-30, -40, -50, -60, -50, -50, -50, -50, -50, -50]
        }, index=pd.date_range('2023-01-01', periods=10, freq='h', name='datetime'))
    }

    timeline = pd.DataFrame(
        index=pd.date_range('2023-01-01', periods=8760, freq='h', name='datetime')
    )

    # Write to database
    # dataframes_to_spine(sample_dfs, "sqlite:///test_flextool.sqlite", import_datetime='2025-10-02_15-30')
