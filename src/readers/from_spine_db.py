"""
Read Spine Toolbox database to dataframes.

This module provides functions to read a Spine database and convert it
to pandas DataFrames, handling entity classes, entities, parameters,
and time series data. This is the inverse of writers/to_spine_db.py.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from spinedb_api import DatabaseMapping
from spinedb_api.filters.scenario_filter import scenario_filter_config
from spinedb_api.filters.tools import append_filter_config

logger = logging.getLogger(__name__)


def _parse_parameter_value(value: Any, value_type: Optional[str] = None) -> Tuple[Any, str]:
    """
    Parse a Spine parameter value to Python/pandas types.

    Handles:
    - Scalars (float, int, str)
    - Time series
    - Maps
    - Arrays

    Returns the parsed value and a type indicator for dataframe naming.
    """
    if value is None:
        return None, 'scalar'

    # Check if it's a special Spine type
    type_name = type(value).__name__

    if type_name == 'TimeSeries':
        return value, 'ts'
    elif type_name == 'Map':
        return value, 'map'
    elif type_name == 'Array':
        return value, 'array'
    elif isinstance(value, (int, float, str, bool)):
        return value, 'scalar'
    else:
        # Unknown type, return as-is
        return value, 'scalar'


def _entity_class_to_df_name(entity_class_name: str, dimension_names: Optional[List[str]] = None) -> str:
    """
    Convert Spine entity class name to dataframe name.

    Spine uses '__' for multi-dimensional classes (e.g., 'unit__node').
    Dataframes use '.' (e.g., 'unit.node').
    """
    if '__' in entity_class_name:
        return entity_class_name.replace('__', '.')
    return entity_class_name


def _build_entity_elements_lookup(
    db_map: DatabaseMapping,
    entity_classes: Dict[str, Any],
) -> Dict[Tuple[str, str], Tuple[str, ...]]:
    """
    Build a lookup mapping (class_name, entity_name) -> element_name_list
    for multi-dimensional entity classes.

    Args:
        db_map: Open Spine database mapping.
        entity_classes: Dict of entity class name -> entity class item.

    Returns:
        Dict mapping (class_name, entity_name) tuples to element name tuples.
    """
    entity_elements: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    for class_name, ec in entity_classes.items():
        dimension_names = ec.get('dimension_name_list', ())
        if dimension_names:
            entities = list(db_map.get_entity_items(entity_class_name=class_name))
            for entity in entities:
                entity_elements[(class_name, entity['name'])] = entity.get('element_name_list', ())
    return entity_elements


def _build_column_key(
    entity_name: str,
    is_multi_dimensional: bool,
    entity_elements: Dict[Tuple[str, str], Tuple[str, ...]],
    class_name: str,
) -> Any:
    """
    Build the column key for a parameter value entry.

    For multi-dimensional entities, returns a tuple of (normalized_name, *element_names).
    For single-dimensional entities, returns the entity name string.

    Args:
        entity_name: The raw entity name from the database.
        is_multi_dimensional: Whether the entity class is multi-dimensional.
        entity_elements: Lookup dict from _build_entity_elements_lookup.
        class_name: The entity class name.

    Returns:
        A string or tuple suitable for use as a DataFrame column key.
    """
    if is_multi_dimensional:
        element_names = entity_elements.get((class_name, entity_name), ())
        entity_name_normalized = entity_name.replace('__', '.')
        return (entity_name_normalized,) + tuple(element_names)
    return entity_name


def _set_multiindex_columns(
    df: pd.DataFrame,
    is_multi_dimensional: bool,
    dimension_names: Tuple[str, ...],
    data_dict: Dict,
) -> pd.DataFrame:
    """
    Convert DataFrame columns to MultiIndex for multi-dimensional entities.

    Args:
        df: The DataFrame to update.
        is_multi_dimensional: Whether the entity class is multi-dimensional.
        dimension_names: Tuple of dimension names for the entity class.
        data_dict: The data dict used to build the DataFrame (to check length).

    Returns:
        The DataFrame with updated columns (possibly MultiIndex).
    """
    if is_multi_dimensional and len(data_dict) > 0:
        level_names = ['name'] + list(dimension_names)
        df.columns = pd.MultiIndex.from_tuples(df.columns.tolist(), names=level_names)
    return df


def _build_entity_dataframes(
    db_map: DatabaseMapping,
) -> Dict[str, pd.DataFrame]:
    """
    Build dataframes for entities and their scalar parameter values.

    Returns dict mapping dataframe names to DataFrames where:
    - Index is entity name (or MultiIndex for multi-dimensional)
    - Columns are parameter names with scalar values
    """
    dataframes: Dict[str, pd.DataFrame] = {}

    # Get all entity classes
    entity_classes = {ec['name']: ec for ec in db_map.get_entity_class_items()}

    for class_name, ec in entity_classes.items():
        df_name = _entity_class_to_df_name(class_name)
        dimension_names = ec.get('dimension_name_list', ())
        is_multi_dimensional = len(dimension_names) > 0

        # Get entities for this class
        entities = list(db_map.get_entity_items(entity_class_name=class_name))

        if not entities:
            continue

        # Get parameter definitions for this class
        param_defs = list(db_map.get_parameter_definition_items(entity_class_name=class_name))
        param_names = [pd_item['name'] for pd_item in param_defs]

        # Build entity data
        entity_data = []

        for entity in entities:
            entity_name = entity['name']
            element_name_list = entity.get('element_name_list', ())

            # Determine index value
            if is_multi_dimensional:
                # For multi-dimensional, use element names as index tuple
                # Prepend the entity name as first element (matches writer convention)
                # Replace __ with . in entity name for CESM format
                entity_name_normalized = entity_name.replace('__', '.')
                index_val = (entity_name_normalized,) + tuple(element_name_list)
            else:
                index_val = entity_name

            # Get parameter values for this entity
            row_data: Dict[str, Any] = {'_index': index_val}

            for param_name in param_names:
                # Get parameter value
                pv_items = list(db_map.get_parameter_value_items(
                    entity_class_name=class_name,
                    entity_name=entity_name,
                    parameter_definition_name=param_name
                ))

                if pv_items:
                    parsed_value = pv_items[0].get('parsed_value')
                    value, value_type = _parse_parameter_value(parsed_value)

                    # Only include scalar values in entity dataframes
                    if value_type == 'scalar':
                        row_data[param_name] = value

            entity_data.append(row_data)

        if entity_data:
            df = pd.DataFrame(entity_data)

            # Set index
            if is_multi_dimensional:
                # Create MultiIndex from tuples
                index_tuples = df['_index'].tolist()
                # First element is entity name ('name'), rest are dimension elements
                index_names = ['name'] + list(dimension_names)
                df.index = pd.MultiIndex.from_tuples(index_tuples, names=index_names)
            else:
                df.index = pd.Index(df['_index'], name=class_name)

            df = df.drop(columns=['_index'])

            # Only add if there are parameter columns
            if len(df.columns) > 0:
                dataframes[df_name] = df
            elif len(entities) > 0:
                # Add empty dataframe to preserve entity information
                dataframes[df_name] = df

    return dataframes


def _build_indexed_dataframes(
    db_map: DatabaseMapping,
    value_type_filter: str,
    df_name_infix: str,
) -> Dict[str, pd.DataFrame]:
    """
    Build dataframes for indexed parameter values (time series, maps, or arrays).

    This is a generic builder that handles all indexed value types. The value_type_filter
    determines which Spine value type to extract, and df_name_infix determines the naming
    convention for the resulting dataframes.

    Args:
        db_map: Open Spine database mapping.
        value_type_filter: The value type to filter for ('ts', 'map', or 'array').
        df_name_infix: The infix used in dataframe naming (e.g., 'ts', 'str', 'array').

    Returns:
        Dict mapping dataframe names (e.g., 'class_name.ts.param_name') to DataFrames where:
        - Index is datetime (for ts), map index (for map/str), or numeric (for array)
        - Columns are entity names (MultiIndex for multi-dimensional entities)

    For multi-dimensional entity classes (e.g., unit__node), columns are
    MultiIndex with levels: [entity_name, dimension1, dimension2, ...]
    This enables dimension transformations in the transformation layer.
    """
    dataframes: Dict[str, pd.DataFrame] = {}

    # Get all entity classes
    entity_classes = {ec['name']: ec for ec in db_map.get_entity_class_items()}

    # Build entity lookup for multi-dimensional classes
    entity_elements = _build_entity_elements_lookup(db_map, entity_classes)

    for class_name, ec in entity_classes.items():
        df_name_base = _entity_class_to_df_name(class_name)
        dimension_names = ec.get('dimension_name_list', ())
        is_multi_dimensional = len(dimension_names) > 0

        # Get parameter definitions
        param_defs = list(db_map.get_parameter_definition_items(entity_class_name=class_name))

        for param_def in param_defs:
            param_name = param_def['name']

            # Get all parameter values for this class and parameter
            pv_items = list(db_map.get_parameter_value_items(
                entity_class_name=class_name,
                parameter_definition_name=param_name
            ))

            collected_data: Dict[Any, Any] = {}
            max_length = 0  # Only used for arrays

            for pv in pv_items:
                parsed_value = pv.get('parsed_value')
                value, vtype = _parse_parameter_value(parsed_value)

                if vtype != value_type_filter:
                    continue

                entity_name = pv['entity_name']

                try:
                    col_key = _build_column_key(
                        entity_name, is_multi_dimensional, entity_elements, class_name
                    )

                    if value_type_filter == 'ts':
                        indexes = list(value.indexes)
                        values = list(value.values)
                        # Convert to datetime if string timestamps
                        if indexes and isinstance(indexes[0], str):
                            indexes = pd.to_datetime(indexes)
                        collected_data[col_key] = pd.Series(values, index=indexes)

                    elif value_type_filter == 'map':
                        collected_data[col_key] = pd.Series(value.values, value.indexes)

                    elif value_type_filter == 'array':
                        values = list(value.values)
                        collected_data[col_key] = values
                        max_length = max(max_length, len(values))

                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(
                        "Could not parse %s for %s.%s entity '%s': %s",
                        value_type_filter, class_name, param_name, entity_name, e
                    )

            if collected_data:
                # For arrays, pad to uniform length before creating DataFrame
                if value_type_filter == 'array':
                    for key in collected_data:
                        current_len = len(collected_data[key])
                        if current_len < max_length:
                            collected_data[key].extend([None] * (max_length - current_len))

                df = pd.DataFrame(collected_data)

                # Set index name for time series
                if value_type_filter == 'ts':
                    df.index.name = 'datetime'

                # Convert columns to MultiIndex for multi-dimensional entities
                df = _set_multiindex_columns(df, is_multi_dimensional, dimension_names, collected_data)

                full_name = f"{df_name_base}.{df_name_infix}.{param_name}"
                dataframes[full_name] = df

    return dataframes


def _looks_like_datetime(value: str) -> bool:
    """Check if a string looks like a datetime value."""
    if not isinstance(value, str):
        return False
    # Check for common datetime patterns
    return ('T' in value and '-' in value) or (len(value) >= 10 and value[4] == '-' and value[7] == '-')


def spine_to_dataframes(
    db_url: str,
    scenario: str,
) -> Dict[str, pd.DataFrame]:
    """
    Read Spine database and convert to dataframes.

    This reader is data-agnostic and does not perform any datetime conversions.
    String-based time indexes (e.g., 't0001') are preserved as-is. Conversion
    to datetime should be handled by the transformation layer.

    Args:
        db_url: Database URL (e.g., "sqlite:///path/to/database.sqlite")
        scenario: Scenario name to filter data (required)

    Returns:
        Dict mapping dataframe names to DataFrames:
        - Entity dataframes: 'class_name' or 'class1.class2' for multi-dimensional
        - Time series: 'class_name.ts.parameter_name'
        - Maps: 'class_name.str.parameter_name'
        - Arrays: 'class_name.array.parameter_name'

    The dataframe structure mirrors the format expected by writers/to_spine_db.py:
    - Entity dataframes have entity names in index, parameters as columns
    - Time series/map/array dataframes have datetime/index in rows, entities as columns
    """
    # Apply scenario filter to URL
    scenario_filter = scenario_filter_config(scenario)
    filtered_url = append_filter_config(db_url, scenario_filter)

    logger.info("Reading from Spine database with scenario filter: %s", scenario)

    dataframes: Dict[str, pd.DataFrame] = {}

    with DatabaseMapping(filtered_url) as db_map:
        # Make things faster by getting everything to memory at once (avoid lot of sql calls)
        db_map.fetch_all()

        # Build entity dataframes with scalar parameters
        logger.info("Reading entity classes and scalar parameters...")
        entity_dfs = _build_entity_dataframes(db_map)
        dataframes.update(entity_dfs)

        # Build time series dataframes
        logger.info("Reading time series parameters...")
        ts_dfs = _build_indexed_dataframes(db_map, value_type_filter='ts', df_name_infix='ts')
        dataframes.update(ts_dfs)

        # Build map dataframes
        logger.info("Reading map parameters...")
        map_dfs = _build_indexed_dataframes(db_map, value_type_filter='map', df_name_infix='str')
        dataframes.update(map_dfs)

        # Build array dataframes
        logger.info("Reading array parameters...")
        array_dfs = _build_indexed_dataframes(db_map, value_type_filter='array', df_name_infix='array')
        dataframes.update(array_dfs)

    logger.info("Read %d dataframes from database", len(dataframes))

    return dataframes


def list_scenarios(db_url: str) -> List[str]:
    """
    List all scenarios in a Spine database.

    Args:
        db_url: Database URL

    Returns:
        List of scenario names
    """
    with DatabaseMapping(db_url) as db_map:
        scenarios = list(db_map.get_scenario_items())
        return [s['name'] for s in scenarios]


def list_alternatives(db_url: str) -> List[str]:
    """
    List all alternatives in a Spine database.

    Args:
        db_url: Database URL

    Returns:
        List of alternative names
    """
    with DatabaseMapping(db_url) as db_map:
        alternatives = list(db_map.get_alternative_items())
        return [a['name'] for a in alternatives]


# Example usage
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python from_spine_db.py <database_url> <scenario>")
        print("Example: python from_spine_db.py sqlite:///test.sqlite base")
        sys.exit(1)

    db_url = sys.argv[1]
    scenario = sys.argv[2]

    # List available scenarios
    print(f"\nAvailable scenarios: {list_scenarios(db_url)}")
    print(f"Available alternatives: {list_alternatives(db_url)}")

    # Read dataframes
    dfs = spine_to_dataframes(db_url, scenario)

    print("\nDataframes read:")
    for name, df in sorted(dfs.items()):
        print(f"  {name}: {df.shape}")
