"""
Read Spine Toolbox database to dataframes.

This module provides functions to read a Spine database and convert it
to pandas DataFrames, handling entity classes, entities, parameters,
and time series data. This is the inverse of writers/to_spine_db.py.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from spinedb_api import DatabaseMapping
from spinedb_api.filters.scenario_filter import scenario_filter_config
from spinedb_api.filters.tools import append_filter_config


def _parse_parameter_value(value: Any, value_type: Optional[str] = None) -> Any:
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


def _build_entity_dataframes(
    db_map: DatabaseMapping
) -> Dict[str, pd.DataFrame]:
    """
    Build dataframes for entities and their scalar parameter values.

    Returns dict mapping dataframe names to DataFrames where:
    - Index is entity name (or MultiIndex for multi-dimensional)
    - Columns are parameter names with scalar values
    """
    dataframes = {}

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
        param_names = [pd['name'] for pd in param_defs]

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
            row_data = {'_index': index_val}

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


def _build_timeseries_dataframes(
    db_map: DatabaseMapping
) -> Dict[str, pd.DataFrame]:
    """
    Build dataframes for time series parameter values.

    Returns dict mapping dataframe names to DataFrames where:
    - Name format: 'class_name.ts.parameter_name'
    - Index is datetime
    - Columns are entity names (MultiIndex for multi-dimensional entities)

    For multi-dimensional entity classes (e.g., unit__node), columns are
    MultiIndex with levels: [entity_name, dimension1, dimension2, ...]
    This enables dimension transformations in the transformation layer.
    """
    dataframes = {}

    # Get all entity classes
    entity_classes = {ec['name']: ec for ec in db_map.get_entity_class_items()}

    # Build entity lookup for multi-dimensional classes
    # Maps (class_name, entity_name) -> element_name_list
    entity_elements = {}
    for class_name, ec in entity_classes.items():
        dimension_names = ec.get('dimension_name_list', ())
        if dimension_names:
            entities = list(db_map.get_entity_items(entity_class_name=class_name))
            for entity in entities:
                entity_elements[(class_name, entity['name'])] = entity.get('element_name_list', ())

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

            # Filter for time series values
            ts_data = {}
            column_tuples = []  # For MultiIndex columns
            datetime_index = None

            for pv in pv_items:
                parsed_value = pv.get('parsed_value')
                value, value_type = _parse_parameter_value(parsed_value)

                if value_type == 'ts':
                    entity_name = pv['entity_name']

                    # Extract time series data
                    try:
                        indexes = list(value.indexes)
                        values = list(value.values)

                        # Convert to datetime if string timestamps
                        if indexes and isinstance(indexes[0], str):
                            indexes = pd.to_datetime(indexes)

                        # Build column key
                        if is_multi_dimensional:
                            element_names = entity_elements.get((class_name, entity_name), ())
                            # Normalize entity name (replace __ with .)
                            entity_name_normalized = entity_name.replace('__', '.')
                            col_key = (entity_name_normalized,) + tuple(element_names)
                            column_tuples.append(col_key)
                        else:
                            col_key = entity_name

                        ts_data[col_key] = pd.Series(values, index=indexes)

                        if datetime_index is None:
                            datetime_index = indexes
                    except Exception as e:
                        print(f"  Warning: Could not parse time series for {entity_name}: {e}")

            if ts_data:
                df = pd.DataFrame(ts_data)
                df.index.name = 'datetime'

                # Convert columns to MultiIndex for multi-dimensional entities
                if is_multi_dimensional and column_tuples:
                    level_names = ['name'] + list(dimension_names)
                    df.columns = pd.MultiIndex.from_tuples(df.columns.tolist(), names=level_names)

                full_name = f"{df_name_base}.ts.{param_name}"
                dataframes[full_name] = df

    return dataframes


def _build_map_dataframes(
    db_map: DatabaseMapping
) -> Dict[str, pd.DataFrame]:
    """
    Build dataframes for Map (string-indexed) parameter values.

    Returns dict mapping dataframe names to DataFrames where:
    - Name format: 'class_name.str.parameter_name'
    - Index is the map index (often datetime strings)
    - Columns are entity names (MultiIndex for multi-dimensional entities)

    For multi-dimensional entity classes (e.g., unit__node), columns are
    MultiIndex with levels: [entity_name, dimension1, dimension2, ...]
    """
    dataframes = {}

    entity_classes = {ec['name']: ec for ec in db_map.get_entity_class_items()}

    # Build entity lookup for multi-dimensional classes
    entity_elements = {}
    for class_name, ec in entity_classes.items():
        dimension_names = ec.get('dimension_name_list', ())
        if dimension_names:
            entities = list(db_map.get_entity_items(entity_class_name=class_name))
            for entity in entities:
                entity_elements[(class_name, entity['name'])] = entity.get('element_name_list', ())

    for class_name, ec in entity_classes.items():
        df_name_base = _entity_class_to_df_name(class_name)
        dimension_names = ec.get('dimension_name_list', ())
        is_multi_dimensional = len(dimension_names) > 0

        param_defs = list(db_map.get_parameter_definition_items(entity_class_name=class_name))

        for param_def in param_defs:
            param_name = param_def['name']

            pv_items = list(db_map.get_parameter_value_items(
                entity_class_name=class_name,
                parameter_definition_name=param_name
            ))

            map_data = {}
            for pv in pv_items:
                parsed_value = pv.get('parsed_value')
                value, value_type = _parse_parameter_value(parsed_value)

                if value_type == 'map':  # Map type
                    entity_name = pv['entity_name']

                    try:
                        # Build column key
                        if is_multi_dimensional:
                            element_names = entity_elements.get((class_name, entity_name), ())
                            entity_name_normalized = entity_name.replace('__', '.')
                            col_key = (entity_name_normalized,) + tuple(element_names)
                        else:
                            col_key = entity_name

                        map_data[col_key] = pd.Series(value.values, value.indexes)
                    except Exception as e:
                        print(f"  Warning: Could not parse map for {entity_name}: {e}")

            if map_data:
                df = pd.DataFrame(map_data)

                # Convert columns to MultiIndex for multi-dimensional entities
                if is_multi_dimensional and len(map_data) > 0:
                    level_names = ['name'] + list(dimension_names)
                    df.columns = pd.MultiIndex.from_tuples(df.columns.tolist(), names=level_names)

                full_name = f"{df_name_base}.str.{param_name}"
                dataframes[full_name] = df

    return dataframes


def _build_array_dataframes(
    db_map: DatabaseMapping
) -> Dict[str, pd.DataFrame]:
    """
    Build dataframes for Array parameter values.

    Returns dict mapping dataframe names to DataFrames where:
    - Name format: 'class_name.array.parameter_name'
    - Index is numeric (0, 1, 2, ...)
    - Columns are entity names (MultiIndex for multi-dimensional entities)

    For multi-dimensional entity classes (e.g., unit__node), columns are
    MultiIndex with levels: [entity_name, dimension1, dimension2, ...]
    """
    dataframes = {}

    entity_classes = {ec['name']: ec for ec in db_map.get_entity_class_items()}

    # Build entity lookup for multi-dimensional classes
    entity_elements = {}
    for class_name, ec in entity_classes.items():
        dimension_names = ec.get('dimension_name_list', ())
        if dimension_names:
            entities = list(db_map.get_entity_items(entity_class_name=class_name))
            for entity in entities:
                entity_elements[(class_name, entity['name'])] = entity.get('element_name_list', ())

    for class_name, ec in entity_classes.items():
        df_name_base = _entity_class_to_df_name(class_name)
        dimension_names = ec.get('dimension_name_list', ())
        is_multi_dimensional = len(dimension_names) > 0

        param_defs = list(db_map.get_parameter_definition_items(entity_class_name=class_name))

        for param_def in param_defs:
            param_name = param_def['name']

            pv_items = list(db_map.get_parameter_value_items(
                entity_class_name=class_name,
                parameter_definition_name=param_name
            ))

            array_data = {}
            max_length = 0

            for pv in pv_items:
                parsed_value = pv.get('parsed_value')
                value, value_type = _parse_parameter_value(parsed_value)

                if value_type == 'array':
                    entity_name = pv['entity_name']

                    try:
                        values = list(value.values)

                        # Build column key
                        if is_multi_dimensional:
                            element_names = entity_elements.get((class_name, entity_name), ())
                            entity_name_normalized = entity_name.replace('__', '.')
                            col_key = (entity_name_normalized,) + tuple(element_names)
                        else:
                            col_key = entity_name

                        array_data[col_key] = values
                        max_length = max(max_length, len(values))
                    except Exception as e:
                        print(f"  Warning: Could not parse array for {entity_name}: {e}")

            if array_data:
                # Pad arrays to same length
                for key in array_data:
                    if len(array_data[key]) < max_length:
                        array_data[key].extend([None] * (max_length - len(array_data[key])))

                df = pd.DataFrame(array_data)

                # Convert columns to MultiIndex for multi-dimensional entities
                if is_multi_dimensional and len(array_data) > 0:
                    level_names = ['name'] + list(dimension_names)
                    df.columns = pd.MultiIndex.from_tuples(df.columns.tolist(), names=level_names)

                full_name = f"{df_name_base}.array.{param_name}"
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
    scenario: str
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

    print(f"Reading from Spine database with scenario filter: {scenario}")

    dataframes = {}

    with DatabaseMapping(filtered_url) as db_map:
        # Make things faster by getting everything to memory at once (avoid lot of sql calls)
        db_map.fetch_all()

        # Build entity dataframes with scalar parameters
        print("  Reading entity classes and scalar parameters...")
        entity_dfs = _build_entity_dataframes(db_map)
        dataframes.update(entity_dfs)

        # Build time series dataframes
        print("  Reading time series parameters...")
        ts_dfs = _build_timeseries_dataframes(db_map)
        dataframes.update(ts_dfs)

        # Build map dataframes
        print("  Reading map parameters...")
        map_dfs = _build_map_dataframes(db_map)
        dataframes.update(map_dfs)

        # Build array dataframes
        print("  Reading array parameters...")
        array_dfs = _build_array_dataframes(db_map)
        dataframes.update(array_dfs)

    print(f"  Read {len(dataframes)} dataframes from database")

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
