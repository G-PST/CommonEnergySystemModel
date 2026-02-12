import ast
import pandas as pd
import yaml
from typing import Dict, List, Any, Tuple, Union, Optional
from datetime import datetime, timedelta
import numpy as np
import logging


def _get_timestep_minutes_from_data(source_dfs: Dict[str, pd.DataFrame]) -> Optional[int]:
    """
    Extract timestep duration in minutes from timeline.str.timestep_duration.

    Reads the first value from the timestep_duration map parameter.

    Args:
        source_dfs: Dictionary of source dataframes

    Returns:
        Timestep duration in minutes, or None if not found
    """
    ts_df_name = 'timeline.str.timestep_duration'
    if ts_df_name not in source_dfs:
        return None

    ts_df = source_dfs[ts_df_name]
    if ts_df.empty:
        return None

    # Get the first non-null value (duration in hours)
    for col in ts_df.columns:
        values = ts_df[col].dropna()
        if len(values) > 0:
            hours = float(values.iloc[0])
            return int(hours * 60)

    return None


def _convert_str_index_to_datetime(
    df: pd.DataFrame,
    source_dfs: Dict[str, pd.DataFrame],
    start_time: datetime
) -> pd.DataFrame:
    """
    Convert string-based timestep index to datetime index.

    Handles indexes like 't0001', 't0002', etc. by extracting the numeric
    part and calculating datetime based on start_time and timestep duration
    from timeline.str.timestep_duration.

    Args:
        df: DataFrame with string index (e.g., 't0001', 't0002')
        source_dfs: Dictionary of source dataframes (to get timestep duration)
        start_time: Datetime of the first timestep

    Returns:
        DataFrame with DatetimeIndex, or original if conversion fails
    """
    if df.empty:
        return df

    # Already has DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # Get timestep duration from source data
    timestep_minutes = _get_timestep_minutes_from_data(source_dfs)
    if timestep_minutes is None:
        logging.warning("Could not determine timestep duration for str→ts conversion")
        return df

    # Try to extract numeric part from index values
    new_index = []
    for idx_val in df.index:
        if isinstance(idx_val, str):
            # Extract numeric part (e.g., 't0001' -> 1, 'step_5' -> 5)
            numeric_part = ''.join(c for c in idx_val if c.isdigit())
            if numeric_part:
                step_num = int(numeric_part) - 1  # Convert to 0-based
                dt = start_time + timedelta(minutes=step_num * timestep_minutes)
                new_index.append(dt)
            else:
                # Can't parse, return original dataframe
                return df
        else:
            # Already numeric or other type, return original
            return df

    df = df.copy()
    df.index = pd.DatetimeIndex(new_index)
    df.index.name = 'datetime'
    return df


def _datetime_index_to_str(index: pd.Index) -> pd.Index:
    """
    Convert a DatetimeIndex to string Index with consistent UTC formatting.

    For DatetimeIndex:
    - Converts timezone-aware timestamps to UTC
    - Uses ISO 8601 format with 'T' separator: '2023-01-01T00:00:00'

    For other index types:
    - Uses default string conversion

    This ensures all timestamps are UTC and consistently formatted.
    """
    if isinstance(index, pd.DatetimeIndex):
        # Convert to UTC if timezone-aware
        if index.tz is not None:
            index = index.tz_convert('UTC').tz_localize(None)
        # Format with 'T' separator, no 'Z' suffix
        return pd.Index(index.strftime('%Y-%m-%dT%H:%M:%S'), name=index.name)
    else:
        return index.astype(str)


def load_config(config_path: str) -> Dict:
    """Load the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def is_timeseries(df: pd.DataFrame) -> bool:
    """Check if DataFrame is a time series (has datetime in index)."""
    return (df.index.name == 'datetime' or 
            (isinstance(df.index, pd.MultiIndex) and 'datetime' in df.index.names))


def get_entity_index(df: pd.DataFrame) -> Union[pd.Index, pd.MultiIndex]:
    """
    Get the entity index from a DataFrame.
    For time series: returns columns (entities are in columns)
    For regular data: returns index (entities are in index)
    """
    if is_timeseries(df):
        return df.columns
    else:
        return df.index


def set_entity_index(df: pd.DataFrame, new_index: Union[pd.Index, pd.MultiIndex], 
                     is_pivoted: bool = False) -> pd.DataFrame:
    """
    Set the entity index for a DataFrame.
    For time series: sets columns
    For regular data: sets index
    """
    result = df.copy()
    if is_pivoted:
        result.columns = new_index
    else:
        result.index = new_index
    return result


def list_of_lists_to_index(data):
    if not data:
        return pd.Index([])
    if len(data[0]) == 1:
        return pd.Index([item[0] for item in data])
    else:
        return pd.MultiIndex.from_tuples([tuple(item) for item in data])

def index_to_names(idx: Union[pd.Index, pd.MultiIndex]) -> List[str]:
    if isinstance(idx, pd.MultiIndex):
        return [list(row) for row in idx]
    else:
        return [[item] for item in idx]

def parse_operation(operation_name: str, config: Dict) -> Tuple[List, List, List]:
    """
    Parse a single operation from the config.

    Returns:
        source_specs: List of source class:attribute pairs or just classes
        target_specs: List of target class:attribute pairs or just classes
        operations: List of additional operations (includes dimensions, match_by, order, etc.)
    """
    operation_config = config[operation_name]

    source_specs = []
    target_specs = []
    operations = []

    for i, item in enumerate(operation_config):
        if i == 0:
            source_specs = parse_spec(item)
        elif i == 1:
            target_specs = parse_spec(item)
        else:
            if isinstance(item, dict):
                operations.append(item)
            elif isinstance(item, list):
                operations.extend(item)

    return source_specs, target_specs, operations


def parse_spec(spec: Any) -> List[Dict]:
    """Parse a source or target specification."""
    result = []
    
    if isinstance(spec, str):
        result.append({'class': spec, 'attribute': None})
    elif isinstance(spec, dict):
        for key, value in spec.items():
            if isinstance(value, dict):
                result.append({'class': key, 'attribute': None, 'rule': value})
            else:
                result.append({'class': key, 'attribute': value})
    elif isinstance(spec, list):
        for item in spec:
            result.extend(parse_spec(item))
    
    return result


def get_operation_type(source_specs: List[Dict], target_specs: List[Dict], 
                       operations: List[Dict]) -> str:
    """Determine the type of operation to perform."""
    source_has_attr = any(s.get('attribute') for s in source_specs)
    target_has_attr = any(t.get('attribute') for t in target_specs)
    
    if not source_has_attr and not target_has_attr:
        return 'copy_entities'
    elif not source_has_attr and target_has_attr and any('value' in d for d in operations):
        return 'create_parameter'
    elif source_has_attr and target_has_attr:
        return 'transform_parameter'
    else:
        return 'unknown'


def copy_entities(source_dfs: Dict[str, pd.DataFrame],
                 source_specs: List[Dict],
                 target_spec: Dict,
                 target_dfs: Dict[str, pd.DataFrame],
                 operations: List[Dict]) -> Dict[str, pd.DataFrame]:
    """Copy entities from source class to target class."""
    # Extract dimensions from operations
    dimensions = None
    for op_dict in operations:
        if 'dimensions' in op_dict:
            dimensions = op_dict['dimensions']
            break

    target_class = target_spec['class']
    for source_spec in source_specs:
        source_class = source_spec['class']
        
        if source_class not in source_dfs:
            return target_dfs
        
        source_df = source_dfs[source_class]
        source_idx = source_df.index
        
        # Build target entities based on order specification
        if 'rule' in target_spec:
            source_names = index_to_names(source_idx).copy()
            if 'if_parameter' in target_spec['rule']:
                filtered_source_names = []
                if_parameters = target_spec['rule']['if_parameter']
                if not isinstance(if_parameters, list):
                    if_parameters = [if_parameters]
                for if_param in if_parameters:
                    splitted_keys = [x.split('.') for x in list(source_dfs.keys())]
                    long_keys = [x for x in splitted_keys if len(x)>2]
                    df_keys_found = ['.'.join(x) for x in long_keys if x[0] == source_class and x[2] == if_param]
                    for df_key_found in df_keys_found:
                        for item in source_names:
                            if tuple(item) in source_dfs[df_key_found].columns:
                                filtered_source_names.append(item)
                    for item in source_names:
                        if if_param in source_df.columns:
                            item_key = tuple(item) if len(item) > 1 else item[0]
                            if pd.notna(source_df.loc[item_key, if_param]):
                                if item not in filtered_source_names:
                                    filtered_source_names.append(item)
                source_names = filtered_source_names
            if 'if_not_parameter' in target_spec['rule']:
                # Exclude entities that have non-null values for specified parameters
                if_not_parameters = target_spec['rule']['if_not_parameter']
                if not isinstance(if_not_parameters, list):
                    if_not_parameters = [if_not_parameters]
                excluded_names = []
                for if_not_param in if_not_parameters:
                    # Check in pivoted dataframes (time series)
                    splitted_keys = [x.split('.') for x in list(source_dfs.keys())]
                    long_keys = [x for x in splitted_keys if len(x) > 2]
                    df_keys_found = ['.'.join(x) for x in long_keys if x[0] == source_class and x[2] == if_not_param]
                    for df_key_found in df_keys_found:
                        for item in source_names:
                            if tuple(item) in source_dfs[df_key_found].columns:
                                if item not in excluded_names:
                                    excluded_names.append(item)
                    # Check in regular dataframes
                    for item in source_names:
                        if if_not_param in source_df.columns:
                            item_key = tuple(item) if len(item) > 1 else item[0]
                            if pd.notna(source_df.loc[item_key, if_not_param]):
                                if item not in excluded_names:
                                    excluded_names.append(item)
                # Filter out excluded names
                source_names = [item for item in source_names if item not in excluded_names]
            # Create target index from filtered source names (if no order specified)
            if source_names:
                target_idx = list_of_lists_to_index(source_names)
            else:
                target_idx = pd.Index([])
            if 'order' in target_spec['rule']:
                order = target_spec['rule']['order']

                # Build target names according to order spec
                target_names = []
                for parts in source_names:
                    target_dims = []
                    for target_dim_spec in order:
                        # Combine source dimensions
                        dim_parts = [parts[idx] for idx in target_dim_spec]
                        target_dims.append('__'.join(dim_parts))
                    target_names.append(target_dims)
                # Create entity index (handle empty list case)
                if target_names:
                    target_idx = list_of_lists_to_index(target_names)
                else:
                    target_idx = pd.Index([])
                    
        else:
            target_idx = source_idx
        
        # Store or merge with existing target dataframe
        if target_class not in target_dfs:
            # Create new dataframe with just the index
            target_dfs[target_class] = pd.DataFrame(index=target_idx)
        else:
            # Append new rows while preserving existing data
            existing_df = target_dfs[target_class]
            combined = existing_df.index.union(target_idx)
            target_dfs[target_class] = existing_df.reindex(combined)
        if not dimensions:  # Take dimensions from target_class names if not provided in the spec
            dimensions = target_class.split('.')
        if len(dimensions) > 1:  # Multi-dimensional target classes need to have name index too
            dimensions = ['name'] + dimensions
        # Only set dimension names if index is not empty
        if len(target_dfs[target_class].index) > 0:
            if len(dimensions) > 1:
                target_dfs[target_class].index.names = dimensions
            elif len(dimensions) == 1:
                target_dfs[target_class].index.name = dimensions[0]
    return target_dfs


def create_parameter(source_dfs: Dict[str, pd.DataFrame],
                    source_specs: List[Dict],
                    target_spec: Dict,
                    target_dfs: Dict[str, pd.DataFrame],
                    operations: List[Dict]) -> Dict[str, pd.DataFrame]:
    """Create a parameter with value from configuration file or source data."""
    target_class = target_spec['class']
    target_attribute = target_spec['attribute']
    for source_spec in source_specs:
        source_class = source_spec['class']

        source_df = source_dfs[source_class]
        source_idx = get_entity_index(source_df)
        source_names = index_to_names(source_idx)

        # Check for if_parameter filter in operations
        for op_dict in operations:
            if 'if_parameter' in op_dict:
                filtered_source_names = []
                if_parameters = op_dict['if_parameter']
                for if_param in if_parameters:
                    # Check in pivoted dataframes (time series)
                    splitted_keys = [x.split('.') for x in list(source_dfs.keys())]
                    long_keys = [x for x in splitted_keys if len(x) > 2]
                    df_keys_found = ['.'.join(x) for x in long_keys if x[0] == source_class and x[2] == if_param]
                    for df_key_found in df_keys_found:
                        for item in source_names:
                            if tuple(item) in source_dfs[df_key_found].columns and item not in filtered_source_names:
                                filtered_source_names.append(item)
                    # Check in regular dataframes
                    for item in source_names:
                        if if_param in source_df.columns and pd.notna(source_df.loc[tuple(item) if len(item) > 1 else item[0], if_param]):
                            if item not in filtered_source_names:
                                filtered_source_names.append(item)
                source_names = filtered_source_names
                source_idx = list_of_lists_to_index(source_names) if source_names else pd.Index([])

        # Skip if no matching entities after filtering
        if len(source_idx) == 0:
            continue

        target_idx = source_idx  # Will be replaced if order is given

        # Get the value from operations
        new_parameter_value = None
        for op_dict in operations:
            if 'order' in op_dict:
                target_idx = reorder_entity_names(source_idx, op_dict['order'])

            if 'value' in op_dict:
                value_spec = op_dict['value']
                # Check if value is a dimension extraction (list with single int)
                if isinstance(value_spec, list) and len(value_spec) == 1 and isinstance(value_spec[0], int):
                    # Extract dimension value from source entity index
                    dim_idx = value_spec[0]
                    new_parameter_value = [name[dim_idx] if len(name) > dim_idx else None
                                          for name in source_names]
                else:
                    # Constant value
                    new_parameter_value = value_spec

            # Apply rename if specified
            if 'rename' in op_dict:
                raise ValueError(f"There should be no renaming for parameters that are freshly created through _value_, target attribute: {target_attribute}")

        # Skip if no value was specified
        if new_parameter_value is None:
            continue

        # Create target index and dataframe
        param_data = pd.DataFrame({target_attribute: new_parameter_value}, index=target_idx)
        
        # Store result in target class dataframe
        if target_class not in target_dfs:
            target_dfs[target_class] = param_data
        else:
            # Add column to existing dataframe
            if target_attribute not in target_dfs[target_class].columns:
                target_dfs[target_class][target_attribute] = None
            
            # Update values for matching indices
            common_idx = target_dfs[target_class].index.intersection(param_data.index)
            target_dfs[target_class].loc[common_idx, target_attribute] = param_data.loc[common_idx, target_attribute]
            
            # Add new indices
            new_idx = param_data.index.difference(target_dfs[target_class].index)
            if len(new_idx) > 0:
                new_data = pd.DataFrame({target_attribute: param_data.loc[new_idx, target_attribute]}, index=new_idx)
                target_dfs[target_class] = pd.concat([target_dfs[target_class], new_data])
    
    return target_dfs


def transform_parameter(source_dfs: Dict[str, pd.DataFrame],
                       source_specs: List[Dict],
                       target_spec: Dict,
                       target_dfs: Dict[str, pd.DataFrame],
                       operations: List[Dict],
                       start_time: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
    """Transform parameters from source to target.

    Args:
        source_dfs: Dictionary of source dataframes
        source_specs: List of source specifications
        target_spec: Target specification
        target_dfs: Dictionary of target dataframes (will be modified)
        operations: List of operations to apply (may include dimensions, match_by, order, etc.)
        start_time: Optional start datetime for str→ts index conversion

    Returns:
        Updated target_dfs dictionary
    """
    # Extract match_by from operations
    match_by = None
    for op_dict in operations:
        if 'match_by' in op_dict:
            match_by = op_dict['match_by']
            break

    target_class = target_spec['class']
    target_attribute = target_spec['attribute']
    is_pivoted = False

    # Compute target class name once before the loop if target_attribute indicates pivoted data
    if isinstance(target_attribute, list) and len(target_attribute) == 2:
        target_class = target_class + '.' + '.'.join(target_attribute[1]) + '.' + target_attribute[0]

    # Gather source data - collect all source dataframes first for multi-source operations
    source_dfs_list = []
    source_attributes = []  # Track source attributes for use in rename operations
    for source_spec in source_specs:
        source_class = source_spec['class']
        source_attribute = source_spec.get('attribute')
        source_attributes.append(source_attribute)

        # List based attributes indicate that the dataframe is pivoted (usually for time series content) and contains only one attribute
        if isinstance(source_attribute, list):
            is_pivoted = True
            if len(source_attribute) == 2:
                source_datatypes = source_attribute[1]
                if not isinstance(target_attribute, list):
                    raise ValueError(f"Source attribute is of list type (indicates pivoted data) and there is a datatype conversion (second item in the attribute list), but target attribute does not have a list with two items. {source_class} {source_attribute}")
                if len(source_datatypes) != len(target_attribute[1]):
                    raise ValueError(f"Source attribute is of list type (indicates pivoted data) and there is a datatype conversion, but target attribute does not have the same number of datatypes in the attribute list. {source_class} {source_attribute}")
                source_class = source_class + '.' + '.'.join(source_attribute[1]) + '.' + source_attribute[0]
            else:
                raise ValueError(f"Source attribute is of list type (indicates pivoted data) but the length of the list is not 2 (the list should have parameter name and data type list for the index columns). {source_class} {source_attribute}")

        # Check if source class exists
        if source_class not in source_dfs:
            if is_pivoted:
                if source_attribute[1][0] == 'ts':
                    source_dfs[source_class] = source_dfs['timeline']
                else:
                    # Missing pivoted data (e.g., arrays) - skip this transformation
                    logging.warning(f"Could not find source dataframe '{source_class}'")
                    return target_dfs
            else:
                logging.warning(f"Could not find source dataframe from class '{source_class}'")
                return target_dfs

        if is_pivoted:
            source_df = source_dfs[source_class]
        else:
            # If attribute specified, extract that column
            if source_attribute in source_dfs[source_class].columns:
                # Use the attribute column
                source_df = source_dfs[source_class][[source_attribute]]
            else:
                logging.warning(f"Could not find source parameter '{source_attribute}' from class '{source_class}'")
                return target_dfs

        source_dfs_list.append(source_df)

    # Use the first source as the starting point
    result = source_dfs_list[0].copy() if source_dfs_list else pd.DataFrame()

    # Apply if_parameter / if_not_parameter filtering
    # This filters entities based on whether they have (or don't have) certain parameters
    # Collect all filter operations first
    if_parameters_all = []
    if_not_parameters_all = []
    for op_dict in operations:
        if 'if_parameter' in op_dict:
            params = op_dict['if_parameter']
            if isinstance(params, list):
                if_parameters_all.extend(params)
            else:
                if_parameters_all.append(params)
        if 'if_not_parameter' in op_dict:
            params = op_dict['if_not_parameter']
            if isinstance(params, list):
                if_not_parameters_all.extend(params)
            else:
                if_not_parameters_all.append(params)

    if if_parameters_all or if_not_parameters_all:
        # Get the base source class (without attribute type suffix)
        base_source_class = source_specs[0]['class']
        base_source_df = source_dfs.get(base_source_class, pd.DataFrame())

        # Get entity index from result
        if is_pivoted:
            entity_names = list(result.columns)
        else:
            entity_names = index_to_names(result.index)

        filtered_names = entity_names.copy()

        # Apply if_parameter filter (include only entities that have the parameter)
        if if_parameters_all:
            included_names = []
            for if_param in if_parameters_all:
                # Check in pivoted dataframes (time series)
                splitted_keys = [x.split('.') for x in list(source_dfs.keys())]
                long_keys = [x for x in splitted_keys if len(x) > 2]
                df_keys_found = ['.'.join(x) for x in long_keys if x[0] == base_source_class and x[2] == if_param]
                for df_key_found in df_keys_found:
                    for item in filtered_names:
                        item_key = item if isinstance(item, str) else (tuple(item) if len(item) > 1 else item[0])
                        if item_key in source_dfs[df_key_found].columns:
                            if item not in included_names:
                                included_names.append(item)
                # Check in regular dataframes
                if if_param in base_source_df.columns:
                    for item in filtered_names:
                        item_key = item if isinstance(item, str) else (tuple(item) if len(item) > 1 else item[0])
                        if item_key in base_source_df.index and pd.notna(base_source_df.loc[item_key, if_param]):
                            if item not in included_names:
                                included_names.append(item)
            filtered_names = included_names

        # Apply if_not_parameter filter (exclude entities that have the parameter)
        if if_not_parameters_all:
            excluded_names = []
            for if_not_param in if_not_parameters_all:
                # Check in pivoted dataframes (time series)
                splitted_keys = [x.split('.') for x in list(source_dfs.keys())]
                long_keys = [x for x in splitted_keys if len(x) > 2]
                df_keys_found = ['.'.join(x) for x in long_keys if x[0] == base_source_class and x[2] == if_not_param]
                for df_key_found in df_keys_found:
                    for item in filtered_names:
                        item_key = item if isinstance(item, str) else (tuple(item) if len(item) > 1 else item[0])
                        if item_key in source_dfs[df_key_found].columns:
                            if item not in excluded_names:
                                excluded_names.append(item)
                # Check in regular dataframes
                if if_not_param in base_source_df.columns:
                    for item in filtered_names:
                        item_key = item if isinstance(item, str) else (tuple(item) if len(item) > 1 else item[0])
                        if item_key in base_source_df.index and pd.notna(base_source_df.loc[item_key, if_not_param]):
                            if item not in excluded_names:
                                excluded_names.append(item)
            filtered_names = [item for item in filtered_names if item not in excluded_names]

        # Apply the filter to result
        if is_pivoted:
            # Filter columns for pivoted data
            cols_to_keep = [col for col in result.columns if col in filtered_names]
            result = result[cols_to_keep]
        else:
            # Filter rows for regular data
            if filtered_names:
                idx_to_keep = list_of_lists_to_index(filtered_names) if isinstance(filtered_names[0], list) else pd.Index(filtered_names)
                result = result.loc[result.index.intersection(idx_to_keep)]
            else:
                result = pd.DataFrame()

    # There is a datatype change operation that needs to be performed
    if is_pivoted:
        target_datatypes = target_attribute[1]
        if isinstance(result.index, pd.MultiIndex):
            for i, source_datatype in enumerate(source_datatypes):
                if source_datatype is not target_datatypes[i]:
                    if target_datatypes[i] == "str":
                        # Use consistent UTC formatting for datetime levels
                        level_index = result.index.levels[i]
                        result.index = result.index.set_levels(
                            _datetime_index_to_str(level_index), level=i
                        )
                    if target_datatypes[i] == "ts":
                        result.index = result.index.set_levels(
                            pd.to_datetime(result.index.levels[i]), level=i
                        )
        else:
            if source_datatypes[0] is not target_datatypes[0]:
                if target_datatypes[0] == "str":
                    # Use consistent UTC formatting for datetime index
                    result.index = _datetime_index_to_str(result.index)
                elif target_datatypes[0] == "ts":
                    # Convert string index to datetime - use helper if start_time provided
                    if start_time is not None:
                        result = _convert_str_index_to_datetime(result, source_dfs, start_time)
                    else:
                        # Try direct conversion (works for ISO datetime strings)
                        try:
                            result.index = pd.to_datetime(result.index)
                        except Exception:
                            logging.warning("Could not convert index to datetime - start_time not provided")
                elif target_datatypes[0] == "array":
                    # Use consistent UTC formatting for datetime index
                    result.index = _datetime_index_to_str(result.index)

    # Apply operations
    for op_dict in operations:
        if 'operation' in op_dict:
            operation = op_dict['operation']

            if operation == 'multiply':
                with_value = op_dict.get('with')
                if with_value is not None and not isinstance(with_value, list):
                    for col in result.select_dtypes(include=[np.number]).columns:
                        result[col] = result[col] * with_value
                else:
                    raise ValueError(f"When trying to do {op_dict}: 'operation' and 'with' work only with constant values - use 'algebra' if operations between two dataframes are needed")

            elif operation == 'divide':
                with_value = op_dict.get('with')
                if with_value is not None and not isinstance(with_value, list):
                    for col in result.select_dtypes(include=[np.number]).columns:
                        result[col] = result[col] / with_value
                else:
                    raise ValueError(f"When trying to do {op_dict}: 'operation' and 'with' work only with constant values - use 'algebra' if operations between two dataframes are needed")

            elif operation == 'add':
                with_value = op_dict.get('with')
                if with_value is not None and not isinstance(with_value, list):
                    for col in result.select_dtypes(include=[np.number]).columns:
                        result[col] = result[col] + with_value
                else:
                    raise ValueError(f"When trying to do {op_dict}: 'operation' and 'with' work only with constant values - use 'algebra' if operations between two dataframes are needed")

            elif operation == 'subtract':
                with_value = op_dict.get('with')
                if with_value is not None and not isinstance(with_value, list):
                    for col in result.select_dtypes(include=[np.number]).columns:
                        result[col] = result[col] - with_value
                else:
                    raise ValueError(f"When trying to do {op_dict}: 'operation' and 'with' work only with constant values - use 'algebra' if operations between two dataframes are needed")

            elif operation == 'sum':
                # Sum all source dataframes
                for df in source_dfs_list[1:]:
                    result = add_dataframes(result, df)

        if 'algebra' in op_dict:
            algebra_expr = op_dict['algebra']
            match_spec = op_dict.get('match', [])
            result = evaluate_algebra_expression(
                algebra_expr,
                source_dfs_list,
                match_spec,
                is_pivoted
            )

        if 'order' in op_dict:
            # Check if order is nested (order: order: [[1]]) or direct (order: [[1]])
            order_spec = op_dict['order']
            if isinstance(order_spec, dict) and 'order' in order_spec:
                order_list = order_spec['order']
                aggregate = order_spec.get('aggregate')
            else:
                order_list = order_spec
                aggregate = op_dict.get('aggregate')

            result = reorder_dimensions(result, order_list, aggregate, is_pivoted)

        if 'rename' in op_dict:
            # Use the first non-pivoted source attribute for rename
            rename_attr = None
            for attr in source_attributes:
                if not isinstance(attr, list):
                    rename_attr = attr
                    break
            if rename_attr is None and source_attributes:
                attr = source_attributes[0]
                rename_attr = attr[0] if isinstance(attr, list) else attr
            if rename_attr:
                result = apply_rename(result, rename_attr, op_dict['rename'], is_pivoted)

    # Rename data column to target attribute if needed
    if not is_pivoted and (len(result.columns) == 1):
        result.columns = [target_attribute]

    # Store result in target class dataframe
    if target_class not in target_dfs:
        target_dfs[target_class] = result
    else:
        # Add column to existing dataframe or update
        if is_pivoted:
            # For pivoted data (time series), add any new entity columns
            for col in result.columns:
                if col not in target_dfs[target_class].columns:
                    target_dfs[target_class][col] = None
        else:
            if target_attribute not in target_dfs[target_class].columns:
                # Add new column
                for col in result.columns:
                    target_dfs[target_class][col] = None

        # Update values for matching indices
        target_idx = target_dfs[target_class].index
        result_idx = result.index

        # Check if we need match_by logic: target is MultiIndex, result is simpler
        if (match_by is not None and
            isinstance(target_idx, pd.MultiIndex) and
            not isinstance(result_idx, pd.MultiIndex)):
            # Extract 'name' level from target (first level of MultiIndex)
            name_level = target_idx.get_level_values('name') if 'name' in target_idx.names else target_idx.get_level_values(0)

            # Build match keys from target names by extracting specified components
            # e.g., 'battery_inverter.west.battery' with match_by=[0] -> 'battery_inverter'
            def extract_match_key(name, match_by_indices):
                parts = name.split('.')
                key_parts = [parts[i] for i in match_by_indices if i < len(parts)]
                return '.'.join(key_parts)

            target_match_keys = [extract_match_key(name, match_by) for name in name_level]

            # For each source entity, find matching target entities and broadcast value
            for source_name in result_idx:
                source_key = str(source_name)
                # Find all target indices where the match key equals source key
                matching_positions = [i for i, key in enumerate(target_match_keys) if key == source_key]
                if matching_positions:
                    matching_target_idx = target_idx[matching_positions]
                    for col in result.columns:
                        if col in target_dfs[target_class].columns:
                            target_dfs[target_class].loc[matching_target_idx, col] = result.loc[source_name, col]
        else:
            # Standard matching logic for compatible indices
            common_idx = target_idx.intersection(result_idx)
            for col in result.columns:
                if col in target_dfs[target_class].columns:
                    target_dfs[target_class].loc[common_idx, col] = result.loc[common_idx, col]

            # Add new indices (only when not using match_by)
            new_idx = result_idx.difference(target_idx)
            if len(new_idx) > 0:
                target_dfs[target_class] = pd.concat([target_dfs[target_class], result.loc[new_idx]])

    return target_dfs


def _do_op(op: ast.operator, a, b):
    """Dispatch arithmetic operator to perform operation on a and b."""
    if isinstance(op, ast.Mult):
        return a * b
    elif isinstance(op, ast.Div):
        return a / b
    elif isinstance(op, ast.Add):
        return a + b
    elif isinstance(op, ast.Sub):
        return a - b
    else:
        raise ValueError(f"Unsupported operator: {type(op).__name__}")


def _apply_scalar_op(op: ast.operator, df: pd.DataFrame, scalar: float,
                     is_pivoted: bool, scalar_on_right: bool = True) -> pd.DataFrame:
    """Apply a scalar operation to a DataFrame."""
    result = df.copy()
    if scalar_on_right:
        if isinstance(op, ast.Mult):
            return result * scalar
        elif isinstance(op, ast.Div):
            return result / scalar
        elif isinstance(op, ast.Add):
            return result + scalar
        elif isinstance(op, ast.Sub):
            return result - scalar
    else:
        # Scalar is on the left side
        if isinstance(op, ast.Mult):
            return scalar * result
        elif isinstance(op, ast.Div):
            return scalar / result
        elif isinstance(op, ast.Add):
            return scalar + result
        elif isinstance(op, ast.Sub):
            return scalar - result
    raise ValueError(f"Unsupported operator: {type(op).__name__}")


def _apply_dataframe_op(op: ast.operator, df1: pd.DataFrame, df2: pd.DataFrame,
                        match_spec: List[List[List[int]]], is_pivoted: bool,
                        source_idx1: int, source_idx2: int) -> pd.DataFrame:
    """
    Apply an operation between two DataFrames with dimension matching.

    match_spec format: [[dims_for_source_1], [dims_for_source_2], ...]
    Each dims list contains dimension indices to use for matching.

    source_idx1 and source_idx2 are 0-based indices into match_spec.
    """
    # Get entity indices
    df1_entity_idx = get_entity_index(df1)
    df2_entity_idx = get_entity_index(df2)

    # Get entity names as list of lists
    df1_names = index_to_names(df1_entity_idx)
    df2_names = index_to_names(df2_entity_idx)

    # Get match dimensions for each source (if provided)
    df1_match_dims = match_spec[source_idx1] if source_idx1 < len(match_spec) else None
    df2_match_dims = match_spec[source_idx2] if source_idx2 < len(match_spec) else None

    # Build match keys for df1
    def build_match_key(name_parts, match_dims):
        if match_dims is None:
            return tuple(name_parts)
        return tuple(name_parts[dim] for dim in match_dims if dim < len(name_parts))

    # Create lookup map from df2 match keys to (index_position, entity_key)
    df2_map = {}
    for i, name in enumerate(df2_names):
        key = build_match_key(name, df2_match_dims)
        entity_key = tuple(name) if len(name) > 1 else name[0]
        df2_map[key] = (i, entity_key)

    # Apply operation for matching entities
    result = df1.copy()

    for i, name in enumerate(df1_names):
        df1_key = build_match_key(name, df1_match_dims)

        if df1_key in df2_map:
            df2_pos, df2_entity_key = df2_map[df1_key]

            if is_pivoted:
                # For pivoted data, entities are in columns
                df1_entity_key = tuple(name) if len(name) > 1 else name[0]
                df1_values = result[df1_entity_key]
                df2_values = df2[df2_entity_key]
                result[df1_entity_key] = _do_op(op, df1_values, df2_values)
            else:
                # For non-pivoted data, entities are in index
                df1_entity_key = tuple(name) if len(name) > 1 else name[0]
                df1_values = result.loc[df1_entity_key]
                df2_values = df2.loc[df2_entity_key]
                result.loc[df1_entity_key] = _do_op(op, df1_values, df2_values)

    return result


def _evaluate_ast_node(node: ast.AST, source_dfs_list: List[pd.DataFrame],
                       match_spec: List[List[List[int]]], is_pivoted: bool,
                       source_indices: List[int] = None) -> Tuple[Any, List[int]]:
    """
    Recursively evaluate an AST node.

    Returns a tuple of (result, source_indices_used).
    - result: DataFrame or scalar
    - source_indices_used: list of 0-based source indices used in this subtree
    """
    if source_indices is None:
        source_indices = []

    if isinstance(node, ast.BinOp):
        # Evaluate left and right operands
        left_result, left_indices = _evaluate_ast_node(
            node.left, source_dfs_list, match_spec, is_pivoted)
        right_result, right_indices = _evaluate_ast_node(
            node.right, source_dfs_list, match_spec, is_pivoted)

        # Combine source indices
        combined_indices = left_indices + right_indices

        # Determine what types we're working with
        left_is_df = isinstance(left_result, pd.DataFrame)
        right_is_df = isinstance(right_result, pd.DataFrame)

        if left_is_df and right_is_df:
            # DataFrame op DataFrame - use dimension matching
            # Get the primary source index from each side
            left_src_idx = left_indices[0] if left_indices else 0
            right_src_idx = right_indices[0] if right_indices else 0
            result = _apply_dataframe_op(
                node.op, left_result, right_result, match_spec, is_pivoted,
                left_src_idx, right_src_idx)
        elif left_is_df:
            # DataFrame op scalar
            result = _apply_scalar_op(node.op, left_result, right_result, is_pivoted, scalar_on_right=True)
        elif right_is_df:
            # scalar op DataFrame
            result = _apply_scalar_op(node.op, right_result, left_result, is_pivoted, scalar_on_right=False)
        else:
            # scalar op scalar
            result = _do_op(node.op, left_result, right_result)

        return result, combined_indices

    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            operand_result, operand_indices = _evaluate_ast_node(
                node.operand, source_dfs_list, match_spec, is_pivoted)
            if isinstance(operand_result, pd.DataFrame):
                return -operand_result, operand_indices
            else:
                return -operand_result, operand_indices
        elif isinstance(node.op, ast.UAdd):
            return _evaluate_ast_node(node.operand, source_dfs_list, match_spec, is_pivoted)
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    elif isinstance(node, ast.Constant):
        # In Python 3.8+, numbers are ast.Constant
        value = node.value
        if isinstance(value, int) and value >= 1:
            # Integer >= 1 is a source reference (1-based)
            source_idx = value - 1  # Convert to 0-based
            if source_idx < len(source_dfs_list):
                return source_dfs_list[source_idx].copy(), [source_idx]
            else:
                raise ValueError(f"Source reference {value} is out of range (only {len(source_dfs_list)} sources)")
        else:
            # Float or other constant
            return value, []

    elif isinstance(node, ast.Num):
        # For Python 3.7 compatibility
        value = node.n
        if isinstance(value, int) and value >= 1:
            source_idx = value - 1
            if source_idx < len(source_dfs_list):
                return source_dfs_list[source_idx].copy(), [source_idx]
            else:
                raise ValueError(f"Source reference {value} is out of range (only {len(source_dfs_list)} sources)")
        else:
            return value, []

    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def evaluate_algebra_expression(expr: str, source_dfs_list: List[pd.DataFrame],
                                match_spec: List[List[List[int]]],
                                is_pivoted: bool) -> pd.DataFrame:
    """
    Evaluate an algebraic expression using source DataFrames.

    expr: String expression like "1*2", "(1+2)*3", "1*0.5"
          Integer values (1, 2, 3...) refer to source DataFrames (1-based)
          Float values are constants

    source_dfs_list: List of source DataFrames

    match_spec: List of dimension indices for matching entities between sources
                Format: [[dims_for_source_1], [dims_for_source_2], ...]
                Each dims list contains indices into the entity name parts

    is_pivoted: Whether data is pivoted (time series in rows, entities in columns)

    Returns: Result DataFrame
    """
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid algebra expression '{expr}': {e}")

    result, _ = _evaluate_ast_node(tree.body, source_dfs_list, match_spec, is_pivoted)

    if not isinstance(result, pd.DataFrame):
        raise ValueError(f"Algebra expression '{expr}' did not produce a DataFrame result")

    return result


def add_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Add two dataframes together, aligning on index/columns."""
    # Align indices
    result = df1.add(df2, fill_value=0)
    return result


def multiply_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                       op_dict: Dict) -> pd.DataFrame:
    """Multiply two dataframes together, with optional dimension matching."""
    
    # Get match specification if provided
    match_spec = op_dict.get('match', {})
    
    if match_spec:
        # Extract dimension indices for matching
        df1_match_dims = [i - 1 for i in match_spec.get('df1', [])]
        df2_match_dims = [i - 1 for i in match_spec.get('df2', [])]
        
        # Get entity names
        df1_names = index_to_names(get_entity_index(df1))
        df2_names = index_to_names(get_entity_index(df2))
        
        # Create match keys
        df1_match_keys = []
        for name in df1_names:
            parts = name.split('.')
            if max(df1_match_dims) < len(parts):
                key = '.'.join([parts[i] for i in df1_match_dims])
                df1_match_keys.append(key)
            else:
                df1_match_keys.append(None)
        
        df2_match_keys = []
        for name in df2_names:
            parts = name.split('.')
            if max(df2_match_dims) < len(parts):
                key = '.'.join([parts[i] for i in df2_match_dims])
                df2_match_keys.append(key)
            else:
                df2_match_keys.append(name)
        
        # Create mapping from match key to values
        df2_map = {}
        for i, key in enumerate(df2_match_keys):
            if key is not None:
                df2_map[key] = df2.iloc[i] if not is_timeseries(df2) else df2.iloc[:, i]
        
        # Multiply matched values
        result = df1.copy()
        for i, key in enumerate(df1_match_keys):
            if key in df2_map:
                if is_timeseries(df1):
                    result.iloc[:, i] = result.iloc[:, i] * df2_map[key]
                else:
                    result.iloc[i] = result.iloc[i] * df2_map[key]
        
        return result
    else:
        # Simple element-wise multiplication
        return df1.multiply(df2, fill_value=1)


def reorder_entity_names(source_idx: pd.Index, order: List[List[int]]) -> pd.Index:
    """
    Reorder entity name dimensions according to order specification.
    order_spec: [[source_dims for target_dim_0], [source_dims for target_dim_1], ...]
    Dimension indices are 1-based in config, convert to 0-based.
    """

    # Build target names according to order spec
    target_names = []
    source_names = index_to_names(source_idx)
    for parts in source_names:
        target_dims = []
        for target_dim_spec in order:
            # Combine source dimensions
            dim_parts = [parts[idx] for idx in target_dim_spec]
            target_dims.append('__'.join(dim_parts))
        target_names.append(target_dims)
    # Create entity index
    target_idx = list_of_lists_to_index(target_names)
   
    return target_idx


def reorder_dimensions(df: pd.DataFrame, order_spec: List[List[int]],
                      aggregate: str = None, is_pivoted: bool = False) -> pd.DataFrame:
    """
    Reorder dimensions in DataFrame index/columns according to order specification.

    Args:
        df: Input DataFrame
        order_spec: List of lists specifying dimension reordering
        aggregate: Aggregation method when collapsing dimensions.
                   Options: 'sum', 'average', 'max', 'min', 'first'
        is_pivoted: Whether data is pivoted (time series in rows, entities in columns)

    Returns:
        DataFrame with reordered dimensions
    """
    if is_pivoted:
        entity_idx = df.columns
    else:
        entity_idx = df.index

    # Reorder entity names
    new_idx = reorder_entity_names(entity_idx, order_spec)

    # If aggregating (collapsing dimensions), need to group
    if aggregate:
        # Create temporary dataframe with new index
        temp_df = df.copy()
        if is_pivoted:
            # For time series, transpose, set index, aggregate, transpose back
            temp_df = temp_df.T
            temp_df.index = new_idx
            grouped = temp_df.groupby(level=list(range(temp_df.index.nlevels)))
            if aggregate == 'sum':
                temp_df = grouped.sum()
            elif aggregate == 'average':
                temp_df = grouped.mean()
            elif aggregate == 'max':
                temp_df = grouped.max()
            elif aggregate == 'min':
                temp_df = grouped.min()
            elif aggregate == 'first':
                temp_df = grouped.first()
            else:
                raise ValueError(f"Unsupported aggregation: {aggregate}")
            temp_df = temp_df.T
            result = temp_df
        else:
            temp_df.index = new_idx
            grouped = temp_df.groupby(level=list(range(temp_df.index.nlevels)))
            if aggregate == 'sum':
                result = grouped.sum()
            elif aggregate == 'average':
                result = grouped.mean()
            elif aggregate == 'max':
                result = grouped.max()
            elif aggregate == 'min':
                result = grouped.min()
            elif aggregate == 'first':
                result = grouped.first()
            else:
                raise ValueError(f"Unsupported aggregation: {aggregate}")
    else:
        # Just reindex
        result = set_entity_index(df, new_idx, is_pivoted)

    return result


def apply_rename(df: pd.DataFrame, source_attribute: str, rename_map: Dict, is_pivoted: bool = False) -> pd.DataFrame:
    """Apply rename mapping to column values."""
    if is_pivoted:
        raise ValueError(f"Unsupported renaming of values in a timeseries object for parameter: {source_attribute}")
    else:
        if source_attribute in df.columns:
            df[source_attribute] = df[source_attribute].replace(rename_map)
    return df

def transform_data(source_dfs: Dict[str, pd.DataFrame],
                   config_path: str,
                   start_time: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
    """
    Main transformer function that reads config and applies transformations.

    Args:
        source_dfs: Dictionary of source dataframes with entity names in index/columns
        config_path: Path to YAML configuration file
        start_time: Optional start datetime for converting string indexes to datetime
                   when transforming from 'str' to 'ts' type parameters.

    Returns:
        Dictionary of transformed target dataframes with entity names in index/columns
    """
    config = load_config(config_path)
    target_dfs = {}

    for operation_name in config.keys():
        print(f"Processing operation: {operation_name}. ", end="")

        source_specs, target_specs, operations = parse_operation(operation_name, config)
        operation_type = get_operation_type(source_specs, target_specs, operations)

        print(f"  Operation type: {operation_type}")

        for target_spec in target_specs:
            if operation_type == 'copy_entities':
                target_dfs = copy_entities(source_dfs, source_specs, target_spec,
                                        target_dfs, operations)
            elif operation_type == 'create_parameter':
                target_dfs = create_parameter(source_dfs, source_specs, target_spec,
                                            target_dfs, operations)
            elif operation_type == 'transform_parameter':
                target_dfs = transform_parameter(source_dfs, source_specs, target_spec,
                                                target_dfs, operations,
                                                start_time=start_time)

    return target_dfs


# Example usage:
if __name__ == "__main__":
    # Load your source dataframes (dfs)
    # result_dfs = transform_data(dfs, 'to_flextool.yaml')
    pass