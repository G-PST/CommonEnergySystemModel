import pandas as pd
import yaml
from typing import Dict, List, Any, Tuple
import numpy as np


def load_config(config_path: str) -> Dict:
    """Load the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_operation(operation_name: str, config: Dict) -> Tuple[List, List, Dict]:
    """
    Parse a single operation from the config.
    
    Returns:
        source_specs: List of source class:attribute pairs or just classes
        target_specs: List of target class:attribute pairs or just classes
        operations: Dict of additional operations (order, rename, operation, with, value)
    """
    operation_config = config[operation_name]
    
    source_specs = []
    target_specs = []
    operations = {}
    
    for i, item in enumerate(operation_config):
        if i == 0:
            # First item is source
            source_specs = parse_spec(item)
        elif i == 1:
            # Second item is target
            target_specs = parse_spec(item)
        elif i == 2:
            # Third item is operations/modifiers
            operations = item
    
    return source_specs, target_specs, operations


def parse_spec(spec: Any) -> List[Dict]:
    """Parse a source or target specification."""
    result = []
    
    if isinstance(spec, str):
        # Simple class name
        result.append({'class': spec, 'attribute': None})
    elif isinstance(spec, dict):
        for key, value in spec.items():
            if isinstance(value, dict):
                # Has additional config like order, rename
                result.append({'class': key, 'attribute': None, 'config': value})
            else:
                # class: attribute format
                result.append({'class': key, 'attribute': value})
    elif isinstance(spec, list):
        # Multiple sources
        for item in spec:
            result.extend(parse_spec(item))
    
    return result


def get_operation_type(source_specs: List[Dict], target_specs: List[Dict], 
                       operations: Dict) -> str:
    """Determine the type of operation to perform."""
    source_has_attr = any(s.get('attribute') for s in source_specs)
    target_has_attr = any(t.get('attribute') for t in target_specs)
    
    if not source_has_attr and not target_has_attr:
        return 'copy_entities'
    elif not source_has_attr and target_has_attr and 'value' in operations:
        return 'create_parameter'
    elif source_has_attr and target_has_attr:
        return 'transform_parameter'
    else:
        return 'unknown'


def copy_entities(source_dfs: Dict[str, pd.DataFrame], 
                 source_specs: List[Dict], 
                 target_specs: List[Dict],
                 target_dfs: Dict[str, pd.DataFrame],
                 operations: Dict) -> Dict[str, pd.DataFrame]:
    """Copy entities from source class to target class."""
    source_class = source_specs[0]['class']
    target_class = target_specs[0]['class']
    
    if source_class not in source_dfs:
        return target_dfs
    
    source_df = source_dfs[source_class]
    
    # Get entity names (assume 'name' column)
    if 'name' not in source_df.columns:
        return target_dfs
    
    entities = source_df[['name']].copy()
    
    # Handle dimension reordering if specified
    if 'order' in operations:
        entities = reorder_dimensions(entities, operations['order'])
    
    # Merge with existing target dataframe
    if target_class not in target_dfs:
        target_dfs[target_class] = entities
    else:
        target_dfs[target_class] = pd.concat([target_dfs[target_class], entities], 
                                              ignore_index=True).drop_duplicates()
    
    return target_dfs


def create_parameter(source_dfs: Dict[str, pd.DataFrame],
                    source_specs: List[Dict],
                    target_specs: List[Dict],
                    target_dfs: Dict[str, pd.DataFrame],
                    operations: Dict) -> Dict[str, pd.DataFrame]:
    """Create a parameter with a fixed value for entities that exist."""
    source_class = source_specs[0]['class']
    target_class = target_specs[0]['class']
    target_attribute = target_specs[0]['attribute']
    value = operations.get('value')
    
    if source_class not in source_dfs:
        return target_dfs
    
    source_df = source_dfs[source_class]
    
    # Initialize target if needed
    if target_class not in target_dfs:
        target_dfs[target_class] = source_df[['name']].copy()
    
    # Add the parameter with the specified value
    target_dfs[target_class][target_attribute] = value
    
    return target_dfs


def transform_parameter(source_dfs: Dict[str, pd.DataFrame],
                       source_specs: List[Dict],
                       target_specs: List[Dict],
                       target_dfs: Dict[str, pd.DataFrame],
                       operations: Dict) -> Dict[str, pd.DataFrame]:
    """Transform parameter values from source to target."""
    target_class = target_specs[0]['class']
    target_attribute = target_specs[0]['attribute']
    
    # Collect source data
    source_data_list = []
    for source_spec in source_specs:
        source_class = source_spec['class']
        source_attribute = source_spec['attribute']
        
        # Check both regular and time series dataframes
        df_key = source_class
        ts_key = f"{source_class}.ts.{source_attribute}"
        
        if df_key in source_dfs and source_attribute in source_dfs[df_key].columns:
            source_data_list.append(source_dfs[df_key][['name', source_attribute]].copy())
        elif ts_key in source_dfs:
            source_data_list.append((ts_key, source_dfs[ts_key]))
    
    if not source_data_list:
        # No source data, create parameter with None
        if target_class not in target_dfs:
            target_dfs[target_class] = pd.DataFrame()
        if target_attribute not in target_dfs[target_class].columns:
            target_dfs[target_class][target_attribute] = None
        return target_dfs
    
    result_data = source_data_list[0]

    if operations:
        if not type(operations) == list:
            operations = [operations]
        
        for operation in operations:
            if 'algebra' in operation:
                result_data = apply_algebra_operation(source_data_list, operation['algebra'], operation['match'], target_attribute)
            elif 'order' in operation:
                result_data = reorder_dimensions(result_data, operation['order'])
            elif 'rename' in operation:
                result_data = apply_rename(result_data, target_attribute, operation['rename'])
            else:
                # Must be a operation with a constant
                if isinstance(result_data, tuple):
                    # Time series data
                    ts_key, ts_df = result_data
                    result_data = apply_operations_to_timeseries(ts_df, operation)
                else:
                    # Regular data
                    result_data = apply_operations_to_data(result_data, source_specs[0]['attribute'], operation)
    
    # Merge into target dataframe
    if target_class not in target_dfs:
        target_dfs[target_class] = result_data
    else:
        # Ensure target attribute exists
        if 'name' in result_data.columns and 'name' in target_dfs[target_class].columns:
            target_dfs[target_class] = target_dfs[target_class].merge(
                result_data, on='name', how='outer', suffixes=('', '_new')
            )
            if f'{target_attribute}_new' in target_dfs[target_class].columns:
                target_dfs[target_class][target_attribute] = target_dfs[target_class][f'{target_attribute}_new']
                target_dfs[target_class].drop(columns=[f'{target_attribute}_new'], inplace=True)
        elif target_attribute not in target_dfs[target_class].columns:
            target_dfs[target_class][target_attribute] = None
    
    return target_dfs


def apply_operations_to_data(data: pd.DataFrame, column: str, operation: Dict) -> pd.DataFrame:
    """Apply mathematical operations to regular data."""
    
    if 'multiply' in operation:
        data[column] = data[column] * float(operation['multiply']['with'])
    elif 'divide' in operation:
        data[column] = data[column] / float(operation['divide']['with'])
    elif 'add' in operation:
        data[column] = data[column] + float(operation['add']['with'])
    elif 'subtract' in operation:
        data[column] = data[column] - float(operation['subtract']['with'])
    
    return data


def apply_operations_to_timeseries(ts_df: pd.DataFrame, operation: Dict) -> pd.DataFrame:
    """Apply mathematical operations to time series data."""
    if 'operation' not in operations:
        return ts_df
    
    with_value = float(list(operation.values())[0]['with'])
    
    result_df = ts_df.copy()
    
    # Apply to all columns except 'datetime' and 'name'
    for col in result_df.columns:
        if col not in ['datetime', 'name']:
            if 'multiply' in operation:
                result_df[col] = result_df[col].apply(
                    lambda x: [v * with_value if v is not None else None for v in x] if isinstance(x, list) else x
                )
            elif 'divide' in operation:
                result_df[col] = result_df[col].apply(
                    lambda x: [v / with_value if v is not None else None for v in x] if isinstance(x, list) else x
                )
            elif 'add' in operation:
                result_df[col] = result_df[col].apply(
                    lambda x: [v + with_value if v is not None else None for v in x] if isinstance(x, list) else x
                )
            elif 'subtract' in operation:
                result_df[col] = result_df[col].apply(
                    lambda x: [v - with_value if v is not None else None for v in x] if isinstance(x, list) else x
                )
    
    return result_df


def apply_algebra_operation(source_data_list: List, operation: str, match: str, target_attribute: str) -> pd.DataFrame:
    """
    Apply algebra operations between multiple sources with dimension matching.
    
    match spec: [[dims_from_source1], [dims_from_source2], ...]
    Each nested list specifies which dimensions (1-based) to extract for matching.
    """
    # For now, just multiply first two sources
    df1 = source_data_list[0]
    df2 = source_data_list[1]
    
    if isinstance(df1, tuple) or isinstance(df2, tuple):
        # Handle time series multiplication later
        return df1 if not isinstance(df1, tuple) else df2
    
    if match and len(match) >= 2:
        # Extract dimensions for matching
        df1_match_dims = [i - 1 for i in match[0]]  # Convert 1-based to 0-based
        df2_match_dims = [i - 1 for i in match[1]]
        
        # Extract matching keys from multi-dimensional names
        df1_match_key = df1['name'].apply(
            lambda x: '__'.join([x.split('__')[i] for i in df1_match_dims]) 
            if len(x.split('__')) > max(df1_match_dims) else None
        )
        df2_match_key = df2['name'].apply(
            lambda x: '__'.join([x.split('__')[i] for i in df2_match_dims])
            if len(x.split('__')) > max(df2_match_dims) else x
        )
        
        # Add match keys to dataframes
        df1_temp = df1.copy()
        df2_temp = df2.copy()
        df1_temp['_match_key'] = df1_match_key
        df2_temp['_match_key'] = df2_match_key
        
        # Remove rows where match key couldn't be extracted
        df1_temp = df1_temp[df1_temp['_match_key'].notna()]
        df2_temp = df2_temp[df2_temp['_match_key'].notna()]
        
        # Merge on match key
        merged = df1_temp.merge(df2_temp, on='_match_key', how='inner', suffixes=('_1', '_2'))
        
        # Get the actual data columns (not 'name' or '_match_key')
        col1 = [c for c in df1_temp.columns if c not in ['name', '_match_key']][0]
        col2 = [c for c in df2_temp.columns if c not in ['name', '_match_key']][0]
        
        # Use the name from the first dataframe (preserves full multi-dimensional structure)
        result = pd.DataFrame({
            'name': merged['name_1'],
            target_attribute: merged[col1] * merged[col2]
        })
        
    else:
        # No match specification, use simple name-based merge
        merged = df1.merge(df2, on='name', how='inner')
        col1 = [c for c in df1.columns if c != 'name'][0]
        col2 = [c for c in df2.columns if c != 'name'][0]
        
        result = pd.DataFrame({
            'name': merged['name'],
            target_attribute: merged[col1] * merged[col2]
        })
    
    return result

def reorder_dimensions(data: pd.DataFrame, order_spec: List[List[int]]) -> pd.DataFrame:
    """
    Reorder dimensions according to order specification.
    order_spec: [[source_dims for target_dim_0], [source_dims for target_dim_1], ...]
    Dimension indices in config are 1-based, convert to 0-based.
    """
    if 'name' not in data.columns:
        return data
    
    # Parse the name column to extract dimensions
    # Assume format like "dim1__dim2" for multi-dimensional entities
    name_parts = data['name'].str.split('__', expand=True)
    
    if name_parts.shape[1] == 1:
        # Single dimension, no reordering needed unless creating multi-dim
        if len(order_spec) > 1:
            # Create multi-dimensional from single dimension
            new_names = []
            for _, row in data.iterrows():
                parts = []
                for target_dim in order_spec:
                    # target_dim is 1-based, convert to 0-based
                    source_indices = [i - 1 for i in target_dim]
                    # All source indices point to same dimension (0)
                    parts.append(name_parts.iloc[_, 0])
                new_names.append('__'.join(parts))
            data['name'] = new_names
    else:
        # Multi-dimensional, reorder
        new_names = []
        for idx, row in data.iterrows():
            parts = []
            for target_dim in order_spec:
                # target_dim is 1-based, convert to 0-based
                source_indices = [i - 1 for i in target_dim]
                if len(source_indices) == 1:
                    parts.append(name_parts.iloc[idx, source_indices[0]])
                else:
                    # Concatenate multiple source dimensions
                    concat_parts = [name_parts.iloc[idx, si] for si in source_indices]
                    parts.append('_'.join(concat_parts))
            new_names.append('__'.join(parts))
        data['name'] = new_names
    
    return data


def apply_rename(data: pd.DataFrame, column: str, rename_map: Dict) -> pd.DataFrame:
    """Apply rename mapping to column values."""
    if column in data.columns:
        data[column] = data[column].replace(rename_map)
    return data


def transform_data(source_dfs: Dict[str, pd.DataFrame], 
                   config_path: str) -> Dict[str, pd.DataFrame]:
    """
    Main transformer function that reads config and applies transformations.
    
    Args:
        source_dfs: Dictionary of source dataframes
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary of transformed target dataframes
    """
    config = load_config(config_path)
    target_dfs = {}
    
    for operation_name in config.keys():
        print(f"Processing operation: {operation_name}")
        
        source_specs, target_specs, operations = parse_operation(operation_name, config)
        operation_type = get_operation_type(source_specs, target_specs, operations)
        
        print(f"  Operation type: {operation_type}")
        
        if operation_type == 'copy_entities':
            target_dfs = copy_entities(source_dfs, source_specs, target_specs, 
                                      target_dfs, operations)
        elif operation_type == 'create_parameter':
            target_dfs = create_parameter(source_dfs, source_specs, target_specs,
                                         target_dfs, operations)
        elif operation_type == 'transform_parameter':
            target_dfs = transform_parameter(source_dfs, source_specs, target_specs,
                                            target_dfs, operations)
    
    return target_dfs


# Example usage:
if __name__ == "__main__":
    # Load your source dataframes (dfs)
    # result_dfs = transform_data(dfs, 'parameters_to_flextool.yaml')
    pass