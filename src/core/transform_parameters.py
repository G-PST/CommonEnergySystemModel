import pandas as pd
import yaml
from typing import Dict, List, Any, Tuple
import numpy as np


def load_config(config_path: str) -> Dict:
    """Load the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def split_dimensions(df: pd.DataFrame, df_key: str, dimensions: List[str] = None) -> pd.DataFrame:
    """
    Split multi-dimensional 'name' column into separate dimension columns.
    Only splits if the df_key contains a dot (indicating multi-dimensional class).
    
    Args:
        df: DataFrame with 'name' column containing dot-separated dimensions
        df_key: Key like 'unit.outputNode' to extract dimension names
        dimensions: Optional list of dimension names to use instead of extracting from df_key
    
    Returns:
        DataFrame with dimensions split into columns named after the classes
    """
    if 'name' not in df.columns:
        return df
    
    # Extract dimension names from df_key (e.g., 'unit.outputNode' -> ['unit', 'outputNode'])
    # Skip '.ts.' parts for time series keys
    if '.ts.' in df_key:
        base_key = df_key.split('.ts.')[0]
    else:
        base_key = df_key
    
    # Only split if the class is multi-dimensional (has a dot)
    if '.' not in base_key:
        return df
    
    # Use provided dimensions list if available, otherwise extract from base_key
    if dimensions:
        dimension_names = dimensions
    else:
        dimension_names = base_key.split('.')
    
    # Split the name column by '.'
    name_parts = df['name'].str.split('.', expand=True)
    
    # Create new columns for each dimension
    for i, dim_name in enumerate(dimension_names):
        if i < name_parts.shape[1]:
            df[dim_name] = name_parts[i]
    
    # Drop the original 'name' column
    df = df.drop(columns=['name'])
    
    # Reorder columns to put dimension columns first
    other_cols = df.columns.difference(dimension_names)
    df = df.reindex(columns=dimension_names + other_cols.tolist())
    
    return df


def parse_operation(operation_name: str, config: Dict) -> Tuple[List, List, Dict, List]:
    """
    Parse a single operation from the config.
    
    Returns:
        source_specs: List of source class:attribute pairs or just classes
        target_specs: List of target class:attribute pairs or just classes
        operations: Dict of additional operations (order, rename, operation, with, value)
        dimensions: List of dimension names (optional)
    """
    operation_config = config[operation_name]
    
    source_specs = []
    target_specs = []
    operations = []
    dimensions = None
    
    for i, item in enumerate(operation_config):
        if i == 0:
            # First item is source
            source_specs = parse_spec(item)
        elif i == 1:
            # Second item is target
            target_specs = parse_spec(item)
        else:
            # Remaining items are operations/modifiers or dimensions
            if isinstance(item, dict):
                if 'dimensions' in item:
                    dimensions = item['dimensions']
                else:
                    operations = [item]
            elif isinstance(item, list):
                for it in item[:]:
                    if 'dimensions' in it:
                        dimensions = it['dimensions']
                        item.remove(it)
                operations = item
    
    return source_specs, target_specs, operations, dimensions


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
                 operations: Dict,
                 dimensions: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Copy entities from source class to target class."""
    source_class = source_specs[0]['class']
    target_class = target_spec['class']
    
    if source_class not in source_dfs:
        return target_dfs
    
    source_df = source_dfs[source_class]
    
    if 'name' not in source_df.columns:
        return target_dfs
    
    # Parse target dimensions from class name
    target_dims = target_class.split('.')
    
    # Build target entities based on order specification
    if 'config' in target_spec and 'order' in target_spec['config']:
        order = target_spec['config']['order']
        
        # Get source entity names split by dimension
        source_entities = source_df['name'].str.split('.', expand=True)
        
        target_entities = []
        for target_dim_spec in order:
            # Combine source dimensions as specified (1-indexed to 0-indexed)
            dim_parts = [source_entities[idx - 1] for idx in target_dim_spec]
            target_entities.append(pd.Series(['__'.join(map(str, parts)) for parts in zip(*dim_parts)]))
        
        # Combine target dimensions with '__'
        entities = pd.DataFrame({
            'name': ['.'.join(parts) for parts in zip(*target_entities)]
        })
    else:
        entities = source_df[['name']].copy()
    
    # Merge with existing target dataframe
    if target_class not in target_dfs:
        target_dfs[target_class] = entities
    else:
        target_dfs[target_class] = pd.concat([target_dfs[target_class], entities], 
                                              ignore_index=True).drop_duplicates()
    
    return target_dfs

def apply_rename_to_timeseries(ts_df: pd.DataFrame, rename_map: Dict) -> pd.DataFrame:
    """Apply rename mapping to time series column names (entity names)."""
    # Rename columns except 'datetime'
    new_columns = {}
    for col in ts_df.columns:
        if col != 'datetime':
            new_columns[col] = rename_map.get(col, col)
    return ts_df.rename(columns=new_columns)
    
def create_parameter(source_dfs: Dict[str, pd.DataFrame],
                    source_specs: List[Dict],
                    target_spec: Dict,
                    target_dfs: Dict[str, pd.DataFrame],
                    operations: Dict,
                    dimensions: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Create a parameter with a fixed value for entities that exist in source classes."""
    target_class = target_spec['class']
    target_attribute = target_spec['attribute']
    value = next((d['value'] for d in operations if 'value' in d), None)
    
    # Collect entities from all source classes
    all_source_entities = []
    for source_spec in source_specs:
        source_class = source_spec['class']
        if source_class not in source_dfs:
            continue
        
        source_df = source_dfs[source_class]
        if 'name' in source_df.columns:
            all_source_entities.extend(source_df['name'].tolist())
    
    if not all_source_entities:
        return target_dfs
    
    # Create dataframe with parameter for these specific entities only
    param_df = pd.DataFrame({
        'name': all_source_entities,
        target_attribute: value
    })
    
    # Initialize target if needed
    if target_class not in target_dfs:
        target_dfs[target_class] = param_df
    else:
        # Merge only for entities that exist in source
        target_dfs[target_class] = target_dfs[target_class].merge(
            param_df, on='name', how='left', suffixes=('', '_new')
        )
        if f'{target_attribute}_new' in target_dfs[target_class].columns:
            # Only update where we have new values (i.e., where source entities exist)
            target_dfs[target_class][target_attribute] = target_dfs[target_class][f'{target_attribute}_new'].combine_first(
                target_dfs[target_class].get(target_attribute)
            )
            target_dfs[target_class].drop(columns=[f'{target_attribute}_new'], inplace=True)
    
    # Replace NaN with None
    target_dfs[target_class] = target_dfs[target_class].replace({np.nan: None})
    
    return target_dfs

def transform_parameter(source_dfs: Dict[str, pd.DataFrame],
                       source_specs: List[Dict],
                       target_spec: List[Dict],
                       target_dfs: Dict[str, pd.DataFrame],
                       operations: Dict,
                       dimensions: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Transform parameter values from source to target."""
    """Needs refactoring to account for the situation where some parameters are time series and others constants"""
    target_class = target_spec['class']
    target_attribute = target_spec['attribute']
    
    # Collect source data
    source_data_list = []
    for source_spec in source_specs:
        source_class = source_spec['class']
        source_attribute = source_spec['attribute']
        
        # Check both regular and time series dataframes
        df_key = source_class
        ts_key = f"{source_class}.ts.{source_attribute}"
        
        if df_key in source_dfs and source_attribute in source_dfs[df_key].columns:
            df = source_dfs[df_key][['name', source_attribute]].copy()
            # Rename to target attribute immediately
            df.rename(columns={source_attribute: target_attribute}, inplace=True)
            source_data_list.append(df)
        elif ts_key in source_dfs:
            target_ts_key = f"{target_spec['class']}.ts.{target_spec['attribute']}"
            source_data_list.append((target_ts_key, source_dfs[ts_key]))
    
    if not source_data_list:
        # No source data, create parameter with None
        if target_class not in target_dfs:
            target_dfs[target_class] = pd.DataFrame()
        if target_attribute not in target_dfs[target_class].columns:
            target_dfs[target_class][target_attribute] = None
        return target_dfs
    
    # Check if this is an algebra operation or a union operation
    is_algebra = operations and any('algebra' in op for op in (operations if isinstance(operations, list) else [operations]))
    
    if len(source_data_list) > 1 and not is_algebra:
        # Union operation - concatenate data from multiple sources
        combined_dfs = []
        if isinstance(source_data_list[0], tuple):
            df_temp = source_data_list[0][1].copy()
        for data in source_data_list[1:]:
            if isinstance(data, tuple):
                # Time series data - skip for now or handle separately
                for df_in_tuple in source_data_list[1:]:
                    df_temp = pd.merge(df_temp, df_in_tuple[1], on='datetime', how='outer')
                result_data = (target_ts_key, df_temp)
            else:
                combined_dfs.append(data)
        
        if combined_dfs:
            result_data = pd.concat(combined_dfs, ignore_index=True).drop_duplicates(subset=['name'])
    else:
        result_data = source_data_list[0]
    
    if operations:
        for operation in operations:
            if 'algebra' in operation:
                # Algebra creates result with target_attribute name
                result_data = apply_algebra_operation(source_data_list, operation['algebra'], operation.get('match'), target_attribute)
            elif 'order' in operation:
                aggregate = None
                if isinstance(operation['order'], list):
                    order = operation['order']
                else:
                    order = operation['order']['order']
                    if 'aggregate' in operation['order']:
                        aggregate = operation['order']['aggregate']
                if isinstance(result_data, tuple):  # It's a time series
                    result_data = (result_data[0], reorder_dimensions_for_timeseries(result_data[1], order))
                else:
                    result_data = reorder_dimensions(result_data, order, aggregate)
            elif 'rename' in operation:
                if isinstance(result_data, tuple):  # It's a time series
                    result_data = apply_rename_to_timeseries(result_data, target_attribute, operation['rename'])
                else: # It's regular
                    result_data = apply_rename(result_data, target_attribute, operation['rename'])
            elif 'to_datatype' in operation:
                # Handle datatype conversion for series data
                to_datatype = operation['to_datatype']
                if isinstance(result_data, tuple) and 'ts' in to_datatype and to_datatype['ts'] == 'table':
                     # Change the key from .ts. to .table.
                    new_key = f"{target_class}.table.{target_attribute}"
                    result_data = (new_key, result_data[1])
                else:
                    raise ValueError(f"Unsupported datatype conversion: {df_key}. Currently only timeries to table is supported.")

            else:
                # Must be a operation with a constant
                if isinstance(result_data, tuple):
                    # Time series data
                    ts_key, ts_df = result_data
                    result_data = apply_operations_to_timeseries(ts_df, operation)
                else:
                    # Regular data - operates on target_attribute
                    result_data = apply_operations_to_data(result_data, target_attribute, operation)
    
    # Handle time series / map data
    if isinstance(result_data, tuple):
        ts_key, ts_df = result_data
        if ts_key not in target_dfs:
            target_dfs[ts_key] = ts_df
        else:
            target_dfs[ts_key] = pd.concat([target_dfs[ts_key], ts_df], 
                                            ignore_index=True).drop_duplicates()
        return target_dfs
    
    # Merge into target dataframe (only for regular data)
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
    
    # Replace NaN with None
    target_dfs[target_class] = target_dfs[target_class].replace({np.nan: None})
    
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
            lambda x: '.'.join([x.split('.')[i] for i in df1_match_dims]) 
            if len(x.split('.')) > max(df1_match_dims) else None
        )
        df2_match_key = df2['name'].apply(
            lambda x: '.'.join([x.split('.')[i] for i in df2_match_dims])
            if len(x.split('.')) > max(df2_match_dims) else x
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

def reorder_dimensions(data: pd.DataFrame, order_spec: List[List[int]], aggregate: str = None) -> pd.DataFrame:
    """
    Reorder dimensions according to order specification.
    order_spec: [[source_dims for target_dim_0], [source_dims for target_dim_1], ...]
    Dimension indices in config are 1-based, convert to 0-based.
    
    Args:
        aggregate: How to aggregate numeric values when collapsing dimensions. 
                   Currently only 'sum' is supported.
    """
    # Parse the name to extract dimensions
    name_parts = data['name'].str.split('.', expand=True)
    
    # Get numeric columns for potential aggregation
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'name' in numeric_cols:
        numeric_cols.remove('name')
    
    if name_parts.shape[1] == 1:
        # Single dimension, no reordering needed unless creating multi-dim
        if len(order_spec) > 1:
            # Create multi-dimensional from single dimension
            new_names = []
            for _, row in data.iterrows():
                parts = []
                for target_dim in order_spec:
                    source_indices = [i - 1 for i in target_dim]
                    parts.append(name_parts.iloc[_, 0])
                new_names.append('.'.join(parts))
            data['name'] = new_names
    else:
        # Multi-dimensional, reorder
        new_names = []
        for idx, row in data.iterrows():
            parts = []
            for target_dim in order_spec:
                source_indices = [i - 1 for i in target_dim]
                if len(source_indices) == 1:
                    parts.append(name_parts.iloc[idx, source_indices[0]])
                else:
                    # Concatenate multiple source dimensions
                    concat_parts = [name_parts.iloc[idx, si] for si in source_indices]
                    parts.append('.'.join(concat_parts))
            new_names.append('.'.join(parts))
        data['name'] = new_names
        
        # Check if we're collapsing dimensions (target has fewer dims than source)
        source_dims = name_parts.shape[1]
        target_dims = len(order_spec)
        
        if target_dims < source_dims and aggregate and numeric_cols:
            # Aggregate numeric columns by the new name
            if aggregate == 'sum':
                agg_dict = {col: 'sum' for col in numeric_cols}
                data = data.groupby('name', as_index=False).agg(agg_dict)
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregate}. Currently only 'sum' is supported.")
    
    return data


def reorder_dimensions_for_timeseries(df: pd.DataFrame, order_spec: List[List[int]]) -> pd.DataFrame:
    """
    Reorder dimensions according to order specification.
    order_spec: [[source_dims for target_dim_0], [source_dims for target_dim_1], ...]
    Dimension indices in config are 1-based, convert to 0-based.
    """
    result = df.copy()
    new_columns = {'datetime': 'datetime'}  # Keep datetime as-is
    
    for col in df.columns:
        if col == 'datetime':
            continue
            
        # Split column name into dimensions
        parts = col.split('.')
        
        # Build new column name based on order_spec
        new_parts = []
        for target_dims in order_spec:
            # Collect source dimensions for this target dimension
            collected = []
            for source_dim_idx in target_dims:
                if source_dim_idx <= len(parts):
                    collected.append(parts[source_dim_idx - 1])  # Convert to 0-indexed
            
            # Join with '__' if multiple dimensions
            new_parts.append('__'.join(collected))
        
        # Join target dimensions with '.'
        new_columns[col] = '.'.join(new_parts)
    
    result.rename(columns=new_columns, inplace=True)
    return result



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
        Dictionary of transformed target dataframes with dimensions split into columns
    """
    config = load_config(config_path)
    target_dfs = {}
    dimensions_map = {}  # Store dimensions metadata
    
    for operation_name in config.keys():
        print(f"Processing operation: {operation_name}")
        
        source_specs, target_specs, operations, dimensions = parse_operation(operation_name, config)
        operation_type = get_operation_type(source_specs, target_specs, operations)
        
        print(f"  Operation type: {operation_type}")
        
        for target_spec in target_specs:
            # Store dimensions for target class
            target_class = target_spec['class']
            if dimensions and '.' in target_class:
                dimensions_map[target_class] = dimensions
            elif '.' in target_class:
                dimensions_map[target_class] = target_class.split('.')
            
            if operation_type == 'copy_entities':
                target_dfs = copy_entities(source_dfs, source_specs, target_spec, 
                                        target_dfs, operations, dimensions)
            elif operation_type == 'create_parameter':
                target_dfs = create_parameter(source_dfs, source_specs, target_spec,
                                            target_dfs, operations, dimensions)
            elif operation_type == 'transform_parameter':
                target_dfs = transform_parameter(source_dfs, source_specs, target_spec,
                                                target_dfs, operations, dimensions)
        
    # Split dimensions for all target dataframes
    final_dfs = {}
    for df_key, df in target_dfs.items():
        # Check if we have custom dimensions for this key
        custom_dims = dimensions_map.get(df_key)
        final_dfs[df_key] = split_dimensions(df, df_key, custom_dims)
    
    return final_dfs


# Example usage:
if __name__ == "__main__":
    # Load your source dataframes (dfs)
    # result_dfs = transform_data(dfs, 'parameters_to_flextool.yaml')
    pass