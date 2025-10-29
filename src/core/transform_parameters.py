import pandas as pd
import yaml
from typing import Dict, List, Any, Tuple, Union
import numpy as np


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
                     is_ts: bool = False) -> pd.DataFrame:
    """
    Set the entity index for a DataFrame.
    For time series: sets columns
    For regular data: sets index
    """
    result = df.copy()
    if is_ts:
        result.columns = new_index
    else:
        result.index = new_index
    return result


def create_index_from_names(names: List[str]) -> Union[pd.Index, pd.MultiIndex]:
    """
    Create appropriate Index or MultiIndex from entity name strings.
    Names with dots become MultiIndex levels.
    """
    if not names:
        return pd.Index([])
    
    # Check if multi-dimensional
    sample = names[0]
    if '.' in sample:
        # Multi-dimensional: split by dots
        split_names = [name.split('.') for name in names]
        return pd.MultiIndex.from_tuples([tuple(parts) for parts in split_names])
    else:
        # Single dimensional
        return pd.Index(names)


def index_to_names(idx: Union[pd.Index, pd.MultiIndex]) -> List[str]:
    """
    Convert Index or MultiIndex to list of dot-separated name strings.
    """
    if isinstance(idx, pd.MultiIndex):
        return ['.'.join(map(str, t)) for t in idx]
    else:
        return list(map(str, idx))


def parse_operation(operation_name: str, config: Dict) -> Tuple[List, List, List, List]:
    """
    Parse a single operation from the config.
    
    Returns:
        source_specs: List of source class:attribute pairs or just classes
        target_specs: List of target class:attribute pairs or just classes
        operations: List of additional operations
        dimensions: List of dimension names (optional)
    """
    operation_config = config[operation_name]
    
    source_specs = []
    target_specs = []
    operations = []
    dimensions = None
    
    for i, item in enumerate(operation_config):
        if i == 0:
            source_specs = parse_spec(item)
        elif i == 1:
            target_specs = parse_spec(item)
        else:
            if isinstance(item, dict):
                if 'dimensions' in item:
                    dimensions = item['dimensions']
                else:
                    operations.append(item)
            elif isinstance(item, list):
                for it in item[:]:
                    if 'dimensions' in it:
                        dimensions = it['dimensions']
                        item.remove(it)
                operations.extend(item)
    
    return source_specs, target_specs, operations, dimensions


def parse_spec(spec: Any) -> List[Dict]:
    """Parse a source or target specification."""
    result = []
    
    if isinstance(spec, str):
        result.append({'class': spec, 'attribute': None})
    elif isinstance(spec, dict):
        for key, value in spec.items():
            if isinstance(value, dict):
                result.append({'class': key, 'attribute': None, 'config': value})
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
                 operations: List[Dict],
                 dimensions: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Copy entities from source class to target class."""
    source_class = source_specs[0]['class']
    target_class = target_spec['class']
    
    if source_class not in source_dfs:
        return target_dfs
    
    source_df = source_dfs[source_class]
    source_idx = get_entity_index(source_df)
    source_names = index_to_names(source_idx)
    
    # Build target entities based on order specification
    if 'config' in target_spec and 'order' in target_spec['config']:
        order = target_spec['config']['order']
        
        # Split source names into dimensions
        source_parts = [name.split('.') for name in source_names]
        
        # Build target names according to order spec
        target_names = []
        for parts in source_parts:
            target_dims = []
            for target_dim_spec in order:
                # Combine source dimensions (1-indexed to 0-indexed)
                dim_parts = [parts[idx - 1] for idx in target_dim_spec]
                target_dims.append('__'.join(dim_parts))
            target_names.append('.'.join(target_dims))
    else:
        target_names = source_names
    
    # Create entity index
    target_idx = create_index_from_names(target_names)
    
    # Store or merge with existing target dataframe
    if target_class not in target_dfs:
        # Create new dataframe with just the index
        target_dfs[target_class] = pd.DataFrame(index=target_idx)
    else:
        # Merge indices
        existing_idx = target_dfs[target_class].index
        combined = existing_idx.append(target_idx).unique()
        target_dfs[target_class] = pd.DataFrame(index=combined)
    if not dimensions:
        dimensions = target_class.split('.')
    if len(dimensions) > 1:
        target_dfs[target_class].index.names = dimensions
    elif len(dimensions) == 1:
        target_dfs[target_class].index.name = dimensions[0]    
    return target_dfs


def create_parameter(source_dfs: Dict[str, pd.DataFrame],
                    source_specs: List[Dict],
                    target_spec: Dict,
                    target_dfs: Dict[str, pd.DataFrame],
                    operations: List[Dict],
                    dimensions: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Create a parameter with value from config or source data."""
    source_class = source_specs[0]['class']
    target_class = target_spec['class']
    target_attribute = target_spec['attribute']

    source_df = source_dfs[source_class]
    source_idx = get_entity_index(source_df)
    entity_names = index_to_names(source_idx)

    # Get the value from operations
    for op_dict in operations:
        if 'order' in op_dict:
            entity_names = reorder_entity_names(entity_names, op_dict['order'])
        
        if 'value' in op_dict:
            new_parameter_value = op_dict['value']

        # Apply rename if specified
        if 'rename' in op_dict:
            raise ValueError(f"There should be no renaming for parameters that are freshly created through _value_, target attribute: {target_attribute}")

    # Create target index and dataframe
    target_idx = create_index_from_names(entity_names)
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
                       dimensions: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Transform parameters from source to target."""
    target_class = target_spec['class']
    target_attribute = target_spec['attribute']
    
    # Check if this is a time series operation
    is_ts = any('.ts.' in spec['class'] for spec in source_specs)
    
    # Gather source data
    source_dfs_list = []
    for source_spec in source_specs:
        source_class = source_spec['class']
        source_attribute = source_spec.get('attribute')
        
        # Check if source class exists
        if source_class not in source_dfs:
            continue
        
        # Get the dataframe
        source_df = source_dfs[source_class]
        
        # If attribute specified, extract that column
        if source_attribute:
            if source_attribute in source_df.columns:
                # Create a dataframe with just this column
                extracted = source_df[[source_attribute]].copy()
                source_dfs_list.append(extracted)
        else:
            # Use entire dataframe
            source_dfs_list.append(source_df.copy())
    
    if not source_dfs_list:
        return target_dfs
    
    # Start with first source
    result = source_dfs_list[0].copy()
    
    # Apply operations
    for op_dict in operations:
        if 'operation' in op_dict:
            operation = op_dict['operation']
            
            if operation == 'multiply':
                # Get multiply factor
                with_value = op_dict.get('with')
                if with_value is not None and not isinstance(with_value, list):
                    # Multiply by constant
                    for col in result.select_dtypes(include=[np.number]).columns:
                        result[col] = result[col] * with_value
                elif len(source_dfs_list) > 1:
                    # Multiply with second dataframe
                    result = multiply_dataframes(result, source_dfs_list[1], op_dict)
                elif isinstance(with_value, list):
                    # Need to fetch the 'with' data
                    with_specs = parse_spec(with_value)
                    for with_spec in with_specs:
                        with_class = with_spec['class']
                        with_attr = with_spec.get('attribute')
                        
                        if with_class in source_dfs:
                            with_df = source_dfs[with_class]
                            
                            # Extract attribute if specified
                            if with_attr and with_attr in with_df.columns:
                                with_data = with_df[[with_attr]].copy()
                            else:
                                with_data = with_df.copy()
                            
                            result = multiply_dataframes(result, with_data, op_dict)
            
            elif operation == 'sum':
                # Sum all source dataframes
                for df in source_dfs_list[1:]:
                    result = add_dataframes(result, df)
        
        if 'multiply' in op_dict and 'with' in op_dict:
            # Multiply all numeric columns by constant
            with_value = op_dict['with']
            for col in result.select_dtypes(include=[np.number]).columns:
                result[col] = result[col] * with_value
        
        if 'order' in op_dict:
            # Check if order is nested (order: order: [[1]]) or direct (order: [[1]])
            order_spec = op_dict['order']
            if isinstance(order_spec, dict) and 'order' in order_spec:
                order_list = order_spec['order']
                aggregate = order_spec.get('aggregate')
            else:
                order_list = order_spec
                aggregate = op_dict.get('aggregate')
            
            result = reorder_dimensions(result, order_list, aggregate, is_ts)
        
        if 'rename' in op_dict:
            result = apply_rename(result, source_attribute, op_dict['rename'], is_ts)
    
    # Rename data column to target attribute if needed
    if not is_ts and len(result.columns) == 1:
        result.columns = [target_attribute]
    
    # Store result in target class dataframe
    if target_class not in target_dfs:
        target_dfs[target_class] = result
    else:
        # Add column to existing dataframe or update
        if target_attribute not in target_dfs[target_class].columns:
            # Add new column
            for col in result.columns:
                target_dfs[target_class][col] = None
        
        # Update values for matching indices
        common_idx = target_dfs[target_class].index.intersection(result.index)
        for col in result.columns:
            if col in target_dfs[target_class].columns:
                target_dfs[target_class].loc[common_idx, col] = result.loc[common_idx, col]
        
        # Add new indices
        new_idx = result.index.difference(target_dfs[target_class].index)
        if len(new_idx) > 0:
            target_dfs[target_class] = pd.concat([target_dfs[target_class], result.loc[new_idx]])
    
    return target_dfs


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


def reorder_entity_names(names: List[str], order_spec: List[List[int]]) -> List[str]:
    """
    Reorder entity name dimensions according to order specification.
    order_spec: [[source_dims for target_dim_0], [source_dims for target_dim_1], ...]
    Dimension indices are 1-based in config, convert to 0-based.
    """
    new_names = []
    
    for name in names:
        parts = name.split('.')
        
        new_parts = []
        for target_dim_spec in order_spec:
            # Get source dimension indices (convert from 1-based to 0-based)
            source_indices = [i - 1 for i in target_dim_spec]
            
            if len(source_indices) == 1:
                if source_indices[0] < len(parts):
                    new_parts.append(parts[source_indices[0]])
            else:
                # Concatenate multiple source dimensions
                concat = []
                for idx in source_indices:
                    if idx < len(parts):
                        concat.append(parts[idx])
                if concat:
                    new_parts.append('__'.join(concat))
        
        if new_parts:
            new_names.append('.'.join(new_parts))
    
    return new_names


def reorder_dimensions(df: pd.DataFrame, order_spec: List[List[int]], 
                      aggregate: str = None, is_ts: bool = False) -> pd.DataFrame:
    """
    Reorder dimensions in DataFrame index/columns according to order specification.
    """
    entity_idx = get_entity_index(df)
    entity_names = index_to_names(entity_idx)
    
    # Reorder entity names
    new_names = reorder_entity_names(entity_names, order_spec)
    
    # Create new index
    new_idx = create_index_from_names(new_names)
    
    # If aggregating (collapsing dimensions), need to group
    if aggregate:
        # Create temporary dataframe with new index
        temp_df = df.copy()
        if is_ts:
            # For time series, transpose, set index, aggregate, transpose back
            temp_df = temp_df.T
            temp_df.index = new_idx
            if aggregate == 'sum':
                temp_df = temp_df.groupby(level=list(range(temp_df.index.nlevels))).sum()
            temp_df = temp_df.T
            result = temp_df
        else:
            temp_df.index = new_idx
            if aggregate == 'sum':
                result = temp_df.groupby(level=list(range(temp_df.index.nlevels))).sum()
            else:
                raise ValueError(f"Unsupported aggregation: {aggregate}")
    else:
        # Just reindex
        result = set_entity_index(df, new_idx, is_ts)
    
    return result


def apply_rename(df: pd.DataFrame, source_attribute: str, rename_map: Dict, is_ts: bool = False) -> pd.DataFrame:
    """Apply rename mapping to entity names in index or columns."""    """Apply rename mapping to column values."""
    if is_ts:
        raise ValueError(f"Unsupported renaming of values in a timeseries object for parameter: {target_attribute}")
    else:
        if source_attribute in df.columns:
            df[source_attribute] = df[source_attribute].replace(rename_map)
    return df

def transform_data(source_dfs: Dict[str, pd.DataFrame], 
                   config_path: str) -> Dict[str, pd.DataFrame]:
    """
    Main transformer function that reads config and applies transformations.
    
    Args:
        source_dfs: Dictionary of source dataframes with entity names in index/columns
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary of transformed target dataframes with entity names in index/columns
    """
    config = load_config(config_path)
    target_dfs = {}
    
    for operation_name in config.keys():
        print(f"Processing operation: {operation_name}")
        
        source_specs, target_specs, operations, dimensions = parse_operation(operation_name, config)
        operation_type = get_operation_type(source_specs, target_specs, operations)
        
        print(f"  Operation type: {operation_type}")
        
        for target_spec in target_specs:
            if operation_type == 'copy_entities':
                target_dfs = copy_entities(source_dfs, source_specs, target_spec, 
                                        target_dfs, operations, dimensions)
            elif operation_type == 'create_parameter':
                target_dfs = create_parameter(source_dfs, source_specs, target_spec,
                                            target_dfs, operations, dimensions)
            elif operation_type == 'transform_parameter':
                target_dfs = transform_parameter(source_dfs, source_specs, target_spec,
                                                target_dfs, operations, dimensions)
    
    return target_dfs


# Example usage:
if __name__ == "__main__":
    # Load your source dataframes (dfs)
    # result_dfs = transform_data(dfs, 'to_flextool.yaml')
    pass