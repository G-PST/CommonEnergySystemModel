import pandas as pd
import inspect
from typing import Dict, List, Any, Tuple
import yaml

def yaml_to_df(database, schema_path: str = None) -> Dict[str, pd.DataFrame]:
    """
    Extract DataFrames from a LinkML-generated database object.
    
    Args:
        database: The database object (generated from LinkML schema)
        schema_path: Optional path to schema YAML for time_dimensional annotation detection
    
    Returns:
        Dictionary with keys as table names and values as DataFrames
        - Class DataFrames: {class_name} (e.g., 'balances')
        - Time-series DataFrames: {class_name}.ts.{attribute} (e.g., 'balances.ts.flow_profile')
    """
    
    # Load schema to identify time_dimensional attributes
    time_dimensional_attrs = set()
    if schema_path:
        time_dimensional_attrs = _extract_time_dimensional_attrs(schema_path)
    
    # Get timeline from database
    timeline = getattr(database, 'timeline', None)
    if timeline is None:
        raise ValueError("Database must have a 'timeline' attribute")
    
    result_dfs = {}
    
    # Get all collection attributes from database (excluding 'timeline', 'id', etc.)
    collection_attrs = _get_collection_attributes(database)
    
    for attr_name in collection_attrs:
        collection = getattr(database, attr_name, [])
        if not collection:
            continue
            
        class_name = attr_name  # e.g., 'balances', 'storages'
        
        # Extract single-dimensional and time-series data separately
        single_dim_data = []
        timeseries_data = {}
        
        for entity in collection:
            entity_dict = _entity_to_dict(entity)
            
            # Get entity identifier for time-series column naming (before filtering)
            entity_name = getattr(entity, 'name', None) or getattr(entity, 'id', None)
            if entity_name is None:
                entity_name = f'entity_{len(single_dim_data)}'
            
            # Always include name/id to guarantee entity exists in table
            single_dim_row = {}
            if 'name' in entity_dict:
                single_dim_row['name'] = entity_dict['name']
            if 'id' in entity_dict:
                single_dim_row['id'] = entity_dict['id']
            
            for key, value in entity_dict.items():
                if _is_timeseries_attribute(key, value, time_dimensional_attrs):
                    # Handle time-series data
                    if value is not None and len(value) > 0:
                        ts_key = f"{class_name}.ts.{key}"
                        if ts_key not in timeseries_data:
                            timeseries_data[ts_key] = {'datetime': timeline}
                        
                        # Use entity name for column name
                        timeseries_data[ts_key][entity_name] = value
                else:
                    # Handle single-dimensional data (skip name/id as already added)
                    if key not in ['name', 'id']:
                        single_dim_row[key] = value
            
            # Always append - even if only name/id exist
            single_dim_data.append(single_dim_row)
        
        # Create class DataFrame
        if single_dim_data:
            df = pd.DataFrame(single_dim_data)
            # Reorder columns: name first, id second, rest alphabetical
            cols = []
            if 'name' in df.columns:
                cols.append('name')
            if 'id' in df.columns:
                cols.append('id')
            # Add remaining columns in alphabetical order
            remaining_cols = sorted([col for col in df.columns if col not in ['name', 'id']])
            cols.extend(remaining_cols)
            result_dfs[f"{class_name}"] = df[cols]
        
        # Create time-series DataFrames
        for ts_key, ts_dict in timeseries_data.items():
            df = pd.DataFrame(ts_dict)
            # Ensure datetime is first column
            cols = ['datetime'] + [col for col in df.columns if col != 'datetime']
            result_dfs[ts_key] = df[cols]
    
    return result_dfs

def _extract_time_dimensional_attrs(schema_path: str) -> set:
    """Extract attributes marked with time_dimensional: true from schema."""
    time_dimensional = set()
    
    try:
        with open(schema_path, 'r') as f:
            schema = yaml.safe_load(f)
        
        def check_attributes(obj_dict):
            if isinstance(obj_dict, dict):
                if 'attributes' in obj_dict:
                    for attr_name, attr_def in obj_dict['attributes'].items():
                        if isinstance(attr_def, dict) and 'annotations' in attr_def:
                            annotations = attr_def['annotations']
                            if annotations and 'time_dimensional' in annotations:
                                time_dimensional.add(attr_name)
                
                # Recursively check nested objects
                for value in obj_dict.values():
                    check_attributes(value)
        
        check_attributes(schema)
        
    except Exception as e:
        print(f"Warning: Could not parse schema file: {e}")
    
    return time_dimensional

def _get_collection_attributes(database) -> List[str]:
    """Get all collection attributes from database object."""
    collection_attrs = []
    
    # Get all attributes that are lists/collections (excluding special ones)
    for attr_name in dir(database):
        if attr_name.startswith('_'):
            continue
        if attr_name in ['timeline', 'id', 'currency_year']:
            continue
            
        attr_value = getattr(database, attr_name, None)
        if isinstance(attr_value, list):
            collection_attrs.append(attr_name)
    
    return collection_attrs

def _entity_to_dict(entity) -> dict:
    """Convert entity object to dictionary, handling nested objects."""
    # Attributes to skip (metadata, not actual data)
    # Keep 'name' and 'id' for entity identification
    skip_attrs = {
        'semantic_id', 'alternative_names', 'description',
        'source', 'sink', 'source_name', 'sink_name', 'node_A', 'node_B',
        'node_type', '_inherited_slots', 'class_class_uri', 'class_class_curie',
        'class_name', 'class_model_uri'
    }
    
    entity_dict = {}
    
    for attr_name in dir(entity):
        if attr_name.startswith('_'):
            continue
        
        # Skip metadata attributes
        if attr_name in skip_attrs:
            continue
            
        try:
            value = getattr(entity, attr_name)
            
            # Skip methods and callable attributes
            if callable(value):
                continue
                
            # Handle None values
            if value is None:
                entity_dict[attr_name] = None
            # Handle simple types
            elif isinstance(value, (str, int, float, bool)):
                entity_dict[attr_name] = value
            # Handle lists (potential time-series)
            elif isinstance(value, list):
                entity_dict[attr_name] = value
            # Handle enum values
            elif hasattr(value, 'text'):
                entity_dict[attr_name] = value.text
            # Handle objects with string representation
            else:
                entity_dict[attr_name] = str(value)
                
        except Exception:
            # Skip attributes that can't be accessed
            continue
    
    return entity_dict

def _is_timeseries_attribute(attr_name: str, value: Any, time_dimensional_attrs: set) -> bool:
    """Determine if an attribute represents time-series data."""
    # Check if explicitly marked as time_dimensional
    if attr_name in time_dimensional_attrs:
        return True
    
    # Check if it's a list of numbers (heuristic)
    if isinstance(value, list) and len(value) > 0:
        # Check if first few elements are numeric
        sample = value[:min(3, len(value))]
        if all(isinstance(x, (int, float)) for x in sample):
            return True
    
    return False

# Example usage:
def example_usage():
    """
    Example of how to use the extractor function.
    """
    from linkml_runtime.loaders import yaml_loader
    # from your_generated_module import Database  # Import your generated class
    
    # Load your data
    # database = yaml_loader.load("your_data.yaml", target_class=Database)
    
    # Extract DataFrames
    # dfs = extract_dataframes_from_database(database, schema_path="cesm.yaml")
    
    # Access the results
    # balances_df = dfs.get('balances')
    # flow_profile_ts = dfs.get('balances.ts.flow_profile')
    
    print("Function ready to use!")

if __name__ == "__main__":
    example_usage()