import pandas as pd
import numpy as np
import inspect
from typing import Dict, List, Any, Tuple
import dataclasses
import yaml
import typing
from linkml_runtime.utils.schemaview import SchemaView
from linkml_runtime.utils.yamlutils import extended_float, extended_int
from linkml_runtime.loaders import yaml_loader

def yaml_to_df(dataset, schema_path: str = None, strict: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Extract DataFrames from a LinkML-generated dataset object.

    Args:
        dataset: The dataset object (generated from LinkML schema)
        schema_path: Optional path to schema YAML for time_dimensional annotation detection
        strict: If True, require timeline attribute. If False, timeline is optional
                (for incremental updates where timeline may already exist in database).
                Default: True.

    Returns:
        Dictionary with keys as table names and values as DataFrames
        - Class DataFrames: {class_name} (e.g., 'balances')
        - Time-series DataFrames: {class_name}.ts.{attribute} (e.g., 'balances.ts.flow_profile')
    """

    # Load schema to identify time_dimensional attributes
    schema = SchemaView(schema_path)

    # Get timeline from dataset
    timeline = getattr(dataset, 'timeline', None)
    if timeline is None:
        if strict:
            raise ValueError("Dataset must have a 'timeline' attribute")
        else:
            print("Warning: No timeline in dataset. Time-series data will be skipped.")

    result_dfs = {}
    if timeline is not None:
        result_dfs['timeline'] = df = pd.DataFrame(index=pd.to_datetime(timeline))
    
    # Get all collection attributes from dataset (excluding 'timeline', 'id', etc.)
    collection_attrs = _get_collection_attributes(dataset)
    
    for attr_name in collection_attrs:
        collection = getattr(dataset, attr_name, [])
        if not collection:
            continue
            
        slot_class_name = attr_name  # e.g., 'balance', 'storage'
        slot = schema.induced_slot(slot_class_name, "Dataset")        
        class_name = slot.range
        dimensions = get_dimensions(schema, class_name)

        if schema.get_class(slot.range):
            range_class_name = slot.range


        # Extract single-dimensional and time-series data separately
        single_dim_data = []
        timeseries_data = {}
        period_data = {}
        array_data = {}
        
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
                datatype = detect_datatype(value)
                if datatype == "list_of_floats":
                    # Handle time-series data (skip if no timeline)
                    if timeline is not None:
                        ts_key = f"{slot_class_name}.ts.{key}"
                        if ts_key not in timeseries_data:
                            timeseries_data[ts_key] = {'datetime': pd.to_datetime(timeline)}

                        # Use entity name for column name
                        timeseries_data[ts_key][entity_name] = np.array(value, dtype=float)
                elif datatype == "list_of_strings":
                    array_key = f"{slot_class_name}.array.{key}"
                    if array_key not in array_data:
                        array_data[array_key] = {}
                    array_data[array_key][entity_name] = np.array(value, dtype=str)
                elif datatype == "lists_of_strings_and_floats":
                    map_key = f"{slot_class_name}.map.{key}"
                    if map_key not in period_data:
                        period_data[map_key] = {'period': value[0]}
                    period_data[map_key][entity_name] = pd.to_numeric(value[1])
                elif datatype == None:
                    single_dim_row[key] = None
                else:
                    # Handle single-dimensional data (skip name/id as already added)
                    if key not in ['name', 'id']:
                        single_dim_row[key] = value
            
            # Always append - even if only name/id exist
            single_dim_data.append(single_dim_row)
        
        # Create class DataFrame
        if single_dim_data:
            if dimensions:
                dim_names = ['name'] + [d['name'] for d in dimensions]
                df = pd.DataFrame(single_dim_data).set_index(dim_names)
            else:
                df = pd.DataFrame(single_dim_data).set_index('name')
                df.index.rename(slot_class_name, inplace=True)
            # Reorder columns: name first, id second, rest alphabetical
            cols = []
            if 'id' in df.columns:
                cols.append('id')
            # Add remaining columns in alphabetical order
            remaining_cols = sorted([col for col in df.columns if col not in ['id']])
            cols.extend(remaining_cols)
            result_dfs[f"{slot_class_name}"] = df[cols]
        
        # Create time-series DataFrames
        for ts_key, ts_dict in timeseries_data.items():
            df = pd.DataFrame(ts_dict).set_index('datetime').astype('float64')
            if dimensions:
                df.columns = pd.MultiIndex.from_tuples(
                    [tuple([col] + col.split('.')) for col in df.columns]
                )
                dim_names = ['name'] + [d['name'] for d in dimensions]
                df.columns.names = dim_names
            result_dfs[ts_key] = df
            if dimensions:
                dim_names = [d['name'] for d in dimensions]
        
        for array_key, array_dict in array_data.items():
            df = pd.DataFrame(array_dict)
            if dimensions:
                df.columns = pd.MultiIndex.from_tuples(
                    [tuple([col] + col.split('.')) for col in df.columns]
                )
                dim_names = ['name'] + [d['name'] for d in dimensions]
                df.columns.names = dim_names
            result_dfs[array_key] = df
            if dimensions:
                dim_names = [d['name'] for d in dimensions]
    
    return result_dfs

def detect_datatype(value):
    if isinstance(value, str):
        return "string"
    
    if isinstance(value, (float, int, extended_float, extended_int)):
        return "float"
    
    if isinstance(value, list):
        # Check for [floats]
        if isinstance(value[0], (float, int, extended_float, extended_int)):
            return "list_of_floats"
        
        # Check for [strings]
        if isinstance(value[0], str):
            return "list_of_strings"
        
        # Check for [[strings], [floats]]
        if (len(value) == 2 and 
            isinstance(value[0], list) and 
            isinstance(value[1], list) and
            isinstance(value[0][0], (str)) and
            isinstance(value[1][0], (float, int, extended_float, extended_int))):
            return "lists_of_strings_and_floats"
    
    return None  # No match

def get_dimensions(schema, class_name: str):
    """Get dimension info from schema"""
    slots = schema.class_induced_slots(class_name)
    dimensions = []
    for slot in slots:
        # Check for dimension annotation
        if slot.annotations and slot.annotations._get('is_dimension'):
            dimensions.append({
                'name': slot.name,
                'range': slot.range
            })
    return dimensions


def get_class_from_field(root_class, field_name):
    """Extract the actual class from a field's type hint"""
    field = root_class.__dataclass_fields__[field_name]
    field_type = field.type
    
    # Parse Union types to find the actual class
    if hasattr(field_type, '__origin__'):
        args = typing.get_args(field_type)
        for arg in args:
            if hasattr(arg, '__origin__'):  # dict, list
                inner_args = typing.get_args(arg)
                for inner in inner_args:
                    # Check if it's a Union (the dict value)
                    if hasattr(inner, '__origin__') and inner.__origin__ is typing.Union:
                        union_args = typing.get_args(inner)
                        for ua in union_args:
                            if isinstance(ua, type) and hasattr(ua, '__dataclass_fields__'):
                                return ua
                    # Direct class reference
                    elif isinstance(inner, type) and hasattr(inner, '__dataclass_fields__'):
                        return inner
    return None


def _get_collection_attributes(dataset) -> List[str]:
    """Get all collection attributes from dataset object."""
    collection_attrs = []
    
    # Get all attributes that are lists/collections (excluding special ones)
    for attr_name in dir(dataset):
        if attr_name.startswith('_'):
            continue
        if attr_name in ['timeline', 'id', 'currency_year']:
            continue
            
        attr_value = getattr(dataset, attr_name, None)
        if isinstance(attr_value, list):
            collection_attrs.append(attr_name)
    
    return collection_attrs

def _entity_to_dict(entity) -> dict:
    """Convert entity object to dictionary, handling nested objects."""
    # Attributes to skip (metadata, not actual data)
    # Keep 'name' and 'id' for entity identification
    skip_attrs = {
        'semantic_id', 'alternative_names', 'description',
        'node_type', '_inherited_slots', 'class_class_uri', 'class_class_curie',
        'class_name', 'class_model_uri', 'linkml_meta'
    }
    
    entity_dict = {}
    
    for attr_name in dir(entity):
        if attr_name.startswith('_'):
            continue
        
        if attr_name.startswith('model_'):
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
    if isinstance(value, list) and len(value) > 1:
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
    # from your_generated_module import dataset  # Import your generated class
    
    # Load your data
    # dataset = yaml_loader.load("your_data.yaml", target_class=dataset)
    
    # Extract DataFrames
    # dfs = extract_dataframes_from_dataset(dataset, schema_path="cesm.yaml")
    
    # Access the results
    # balances_df = dfs.get('balances')
    # flow_profile_ts = dfs.get('balances.ts.flow_profile')
    
    print("Function ready to use!")

if __name__ == "__main__":
    example_usage()