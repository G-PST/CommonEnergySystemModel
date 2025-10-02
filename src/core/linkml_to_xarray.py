import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Type
from linkml_runtime.utils.yamlutils import YAMLRoot
from generated.cesm import Database, Entity

# Configuration for multi-dimensional classes
MULTI_DIMENSIONAL_CLASSES = {
    'unit_to_node': {
        'dimensions': ['unit', 'node'],
        'source_field': 'source_name',
        'sink_field': 'sink_name'
    },
    'node_to_unit': {
        'dimensions': ['node', 'unit'],
        'source_field': 'source_name', 
        'sink_field': 'sink_name'
    },
    'link': {
        'dimensions': ['node', 'node'],
        'source_field': 'node_A',
        'sink_field': 'node_B'
    }
}

def linkml_to_xarray(database: Database) -> Dict[str, xr.DataArray]:
    """
    Convert LinkML Database object to dictionary of xarray DataArrays.
    
    Parameters:
    -----------
    database : Database
        The LinkML Database object containing all entities and timeline
        
    Returns:
    --------
    Dict[str, xr.DataArray]
        Dictionary of DataArrays preserving sparseness for multi-dimensional data
    """
    
    # Get timeline from database (special hardcoded case)
    timeline = database.timeline if database.timeline else []
    time_coord = np.array(timeline)
    timeline_length = len(time_coord)  # Cache timeline length
    
    # Discover all class collections dynamically
    class_collections = _discover_class_collections(database)
    
    # Create all DataArrays in a single dictionary
    all_dataarrays = {}
    
    # Process all classes (single and multi-dimensional)
    for collection_name, entities_dict in class_collections.items():
        if entities_dict:
            class_dataarrays = _create_class_dataarrays(
                collection_name, entities_dict, time_coord, timeline_length
            )
            all_dataarrays.update(class_dataarrays)
    
    return all_dataarrays


def _create_class_dataarrays(collection_name: str, entities_dict: Dict, 
                           time_coord: np.ndarray, timeline_length: int) -> Dict[str, xr.DataArray]:
    """Create DataArrays for any class using unified sparse approach."""
    
    # Get parameter information from the class
    first_entity = next(iter(entities_dict.values()))
    entity_class = type(first_entity)
    param_info = _get_parameter_info(entity_class)
    
    data_arrays = {}
    is_multi_dimensional = collection_name in MULTI_DIMENSIONAL_CLASSES
    
    # Get dimension configuration
    if is_multi_dimensional:
        config = MULTI_DIMENSIONAL_CLASSES[collection_name]
        dimensions = config['dimensions']
        source_field = config['source_field']
        sink_field = config['sink_field']
        
        # Handle same dimension case
        if dimensions[0] == dimensions[1]:
            level_names = [f"{dimensions[0]}_from", f"{dimensions[1]}_to"]
        else:
            level_names = dimensions
        
        coord_fields = [source_field, sink_field]
    else:
        # Single dimension - entity name is the only coordinate
        dim_name = collection_name.rstrip('s')
        level_names = [dim_name]
        coord_fields = [None]  # No field lookup needed
    
    # Define coordinate name for all parameters
    coord_name = f"{collection_name}"
    
    # Process each parameter
    for param_name, is_time_dimensional in param_info.items():
        
        # Check if any entity actually has time-dimensional data
        actual_has_time_dimension = False
        for entity in entities_dict.values():
            value = getattr(entity, param_name, None)
            if value is not None and isinstance(value, list) and is_time_dimensional:
                actual_has_time_dimension = True
                break
        
        # Collect sparse data
        sparse_coords = []
        sparse_values = []
        
        for entity_key, entity in entities_dict.items():
            # Get coordinate values
            if is_multi_dimensional:
                # Multi-dimensional: get from entity fields
                coord_values = []
                for field in coord_fields:
                    val = getattr(entity, field, None)
                    if val is None:
                        break
                    coord_values.append(str(val))
                
                if len(coord_values) != len(coord_fields):
                    continue  # Skip if missing coordinate values
                    
                coord_tuple = tuple(coord_values)
            else:
                # Single-dimensional: use entity key
                coord_tuple = (str(entity_key),)
            
            # Get parameter value
            value = getattr(entity, param_name, None)
            
            if value is not None:
                if actual_has_time_dimension and isinstance(value, list) and len(value) > 0:
                    aligned_value = _align_with_timeline(value, timeline_length)
                    sparse_coords.append(coord_tuple)
                    sparse_values.append(aligned_value)
                elif not actual_has_time_dimension:
                    if not isinstance(value, list):
                        # Convert enums to strings, but preserve numeric types
                        if hasattr(value, 'text'):
                            value = str(value.text)
                        elif hasattr(value, 'value'):
                            value = value.value  # Keep original type for numeric values
                        elif hasattr(value, '__class__') and 'Enum' in str(type(value)):
                            value = str(value)
                        # else: keep original value and type
                    
                    if not (isinstance(value, list) and len(value) == 0):
                        sparse_coords.append(coord_tuple)
                        sparse_values.append(value)
        
        # Create sparse DataArray if we have data
        if sparse_coords and sparse_values:
            # Create coordinate index
            if len(level_names) == 1:
                # For single dimension, extract values and use coord_name as index name
                coord_values = [coord[0] for coord in sparse_coords]
                coord_index = pd.Index(coord_values, name=coord_name)
            else:
                coord_index = pd.MultiIndex.from_tuples(sparse_coords, names=level_names)
            
            var_name = f"{collection_name}.{param_name}"
            
            if actual_has_time_dimension:
                data_array = xr.DataArray(
                    np.array(sparse_values),
                    dims=[coord_name, 'time'],
                    coords={coord_name: coord_index, 'time': time_coord},
                    name=var_name
                )
            else:
                data_array = xr.DataArray(
                    sparse_values,
                    dims=[coord_name],
                    coords=[coord_index],
                    name=var_name
                )
            
            data_arrays[var_name] = data_array
    
    return data_arrays

def _discover_class_collections(database: Database) -> Dict[str, Dict]:
    """Dynamically discover all class collections in the database."""
    collections = {}
    
    for attr_name in dir(database):
        if attr_name.startswith('_') or attr_name in ['id', 'timeline', 'currency_year', 'entity']:
            continue
            
        attr_value = getattr(database, attr_name)
        
        # Check if it's a list of entities (class collection)
        if isinstance(attr_value, list) and attr_value:
            first_value = attr_value[0]
            if isinstance(first_value, Entity):
                # Convert list to dictionary keyed by name
                collections[attr_name] = {entity.name: entity for entity in attr_value}
    
    return collections


def _get_parameter_info(entity_class: Type) -> Dict[str, bool]:
    """
    Extract parameter information from entity class.
    Returns dict of {parameter_name: is_time_dimensional}
    """
    excluded_fields = {
        'name', 'id', 'semantic_id', 'alternative_names', 'description', 
        'source', 'sink', 'source_name', 'sink_name', 'node_A', 'node_B',
        'node_type', '_inherited_slots', 'class_class_uri', 'class_class_curie',
        'class_name', 'class_model_uri'  # Exclude class metadata attributes
    }
    
    param_info = {}
    
    if hasattr(entity_class, '__dataclass_fields__'):
        for field_name, field_info in entity_class.__dataclass_fields__.items():
            if field_name in excluded_fields:
                continue
                
            # Inline time-dimensional heuristic
            is_time_dimensional = (
                field_name in {'flow_profile', 'profile_limit_upper', 'profile_limit_lower'} or
                any(keyword in field_name.lower() for keyword in ['profile', 'series', 'timeline', 'temporal'])
            )
            param_info[field_name] = is_time_dimensional
    
    return param_info


def _align_with_timeline(values: List[float], timeline_length: int) -> List[float]:
    """Align parameter values with timeline length."""
    if len(values) == timeline_length:
        return values
    elif len(values) < timeline_length:
        last_val = values[-1] if values else np.nan
        return values + [last_val] * (timeline_length - len(values))
    else:
        return values[:timeline_length]