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

def linkml_to_xarray(database: Database) -> Dict[str, xr.Dataset]:
    """
    Convert LinkML Database object to multiple xarray Datasets.
    
    Parameters:
    -----------
    database : Database
        The LinkML Database object containing all entities and timeline
        
    Returns:
    --------
    Dict[str, xr.Dataset]
        Dictionary of datasets:
        - 'base': Dataset with single-dimensional classes
        - 'unit_to_node': Dataset with unit_to_node parameters
        - 'node_to_unit': Dataset with node_to_unit parameters  
        - 'link': Dataset with link parameters
    """
    
    # Get timeline from database (special hardcoded case)
    timeline = database.timeline if database.timeline else []
    time_coord = np.array(timeline)
    
    # Discover all class collections dynamically
    class_collections = _discover_class_collections(database)
    
    # Create base dataset for single-dimensional classes
    base_dataset = _create_base_dataset(class_collections, time_coord)
    
    # Create separate datasets for each multi-dimensional class
    multi_datasets = {}
    for collection_name in class_collections:
        if _is_multi_dimensional_class(collection_name):
            dataset = _create_multi_dimensional_dataset(
                collection_name, class_collections[collection_name], time_coord
            )
            if dataset is not None:
                multi_datasets[collection_name] = dataset
    
    # Combine results
    results = {'base': base_dataset}
    results.update(multi_datasets)
    
    return results


def _create_base_dataset(class_collections: Dict, time_coord: np.ndarray) -> xr.Dataset:
    """Create dataset for single-dimensional classes."""
    coords = {'time': time_coord}
    data_vars = {}
    
    # Process each single-dimensional class collection
    for collection_name, entities_dict in class_collections.items():
        if not entities_dict or _is_multi_dimensional_class(collection_name):
            continue
            
        # Create coordinate for this class dimension
        entity_names = list(entities_dict.keys())
        dim_name = collection_name.rstrip('s')  # Remove plural 's'
        coords[dim_name] = np.array(entity_names)
        
        # Get parameter information from the class
        first_entity = next(iter(entities_dict.values()))
        entity_class = type(first_entity)
        param_info = _get_parameter_info(entity_class)
        
        # Process each parameter
        for param_name, is_time_dimensional in param_info.items():
            param_values = []
            actual_has_time_dimension = False
            
            # First pass: check if any entity actually has time-dimensional data
            for entity_name in entity_names:
                entity = entities_dict[entity_name]
                value = getattr(entity, param_name, None)
                if value is not None and isinstance(value, list) and is_time_dimensional:
                    actual_has_time_dimension = True
                    break
            
            # Second pass: collect values consistently
            for entity_name in entity_names:
                entity = entities_dict[entity_name]
                value = getattr(entity, param_name, None)
                
                if value is None:
                    if actual_has_time_dimension:
                        param_values.append([np.nan] * len(time_coord))
                    else:
                        param_values.append(np.nan)
                elif isinstance(value, list) and is_time_dimensional:
                    aligned_value = _align_with_timeline(value, len(time_coord))
                    param_values.append(aligned_value)
                elif isinstance(value, list) and not is_time_dimensional:
                    first_val = value[0] if value else np.nan
                    param_values.append(_convert_enum_to_string(first_val))
                else:
                    # Scalar parameter
                    processed_value = _convert_enum_to_string(value)
                    if actual_has_time_dimension:
                        param_values.append([processed_value] * len(time_coord))
                    else:
                        param_values.append(processed_value)
            
            # Clean enum objects before creating DataArray
            cleaned_param_values = _clean_enum_objects(param_values)
            
            # Create DataArray
            if actual_has_time_dimension:
                data_array = xr.DataArray(
                    np.array(cleaned_param_values),
                    dims=[dim_name, 'time'],
                    coords={dim_name: coords[dim_name], 'time': time_coord},
                    name=f"{collection_name}_{param_name}"
                )
            else:
                data_array = xr.DataArray(
                    np.array(cleaned_param_values),
                    dims=[dim_name],
                    coords={dim_name: coords[dim_name]},
                    name=f"{collection_name}_{param_name}"
                )
            
            var_name = f"{collection_name}_{param_name}"
            data_vars[var_name] = data_array
    
    return xr.Dataset(data_vars, coords=coords)


def _create_multi_dimensional_dataset(collection_name: str, entities_dict: Dict, 
                                     time_coord: np.ndarray) -> xr.Dataset:
    """Create dataset for a single multi-dimensional class."""
    mapping = _get_dimension_mapping(collection_name)
    if not mapping:
        return None
        
    dim1, dim2, source_field, sink_field = mapping
    
    # Get parameter information
    first_entity = next(iter(entities_dict.values()))
    entity_class = type(first_entity)
    param_info = _get_parameter_info(entity_class)
    
    coords = {'time': time_coord}
    data_vars = {}
    
    # Process each parameter
    for param_name, is_time_dimensional in param_info.items():
        
        # Check if any entity actually has time-dimensional data
        actual_has_time_dimension = False
        for entity in entities_dict.values():
            value = getattr(entity, param_name, None)
            if value is not None and isinstance(value, list) and is_time_dimensional:
                actual_has_time_dimension = True
                break
        
        # Collect sparse data - only store actual connections
        sparse_coords = []
        sparse_values = []
        
        for entity in entities_dict.values():
            source_val = getattr(entity, source_field, None)
            sink_val = getattr(entity, sink_field, None)
            value = getattr(entity, param_name, None)
            
            if source_val and sink_val and value is not None:
                # Convert enums before storing
                processed_value = _convert_enum_to_string(value)
                
                if actual_has_time_dimension and isinstance(value, list):
                    aligned_value = _align_with_timeline(value, len(time_coord))
                    sparse_coords.append((str(source_val), str(sink_val)))
                    sparse_values.append(aligned_value)
                elif actual_has_time_dimension and processed_value is not None:
                    # Scalar value broadcast to time dimension
                    broadcast_value = [processed_value] * len(time_coord)
                    sparse_coords.append((str(source_val), str(sink_val)))
                    sparse_values.append(broadcast_value)
                elif not actual_has_time_dimension and processed_value is not None:
                    # Scalar value
                    if not (isinstance(processed_value, list) and len(processed_value) == 0):
                        sparse_coords.append((str(source_val), str(sink_val)))
                        sparse_values.append(processed_value)
        
        # Create sparse DataArray only if we have data
        if sparse_coords and sparse_values:
            # Clean enum objects from sparse values
            cleaned_sparse_values = [_convert_enum_to_string(val) for val in sparse_values]
            
            # Create MultiIndex for sparse coordinates
            if dim1 == dim2:
                # Handle case where both dimensions are the same (e.g., link: node->node)
                level_names = [f"{dim1}_from", f"{dim2}_to"]
            else:
                level_names = [dim1, dim2]
            
            multi_index = pd.MultiIndex.from_tuples(
                sparse_coords, 
                names=level_names
            )
            
            coord_name = f"{collection_name}_coords"
            
            if actual_has_time_dimension:
                # Create 2D array: connections x time
                data_array = xr.DataArray(
                    np.array(cleaned_sparse_values),
                    dims=[coord_name, 'time'],
                    coords={
                        coord_name: multi_index, 
                        'time': time_coord
                    },
                    name=f"{collection_name}_{param_name}"
                )
            else:
                # Create 1D array: connections only
                data_array = xr.DataArray(
                    cleaned_sparse_values,
                    dims=[coord_name],
                    coords=[multi_index],
                    name=f"{collection_name}_{param_name}"
                )
            
            var_name = f"{collection_name}_{param_name}"
            data_vars[var_name] = data_array
    
    # Only return dataset if we have data variables
    if data_vars:
        return xr.Dataset(data_vars, coords=coords)
    else:
        return None


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
                entity_dict = {entity.name: entity for entity in attr_value}
                collections[attr_name] = entity_dict
    
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
                
            is_time_dimensional = _is_time_dimensional_heuristic(field_name, field_info)
            param_info[field_name] = is_time_dimensional
    
    return param_info


def _is_time_dimensional_heuristic(field_name: str, field_info) -> bool:
    """
    Heuristic to determine if a field is time dimensional.
    This should eventually be replaced by schema annotations.
    """
    time_dimensional_patterns = {
        'flow_profile', 'profile_limit_upper', 'profile_limit_lower'
    }
    
    if field_name in time_dimensional_patterns:
        return True
        
    time_keywords = ['profile', 'series', 'timeline', 'temporal']
    if any(keyword in field_name.lower() for keyword in time_keywords):
        return True
    
    return False


def _convert_enum_to_string(value):
    """Convert enum objects to strings."""
    if hasattr(value, 'text'):
        return str(value.text)
    elif hasattr(value, 'value'):
        return str(value.value)
    elif 'Enum' in str(type(value)):
        return str(value)
    else:
        return value


def _clean_enum_objects(param_values):
    """Force convert any remaining enum objects to strings."""
    cleaned_param_values = []
    for val in param_values:
        if hasattr(val, 'text') or hasattr(val, 'value') or 'Enum' in str(type(val)):
            cleaned_param_values.append(_convert_enum_to_string(val))
        elif isinstance(val, list):
            cleaned_list = [_convert_enum_to_string(item) for item in val]
            cleaned_param_values.append(cleaned_list)
        else:
            cleaned_param_values.append(val)
    return cleaned_param_values


def _is_multi_dimensional_class(collection_name: str) -> bool:
    """Check if a class is configured as multi-dimensional."""
    return collection_name in MULTI_DIMENSIONAL_CLASSES


def _get_dimension_mapping(collection_name: str) -> tuple:
    """Get dimension mapping for multi-dimensional classes from configuration."""
    if collection_name not in MULTI_DIMENSIONAL_CLASSES:
        return None
    
    config = MULTI_DIMENSIONAL_CLASSES[collection_name]
    dimensions = config['dimensions']
    source_field = config['source_field']
    sink_field = config['sink_field']
    
    return (dimensions[0], dimensions[1], source_field, sink_field)


def _align_with_timeline(values: List[float], timeline_length: int) -> List[float]:
    """Align parameter values with timeline length."""
    if len(values) == timeline_length:
        return values
    elif len(values) < timeline_length:
        last_val = values[-1] if values else np.nan
        return values + [last_val] * (timeline_length - len(values))
    else:
        return values[:timeline_length]


def _process_multi_dimensional_classes(database: Database, data_vars: Dict, 
                                     time_coord: np.ndarray, class_collections: Dict):
    """Process classes that have multiple dimensions using sparse representation."""
    
    for collection_name, entities_dict in class_collections.items():
        if not entities_dict or not _is_multi_dimensional_class(collection_name):
            continue
            
        mapping = _get_dimension_mapping(collection_name)
        if not mapping:
            continue
            
        dim1, dim2, source_field, sink_field = mapping
        
        # Get parameter information
        first_entity = next(iter(entities_dict.values()))
        entity_class = type(first_entity)
        param_info = _get_parameter_info(entity_class)
        
        # Process each parameter with sparse representation
        for param_name, is_time_dimensional in param_info.items():
            
            # Check if any entity actually has time-dimensional data
            actual_has_time_dimension = False
            for entity in entities_dict.values():
                value = getattr(entity, param_name, None)
                if value is not None and isinstance(value, list) and is_time_dimensional:
                    actual_has_time_dimension = True
                    break
            
            # Collect sparse data - only store actual connections
            sparse_coords = []
            sparse_values = []
            
            for entity in entities_dict.values():
                source_val = getattr(entity, source_field, None)
                sink_val = getattr(entity, sink_field, None)
                value = getattr(entity, param_name, None)
                
                if source_val and sink_val and value is not None:
                    # Convert enums before storing
                    processed_value = _convert_enum_to_string(value)
                    
                    if actual_has_time_dimension and isinstance(value, list):
                        aligned_value = _align_with_timeline(value, len(time_coord))
                        # Store as (source, sink) pairs, with separate time dimension
                        sparse_coords.append((str(source_val), str(sink_val)))
                        sparse_values.append(aligned_value)
                    elif actual_has_time_dimension and processed_value is not None:
                        # Scalar value broadcast to time dimension
                        broadcast_value = [processed_value] * len(time_coord)
                        sparse_coords.append((str(source_val), str(sink_val)))
                        sparse_values.append(broadcast_value)
                    elif not actual_has_time_dimension and processed_value is not None:
                        # Scalar value
                        if not (isinstance(processed_value, list) and len(processed_value) == 0):
                            sparse_coords.append((str(source_val), str(sink_val)))
                            sparse_values.append(processed_value)
            
            # Create sparse DataArray only if we have data
            if sparse_coords and sparse_values:
                # Clean enum objects from sparse values
                cleaned_sparse_values = [_convert_enum_to_string(val) for val in sparse_values]
                
                # Create MultiIndex for sparse coordinates - use simple level names
                if dim1 == dim2:
                    # Handle case where both dimensions are the same (e.g., link: node->node)
                    level_names = [f"{dim1}_from", f"{dim2}_to"]
                else:
                    level_names = [dim1, dim2]
                
                multi_index = pd.MultiIndex.from_tuples(
                    sparse_coords, 
                    names=level_names
                )
                
                coord_name = f"{collection_name}_coords"
                
                if actual_has_time_dimension:
                    # Create 2D array: connections x time
                    data_array = xr.DataArray(
                        np.array(cleaned_sparse_values),
                        dims=[coord_name, 'time'],
                        coords={
                            coord_name: multi_index, 
                            'time': time_coord
                        },
                        name=f"{collection_name}_{param_name}"
                    )
                else:
                    # Create 1D array: connections only
                    data_array = xr.DataArray(
                        cleaned_sparse_values,
                        dims=[coord_name],
                        coords=[multi_index],
                        name=f"{collection_name}_{param_name}"
                    )
                
                var_name = f"{collection_name}_{param_name}"
                data_vars[var_name] = data_array