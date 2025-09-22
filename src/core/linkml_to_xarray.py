import xarray as xr
import numpy as np
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

def linkml_to_xarray(database: Database) -> xr.Dataset:
    """
    Convert LinkML Database object to xarray Dataset.
    
    Parameters:
    -----------
    database : Database
        The LinkML Database object containing all entities and timeline
        
    Returns:
    --------
    xr.Dataset
        Dataset with coordinates based on class dimensions and parameters as DataArrays
    """
    
    # Get timeline from database (special hardcoded case)
    timeline = database.timeline if database.timeline else []
    time_coord = np.array(timeline)
    
    # Initialize coordinates dictionary
    coords = {'time': time_coord}
    
    # Initialize data variables dictionary
    data_vars = {}
    
    # Discover all class collections dynamically
    class_collections = _discover_class_collections(database)
    
    # Process each class collection
    for collection_name, entities_dict in class_collections.items():
        if not entities_dict:
            continue
            
        # Skip if this is a multi-dimensional class (handle separately)
        if _is_multi_dimensional_class(collection_name):
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
                        param_values.append([np.nan] * len(timeline))
                    else:
                        param_values.append(np.nan)
                elif isinstance(value, list) and is_time_dimensional:
                    aligned_value = _align_with_timeline(value, len(timeline))
                    param_values.append(aligned_value)
                elif isinstance(value, list) and not is_time_dimensional:
                    first_val = value[0] if value else np.nan
                    param_values.append(_convert_enum_to_string(first_val))
                else:
                    # Scalar parameter
                    processed_value = _convert_enum_to_string(value)
                    if actual_has_time_dimension:
                        param_values.append([processed_value] * len(timeline))
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
    
    # Handle multi-dimensional classes
    print("Processing multi-dimensional classes...")
    _process_multi_dimensional_classes(database, coords, data_vars, time_coord, class_collections)
    
    # Create Dataset
    dataset = xr.Dataset(data_vars, coords=coords)
    
    return dataset


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


def _process_multi_dimensional_classes(database: Database, coords: Dict, 
                                     data_vars: Dict, time_coord: np.ndarray,
                                     class_collections: Dict):
    """Process classes that have multiple dimensions using sparse representation."""
    
    for collection_name, entities_dict in class_collections.items():
        print(f"Checking collection '{collection_name}': multi-dim = {_is_multi_dimensional_class(collection_name)}")
        if not entities_dict or not _is_multi_dimensional_class(collection_name):
            continue
            
        print(f"Processing multi-dimensional collection: {collection_name}")
        mapping = _get_dimension_mapping(collection_name)
        if not mapping:
            print(f"  No mapping found for {collection_name}")
            continue
            
        print(f"  Mapping: {mapping}")
        dim1, dim2, source_field, sink_field = mapping
        
        print(f"  Starting coordinate collection...")
        
        # Collect all possible coordinate values from all collections
        all_dim1_values = set()
        all_dim2_values = set()
        
        # Get values from this collection
        for entity in entities_dict.values():
            source_val = getattr(entity, source_field, None)
            sink_val = getattr(entity, sink_field, None)
            if source_val:
                all_dim1_values.add(str(source_val))
            if sink_val:
                all_dim2_values.add(str(sink_val))
        
        print(f"  Found {dim1} values: {all_dim1_values}")
        print(f"  Found {dim2} values: {all_dim2_values}")
        
        # Also collect from other collections to get complete coordinate space
        for other_collection_name, other_entities_dict in class_collections.items():
            if other_collection_name == collection_name:
                continue
                
            # Add entities of matching types
            if dim1 == 'unit' and 'unit' in other_collection_name:
                all_dim1_values.update(other_entities_dict.keys())
            elif dim1 == 'node' and any(node_type in other_collection_name for node_type in ['balance', 'storage', 'commodity']):
                all_dim1_values.update(other_entities_dict.keys())
                
            if dim2 == 'unit' and 'unit' in other_collection_name:
                all_dim2_values.update(other_entities_dict.keys())
            elif dim2 == 'node' and any(node_type in other_collection_name for node_type in ['balance', 'storage', 'commodity']):
                all_dim2_values.update(other_entities_dict.keys())
        
        print(f"  Final {dim1} values: {all_dim1_values}")
        print(f"  Final {dim2} values: {all_dim2_values}")
        
        # Don't add multi-dimensional coordinates to main coords dict - they'll be handled separately
        # Each sparse representation will have its own coordinate space
        
        # Get parameter information
        first_entity = next(iter(entities_dict.values()))
        entity_class = type(first_entity)
        param_info = _get_parameter_info(entity_class)
        
        # Process each parameter with sparse representation
        for param_name, is_time_dimensional in param_info.items():
            if dim1 not in coords or dim2 not in coords:
                continue
            
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
                        for t_idx, time_val in enumerate(aligned_value):
                            if not (np.isnan(time_val) if isinstance(time_val, (int, float)) else False):  # Only store non-NaN values
                                sparse_coords.append((str(source_val), str(sink_val), t_idx))
                                sparse_values.append(time_val)
                    elif actual_has_time_dimension and processed_value is not None:
                        # Scalar value broadcast to time dimension
                        for t_idx in range(len(time_coord)):
                            sparse_coords.append((str(source_val), str(sink_val), t_idx))
                            sparse_values.append(processed_value)
                    elif not actual_has_time_dimension and processed_value is not None:
                        # Scalar value - only add if not None and not empty list
                        if not (isinstance(processed_value, list) and len(processed_value) == 0):
                            sparse_coords.append((str(source_val), str(sink_val)))
                            sparse_values.append(processed_value)
            
            # Create sparse DataArray only if we have data
            if sparse_coords and sparse_values:
                # Clean enum objects from sparse values
                cleaned_sparse_values = [_convert_enum_to_string(val) for val in sparse_values]
                
                # Debug: Check if any problematic values remain
                for i, val in enumerate(cleaned_sparse_values):
                    if hasattr(val, 'text') or 'Enum' in str(type(val)):
                        print(f"WARNING: Enum object still present at index {i}: {val} (type: {type(val)})")
                
                print(f"Creating sparse array for {param_name}: {len(cleaned_sparse_values)} values")
                print(f"Sample values: {cleaned_sparse_values[:3]}")
                print(f"Value types: {[type(v) for v in cleaned_sparse_values[:3]]}")
                
                if actual_has_time_dimension:
                    # Create MultiIndex for sparse 3D data
                    import pandas as pd
                    
                    multi_index = pd.MultiIndex.from_tuples(
                        sparse_coords, 
                        names=[dim1, dim2, 'time']
                    )
                    
                    coord_name = f"{collection_name}_coords"
                    data_array = xr.DataArray(
                        cleaned_sparse_values,
                        coords={coord_name: multi_index},
                        dims=[coord_name],
                        name=f"{collection_name}_{param_name}"
                    )
                else:
                    # Create MultiIndex for sparse 2D data
                    import pandas as pd
                    
                    multi_index = pd.MultiIndex.from_tuples(
                        sparse_coords, 
                        names=[dim1, dim2]
                    )
                    
                    coord_name = f"{collection_name}_coords"
                    data_array = xr.DataArray(
                        cleaned_sparse_values,
                        coords={coord_name: multi_index},
                        dims=[coord_name],
                        name=f"{collection_name}_{param_name}"
                    )
                
                var_name = f"{collection_name}_{param_name}"
                data_vars[var_name] = data_array