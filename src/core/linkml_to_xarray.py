import xarray as xr
import numpy as np
import inspect
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

        # Get the class type from the first entity
        first_entity = next(iter(entities_dict.values()))
        entity_class = type(first_entity)

        # Skip if this is a multi-dimensional class (handle separately)
        if _is_multi_dimensional_class(collection_name):
            continue

        # Create coordinate for this class dimension
        entity_names = list(entities_dict.keys())
        dim_name = collection_name.rstrip('s')  # Remove plural 's'
        coords[dim_name] = np.array(entity_names)

        # Get parameter information from the class
        param_info = _get_parameter_info(entity_class)

        # Process each parameter
        for param_name, is_time_dimensional in param_info.items():
            param_values = []
            has_time_dimension = is_time_dimensional

            # Collect values for all entities in this class
            for entity_name in entity_names:
                entity = entities_dict[entity_name]
                value = getattr(entity, param_name, None)

                if value is None:
                    # Handle missing values
                    if has_time_dimension:
                        param_values.append([np.nan] * len(timeline))
                    else:
                        param_values.append(np.nan)
                elif isinstance(value, list) and has_time_dimension:
                    # Time series parameter
                    aligned_value = _align_with_timeline(value, len(timeline))
                    param_values.append(aligned_value)
                elif isinstance(value, list) and not has_time_dimension:
                    # Non-time multivalued parameter - take first value or convert to scalar
                    param_values.append(value[0] if value else np.nan)
                else:
                    # Scalar parameter
                    if has_time_dimension:
                        # Broadcast scalar to time dimension
                        param_values.append([value] * len(timeline))
                    else:
                        param_values.append(value)

            # Create DataArray
            if has_time_dimension:
                # Parameter has time dimension
                data_array = xr.DataArray(
                    np.array(param_values),
                    dims=[dim_name, 'time'],
                    coords={dim_name: coords[dim_name], 'time': time_coord},
                    name=f"{collection_name}_{param_name}"
                )
            else:
                # Parameter is scalar for this class
                data_array = xr.DataArray(
                    np.array(param_values),
                    dims=[dim_name],
                    coords={dim_name: coords[dim_name]},
                    name=f"{collection_name}_{param_name}"
                )

            # Add to data_vars with unique name
            var_name = f"{collection_name}_{param_name}"
            data_vars[var_name] = data_array

    # Handle multi-dimensional classes
    _process_multi_dimensional_classes(database, coords, data_vars, time_coord, class_collections)

    # Create Dataset
    dataset = xr.Dataset(data_vars, coords=coords)

    return dataset


def _discover_class_collections(database: Database) -> Dict[str, Dict]:
    """Dynamically discover all class collections in the database."""
    collections = {}

    # Get all attributes of the database object
    for attr_name in dir(database):
        if attr_name.startswith('_') or attr_name in ['id', 'timeline', 'currency_year', 'entity']:
            continue

        attr_value = getattr(database, attr_name)

        # Check if it's a dictionary of entities (class collection)
        if isinstance(attr_value, dict) and attr_value:
            # Check if values are Entity subclasses
            first_value = next(iter(attr_value.values()))
            if isinstance(first_value, Entity):
                collections[attr_name] = attr_value

    return collections


def _get_parameter_info(entity_class: Type) -> Dict[str, bool]:
    """
    Extract parameter information from entity class.
    Returns dict of {parameter_name: is_time_dimensional}
    """
    excluded_fields = {
        'name', 'id', 'semantic_id', 'alternative_names', 'description',
        'source', 'sink', 'source_name', 'sink_name', 'node_A', 'node_B',
        'node_type'  # Enum fields that are metadata
    }

    param_info = {}

    # Get all dataclass fields
    if hasattr(entity_class, '__dataclass_fields__'):
        for field_name, field_info in entity_class.__dataclass_fields__.items():
            if field_name in excluded_fields:
                continue

            # Check if field is time dimensional
            # For now, use heuristic - in future this would come from schema annotations
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

    # Check if field name suggests time dimension
    if field_name in time_dimensional_patterns:
        return True

    # Check if field name contains time-related keywords
    time_keywords = ['profile', 'series', 'timeline', 'temporal']
    if any(keyword in field_name.lower() for keyword in time_keywords):
        return True

    return False


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
        # Pad with last value or NaN
        last_val = values[-1] if values else np.nan
        return values + [last_val] * (timeline_length - len(values))
    else:
        # Truncate to timeline length
        return values[:timeline_length]


def _process_multi_dimensional_classes(database: Database, coords: Dict,
                                     data_vars: Dict, time_coord: np.ndarray,
                                     class_collections: Dict):
    """Process classes that have multiple dimensions."""

    for collection_name, entities_dict in class_collections.items():
        if not entities_dict:
            continue

        first_entity = next(iter(entities_dict.values()))
        entity_class = type(first_entity)

        if not _is_multi_dimensional_class(collection_name):
            continue

        # Get dimension mapping for this class
        mapping = _get_dimension_mapping(collection_name)
        if not mapping:
            continue

        dim1, dim2, source_field, sink_field = mapping

        # Get all unique source and sink names
        sources = set()
        sinks = set()

        for entity in entities_dict.values():
            source_val = getattr(entity, source_field, None)
            sink_val = getattr(entity, sink_field, None)

            if source_val:
                sources.add(str(source_val))
            if sink_val:
                sinks.add(str(sink_val))

        # Create coordinates if they don't exist
        if dim1 not in coords and sources:
            coords[dim1] = np.array(sorted(sources))
        if dim2 not in coords and sinks:
            coords[dim2] = np.array(sorted(sinks))

        # Get parameter information
        param_info = _get_parameter_info(entity_class)

        # Process each parameter
        for param_name, is_time_dimensional in param_info.items():
            if dim1 not in coords or dim2 not in coords:
                continue

            source_list = list(coords[dim1])
            sink_list = list(coords[dim2])

            if is_time_dimensional:
                param_data = np.full((len(source_list), len(sink_list), len(time_coord)), np.nan)
            else:
                param_data = np.full((len(source_list), len(sink_list)), np.nan)

            # Fill in actual values
            for entity in entities_dict.values():
                source_val = getattr(entity, source_field, None)
                sink_val = getattr(entity, sink_field, None)

                if source_val and sink_val:
                    try:
                        source_idx = source_list.index(str(source_val))
                        sink_idx = sink_list.index(str(sink_val))

                        value = getattr(entity, param_name, None)
                        if value is not None:
                            if is_time_dimensional and isinstance(value, list):
                                aligned_value = _align_with_timeline(value, len(time_coord))
                                param_data[source_idx, sink_idx, :] = aligned_value
                            elif is_time_dimensional:
                                param_data[source_idx, sink_idx, :] = value
                            else:
                                param_data[source_idx, sink_idx] = value
                    except ValueError:
                        # Skip if source/sink not found in coordinates
                        continue

            # Create DataArray
            if is_time_dimensional:
                dims = [dim1, dim2, 'time']
                coords_dict = {dim1: coords[dim1], dim2: coords[dim2], 'time': time_coord}
            else:
                dims = [dim1, dim2]
                coords_dict = {dim1: coords[dim1], dim2: coords[dim2]}

            data_array = xr.DataArray(
                param_data,
                dims=dims,
                coords=coords_dict,
                name=f"{collection_name}_{param_name}"
            )

            var_name = f"{collection_name}_{param_name}"
            data_vars[var_name] = data_array


# Example usage function
def example_usage():
    """Example of how to use the converter with your sample data."""

    # Assuming you have loaded your YAML data into a Database object
    # database = Database.from_yaml("ines-sample.yaml")

    # Convert to xarray
    # dataset = linkml_to_xarray(database)

    # The function now automatically discovers:
    # - All class collections in the database
    # - All parameters in each class (excluding metadata)
    # - Multi-dimensional relationships
    # - Time-dimensional parameters (via heuristics for now)

    # Example operations you can now do:
    # print(dataset)
    # print(dataset.coords)
    # print(dataset.data_vars)

    pass