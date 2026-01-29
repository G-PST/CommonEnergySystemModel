"""
Transformer for converting source energy system data to target grid database format.

This module converts dict of source dataframes to target dataframes matching SQL schema.
Uses vectorized operations and multi-index alignment for efficiency.
"""

import pandas as pd
import numpy as np
import uuid
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from linkml_runtime.utils.schemaview import SchemaView


def safe_filter(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Safely check if column exists and return boolean mask.
    
    Args:
        df: DataFrame to check
        col: Column name
        
    Returns:
        Boolean series with True where column exists and is not null,
        False for all rows if column doesn't exist
    """
    return df[col].notna() if col in df.columns else pd.Series(False, index=df.index)


def safe_get(df: pd.DataFrame, idx, col: str, default=None):
    """
    Safely get a value from dataframe with fallback.
    
    Args:
        df: DataFrame
        idx: Row index
        col: Column name
        default: Default value if column doesn't exist
        
    Returns:
        Value at df.loc[idx, col] or default
    """
    if col in df.columns:
        return df.loc[idx, col]
    return default


@dataclass
class TransformationErrors:
    """Collector for transformation errors and warnings."""
    errors: List[str] = field(default_factory=list)
    missing_transformations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str):
        self.errors.append(message)
    
    def add_missing(self, message: str):
        self.missing_transformations.append(message)
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def report(self):
        """Print all collected errors and warnings."""
        if self.errors:
            print("\n" + "="*80)
            print("TRANSFORMATION ERRORS:")
            print("="*80)
            for err in self.errors:
                print(f"  • {err}")
        
        if self.warnings:
            print("\n" + "="*80)
            print("TRANSFORMATION WARNINGS:")
            print("="*80)
            for warn in self.warnings:
                print(f"  • {warn}")
        
        if self.missing_transformations:
            print("\n" + "="*80)
            print("MISSING TRANSFORMATIONS (source→target):")
            print("="*80)
            for miss in self.missing_transformations:
                print(f"  • {miss}")


class IDGenerator:
    """Generate globally unique IDs starting from 1."""

    def __init__(self):
        self.current_id = 0
        self.entity_map = {}  # Maps (source_table, key) to id
        self.name_map = {}    # Maps (source_table, key) to name
        self.type_map = {}    # Maps entity_id to entity_type (PowerSystems type)

    def next_id(self) -> int:
        self.current_id += 1
        return self.current_id

    def get_or_create(self, source_table: str, key: str, name: str = None, entity_type: str = None) -> int:
        """Get existing ID or create new one for entity."""
        lookup_key = (source_table, key)
        if lookup_key not in self.entity_map:
            entity_id = self.next_id()
            self.entity_map[lookup_key] = entity_id
            if name:
                self.name_map[lookup_key] = name
            if entity_type:
                self.type_map[entity_id] = entity_type
        return self.entity_map[lookup_key]

    def get(self, source_table: str, key: str) -> int:
        """Get existing ID (raises KeyError if not found)."""
        return self.entity_map[(source_table, key)]

    def get_name(self, source_table: str, key: str) -> str:
        """Get the name for an entity (returns key if name not found)."""
        return self.name_map.get((source_table, key), key)

    def get_type(self, entity_id: int) -> str:
        """Get the entity_type for an entity_id (returns None if not found)."""
        return self.type_map.get(entity_id)


def create_composite_name(index_tuple) -> str:
    """Create composite name from multi-index tuple using double underscores."""
    if isinstance(index_tuple, tuple):
        return "__".join(str(x) for x in index_tuple)
    return str(index_tuple)


def determine_prime_mover(source: Dict[str, pd.DataFrame], unit_name: str) -> str:
    """
    Determine prime mover type for a unit: STEAM (fuel-based) or ROR (renewable).

    Classification rules:
    - Has fuel input → STEAM (thermal generation)
    - No fuel but has profile_limit_upper → ROR (renewable with capacity limits)
    - Neither → STEAM (default thermal)

    Args:
        source: Dictionary of source dataframes
        unit_name: Name of the unit to classify

    Returns:
        'STEAM' or 'ROR'
    """
    # Check if unit has fuel input (via node_to_unit connections)
    if 'node_to_unit' in source and 'commodity' in source:
        node_to_unit = source['node_to_unit']
        commodities = source['commodity']

        # Find inputs to this unit
        unit_inputs = node_to_unit[node_to_unit.index.get_level_values('sink') == unit_name]

        for input_idx in unit_inputs.index:
            # node_to_unit index is a 3-tuple: (name, source, sink)
            if isinstance(input_idx, tuple) and len(input_idx) >= 3:
                _, input_node, _ = input_idx
            else:
                continue

            # Check if input is a fuel commodity
            if input_node in commodities.index:
                commodity_type = safe_get(commodities, input_node, 'commodity_type')
                if commodity_type == 'fuel':
                    return 'STEAM'  # Has fuel → thermal generation

    # Check if unit has profile_limit_upper time series
    # This indicates renewable generation with capacity factor limits
    has_profile = False
    for key in source.keys():
        if key.startswith('unit_to_node.ts.profile_limit_'):
            ts_df = source[key]
            # Check if this unit appears in the time series columns
            if hasattr(ts_df, 'columns'):
                if isinstance(ts_df.columns, pd.MultiIndex):
                    # MultiIndex: check if unit_name appears in 'source' level (level 1)
                    # Columns are typically (name, source, sink)
                    if unit_name in ts_df.columns.get_level_values(1):
                        has_profile = True
                        break
                else:
                    # Simple index
                    if unit_name in ts_df.columns:
                        has_profile = True
                        break

    if has_profile:
        return 'ROR'  # Renewable with capacity limits

    # Default: thermal generation
    return 'STEAM'


def identify_power_grid_nodes(source: Dict[str, pd.DataFrame], errors: TransformationErrors) -> set:
    """Identify balance nodes that are part of power_grid groups."""
    power_grid_nodes = set()
    
    if 'group' not in source or 'group_entity' not in source:
        errors.add_warning("No group or group_entity data - cannot identify power_grid nodes")
        return power_grid_nodes
    
    # Find groups with group_type == 'power_grid'
    groups = source['group']
    power_grid_groups = groups[groups['group_type'] == 'power_grid'].index.tolist()
    
    if not power_grid_groups:
        errors.add_warning("No power_grid groups found")
        return power_grid_nodes
    
    # Find all entities in these groups
    group_entities = source['group_entity']
    for group in power_grid_groups:
        try:
            # Access using xs() for multi-level index
            group_data = group_entities.xs(group, level='group')
            # Extract entity names from the 'entity' level of the index
            entities = group_data.index.get_level_values('entity').tolist()
            power_grid_nodes.update(entities)
        except KeyError:
            # Group not found in group_entities
            errors.add_warning(f"Power grid group '{group}' has no entities in group_entity table")
            continue
    
    return power_grid_nodes


def transform_entities_and_ids(source: Dict[str, pd.DataFrame],
                                id_gen: IDGenerator,
                                errors: TransformationErrors) -> pd.DataFrame:
    """Create entities table and populate ID mappings."""
    entities_data = []

    # Helper to add entity
    def add_entity(source_table: str, entity_type: str, key: str, name: str = None):
        # Store name and entity_type in IDGenerator for later TEXT FK lookups and type resolution
        entity_name = name or create_composite_name(key)
        entity_id = id_gen.get_or_create(source_table, key, entity_name, entity_type)
        entities_data.append({
            'id': entity_id,
            'entity_table': source_table,  # Renamed from source_table
            'entity_type': entity_type
            # Removed: name, description, user_data fields
        })
        return entity_id
    
    # 1. Prime mover types (STEAM, ROR, STORAGE will be created by transform_prime_mover_types)
    # No need to create placeholder here - schema now has INSERT statements

    # 2. Fuels from commodities
    if 'commodity' in source:
        commodities = source['commodity']
        fuels = commodities[commodities['commodity_type'] == 'fuel']
        for commodity_name in fuels.index:
            # Entity type doesn't matter for fuels lookup table
            add_entity('fuels', 'ThermalStandard', commodity_name, commodity_name)

    # 3. Planning regions from groups with type 'node'
    if 'group' in source:
        groups = source['group']
        planning_groups = groups[groups['group_type'] == 'node']
        for group_name in planning_groups.index:
            add_entity('planning_regions', 'Area', group_name, group_name)

    # 4. Balancing topologies from balance nodes in power_grid
    power_grid_nodes = identify_power_grid_nodes(source, errors)
    if 'balance' in source:
        for balance_name in source['balance'].index:
            if balance_name in power_grid_nodes:
                add_entity('balancing_topologies', 'LoadZone', balance_name, balance_name)

    # 5. Generation units from units with units_existing
    if 'unit' in source:
        units = source['unit']
        existing_units = units[safe_filter(units, 'units_existing') & (units['units_existing'] > 0)]
        for unit_name in existing_units.index:
            # Determine entity_type based on prime_mover classification
            prime_mover = determine_prime_mover(source, unit_name)
            entity_type = 'ThermalStandard' if prime_mover == 'STEAM' else 'RenewableDispatch'
            add_entity('generation_units', entity_type, unit_name, unit_name)

    # 6. Supply technologies (investment candidates) from units with investment data
    if 'unit' in source:
        units = source['unit']
        investment_units = units[
            safe_filter(units, 'investment_cost') &
            safe_filter(units, 'discount_rate')
        ]
        for unit_name in investment_units.index:
            # Use _candidate suffix to distinguish from existing units
            key = f"supply_tech_{unit_name}_candidate"
            candidate_name = f"{unit_name}_candidate"
            # Determine entity_type based on prime_mover classification
            prime_mover = determine_prime_mover(source, unit_name)
            entity_type = 'ThermalStandard' if prime_mover == 'STEAM' else 'RenewableDispatch'
            add_entity('supply_technologies', entity_type, key, candidate_name)

    # 7. Arcs from links
    if 'link' in source:
        links = source['link']
        arc_pairs = set()
        for idx in links.index:
            # Link index is always a 3-tuple: (name, node_a, node_b)
            if isinstance(idx, tuple) and len(idx) == 3:
                _, node_a, node_b = idx
            else:
                errors.add_warning(f"Unexpected link index format (expected 3-tuple): {idx}")
                continue

            if node_b and (node_a, node_b) not in arc_pairs and (node_b, node_a) not in arc_pairs:
                arc_pairs.add((node_a, node_b))
                arc_key = f"{node_a}_{node_b}"
                add_entity('arcs', 'Arc', arc_key, arc_key)

    # 8. Transmission lines from links with links_existing
    if 'link' in source:
        links = source['link']
        existing_links = links[safe_filter(links, 'links_existing') & (links['links_existing'] > 0)]
        for idx in existing_links.index:
            link_key = create_composite_name(idx)
            add_entity('transmission_lines', 'Line', link_key, link_key)

    # 8b. Transport technologies (investment candidates) from links with investment data
    if 'link' in source:
        links = source['link']
        investment_links = links[
            safe_filter(links, 'investment_cost') &
            safe_filter(links, 'discount_rate')
        ]
        for idx in investment_links.index:
            link_key = create_composite_name(idx)
            # Use _candidate suffix to distinguish from existing lines
            candidate_key = f"{link_key}_candidate"
            add_entity('transport_technologies', 'Line', candidate_key, candidate_key)

    # 9. Storage units from storage with storages_existing
    if 'storage' in source:
        storages = source['storage']
        existing_storages = storages[safe_filter(storages, 'storages_existing') & (storages['storages_existing'] > 0)]
        for storage_name in existing_storages.index:
            add_entity('storage_units', 'HydroPumpedStorage', storage_name, storage_name)

    # 10. Storage technologies (investment candidates) from storage with investment data
    if 'storage' in source:
        storages = source['storage']
        investment_storages = storages[
            safe_filter(storages, 'investment_cost') &
            safe_filter(storages, 'discount_rate')
        ]
        for storage_name in investment_storages.index:
            # Use _candidate suffix to distinguish from existing storage units
            key = f"storage_tech_{storage_name}_candidate"
            candidate_name = f"{storage_name}_candidate"
            add_entity('storage_technologies', 'HydroPumpedStorage', key, candidate_name)

    # 11. Transmission interchanges from groups with link members
    if 'group' in source and 'group_entity' in source:
        group_entities = source['group_entity']
        for group_name in group_entities.index.get_level_values('group').unique():
            # Access using xs() for multi-level index
            try:
                entities = group_entities.xs(group_name, level='group')
                # Check if members are links (simplification: assume they are if not in balance)
                key = f"interchange_{group_name}"
                add_entity('transmission_interchanges', 'Arc', key, key)
            except KeyError:
                continue
    
    if not entities_data:
        errors.add_error("No entities created from source data")
        return pd.DataFrame()
    
    return pd.DataFrame(entities_data)


def transform_prime_mover_types(id_gen: IDGenerator, errors: TransformationErrors) -> pd.DataFrame:
    """
    Create prime_mover_types table with intermediate GridDB classifications.

    These map to PowerSystems.jl types via db_parser.jl:
    - STEAM → ThermalStandard
    - ROR → RenewableDispatch
    - STORAGE → HydroPumpedStorage / EnergyReservoirStorage

    Note: Schema also has INSERT statements for these values.
    This function returns empty DataFrame since schema populates the table.
    """
    # Return empty DataFrame - schema INSERTs handle population
    return pd.DataFrame(columns=['id', 'name', 'description'])


def transform_fuels(source: Dict[str, pd.DataFrame],
                     id_gen: IDGenerator,
                     errors: TransformationErrors) -> pd.DataFrame:
    """Create fuels table."""
    if 'commodity' not in source:
        return pd.DataFrame(columns=['id', 'name', 'description'])

    commodities = source['commodity']
    fuels = commodities[commodities['commodity_type'] == 'fuel']

    fuels_data = []
    for commodity_name in fuels.index:
        fuel_id = id_gen.get('fuels', commodity_name)
        fuel_name = id_gen.get_name('fuels', commodity_name)
        fuels_data.append({
            'id': fuel_id,
            'name': fuel_name,  # Required field in new schema
            'description': f"Fuel: {commodity_name}"
        })

    errors.add_missing("commodity.price_per_unit → target has no fuel price field")

    return pd.DataFrame(fuels_data)


def transform_planning_regions(source: Dict[str, pd.DataFrame],
                                 id_gen: IDGenerator,
                                 errors: TransformationErrors) -> pd.DataFrame:
    """Create planning_regions table."""
    if 'group' not in source:
        return pd.DataFrame(columns=['id', 'name', 'description'])

    groups = source['group']
    planning_groups = groups[groups['group_type'] == 'node']

    regions_data = []
    for group_name in planning_groups.index:
        region_id = id_gen.get('planning_regions', group_name)
        region_name = id_gen.get_name('planning_regions', group_name)
        regions_data.append({
            'id': region_id,
            'name': region_name,  # Required field in new schema
            'description': f"Planning region: {group_name}"
        })

    return pd.DataFrame(regions_data)


def transform_balancing_topologies(source: Dict[str, pd.DataFrame],
                                     id_gen: IDGenerator,
                                     errors: TransformationErrors) -> pd.DataFrame:
    """Create balancing_topologies table."""
    if 'balance' not in source:
        return pd.DataFrame(columns=['id', 'name', 'area', 'description'])

    power_grid_nodes = identify_power_grid_nodes(source, errors)

    topologies_data = []
    for balance_name in source['balance'].index:
        if balance_name not in power_grid_nodes:
            continue

        topo_id = id_gen.get('balancing_topologies', balance_name)
        topo_name = id_gen.get_name('balancing_topologies', balance_name)

        # Find planning region (area) via group_entity
        area_id = None
        if 'group_entity' in source and 'group' in source:
            group_entities = source['group_entity']
            groups = source['group']

            for group_name in groups[groups['group_type'] == 'node'].index:
                if group_name in group_entities.index.get_level_values('group'):
                    try:
                        # Access using xs() for multi-level index
                        entities_data = group_entities.xs(group_name, level='group')
                        entities = entities_data.index.get_level_values('entity').tolist()
                        if balance_name in entities:
                            area_id = id_gen.get('planning_regions', group_name)
                            break
                    except KeyError:
                        continue

        topologies_data.append({
            'id': topo_id,
            'name': topo_name,  # Required field in new schema
            'area': area_id,
            'description': f"Balancing topology: {balance_name}"
        })

    return pd.DataFrame(topologies_data)


def transform_arcs(source: Dict[str, pd.DataFrame],
                   id_gen: IDGenerator,
                   errors: TransformationErrors) -> pd.DataFrame:
    """Create arcs table from links."""
    if 'link' not in source:
        return pd.DataFrame(columns=['id', 'from_id', 'to_id'])
    
    links = source['link']
    arc_pairs = set()
    arcs_data = []
    
    for idx in links.index:
        # Link index is always a 3-tuple: (name, node_a, node_b)
        if isinstance(idx, tuple) and len(idx) == 3:
            _, node_a, node_b = idx
        else:
            errors.add_warning(f"Link index {idx} is not a 3-tuple, skipping")
            continue
        
        # Deduplicate bidirectional arcs
        if (node_a, node_b) in arc_pairs or (node_b, node_a) in arc_pairs:
            continue
        
        arc_pairs.add((node_a, node_b))
        arc_key = f"{node_a}_{node_b}"
        arc_id = id_gen.get('arcs', arc_key)
        
        # Get node IDs (try balancing topologies first)
        try:
            from_id = id_gen.get('balancing_topologies', node_a)
        except KeyError:
            errors.add_warning(f"Node {node_a} not found in balancing_topologies")
            continue
        
        try:
            to_id = id_gen.get('balancing_topologies', node_b)
        except KeyError:
            errors.add_warning(f"Node {node_b} not found in balancing_topologies")
            continue
        
        arcs_data.append({
            'id': arc_id,
            'from_id': from_id,
            'to_id': to_id
        })
    
    return pd.DataFrame(arcs_data)


def transform_transmission_lines(source: Dict[str, pd.DataFrame],
                                   id_gen: IDGenerator,
                                   errors: TransformationErrors) -> pd.DataFrame:
    """Create transmission_lines table."""
    if 'link' not in source:
        return pd.DataFrame(columns=['id', 'name', 'arc_id', 'continuous_rating', 'ste_rating',
                                     'lte_rating', 'line_length'])

    links = source['link']
    existing_links = links[safe_filter(links, 'links_existing') & (links['links_existing'] > 0)].copy()

    lines_data = []
    for idx in existing_links.index:
        link_key = create_composite_name(idx)
        line_id = id_gen.get('transmission_lines', link_key)
        line_name = id_gen.get_name('transmission_lines', link_key)

        # Get arc_id
        # Link index is always a 3-tuple: (name, node_a, node_b)
        if isinstance(idx, tuple) and len(idx) == 3:
            _, node_a, node_b = idx
            arc_key = f"{node_a}_{node_b}"
            try:
                arc_id = id_gen.get('arcs', arc_key)
            except KeyError:
                # Try reverse direction
                arc_key = f"{node_b}_{node_a}"
                try:
                    arc_id = id_gen.get('arcs', arc_key)
                except KeyError:
                    errors.add_warning(f"Arc not found for link {idx}")
                    continue
        else:
            errors.add_warning(f"Link index {idx} format unexpected (expected 3-tuple)")
            continue

        # Calculate continuous_rating
        links_existing = safe_get(existing_links, idx, 'links_existing', 1.0)
        capacity = safe_get(existing_links, idx, 'capacity', 1.0)
        continuous_rating = links_existing * capacity

        lines_data.append({
            'id': line_id,
            'name': line_name,  # Required field in new schema
            'arc_id': arc_id,
            'continuous_rating': continuous_rating,
            'ste_rating': None,
            'lte_rating': None,
            'line_length': None
        })

    errors.add_missing("link transmission parameters (ste_rating, lte_rating, line_length) not mapped")

    return pd.DataFrame(lines_data)


def transform_generation_units(source: Dict[str, pd.DataFrame],
                                 id_gen: IDGenerator,
                                 errors: TransformationErrors) -> pd.DataFrame:
    """Create generation_units table."""
    if 'unit' not in source or 'unit_to_node' not in source:
        return pd.DataFrame(columns=['id', 'name', 'prime_mover', 'fuel', 'balancing_topology',
                                     'rating', 'base_power'])
    
    units = source['unit']
    existing_units = units[safe_filter(units, 'units_existing') & (units['units_existing'] > 0)].copy()
    unit_to_node = source['unit_to_node']

    power_grid_nodes = identify_power_grid_nodes(source, errors)

    gen_units_data = []

    for unit_name in existing_units.index:
        unit_id = id_gen.get('generation_units', unit_name)
        unit_name_str = id_gen.get_name('generation_units', unit_name)
        units_existing = safe_get(existing_units, unit_name, 'units_existing', 1.0)

        # Determine prime mover type (STEAM or ROR)
        prime_mover = determine_prime_mover(source, unit_name)

        # Find unit_to_node entries for this unit
        unit_to_nodes = unit_to_node[unit_to_node.index.get_level_values('source') == unit_name]

        if len(unit_to_nodes) == 0:
            errors.add_warning(f"Unit {unit_name} has no unit_to_node connections")
            continue

        # Filter to power_grid nodes only
        power_grid_connections = []
        for idx in unit_to_nodes.index:
            # unit_to_node index is a 3-tuple: (name, source, sink)
            if isinstance(idx, tuple) and len(idx) == 3:
                _, source_node, sink_node = idx
            else:
                errors.add_warning(f"unit_to_node index {idx} is not a 3-tuple, skipping")
                continue
            if sink_node in power_grid_nodes:
                power_grid_connections.append((idx, sink_node))

        if len(power_grid_connections) == 0:
            errors.add_warning(f"Unit {unit_name} not connected to power_grid balance nodes")
            continue

        # Use first power_grid connection (ignore others as per instructions)
        idx, balance_node = power_grid_connections[0]
        capacity = safe_get(unit_to_nodes, idx, 'capacity', 1.0)

        rating = base_power = units_existing * capacity

        # Get balancing topology ID
        try:
            balancing_topology = id_gen.get('balancing_topologies', balance_node)
        except KeyError:
            errors.add_error(f"Balancing topology {balance_node} not found for unit {unit_name}")
            continue

        # Find fuel (from node_to_unit where node is fuel commodity)
        fuel_name = None
        if 'node_to_unit' in source and 'commodity' in source:
            node_to_unit = source['node_to_unit']
            commodities = source['commodity']

            unit_inputs = node_to_unit[node_to_unit.index.get_level_values('sink') == unit_name]
            for input_idx in unit_inputs.index:
               # node_to_unit index is a 3-tuple: (name, source, sink)
                _, input_node, _ = input_idx
                if input_node in commodities.index:
                    if safe_get(commodities, input_node, 'commodity_type') == 'fuel':
                        try:
                            # Use TEXT name instead of INTEGER id
                            fuel_name = id_gen.get_name('fuels', input_node)
                            break
                        except KeyError:
                            pass

        gen_units_data.append({
            'id': unit_id,
            'name': unit_name_str,  # Required field in new schema
            'prime_mover': prime_mover,  # 'STEAM' or 'ROR' from determine_prime_mover()
            'fuel': fuel_name,  # TEXT name instead of INTEGER id
            'balancing_topology': balancing_topology,
            'rating': rating,
            'base_power': base_power
        })
    
    return pd.DataFrame(gen_units_data)


def transform_storage_units(source: Dict[str, pd.DataFrame],
                              id_gen: IDGenerator,
                              errors: TransformationErrors) -> pd.DataFrame:
    """Create storage_units table."""
    if 'storage' not in source or 'link' not in source:
        return pd.DataFrame(columns=['id', 'name', 'prime_mover', 'max_capacity', 'balancing_topology',
                                     'efficiency_up', 'efficiency_down', 'rating', 'base_power'])
    
    storages = source['storage']
    existing_storages = storages[safe_filter(storages, 'storages_existing') & (storages['storages_existing'] > 0)].copy()
    links = source['link']

    power_grid_nodes = identify_power_grid_nodes(source, errors)

    storage_units_data = []

    for storage_name in existing_storages.index:
        storage_id = id_gen.get('storage_units', storage_name)
        storage_name_str = id_gen.get_name('storage_units', storage_name)
        max_capacity = safe_get(existing_storages, storage_name, 'storage_capacity', 1.0)

        # Find link connecting this storage to a balance node
        storage_links = []
        for link_idx in links.index:
            # Link index is always a 3-tuple: (name, node_a, node_b)
            if isinstance(link_idx, tuple) and len(link_idx) == 3:
                _, node_a, node_b = link_idx
                if node_a == storage_name and node_b in power_grid_nodes:
                    storage_links.append((link_idx, node_b))
                elif node_b == storage_name and node_a in power_grid_nodes:
                    storage_links.append((link_idx, node_a))

        if len(storage_links) == 0:
            errors.add_warning(f"Storage {storage_name} has no link to balance node")
            continue

        if len(storage_links) > 1:
            errors.add_error(f"Storage {storage_name} has multiple links - only one expected")
            continue

        link_idx, balance_node = storage_links[0]
        capacity = safe_get(links, link_idx, 'capacity', 1.0)

        # Get efficiency if available
        efficiency = safe_get(links, link_idx, 'efficiency', 1.0)

        rating = base_power = capacity

        try:
            balancing_topology = id_gen.get('balancing_topologies', balance_node)
        except KeyError:
            errors.add_error(f"Balancing topology {balance_node} not found for storage {storage_name}")
            continue

        storage_units_data.append({
            'id': storage_id,
            'name': storage_name_str,  # Required field in new schema
            'prime_mover': 'STORAGE',  # Storage units always use STORAGE prime mover
            'max_capacity': max_capacity,
            'balancing_topology': balancing_topology,
            'efficiency_up': efficiency,
            'efficiency_down': efficiency,
            'rating': rating,
            'base_power': base_power
        })
    
    errors.add_missing("storage efficiency parameters not fully mapped (using link efficiency as proxy)")
    
    return pd.DataFrame(storage_units_data)


def transform_supply_technologies(source: Dict[str, pd.DataFrame],
                                    id_gen: IDGenerator,
                                    errors: TransformationErrors) -> pd.DataFrame:
    """Create supply_technologies table."""
    if 'unit' not in source:
        return pd.DataFrame(columns=['id', 'prime_mover', 'fuel', 'area',
                                     'balancing_topology', 'scenario'])

    units = source['unit']
    investment_units = units[
        safe_filter(units, 'investment_cost') &
        safe_filter(units, 'discount_rate')
    ].copy()

    supply_tech_data = []

    for unit_name in investment_units.index:
        # Use _candidate suffix to match entity creation
        key = f"supply_tech_{unit_name}_candidate"
        tech_id = id_gen.get('supply_technologies', key)

        # Determine prime mover type (STEAM or ROR)
        prime_mover = determine_prime_mover(source, unit_name)

        # Find fuel (same logic as generation_units)
        fuel_name = None
        if 'node_to_unit' in source and 'commodity' in source:
            node_to_unit = source['node_to_unit']
            commodities = source['commodity']

            unit_inputs = node_to_unit[node_to_unit.index.get_level_values('sink') == unit_name]
            for input_idx in unit_inputs.index:
                _, input_node, _ = input_idx
                if input_node in commodities.index:
                    if safe_get(commodities, input_node, 'commodity_type') == 'fuel':
                        try:
                            # Use TEXT name instead of INTEGER id
                            fuel_name = id_gen.get_name('fuels', input_node)
                            break
                        except KeyError:
                            pass

        # Find balancing topology (from unit_to_node) - lookup name from source data
        balancing_topology_name = None
        if 'unit_to_node' in source:
            unit_to_node = source['unit_to_node']
            power_grid_nodes = identify_power_grid_nodes(source, errors)

            unit_to_nodes = unit_to_node[unit_to_node.index.get_level_values('source') == unit_name]
            for idx in unit_to_nodes.index:
                if isinstance(idx, tuple) and len(idx) == 3:
                    _, _, sink_node = idx
                else:
                    continue
                if sink_node in power_grid_nodes:
                    try:
                        # Use TEXT name instead of INTEGER id
                        balancing_topology_name = id_gen.get_name('balancing_topologies', sink_node)
                        break
                    except KeyError:
                        pass

        supply_tech_data.append({
            'id': tech_id,
            'prime_mover': prime_mover,  # 'STEAM' or 'ROR' from determine_prime_mover()
            'fuel': fuel_name,  # TEXT name instead of INTEGER id
            'area': None,  # Keep as None (Option A)
            'balancing_topology': balancing_topology_name,  # TEXT name instead of INTEGER id
            'scenario': None
        })

    return pd.DataFrame(supply_tech_data)


def transform_storage_technologies(source: Dict[str, pd.DataFrame],
                                     id_gen: IDGenerator,
                                     errors: TransformationErrors) -> pd.DataFrame:
    """Create storage_technologies table."""
    if 'storage' not in source:
        return pd.DataFrame(columns=['id', 'prime_mover', 'storage_technology_type',
                                     'area', 'balancing_topology', 'scenario'])

    storages = source['storage']
    investment_storages = storages[
        safe_filter(storages, 'investment_cost') &
        safe_filter(storages, 'discount_rate')
    ].copy()

    power_grid_nodes = identify_power_grid_nodes(source, errors)
    storage_tech_data = []

    for storage_name in investment_storages.index:
        # Use _candidate suffix to match entity creation
        key = f"storage_tech_{storage_name}_candidate"
        tech_id = id_gen.get('storage_technologies', key)

        # Lookup balancing topology from link connections
        balancing_topology_name = None
        if 'link' in source:
            links = source['link']
            for link_idx in links.index:
                if isinstance(link_idx, tuple) and len(link_idx) == 3:
                    _, node_a, node_b = link_idx
                    if node_a == storage_name and node_b in power_grid_nodes:
                        try:
                            balancing_topology_name = id_gen.get_name('balancing_topologies', node_b)
                            break
                        except KeyError:
                            pass
                    elif node_b == storage_name and node_a in power_grid_nodes:
                        try:
                            balancing_topology_name = id_gen.get_name('balancing_topologies', node_a)
                            break
                        except KeyError:
                            pass

        storage_tech_data.append({
            'id': tech_id,
            'prime_mover': 'STORAGE',  # Storage technologies always use STORAGE prime mover
            'storage_technology_type': None,
            'area': None,  # Keep as None (Option A)
            'balancing_topology': balancing_topology_name,  # TEXT name instead of INTEGER id
            'scenario': None
        })

    errors.add_missing("storage investment parameters (investment_cost, etc.) - unclear target placement")

    return pd.DataFrame(storage_tech_data)


def transform_transport_technologies(source: Dict[str, pd.DataFrame],
                                       id_gen: IDGenerator,
                                       errors: TransformationErrors) -> pd.DataFrame:
    """Create transport_technologies table for transmission investment candidates."""
    if 'link' not in source:
        return pd.DataFrame(columns=['id', 'arc_id', 'scenario'])

    links = source['link']
    investment_links = links[
        safe_filter(links, 'investment_cost') &
        safe_filter(links, 'discount_rate')
    ]

    transport_tech_data = []

    for idx in investment_links.index:
        link_key = create_composite_name(idx)
        # Use _candidate suffix to match entity creation
        candidate_key = f"{link_key}_candidate"

        try:
            tech_id = id_gen.get('transport_technologies', candidate_key)
        except KeyError:
            errors.add_warning(f"Transport technology entity not found for {candidate_key}")
            continue

        # Find arc_id for this link
        arc_id = None
        if isinstance(idx, tuple) and len(idx) == 3:
            _, node_a, node_b = idx
            arc_key = f"{node_a}_{node_b}"
            try:
                arc_id = id_gen.get('arcs', arc_key)
            except KeyError:
                # Try reverse direction
                arc_key = f"{node_b}_{node_a}"
                try:
                    arc_id = id_gen.get('arcs', arc_key)
                except KeyError:
                    errors.add_warning(f"Arc not found for transport technology {candidate_key}")

        transport_tech_data.append({
            'id': tech_id,
            'arc_id': arc_id,
            'scenario': None
        })

    return pd.DataFrame(transport_tech_data)


def transform_attributes(source: Dict[str, pd.DataFrame],
                         id_gen: IDGenerator,
                         errors: TransformationErrors) -> pd.DataFrame:
    """
    Create attributes table with investment parameters as TechnologyFinancialData JSON.

    Maps CESM investment parameters to GridDB Attributes table:
    - Units with investment data → supply_technologies attributes
    - Storages with investment data → storage_technologies attributes
    - Links with investment data → transport_technologies attributes
    """
    attributes_data = []
    attr_id = 0

    # Process supply technologies (investment units)
    if 'unit' in source:
        units = source['unit']
        investment_units = units[
            safe_filter(units, 'investment_cost') &
            safe_filter(units, 'discount_rate')
        ]

        for unit_name in investment_units.index:
            key = f"supply_tech_{unit_name}_candidate"

            try:
                entity_id = id_gen.get('supply_technologies', key)
            except KeyError:
                continue

            # Build TechnologyFinancialData JSON
            tech_financial_data = {}

            if 'investment_cost' in units.columns:
                investment_cost = safe_get(investment_units, unit_name, 'investment_cost')
                if investment_cost is not None:
                    tech_financial_data['investment_cost'] = float(investment_cost)

            if 'discount_rate' in units.columns:
                discount_rate = safe_get(investment_units, unit_name, 'discount_rate')
                if discount_rate is not None:
                    tech_financial_data['discount_rate'] = float(discount_rate)

            if 'capital_recovery_period' in units.columns:
                capital_recovery_period = safe_get(investment_units, unit_name, 'capital_recovery_period')
                if capital_recovery_period is not None:
                    tech_financial_data['capital_recovery_period'] = float(capital_recovery_period)

            if 'fixed_cost' in units.columns:
                fixed_cost = safe_get(investment_units, unit_name, 'fixed_cost')
                if fixed_cost is not None:
                    tech_financial_data['fixed_cost'] = float(fixed_cost)

            if tech_financial_data:
                attr_id += 1
                attributes_data.append({
                    'id': attr_id,
                    'entity_id': entity_id,
                    'TYPE': 'TechnologyFinancialData',
                    'name': 'financial_parameters',
                    'value': json.dumps(tech_financial_data)
                })

    # Process storage technologies
    if 'storage' in source:
        storages = source['storage']
        investment_storages = storages[
            safe_filter(storages, 'investment_cost') &
            safe_filter(storages, 'discount_rate')
        ]

        for storage_name in investment_storages.index:
            key = f"storage_tech_{storage_name}_candidate"

            try:
                entity_id = id_gen.get('storage_technologies', key)
            except KeyError:
                continue

            # Build TechnologyFinancialData JSON
            tech_financial_data = {}

            if 'investment_cost' in storages.columns:
                investment_cost = safe_get(investment_storages, storage_name, 'investment_cost')
                if investment_cost is not None:
                    tech_financial_data['investment_cost_energy'] = float(investment_cost)

            if 'discount_rate' in storages.columns:
                discount_rate = safe_get(investment_storages, storage_name, 'discount_rate')
                if discount_rate is not None:
                    tech_financial_data['discount_rate'] = float(discount_rate)

            if 'capital_recovery_period' in storages.columns:
                capital_recovery_period = safe_get(investment_storages, storage_name, 'capital_recovery_period')
                if capital_recovery_period is not None:
                    tech_financial_data['capital_recovery_period'] = float(capital_recovery_period)

            if 'fixed_cost' in storages.columns:
                fixed_cost = safe_get(investment_storages, storage_name, 'fixed_cost')
                if fixed_cost is not None:
                    tech_financial_data['fixed_cost_energy'] = float(fixed_cost)

            if tech_financial_data:
                attr_id += 1
                attributes_data.append({
                    'id': attr_id,
                    'entity_id': entity_id,
                    'TYPE': 'TechnologyFinancialData',
                    'name': 'financial_parameters',
                    'value': json.dumps(tech_financial_data)
                })

    # Process transport technologies (investment links)
    if 'link' in source:
        links = source['link']
        investment_links = links[
            safe_filter(links, 'investment_cost') &
            safe_filter(links, 'discount_rate')
        ]

        for idx in investment_links.index:
            link_key = create_composite_name(idx)
            candidate_key = f"{link_key}_candidate"

            try:
                entity_id = id_gen.get('transport_technologies', candidate_key)
            except KeyError:
                continue

            # Build TechnologyFinancialData JSON
            tech_financial_data = {}

            if 'investment_cost' in links.columns:
                investment_cost = safe_get(investment_links, idx, 'investment_cost')
                if investment_cost is not None:
                    tech_financial_data['investment_cost'] = float(investment_cost)

            if 'discount_rate' in links.columns:
                discount_rate = safe_get(investment_links, idx, 'discount_rate')
                if discount_rate is not None:
                    tech_financial_data['discount_rate'] = float(discount_rate)

            if 'capital_recovery_period' in links.columns:
                capital_recovery_period = safe_get(investment_links, idx, 'capital_recovery_period')
                if capital_recovery_period is not None:
                    tech_financial_data['capital_recovery_period'] = float(capital_recovery_period)

            if tech_financial_data:
                attr_id += 1
                attributes_data.append({
                    'id': attr_id,
                    'entity_id': entity_id,
                    'TYPE': 'TechnologyFinancialData',
                    'name': 'financial_parameters',
                    'value': json.dumps(tech_financial_data)
                })

    return pd.DataFrame(attributes_data)


def transform_operational_data(source: Dict[str, pd.DataFrame],
                                 id_gen: IDGenerator,
                                 errors: TransformationErrors) -> pd.DataFrame:
    """Create operational_data table."""
    if 'unit' not in source:
        return pd.DataFrame(columns=['id', 'entity_id', 'active_power_limit_min', 'must_run',
                                     'uptime', 'downtime', 'ramp_up', 'ramp_down', 
                                     'operational_cost'])
    
    units = source['unit']
    operational_data = []
    op_id = 0
    
    for unit_name in units.index:
        # Create operational data for all units
        try:
            entity_id = id_gen.get('generation_units', unit_name)
        except KeyError:
            # Try supply technologies
            try:
                entity_id = id_gen.get('supply_technologies', f"supply_tech_{unit_name}")
            except KeyError:
                continue
        
        op_id += 1
        
        # Get efficiency if available (simplified for now)
        efficiency = safe_get(units, unit_name, 'efficiency')
        
        operational_data.append({
            'id': op_id,
            'entity_id': entity_id,
            'active_power_limit_min': 0.0,  # Default, need better mapping
            'must_run': False,
            'uptime': 0.0,
            'downtime': 0.0,
            'ramp_up': 1.0,  # Default
            'ramp_down': 1.0,  # Default
            'operational_cost': None
            # Note: operational_cost_type is a generated column in schema, exclude from insert
        })
    
    errors.add_missing("operational_data.active_power_limit_min - no clear source mapping")
    errors.add_missing("operational_data fields from unit.efficiency - complex cases not handled")
    
    return pd.DataFrame(operational_data)


def transform_transmission_interchanges(source: Dict[str, pd.DataFrame],
                                          id_gen: IDGenerator,
                                          errors: TransformationErrors) -> pd.DataFrame:
    """Create transmission_interchanges table."""
    interchanges_data = []

    if 'group' in source and 'group_entity' in source:
        groups = source['group']
        group_entities = source['group_entity']

        for group_name in group_entities.index.get_level_values('group').unique():
            key = f"interchange_{group_name}"
            try:
                interchange_id = id_gen.get('transmission_interchanges', key)
                interchange_name = id_gen.get_name('transmission_interchanges', key)
            except KeyError:
                continue

            # Verify this group contains link members
            try:
                entities_data = group_entities.xs(group_name, level='group')
                entities = entities_data.index.get_level_values('entity').tolist()

                # Check if any entities are links
                has_links = False
                if 'link' in source:
                    links = source['link']
                    for entity in entities:
                        # Check if entity appears in link index
                        for link_idx in links.index:
                            if isinstance(link_idx, tuple) and len(link_idx) >= 1:
                                link_name = link_idx[0]
                                if entity == link_name:
                                    has_links = True
                                    break
                        if has_links:
                            break

                if not has_links:
                    continue

            except KeyError:
                continue

            # Try to extract max_flow from group data
            max_flow_from = None
            max_flow_to = None
            if group_name in groups.index:
                max_flow_from = safe_get(groups, group_name, 'max_flow_from')
                max_flow_to = safe_get(groups, group_name, 'max_flow_to')

            interchanges_data.append({
                'id': interchange_id,
                'name': interchange_name,
                'arc_id': None,  # Would need to determine from link members
                'max_flow_from': max_flow_from if max_flow_from is not None else 0.0,
                'max_flow_to': max_flow_to if max_flow_to is not None else 0.0
            })

    return pd.DataFrame(interchanges_data)


def get_parameter_unit(param_name: str, schema_path: str = 'model/cesm.yaml') -> str:
    """
    Extract unit annotation from CESM schema for given parameter.

    Args:
        param_name: Name of the parameter (e.g., 'flow_profile', 'profile_limit_upper')
        schema_path: Path to the CESM schema YAML file

    Returns:
        Unit string from schema annotations (e.g., 'PERCENT', 'UNITLESS'),
        or 'UNKNOWN' if not found
    """
    try:
        schema = SchemaView(schema_path)
        slot = schema.get_slot(param_name)

        if slot and slot.annotations:
            qudt_unit = slot.annotations.get('qudt_unit')
            if qudt_unit:
                # Convert 'unit:PERCENT' to 'PERCENT'
                return qudt_unit.value.replace('unit:', '')
    except Exception:
        # If schema loading fails or slot not found, return UNKNOWN
        pass

    return 'UNKNOWN'


def get_entity_type(entity_id: int, id_gen: IDGenerator) -> str:
    """
    Get PowerSystems.jl owner_type for a given entity_id.

    Uses the entity_type stored in IDGenerator, which accounts for prime_mover
    classification (STEAM→ThermalStandard, ROR→RenewableDispatch, STORAGE→HydroPumpedStorage).

    Args:
        entity_id: The entity ID to look up
        id_gen: IDGenerator instance containing entity mappings

    Returns:
        PowerSystems.jl component type string (e.g., 'ThermalStandard', 'RenewableDispatch')
    """
    # Try to get entity_type directly from id_gen
    entity_type = id_gen.get_type(entity_id)
    if entity_type:
        return entity_type

    # Fallback: lookup by source_table (for entities without explicit type)
    TABLE_TO_OWNER_TYPE = {
        'balancing_topologies': 'LoadZone',
        'planning_regions': 'Area',
        'transmission_lines': 'Line',
        'transmission_interchanges': 'Arc',
        'loads': 'PowerLoad',
        'fuels': 'Component',
        'arcs': 'Arc'
    }

    # Reverse lookup in id_gen.entity_map to find source_table for this entity_id
    for (source_table, key), eid in id_gen.entity_map.items():
        if eid == entity_id:
            return TABLE_TO_OWNER_TYPE.get(source_table, 'Component')

    return 'Component'  # Fallback for unknown entities


def encode_attribute_value(value, data_type: str, units: str) -> str:
    """
    Helper to create JSON-encoded attribute value.

    Args:
        value: The numeric or string value
        data_type: Type identifier ("float", "int", "string")
        units: Unit string (e.g., "per unit", "$/kWh", "years")

    Returns:
        JSON string with structure: {"type": "float", "value": X, "units": "..."}
    """
    return json.dumps({
        "type": data_type,
        "value": value,
        "units": units
    })


def transform_supplemental_attributes(source: Dict[str, pd.DataFrame],
                                        id_gen: IDGenerator,
                                        errors: TransformationErrors) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create supplemental_attributes and supplemental_attributes_association tables.

    Maps CESM fields that don't fit in core GridDB schema:
    - Transmission line financial/physical parameters
    - Storage investment parameters

    Returns:
        Tuple of (supplemental_attributes DataFrame, associations DataFrame)
    """
    attributes_data = []
    associations_data = []
    attr_id = 0

    # Process transmission line attributes
    if 'link' in source:
        links = source['link']
        existing_links = links[safe_filter(links, 'links_existing') & (links['links_existing'] > 0)]

        for idx in existing_links.index:
            link_key = create_composite_name(idx)

            try:
                entity_id = id_gen.get('transmission_lines', link_key)
            except KeyError:
                continue

            # Map efficiency
            if 'efficiency' in links.columns:
                efficiency = safe_get(existing_links, idx, 'efficiency')
                if efficiency is not None:
                    attr_id += 1
                    attributes_data.append({
                        'id': attr_id,
                        'TYPE': 'transmission_efficiency',
                        'value': encode_attribute_value(efficiency, "float", "per unit")
                    })
                    associations_data.append({
                        'attribute_id': attr_id,
                        'entity_id': entity_id
                    })

            # Map discount_rate
            if 'discount_rate' in links.columns:
                discount_rate = safe_get(existing_links, idx, 'discount_rate')
                if discount_rate is not None:
                    attr_id += 1
                    attributes_data.append({
                        'id': attr_id,
                        'TYPE': 'discount_rate',
                        'value': encode_attribute_value(discount_rate, "float", "per unit")
                    })
                    associations_data.append({
                        'attribute_id': attr_id,
                        'entity_id': entity_id
                    })

            # Map capital_recovery_period
            if 'capital_recovery_period' in links.columns:
                capital_recovery_period = safe_get(existing_links, idx, 'capital_recovery_period')
                if capital_recovery_period is not None:
                    attr_id += 1
                    attributes_data.append({
                        'id': attr_id,
                        'TYPE': 'capital_recovery_period',
                        'value': encode_attribute_value(capital_recovery_period, "float", "years")
                    })
                    associations_data.append({
                        'attribute_id': attr_id,
                        'entity_id': entity_id
                    })

    # Note: Storage and unit investment parameters are now handled in transform_attributes()
    # as TechnologyFinancialData JSON for investment candidates

    return pd.DataFrame(attributes_data), pd.DataFrame(associations_data)


def transform_time_series(source: Dict[str, pd.DataFrame],
                           id_gen: IDGenerator,
                           errors: TransformationErrors) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create time_series_associations and static_time_series tables (new schema)."""
    associations_data = []
    static_data = []
    ts_assoc_id = 0
    static_row_id = 0

    # Process profile data from unit_to_node.ts.profile_limit_*
    for key in source.keys():
        if key.startswith('unit_to_node.ts.profile_limit_'):
            ts_df = source[key]
            param_name = key.split('unit_to_node.ts.')[1]

            # Validate datetime index (required for initial_timestamp and resolution)
            if not isinstance(ts_df.index, pd.DatetimeIndex):
                errors.add_error(f"Time series '{key}' missing datetime index (required for initial_timestamp/resolution)")
                continue

            if len(ts_df.index) == 0:
                errors.add_warning(f"Time series '{key}' is empty, skipping")
                continue

            # Get units from schema
            unit_str = get_parameter_unit(param_name)

            # Handle potential multi-index columns
            columns = ts_df.columns
            if isinstance(columns, pd.MultiIndex):
                for col_tuple in columns:
                    ts_assoc_id += 1

                    # Try to find corresponding generation unit
                    try:
                        unit_name = col_tuple[0] if isinstance(col_tuple, tuple) else col_tuple
                        owner_id = id_gen.get('generation_units', unit_name)
                    except KeyError:
                        errors.add_warning(f"Cannot find generation unit for profile {unit_name}")
                        continue

                    # Get dynamic owner type
                    owner_type = get_entity_type(owner_id, id_gen)

                    # Generate proper UUIDs
                    time_series_uuid = str(uuid.uuid4())
                    metadata_uuid = str(uuid.uuid4())

                    # Extract time information (now guaranteed to have datetime index)
                    initial_timestamp = ts_df.index[0].isoformat()
                    resolution = None
                    if len(ts_df.index) > 1:
                        time_diff = ts_df.index[1] - ts_df.index[0]
                        resolution = f"{int(time_diff.total_seconds() / 3600)}h"

                    associations_data.append({
                        'id': ts_assoc_id,
                        'time_series_uuid': time_series_uuid,
                        'time_series_type': 'SingleTimeSeries',
                        'initial_timestamp': initial_timestamp,
                        'resolution': resolution,
                        'horizon': None,
                        'interval': None,
                        'window_count': None,
                        'length': len(ts_df),
                        'name': param_name,
                        'owner_id': owner_id,
                        'owner_type': owner_type,  # Dynamic from entity type
                        'owner_category': 'Component',
                        'features': '',
                        'scaling_factor_multiplier': None,
                        'metadata_uuid': metadata_uuid,
                        'units': unit_str  # From schema annotations
                    })

                    # Add static time series data points with idx-based storage
                    for idx, (timestamp, value) in enumerate(ts_df[col_tuple].items(), start=1):
                        if pd.notna(value):
                            static_row_id += 1
                            static_data.append({
                                'id': static_row_id,
                                'uuid': time_series_uuid,
                                'idx': idx,
                                'value': float(value) if value != 'object' else 0.0
                            })
            else:
                # Simple column index
                for col in columns:
                    ts_assoc_id += 1

                    try:
                        owner_id = id_gen.get('generation_units', col)
                    except KeyError:
                        errors.add_warning(f"Cannot find generation unit for profile {col}")
                        continue

                    # Get dynamic owner type
                    owner_type = get_entity_type(owner_id, id_gen)

                    # Generate proper UUIDs
                    time_series_uuid = str(uuid.uuid4())
                    metadata_uuid = str(uuid.uuid4())

                    # Extract time information (now guaranteed to have datetime index)
                    initial_timestamp = ts_df.index[0].isoformat()
                    resolution = None
                    if len(ts_df.index) > 1:
                        time_diff = ts_df.index[1] - ts_df.index[0]
                        resolution = f"{int(time_diff.total_seconds() / 3600)}h"

                    associations_data.append({
                        'id': ts_assoc_id,
                        'time_series_uuid': time_series_uuid,
                        'time_series_type': 'SingleTimeSeries',
                        'initial_timestamp': initial_timestamp,
                        'resolution': resolution,
                        'horizon': None,
                        'interval': None,
                        'window_count': None,
                        'length': len(ts_df),
                        'name': param_name,
                        'owner_id': owner_id,
                        'owner_type': owner_type,  # Dynamic from entity type
                        'owner_category': 'Component',
                        'features': '',
                        'scaling_factor_multiplier': None,
                        'metadata_uuid': metadata_uuid,
                        'units': unit_str  # From schema annotations
                    })

                    # Add static time series data points with idx-based storage
                    for idx, (timestamp, value) in enumerate(ts_df[col].items(), start=1):
                        if pd.notna(value):
                            static_row_id += 1
                            static_data.append({
                                'id': static_row_id,
                                'uuid': time_series_uuid,
                                'idx': idx,
                                'value': float(value) if value != 'object' else 0.0
                            })

    # Process flow profile data from balance.ts.flow_profile
    if 'balance.ts.flow_profile' in source:
        flow_df = source['balance.ts.flow_profile']
        param_name = 'flow_profile'

        # Validate datetime index
        if not isinstance(flow_df.index, pd.DatetimeIndex):
            errors.add_error("Time series 'balance.ts.flow_profile' missing datetime index (required)")
        elif len(flow_df.index) == 0:
            errors.add_warning("Time series 'balance.ts.flow_profile' is empty, skipping")
        else:
            # Get units from schema
            unit_str = get_parameter_unit(param_name)

            for balance_name in flow_df.columns:
                ts_assoc_id += 1

                try:
                    owner_id = id_gen.get('balancing_topologies', balance_name)
                except KeyError:
                    errors.add_warning(f"Cannot find balancing topology for flow profile {balance_name}")
                    continue

                # Get dynamic owner type
                owner_type = get_entity_type(owner_id, id_gen)

                # Generate proper UUIDs
                time_series_uuid = str(uuid.uuid4())
                metadata_uuid = str(uuid.uuid4())

                # Extract time information (now guaranteed to have datetime index)
                initial_timestamp = flow_df.index[0].isoformat()
                resolution = None
                if len(flow_df.index) > 1:
                    time_diff = flow_df.index[1] - flow_df.index[0]
                    resolution = f"{int(time_diff.total_seconds() / 3600)}h"

                associations_data.append({
                    'id': ts_assoc_id,
                    'time_series_uuid': time_series_uuid,
                    'time_series_type': 'SingleTimeSeries',
                    'initial_timestamp': initial_timestamp,
                    'resolution': resolution,
                    'horizon': None,
                    'interval': None,
                    'window_count': None,
                    'length': len(flow_df),
                    'name': param_name,
                    'owner_id': owner_id,
                    'owner_type': owner_type,  # Dynamic from entity type
                    'owner_category': 'Component',
                    'features': '',
                    'scaling_factor_multiplier': None,
                    'metadata_uuid': metadata_uuid,
                    'units': unit_str  # From schema annotations
                })

                # Add static time series data points with idx-based storage
                for idx, (timestamp, value) in enumerate(flow_df[balance_name].items(), start=1):
                    if pd.notna(value):
                        static_row_id += 1
                        static_data.append({
                            'id': static_row_id,
                            'uuid': time_series_uuid,
                            'idx': idx,
                            'value': float(value) if value != 'object' else 0.0
                        })

    # Process flow profile data from storage.ts.flow_profile
    if 'storage.ts.flow_profile' in source:
        storage_flow_df = source['storage.ts.flow_profile']
        param_name = 'flow_profile'

        # Validate datetime index
        if not isinstance(storage_flow_df.index, pd.DatetimeIndex):
            errors.add_error("Time series 'storage.ts.flow_profile' missing datetime index (required)")
        elif len(storage_flow_df.index) == 0:
            errors.add_warning("Time series 'storage.ts.flow_profile' is empty, skipping")
        else:
            # Get units from schema
            unit_str = get_parameter_unit(param_name)

            for storage_name in storage_flow_df.columns:
                ts_assoc_id += 1

                try:
                    owner_id = id_gen.get('storage_units', storage_name)
                except KeyError:
                    errors.add_warning(f"Cannot find storage unit for flow profile {storage_name}")
                    continue

                # Get dynamic owner type
                owner_type = get_entity_type(owner_id, id_gen)

                # Generate proper UUIDs
                time_series_uuid = str(uuid.uuid4())
                metadata_uuid = str(uuid.uuid4())

                # Extract time information (now guaranteed to have datetime index)
                initial_timestamp = storage_flow_df.index[0].isoformat()
                resolution = None
                if len(storage_flow_df.index) > 1:
                    time_diff = storage_flow_df.index[1] - storage_flow_df.index[0]
                    resolution = f"{int(time_diff.total_seconds() / 3600)}h"

                associations_data.append({
                    'id': ts_assoc_id,
                    'time_series_uuid': time_series_uuid,
                    'time_series_type': 'SingleTimeSeries',
                    'initial_timestamp': initial_timestamp,
                    'resolution': resolution,
                    'horizon': None,
                    'interval': None,
                    'window_count': None,
                    'length': len(storage_flow_df),
                    'name': param_name,
                    'owner_id': owner_id,
                    'owner_type': owner_type,  # Dynamic from entity type
                    'owner_category': 'Component',
                    'features': '',
                    'scaling_factor_multiplier': None,
                    'metadata_uuid': metadata_uuid,
                    'units': unit_str  # From schema annotations
                })

                # Add static time series data points with idx-based storage
                for idx, (timestamp, value) in enumerate(storage_flow_df[storage_name].items(), start=1):
                    if pd.notna(value):
                        static_row_id += 1
                        static_data.append({
                            'id': static_row_id,
                            'uuid': time_series_uuid,
                            'idx': idx,
                            'value': float(value) if value != 'object' else 0.0
                        })

    errors.add_missing("time_series features parameters - ignored")

    # Return with correct table name
    return pd.DataFrame(associations_data), pd.DataFrame(static_data)


def to_griddb(source: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Main transformation function.
    
    Args:
        source: Dictionary of source dataframes
        
    Returns:
        Dictionary of target dataframes matching SQL schema
    """
    errors = TransformationErrors()
    id_gen = IDGenerator()
    target = {}
    
    try:
        # Phase 1: Create entities and populate ID mappings
        print("Phase 1: Creating entities and ID mappings...")
        target['entities'] = transform_entities_and_ids(source, id_gen, errors)
        
        # Phase 2: Create all other tables
        print("Phase 2: Creating target tables...")
        target['prime_mover_types'] = transform_prime_mover_types(id_gen, errors)
        target['fuels'] = transform_fuels(source, id_gen, errors)
        target['planning_regions'] = transform_planning_regions(source, id_gen, errors)
        target['balancing_topologies'] = transform_balancing_topologies(source, id_gen, errors)
        target['arcs'] = transform_arcs(source, id_gen, errors)
        target['transmission_lines'] = transform_transmission_lines(source, id_gen, errors)
        target['transmission_interchanges'] = transform_transmission_interchanges(source, id_gen, errors)
        target['generation_units'] = transform_generation_units(source, id_gen, errors)
        target['storage_units'] = transform_storage_units(source, id_gen, errors)
        target['supply_technologies'] = transform_supply_technologies(source, id_gen, errors)
        target['storage_technologies'] = transform_storage_technologies(source, id_gen, errors)
        target['transport_technologies'] = transform_transport_technologies(source, id_gen, errors)
        target['operational_data'] = transform_operational_data(source, id_gen, errors)
        
        # Phase 3: Time series
        print("Phase 3: Creating time series data...")
        target['time_series_associations'], target['static_time_series'] = \
            transform_time_series(source, id_gen, errors)

        # Phase 4: Attributes and supplemental attributes
        print("Phase 4: Creating attributes and supplemental attributes...")
        target['attributes'] = transform_attributes(source, id_gen, errors)
        target['supplemental_attributes'], target['supplemental_attributes_association'] = \
            transform_supplemental_attributes(source, id_gen, errors)

        # Phase 5: Empty tables for completeness
        target['hydro_reservoir'] = pd.DataFrame(columns=['id', 'name'])
        target['hydro_reservoir_connections'] = pd.DataFrame(columns=['turbine_id', 'reservoir_id'])
        target['loads'] = pd.DataFrame(columns=['id', 'name', 'balancing_topology', 'base_power'])
        # Note: deterministic_forecast_data table removed in new schema

        errors.add_missing("hydro_reservoir and hydro_reservoir_connections - not processed")
        errors.add_missing("loads - cannot generate from flexible source units (no clear base_power)")
        
    except Exception as e:
        errors.add_error(f"Critical transformation error: {str(e)}")
        import traceback
        errors.add_error(traceback.format_exc())
    
    # Report all errors and warnings
    errors.report()
    
    return target