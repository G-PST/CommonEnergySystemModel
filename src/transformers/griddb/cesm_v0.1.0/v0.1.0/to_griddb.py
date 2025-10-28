"""
Transformer for converting source energy system data to target grid database format.

This module converts dict of source dataframes to target dataframes matching SQL schema.
Uses vectorized operations and multi-index alignment for efficiency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


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
    
    def next_id(self) -> int:
        self.current_id += 1
        return self.current_id
    
    def get_or_create(self, source_table: str, key: str) -> int:
        """Get existing ID or create new one for entity."""
        lookup_key = (source_table, key)
        if lookup_key not in self.entity_map:
            self.entity_map[lookup_key] = self.next_id()
        return self.entity_map[lookup_key]
    
    def get(self, source_table: str, key: str) -> int:
        """Get existing ID (raises KeyError if not found)."""
        return self.entity_map[(source_table, key)]


def create_composite_name(index_tuple) -> str:
    """Create composite name from multi-index tuple."""
    if isinstance(index_tuple, tuple):
        return ".".join(str(x) for x in index_tuple)
    return str(index_tuple)


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
        if group in group_entities.index.get_level_values('group'):
            entities = group_entities.loc[group].index.tolist()
            power_grid_nodes.update(entities)
    
    return power_grid_nodes


def transform_entities_and_ids(source: Dict[str, pd.DataFrame], 
                                id_gen: IDGenerator,
                                errors: TransformationErrors) -> pd.DataFrame:
    """Create entities table and populate ID mappings."""
    entities_data = []
    
    # Helper to add entity
    def add_entity(source_table: str, entity_type: str, key: str, name: str = None, description: str = None):
        entity_id = id_gen.get_or_create(source_table, key)
        entities_data.append({
            'id': entity_id,
            'entity_type': entity_type,
            'source_table': source_table,
            'name': name or create_composite_name(key),
            'description': description
        })
        return entity_id
    
    # 1. Placeholder prime mover type
    add_entity('prime_mover_types', 'PrimeMovers', 'temp_generator', 'temp_generator', 
               'Temporary generator type placeholder')
    
    # 2. Fuels from commodities
    if 'commodity' in source:
        commodities = source['commodity']
        fuels = commodities[commodities['commodity_type'] == 'fuel']
        for commodity_name in fuels.index:
            add_entity('fuels', 'ThermalFuels', commodity_name, commodity_name)
    
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
            add_entity('generation_units', 'ThermalStandard', unit_name, unit_name)
    
    # 6. Supply technologies from units with investment data
    if 'unit' in source:
        units = source['unit']
        investment_units = units[
            safe_filter(units, 'investment_cost') & 
            safe_filter(units, 'discount_rate')
        ]
        for unit_name in investment_units.index:
            key = f"supply_tech_{unit_name}"
            add_entity('supply_technologies', 'ThermalStandard', key, f"Supply Tech {unit_name}")
    
    # 7. Arcs from links
    if 'link' in source:
        links = source['link']
        arc_pairs = set()
        for idx in links.index:
            node_a, node_b = idx if isinstance(idx, tuple) else (idx, None)
            if node_b and (node_a, node_b) not in arc_pairs and (node_b, node_a) not in arc_pairs:
                arc_pairs.add((node_a, node_b))
                arc_key = f"{node_a}_{node_b}"
                add_entity('arcs', 'Arc', arc_key, f"Arc {node_a} to {node_b}")
    
    # 8. Transmission lines from links with links_existing
    if 'link' in source:
        links = source['link']
        existing_links = links[safe_filter(links, 'links_existing') & (links['links_existing'] > 0)]
        for idx in existing_links.index:
            link_key = create_composite_name(idx)
            add_entity('transmission_lines', 'Line', link_key, f"Line {link_key}")
    
    # 9. Storage units from storage with storages_existing
    if 'storage' in source:
        storages = source['storage']
        existing_storages = storages[safe_filter(storages, 'storages_existing') & (storages['storages_existing'] > 0)]
        for storage_name in existing_storages.index:
            add_entity('storage_units', 'HydroPumpedStorage', storage_name, storage_name)
    
    # 10. Storage technologies from storage with investment data
    if 'storage' in source:
        storages = source['storage']
        investment_storages = storages[
            safe_filter(storages, 'investment_cost') & 
            safe_filter(storages, 'discount_rate')
        ]
        for storage_name in investment_storages.index:
            key = f"storage_tech_{storage_name}"
            add_entity('storage_technologies', 'HydroPumpedStorage', key, f"Storage Tech {storage_name}")
    
    # 11. Transmission interchanges from groups with link members
    if 'group' in source and 'group_entity' in source:
        group_entities = source['group_entity']
        for group_name in group_entities.index.get_level_values('group').unique():
            entities = group_entities.loc[group_name]
            # Check if members are links (simplification: assume they are if not in balance)
            key = f"interchange_{group_name}"
            add_entity('transmission_interchanges', 'Arc', key, f"Interchange {group_name}")
    
    if not entities_data:
        errors.add_error("No entities created from source data")
        return pd.DataFrame()
    
    return pd.DataFrame(entities_data)


def transform_prime_mover_types(id_gen: IDGenerator, errors: TransformationErrors) -> pd.DataFrame:
    """Create prime_mover_types table."""
    pm_id = id_gen.get('prime_mover_types', 'temp_generator')
    return pd.DataFrame([{
        'id': pm_id,
        'description': 'Temporary generator type'
    }])


def transform_fuels(source: Dict[str, pd.DataFrame], 
                     id_gen: IDGenerator,
                     errors: TransformationErrors) -> pd.DataFrame:
    """Create fuels table."""
    if 'commodity' not in source:
        return pd.DataFrame(columns=['id', 'description'])
    
    commodities = source['commodity']
    fuels = commodities[commodities['commodity_type'] == 'fuel']
    
    fuels_data = []
    for commodity_name in fuels.index:
        fuel_id = id_gen.get('fuels', commodity_name)
        fuels_data.append({
            'id': fuel_id,
            'description': f"Fuel: {commodity_name}"
        })
    
    errors.add_missing("commodity.price_per_unit → target has no fuel price field")
    
    return pd.DataFrame(fuels_data)


def transform_planning_regions(source: Dict[str, pd.DataFrame],
                                 id_gen: IDGenerator,
                                 errors: TransformationErrors) -> pd.DataFrame:
    """Create planning_regions table."""
    if 'group' not in source:
        return pd.DataFrame(columns=['id', 'description'])
    
    groups = source['group']
    planning_groups = groups[groups['group_type'] == 'node']
    
    regions_data = []
    for group_name in planning_groups.index:
        region_id = id_gen.get('planning_regions', group_name)
        regions_data.append({
            'id': region_id,
            'description': f"Planning region: {group_name}"
        })
    
    return pd.DataFrame(regions_data)


def transform_balancing_topologies(source: Dict[str, pd.DataFrame],
                                     id_gen: IDGenerator,
                                     errors: TransformationErrors) -> pd.DataFrame:
    """Create balancing_topologies table."""
    if 'balance' not in source:
        return pd.DataFrame(columns=['id', 'area', 'description'])
    
    power_grid_nodes = identify_power_grid_nodes(source, errors)
    
    topologies_data = []
    for balance_name in source['balance'].index:
        if balance_name not in power_grid_nodes:
            continue
        
        topo_id = id_gen.get('balancing_topologies', balance_name)
        
        # Find planning region (area) via group_entity
        area_id = None
        if 'group_entity' in source and 'group' in source:
            group_entities = source['group_entity']
            groups = source['group']
            
            for group_name in groups[groups['group_type'] == 'node'].index:
                if group_name in group_entities.index.get_level_values('group'):
                    entities = group_entities.loc[group_name].index.tolist()
                    if balance_name in entities:
                        area_id = id_gen.get('planning_regions', group_name)
                        break
        
        topologies_data.append({
            'id': topo_id,
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
        if isinstance(idx, tuple) and len(idx) == 2:
            node_a, node_b = idx
        else:
            errors.add_warning(f"Link index {idx} is not a 2-tuple, skipping")
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
        return pd.DataFrame(columns=['id', 'arc_id', 'continuous_rating', 'ste_rating', 
                                     'lte_rating', 'line_length'])
    
    links = source['link']
    existing_links = links[safe_filter(links, 'links_existing') & (links['links_existing'] > 0)].copy()
    
    lines_data = []
    for idx in existing_links.index:
        link_key = create_composite_name(idx)
        line_id = id_gen.get('transmission_lines', link_key)
        
        # Get arc_id
        if isinstance(idx, tuple) and len(idx) == 2:
            node_a, node_b = idx
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
            errors.add_warning(f"Link index {idx} format unexpected")
            continue
        
        # Calculate continuous_rating
        links_existing = safe_get(existing_links, idx, 'links_existing', 1.0)
        capacity = safe_get(existing_links, idx, 'capacity', 1.0)
        continuous_rating = links_existing * capacity
        
        lines_data.append({
            'id': line_id,
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
        return pd.DataFrame(columns=['id', 'prime_mover', 'fuel', 'balancing_topology', 
                                     'rating', 'base_power'])
    
    units = source['unit']
    existing_units = units[safe_filter(units, 'units_existing') & (units['units_existing'] > 0)].copy()
    unit_to_node = source['unit_to_node']
    
    power_grid_nodes = identify_power_grid_nodes(source, errors)
    pm_id = id_gen.get('prime_mover_types', 'temp_generator')
    
    gen_units_data = []
    
    for unit_name in existing_units.index:
        unit_id = id_gen.get('generation_units', unit_name)
        units_existing = safe_get(existing_units, unit_name, 'units_existing', 1.0)
        
        # Find unit_to_node entries for this unit
        unit_to_nodes = unit_to_node[unit_to_node.index.get_level_values('source') == unit_name]
        
        if len(unit_to_nodes) == 0:
            errors.add_warning(f"Unit {unit_name} has no unit_to_node connections")
            continue
        
        # Filter to power_grid nodes only
        power_grid_connections = []
        for idx in unit_to_nodes.index:
            source_node, sink_node = idx
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
        fuel_id = None
        if 'node_to_unit' in source and 'commodity' in source:
            node_to_unit = source['node_to_unit']
            commodities = source['commodity']
            
            unit_inputs = node_to_unit[node_to_unit.index.get_level_values('sink') == unit_name]
            for input_idx in unit_inputs.index:
                input_node, _ = input_idx
                if input_node in commodities.index:
                    if safe_get(commodities, input_node, 'commodity_type') == 'fuel':
                        try:
                            fuel_id = id_gen.get('fuels', input_node)
                            break
                        except KeyError:
                            pass
        
        gen_units_data.append({
            'id': unit_id,
            'prime_mover': pm_id,
            'fuel': fuel_id,
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
        return pd.DataFrame(columns=['id', 'prime_mover', 'max_capacity', 'balancing_topology',
                                     'efficiency_up', 'efficiency_down', 'rating', 'base_power'])
    
    storages = source['storage']
    existing_storages = storages[safe_filter(storages, 'storages_existing') & (storages['storages_existing'] > 0)].copy()
    links = source['link']
    
    pm_id = id_gen.get('prime_mover_types', 'temp_generator')
    power_grid_nodes = identify_power_grid_nodes(source, errors)
    
    storage_units_data = []
    
    for storage_name in existing_storages.index:
        storage_id = id_gen.get('storage_units', storage_name)
        max_capacity = safe_get(existing_storages, storage_name, 'storage_capacity', 1.0)
        
        # Find link connecting this storage to a balance node
        storage_links = []
        for link_idx in links.index:
            if isinstance(link_idx, tuple) and len(link_idx) == 2:
                node_a, node_b = link_idx
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
            'prime_mover': pm_id,
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
    
    pm_id = id_gen.get('prime_mover_types', 'temp_generator')
    supply_tech_data = []
    
    for unit_name in investment_units.index:
        key = f"supply_tech_{unit_name}"
        tech_id = id_gen.get('supply_technologies', key)
        
        # Find fuel (same logic as generation_units)
        fuel_id = None
        if 'node_to_unit' in source and 'commodity' in source:
            node_to_unit = source['node_to_unit']
            commodities = source['commodity']
            
            unit_inputs = node_to_unit[node_to_unit.index.get_level_values('sink') == unit_name]
            for input_idx in unit_inputs.index:
                input_node, _ = input_idx
                if input_node in commodities.index:
                    if safe_get(commodities, input_node, 'commodity_type') == 'fuel':
                        try:
                            fuel_id = id_gen.get('fuels', input_node)
                            break
                        except KeyError:
                            pass
        
        # Find balancing topology (from unit_to_node)
        balancing_topology = None
        if 'unit_to_node' in source:
            unit_to_node = source['unit_to_node']
            power_grid_nodes = identify_power_grid_nodes(source, errors)
            
            unit_to_nodes = unit_to_node[unit_to_node.index.get_level_values('source') == unit_name]
            for idx in unit_to_nodes.index:
                _, sink_node = idx
                if sink_node in power_grid_nodes:
                    try:
                        balancing_topology = id_gen.get('balancing_topologies', sink_node)
                        break
                    except KeyError:
                        pass
        
        supply_tech_data.append({
            'id': tech_id,
            'prime_mover': pm_id,
            'fuel': fuel_id,
            'area': None,
            'balancing_topology': balancing_topology,
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
    
    pm_id = id_gen.get('prime_mover_types', 'temp_generator')
    storage_tech_data = []
    
    for storage_name in investment_storages.index:
        key = f"storage_tech_{storage_name}"
        tech_id = id_gen.get('storage_technologies', key)
        
        storage_tech_data.append({
            'id': tech_id,
            'prime_mover': pm_id,
            'storage_technology_type': None,
            'area': None,
            'balancing_topology': None,
            'scenario': None
        })
    
    errors.add_missing("storage investment parameters (investment_cost, etc.) - unclear target placement")
    
    return pd.DataFrame(storage_tech_data)


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
    # Simplified: create from groups with link members
    interchanges_data = []
    
    if 'group' in source and 'group_entity' in source:
        group_entities = source['group_entity']
        for group_name in group_entities.index.get_level_values('group').unique():
            key = f"interchange_{group_name}"
            try:
                interchange_id = id_gen.get('transmission_interchanges', key)
            except KeyError:
                continue
            
            # Find corresponding arc (simplified)
            interchanges_data.append({
                'id': interchange_id,
                'arc_id': None,  # Would need to determine from link members
                'max_flow_from': 0.0,
                'max_flow_to': 0.0
            })
    
    errors.add_missing("transmission_interchanges.max_flow_from and max_flow_to - not processed")
    
    return pd.DataFrame(interchanges_data)


def transform_time_series(source: Dict[str, pd.DataFrame],
                           id_gen: IDGenerator,
                           errors: TransformationErrors) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create time_series_associations and static_time_series_data tables."""
    associations_data = []
    static_data = []
    ts_id = 0
    static_id = 0
    
    # Process profile data from unit_to_node.ts.profile_limit_*
    for key in source.keys():
        if key.startswith('unit_to_node.ts.profile_limit_'):
            ts_df = source[key]
            param_name = key.split('unit_to_node.ts.')[1]
            
            # Handle potential multi-index columns
            columns = ts_df.columns
            if isinstance(columns, pd.MultiIndex):
                for col_tuple in columns:
                    ts_id += 1
                    owner_name = create_composite_name(col_tuple)
                    
                    # Try to find corresponding generation unit
                    try:
                        unit_name = col_tuple[0] if isinstance(col_tuple, tuple) else col_tuple
                        owner_uuid = id_gen.get('generation_units', unit_name)
                    except KeyError:
                        errors.add_warning(f"Cannot find generation unit for profile {owner_name}")
                        continue
                    
                    associations_data.append({
                        'id': ts_id,
                        'time_series_uuid': f"ts_{ts_id}",
                        'time_series_type': 'static',
                        'initial_timestamp': None,
                        'resolution': None,
                        'horizon': None,
                        'interval': None,
                        'window_count': None,
                        'length': len(ts_df),
                        'name': param_name,
                        'owner_uuid': str(owner_uuid),
                        'owner_type': 'generation_units',
                        'owner_category': 'operational_data',
                        'features': '',
                        'scaling_factor_multiplier': None,
                        'metadata_uuid': None,
                        'units': None
                    })
                    
                    # Add static time series data points
                    for timestamp, value in ts_df[col_tuple].items():
                        if pd.notna(value):
                            static_id += 1
                            static_data.append({
                                'id': static_id,
                                'time_series_id': ts_id,
                                'timestamp': str(timestamp),
                                'value': float(value) if value != 'object' else 0.0
                            })
            else:
                # Simple column index
                for col in columns:
                    ts_id += 1
                    
                    try:
                        owner_uuid = id_gen.get('generation_units', col)
                    except KeyError:
                        errors.add_warning(f"Cannot find generation unit for profile {col}")
                        continue
                    
                    associations_data.append({
                        'id': ts_id,
                        'time_series_uuid': f"ts_{ts_id}",
                        'time_series_type': 'static',
                        'initial_timestamp': None,
                        'resolution': None,
                        'horizon': None,
                        'interval': None,
                        'window_count': None,
                        'length': len(ts_df),
                        'name': param_name,
                        'owner_uuid': str(owner_uuid),
                        'owner_type': 'generation_units',
                        'owner_category': 'operational_data',
                        'features': '',
                        'scaling_factor_multiplier': None,
                        'metadata_uuid': None,
                        'units': None
                    })
                    
                    # Add static time series data points
                    for timestamp, value in ts_df[col].items():
                        if pd.notna(value):
                            static_id += 1
                            static_data.append({
                                'id': static_id,
                                'time_series_id': ts_id,
                                'timestamp': str(timestamp),
                                'value': float(value) if value != 'object' else 0.0
                            })
    
    # Process flow profile data from balance.ts.flow_profile
    if 'balance.ts.flow_profile' in source:
        flow_df = source['balance.ts.flow_profile']
        
        for balance_name in flow_df.columns:
            ts_id += 1
            
            try:
                owner_uuid = id_gen.get('balancing_topologies', balance_name)
            except KeyError:
                errors.add_warning(f"Cannot find balancing topology for flow profile {balance_name}")
                continue
            
            associations_data.append({
                'id': ts_id,
                'time_series_uuid': f"ts_{ts_id}",
                'time_series_type': 'static',
                'initial_timestamp': None,
                'resolution': None,
                'horizon': None,
                'interval': None,
                'window_count': None,
                'length': len(flow_df),
                'name': 'flow_profile',
                'owner_uuid': str(owner_uuid),
                'owner_type': 'balancing_topologies',
                'owner_category': 'operational_data',
                'features': '',
                'scaling_factor_multiplier': None,
                'metadata_uuid': None,
                'units': None
            })
            
            # Add static time series data points
            for timestamp, value in flow_df[balance_name].items():
                if pd.notna(value):
                    static_id += 1
                    static_data.append({
                        'id': static_id,
                        'time_series_id': ts_id,
                        'timestamp': str(timestamp),
                        'value': float(value) if value != 'object' else 0.0
                    })
    
    errors.add_missing("deterministic_forecast_data - not yet implemented")
    errors.add_missing("time_series features parameters - ignored")
    
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
        target['operational_data'] = transform_operational_data(source, id_gen, errors)
        
        # Phase 3: Time series
        print("Phase 3: Creating time series data...")
        target['time_series_associations'], target['static_time_series_data'] = \
            transform_time_series(source, id_gen, errors)
        
        # Phase 4: Empty tables for completeness
        target['hydro_reservoir'] = pd.DataFrame(columns=['id'])
        target['hydro_reservoir_connections'] = pd.DataFrame(columns=['turbine_id', 'reservoir_id'])
        target['transport_technologies'] = pd.DataFrame(columns=['id', 'arc_id', 'scenario'])
        target['loads'] = pd.DataFrame(columns=['id', 'balancing_topology', 'base_power'])
        target['attributes'] = pd.DataFrame(columns=['id', 'entity_id', 'TYPE', 'name', 'value'])
        target['supplemental_attributes'] = pd.DataFrame(columns=['id', 'TYPE', 'value'])
        target['supplemental_attributes_association'] = pd.DataFrame(columns=['attribute_id', 'entity_id'])
        target['deterministic_forecast_data'] = pd.DataFrame(columns=['id', 'time_series_id', 'timestamp', 'forecast_values'])
        
        errors.add_missing("hydro_reservoir and hydro_reservoir_connections - not processed")
        errors.add_missing("loads - cannot generate from flexible source units (no clear base_power)")
        errors.add_missing("transport_technologies - no clear source mapping")
        
    except Exception as e:
        errors.add_error(f"Critical transformation error: {str(e)}")
        import traceback
        errors.add_error(traceback.format_exc())
    
    # Report all errors and warnings
    errors.report()
    
    return target