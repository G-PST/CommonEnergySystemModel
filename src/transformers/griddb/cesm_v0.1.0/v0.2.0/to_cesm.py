"""
Reverse transformer for converting GridDB SQLite database to CESM dataframe format.

This module converts GridDB SQLite tables to CESM-format DataFrames matching the
structure produced by linkml_to_dataframes.yaml_to_df().

v0.2.0 - Initial implementation

Key mappings:
- balancing_topologies -> balance (LoadZones become balance nodes)
- loads + max_active_power -> balance.ts.flow_profile (demand as negative)
- generation_units -> unit (prime_mover stored in description)
- generation_units connections -> unit_to_node ports
- generation_units fuel FK -> node_to_unit, commodity
- storage_units -> storage
- storage_units connections -> link (bidirectional)
- transmission_lines + arcs -> link
- planning_regions -> group (type=node)
- fuels -> commodity (type=fuel)
- time_series (hourly) -> *.ts.* DataFrames
- time_series (yearly) -> PeriodFloat attributes
- heatrate -> efficiency = heatrate_conversion_factor / heatrate

Configuration is loaded from to_cesm_config.yaml in the same directory.
"""

import pandas as pd
import numpy as np
import sqlite3
import yaml
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), 'to_cesm_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_prime_mover_fuel_mapping(yaml_mapping: Dict[str, str]) -> Dict[Tuple[str, Optional[str]], str]:
    """
    Parse YAML mapping format to tuple-keyed dictionary.

    YAML format: "prime_mover:fuel" -> owner_type
    Python format: (prime_mover, fuel) -> owner_type (where fuel can be None)
    """
    result = {}
    for key, value in yaml_mapping.items():
        parts = key.split(':')
        prime_mover = parts[0]
        fuel = parts[1] if len(parts) > 1 else None
        if fuel == 'null':
            fuel = None
        result[(prime_mover, fuel)] = value
    return result


# Load configuration
_CONFIG = load_config()

# Extract configuration values
HEATRATE_CONVERSION_FACTOR = _CONFIG.get('heatrate_conversion_factor', 3.412)
DEFAULT_EFFICIENCIES = _CONFIG.get('default_efficiencies', {})
PRIME_MOVER_DESCRIPTIONS = _CONFIG.get('prime_mover_descriptions', {})
FUEL_NAME_REVERSE_MAP = _CONFIG.get('fuel_name_reverse_map', {})
PRIME_MOVER_FUEL_TO_TS_OWNER = parse_prime_mover_fuel_mapping(
    _CONFIG.get('prime_mover_fuel_to_ts_owner', {}))
UNIT_NAME_FUEL_HINTS = _CONFIG.get('unit_name_fuel_hints', {})


@dataclass
class TransformationWarnings:
    """Collector for transformation warnings."""
    warnings: List[str] = field(default_factory=list)
    missing_data: List[str] = field(default_factory=list)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def add_missing(self, message: str):
        self.missing_data.append(message)

    def report(self):
        """Print all collected warnings."""
        if self.warnings:
            print("\n" + "=" * 80)
            print("TRANSFORMATION WARNINGS:")
            print("=" * 80)
            for warn in self.warnings:
                print(f"  * {warn}")

        if self.missing_data:
            print("\n" + "=" * 80)
            print("MISSING DATA (GridDB -> CESM):")
            print("=" * 80)
            for miss in self.missing_data:
                print(f"  * {miss}")


def load_griddb_tables(db_path: str) -> Dict[str, pd.DataFrame]:
    """Load all tables from GridDB SQLite database."""
    conn = sqlite3.connect(db_path)

    # Get list of all tables
    tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
    tables = pd.read_sql_query(tables_query, conn)['name'].tolist()

    griddb = {}
    for table in tables:
        try:
            griddb[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception as e:
            print(f"Warning: Could not load table {table}: {e}")

    conn.close()
    return griddb


def load_time_series_data(db_path: str, uuid: str) -> pd.Series:
    """Load time series data for a specific UUID from static_time_series."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT idx, value
        FROM static_time_series
        WHERE uuid = ?
        ORDER BY idx
    """
    df = pd.read_sql_query(query, conn, params=[uuid])
    conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    return pd.Series(df['value'].values, index=df['idx'].values)


def is_constant_series(series: pd.Series, tolerance: float = 1e-10) -> bool:
    """Check if a series has effectively constant values."""
    if len(series) <= 1:
        return True
    return series.std() < tolerance


def map_fuel_name_reverse(griddb_fuel: str) -> str:
    """Map GridDB standardized fuel name to CESM common name."""
    if griddb_fuel is None:
        return None
    return FUEL_NAME_REVERSE_MAP.get(griddb_fuel, griddb_fuel.lower())


def get_prime_mover_description(prime_mover_code: str) -> str:
    """Get descriptive text for prime mover code."""
    return PRIME_MOVER_DESCRIPTIONS.get(prime_mover_code, f"Unknown ({prime_mover_code})")


# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def _normalize_to_utc(dt_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Normalize a DatetimeIndex to UTC (timezone-naive).

    - If timezone-aware, convert to UTC and remove timezone info
    - If timezone-naive, assume already UTC
    Returns a timezone-naive DatetimeIndex representing UTC times.
    """
    if dt_index.tz is not None:
        # Convert to UTC, then remove timezone info
        return dt_index.tz_convert('UTC').tz_localize(None)
    return dt_index


def transform_timeline(griddb: Dict[str, pd.DataFrame],
                       db_path: str,
                       warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Extract timeline from time series associations.

    Finds the longest hourly time series and constructs a datetime index.
    All timestamps are normalized to UTC (timezone-naive).
    """
    if 'time_series_associations' not in griddb:
        warnings.add_missing("No time_series_associations table - cannot construct timeline")
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    tsa = griddb['time_series_associations']

    # Find hourly time series (length > 100, typically 8760 or 8784)
    hourly_ts = tsa[tsa['length'] > 100]

    if hourly_ts.empty:
        warnings.add_warning("No hourly time series found - using default timeline")
        # Default: 8760 hours starting 2020-01-01 UTC
        timeline = pd.date_range(start='2020-01-01', periods=8760, freq='h')
        return pd.DataFrame(index=timeline)

    # Get the longest time series
    max_length = hourly_ts['length'].max()
    sample_ts = hourly_ts[hourly_ts['length'] == max_length].iloc[0]

    # Parse initial_timestamp and resolution
    try:
        initial_ts = pd.to_datetime(sample_ts['initial_timestamp'])
        # Normalize to UTC if timezone-aware
        if initial_ts.tz is not None:
            initial_ts = initial_ts.tz_convert('UTC').tz_localize(None)

        resolution = sample_ts['resolution']
        # Parse resolution like "1h" or "1H"
        if resolution and 'h' in resolution.lower():
            freq = 'h'
        else:
            freq = 'h'  # Default to hourly

        timeline = pd.date_range(start=initial_ts, periods=int(max_length), freq=freq)
    except Exception as e:
        warnings.add_warning(f"Could not parse timeline from time series: {e}")
        timeline = pd.date_range(start='2020-01-01', periods=int(max_length), freq='h')

    # Ensure timeline is UTC (timezone-naive)
    timeline = _normalize_to_utc(timeline)

    return pd.DataFrame(index=timeline)


def transform_periods(griddb: Dict[str, pd.DataFrame],
                      db_path: str,
                      warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Extract period entities from year-indexed time series.

    Periods are auto-created from unique years found in yearly time series data.
    """
    if 'time_series_associations' not in griddb:
        warnings.add_missing("No time_series_associations - cannot determine periods")
        return pd.DataFrame(columns=['years_represented']).rename_axis('period')

    tsa = griddb['time_series_associations']

    # Find yearly time series (length typically 20-50, indexed by year)
    yearly_ts = tsa[(tsa['length'] > 10) & (tsa['length'] < 100)]

    if yearly_ts.empty:
        warnings.add_warning("No yearly time series found - no periods created")
        return pd.DataFrame(columns=['years_represented']).rename_axis('period')

    # Load one yearly time series to get years
    sample_uuid = yearly_ts.iloc[0]['time_series_uuid']
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT idx FROM static_time_series WHERE uuid = ? ORDER BY idx"
    years_df = pd.read_sql_query(query, conn, params=[sample_uuid])
    conn.close()

    if years_df.empty:
        return pd.DataFrame(columns=['years_represented']).rename_axis('period')

    # Create period entries - use year as period name
    years = years_df['idx'].values
    periods_data = []
    for i, year in enumerate(years):
        periods_data.append({
            'name': str(int(year)),
            'years_represented': 1.0  # Each period represents 1 year
        })

    df = pd.DataFrame(periods_data)
    df = df.set_index('name')
    df.index.name = 'period'
    return df


def transform_balance(griddb: Dict[str, pd.DataFrame],
                      warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Transform balancing_topologies to CESM balance nodes.

    balancing_topologies (LoadZones) become balance nodes where energy
    balance must be maintained.
    """
    if 'balancing_topologies' not in griddb:
        warnings.add_missing("No balancing_topologies table")
        return pd.DataFrame(columns=['flow_annual', 'flow_scaling_method',
                                     'penalty_downward', 'penalty_upward']).rename_axis('balance')

    bt = griddb['balancing_topologies']

    balance_data = []
    for _, row in bt.iterrows():
        balance_data.append({
            'name': row['name'],
            'flow_annual': None,
            'flow_scaling_method': None,
            'penalty_downward': None,
            'penalty_upward': 10000.0,  # Default penalty for unserved energy
        })

    df = pd.DataFrame(balance_data)
    if df.empty:
        return pd.DataFrame(columns=['flow_annual', 'flow_scaling_method',
                                     'penalty_downward', 'penalty_upward']).rename_axis('balance')

    df = df.set_index('name')
    df.index.name = 'balance'
    return df


def transform_balance_flow_profile(griddb: Dict[str, pd.DataFrame],
                                   db_path: str,
                                   timeline: pd.DataFrame,
                                   warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Transform load time series to balance.ts.flow_profile.

    The entity relationship chain is:
    - planning_regions (LoadZone entities) have peak_active_power attribute
    - balancing_topologies.area -> planning_regions.id
    - loads.balancing_topology -> balancing_topologies.id

    Time series are attached to PowerLoad entities (loads table). Each PowerLoad belongs
    to a balancing_topology, which in turn belongs to a planning_region (LoadZone) via
    the balancing_topologies.area foreign key.

    The time series values are fractions (0-1) that need to be multiplied by
    peak_active_power to get actual MW values. When multiple PowerLoads belong to
    the same LoadZone (via their balancing_topologies), their base_power values are
    used as weights to distribute the peak_active_power among them.

    Only balancing_topologies whose area (planning_region/LoadZone) has peak_active_power
    attribute will have flow profiles. Results are negative (consumption).
    """
    if 'loads' not in griddb or 'time_series_associations' not in griddb:
        warnings.add_missing("No loads or time_series_associations for flow_profile")
        return pd.DataFrame(index=timeline.index).rename_axis('datetime')

    if 'balancing_topologies' not in griddb:
        warnings.add_missing("No balancing_topologies for flow_profile")
        return pd.DataFrame(index=timeline.index).rename_axis('datetime')

    loads = griddb['loads']
    tsa = griddb['time_series_associations']
    bt = griddb['balancing_topologies']

    # Create mappings for balancing_topologies
    bt_id_to_name = dict(zip(bt['id'], bt['name']))
    # Map balancing_topology id -> area (planning_region) id
    bt_id_to_area_id = dict(zip(bt['id'], bt['area']))

    # Get peak_active_power from attributes table for planning_regions (LoadZone entities)
    # planning_regions are the LoadZone entities that have peak_active_power
    peak_power_by_area_id = {}
    if 'attributes' in griddb and 'planning_regions' in griddb:
        attrs = griddb['attributes']
        pr = griddb['planning_regions']
        pr_ids = set(pr['id'])

        peak_power_attrs = attrs[attrs['name'] == 'peak_active_power']
        for _, attr_row in peak_power_attrs.iterrows():
            entity_id = attr_row['entity_id']
            # Check if this entity is a planning_region (LoadZone)
            if entity_id in pr_ids:
                value = attr_row['value']
                # Handle JSON value - could be a number or a JSON string
                if isinstance(value, str):
                    try:
                        import json
                        value = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if isinstance(value, (int, float)):
                    peak_power_by_area_id[entity_id] = float(value)

    if not peak_power_by_area_id:
        warnings.add_warning("No peak_active_power attributes found for planning_regions (LoadZones)")
        return pd.DataFrame(index=timeline.index).rename_axis('datetime')

    # Calculate sum of base_power for each area (planning_region/LoadZone) for weighting
    # All loads in balancing_topologies that share the same area are grouped together
    area_base_power_sum = {}
    for _, load_row in loads.iterrows():
        bt_id = load_row['balancing_topology']
        area_id = bt_id_to_area_id.get(bt_id)
        if area_id is None:
            continue
        base_power = load_row.get('base_power', 1.0) or 1.0
        if area_id in area_base_power_sum:
            area_base_power_sum[area_id] += base_power
        else:
            area_base_power_sum[area_id] = base_power

    # Find load time series (max_active_power for PowerLoad)
    load_ts = tsa[(tsa['owner_type'].str.contains('Load', case=False, na=False)) &
                  (tsa['name'] == 'max_active_power')]

    if load_ts.empty:
        warnings.add_warning("No load time series found")
        return pd.DataFrame(index=timeline.index).rename_axis('datetime')

    # Build flow_profile DataFrame - aggregate loads by balancing_topology
    flow_data = {}

    for _, load_row in loads.iterrows():
        load_id = load_row['id']
        bt_id = load_row['balancing_topology']
        bt_name = bt_id_to_name.get(bt_id, f"zone_{bt_id}")
        area_id = bt_id_to_area_id.get(bt_id)
        base_power = load_row.get('base_power', 1.0) or 1.0

        # Check if the area (planning_region/LoadZone) has peak_active_power
        if area_id is None or area_id not in peak_power_by_area_id:
            # No peak_active_power for this area - skip (this is expected for some)
            continue

        peak_power = peak_power_by_area_id[area_id]
        total_base_power = area_base_power_sum.get(area_id, base_power)

        # Calculate weight for this load (fraction of total base_power in the area)
        weight = base_power / total_base_power if total_base_power > 0 else 1.0

        # Find time series for this load
        load_tsa = load_ts[load_ts['owner_id'] == load_id]
        if load_tsa.empty:
            continue

        ts_uuid = load_tsa.iloc[0]['time_series_uuid']
        ts_data = load_time_series_data(db_path, ts_uuid)

        if ts_data.empty:
            continue

        # Time series values are fractions of peak_active_power
        # Multiply by (weight * peak_power) to get actual MW for this load
        # Make negative (consumption)
        ts_values = -1 * ts_data.values * weight * peak_power

        # Aggregate by balancing_topology (sum multiple loads in same bt)
        if bt_name in flow_data:
            # Add to existing (element-wise sum)
            min_len = min(len(flow_data[bt_name]), len(ts_values))
            flow_data[bt_name][:min_len] += ts_values[:min_len]
        else:
            flow_data[bt_name] = ts_values

    if not flow_data:
        warnings.add_warning("No flow profiles created - no loads with time series in areas that have peak_active_power")
        return pd.DataFrame(index=timeline.index).rename_axis('datetime')

    # Build DataFrame with timeline index
    df = pd.DataFrame(flow_data, index=timeline.index[:len(next(iter(flow_data.values())))])
    df.index.name = 'datetime'
    return df


def transform_commodity(griddb: Dict[str, pd.DataFrame],
                        db_path: str,
                        warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Transform fuel_price time series to CESM commodity entities.

    Creates commodity nodes from fuel_price time series owner_types.
    Each unique owner_type with fuel_price becomes a fuel commodity.
    """
    if 'time_series_associations' not in griddb:
        warnings.add_missing("No time_series_associations - cannot create commodities from fuel_price")
        return pd.DataFrame(columns=['commodity_type', 'price_per_unit']).rename_axis('commodity')

    tsa = griddb['time_series_associations']
    fuel_price_ts = tsa[tsa['name'] == 'fuel_price']

    if fuel_price_ts.empty:
        warnings.add_warning("No fuel_price time series found")
        return pd.DataFrame(columns=['commodity_type', 'price_per_unit']).rename_axis('commodity')

    commodity_data = []
    for _, ts_row in fuel_price_ts.iterrows():
        owner_type = ts_row['owner_type']  # e.g., 'Coal', 'NGCC', 'Nuclear'
        resolution = ts_row.get('resolution', '')

        # Load fuel price data
        ts_data = load_time_series_data(db_path, ts_row['time_series_uuid'])
        if ts_data.empty:
            price = None
        elif resolution == 'P1Y':
            # Yearly data - use first year's value
            price = float(ts_data.iloc[0])
        elif is_constant_series(ts_data):
            price = float(ts_data.iloc[0])
        else:
            price = float(ts_data.mean())

        # Use owner_type as commodity name (lowercase for CESM convention)
        cesm_name = owner_type.lower()

        commodity_data.append({
            'name': cesm_name,
            'commodity_type': 'fuel',
            'price_per_unit': price,
        })

    df = pd.DataFrame(commodity_data)
    if df.empty:
        return pd.DataFrame(columns=['commodity_type', 'price_per_unit']).rename_axis('commodity')

    df = df.set_index('name')
    df.index.name = 'commodity'
    return df


def transform_groups(griddb: Dict[str, pd.DataFrame],
                     warnings: TransformationWarnings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform planning_regions to CESM groups and group_entity.

    planning_regions become groups with group_type='node'.
    balancing_topologies with area references become group members.
    Also create a 'power_grid' group containing all balancing_topologies.
    """
    group_data = []
    group_entity_data = []

    # Create power_grid group containing all balancing_topologies
    if 'balancing_topologies' in griddb:
        bt = griddb['balancing_topologies']
        group_data.append({
            'name': 'power_grid',
            'group_type': 'power_grid',
            'invest_max_total': None,
        })
        for _, row in bt.iterrows():
            group_entity_data.append({
                'name': f"power_grid__{row['name']}",
                'group': 'power_grid',
                'entity': row['name'],
            })

    # Create planning region groups
    if 'planning_regions' in griddb and 'balancing_topologies' in griddb:
        pr = griddb['planning_regions']
        bt = griddb['balancing_topologies']

        for _, region_row in pr.iterrows():
            region_name = region_row['name']
            group_data.append({
                'name': region_name,
                'group_type': 'node',
                'invest_max_total': None,
            })

            # Find balancing_topologies in this region
            region_bt = bt[bt['area'] == region_row['id']]
            for _, bt_row in region_bt.iterrows():
                group_entity_data.append({
                    'name': f"{region_name}__{bt_row['name']}",
                    'group': region_name,
                    'entity': bt_row['name'],
                })

    # Create group DataFrame
    group_df = pd.DataFrame(group_data)
    if not group_df.empty:
        group_df = group_df.set_index('name')
        group_df.index.name = 'group'
    else:
        group_df = pd.DataFrame(columns=['group_type', 'invest_max_total']).rename_axis('group')

    # Create group_entity DataFrame with multi-index
    group_entity_df = pd.DataFrame(group_entity_data)
    if not group_entity_df.empty:
        group_entity_df = group_entity_df.set_index(['name', 'group', 'entity'])
    else:
        group_entity_df = pd.DataFrame().set_index(
            pd.MultiIndex.from_tuples([], names=['name', 'group', 'entity']))

    return group_df, group_entity_df


def get_ts_owner_for_unit(prime_mover: str, fuel: Optional[str],
                          unit_name: Optional[str] = None) -> Optional[str]:
    """
    Get the time series owner_type name for a unit based on prime_mover and fuel.

    Uses PRIME_MOVER_FUEL_TO_TS_OWNER mapping to find the correct time series
    association (e.g., 'NGCC' for CC + NATURAL_GAS).

    When fuel is None, checks unit_name for hints (e.g., 'NUCLEAR' in name).
    """
    # If fuel is None, try to infer from unit name
    inferred_owner = None
    if fuel is None and unit_name:
        unit_name_upper = unit_name.upper()
        for pattern, owner in UNIT_NAME_FUEL_HINTS.items():
            if pattern.upper() in unit_name_upper:
                inferred_owner = owner
                break

    # If we inferred an owner from the name, use it directly
    if inferred_owner:
        return inferred_owner

    # Try exact match first
    key = (prime_mover, fuel)
    if key in PRIME_MOVER_FUEL_TO_TS_OWNER:
        return PRIME_MOVER_FUEL_TO_TS_OWNER[key]

    # Try with None fuel (default)
    key_none = (prime_mover, None)
    if key_none in PRIME_MOVER_FUEL_TO_TS_OWNER:
        return PRIME_MOVER_FUEL_TO_TS_OWNER[key_none]

    # Try with OTHER fuel
    key_other = (prime_mover, 'OTHER')
    if key_other in PRIME_MOVER_FUEL_TO_TS_OWNER:
        return PRIME_MOVER_FUEL_TO_TS_OWNER[key_other]

    return None


def transform_unit(griddb: Dict[str, pd.DataFrame],
                   db_path: str,
                   warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Transform generation_units to CESM unit entities.

    - prime_mover code stored in description
    - heatrate converted to efficiency using PRIME_MOVER_FUEL_TO_TS_OWNER mapping
    - For yearly data (resolution='P1Y'), use values directly without averaging
    - units_existing from rating/base_power
    """
    if 'generation_units' not in griddb:
        warnings.add_missing("No generation_units table")
        return pd.DataFrame(columns=['availability', 'conversion_method', 'discount_rate',
                                     'efficiency', 'investment_method', 'payback_time',
                                     'units_existing']).rename_axis('unit')

    gu = griddb['generation_units']

    # Load heatrate time series indexed by owner_type
    # owner_type is the key (e.g., 'NGCC', 'Coal', 'Nuclear')
    heatrates = {}
    if 'time_series_associations' in griddb:
        tsa = griddb['time_series_associations']
        heatrate_ts = tsa[tsa['name'] == 'heatrate']

        for _, ts_row in heatrate_ts.iterrows():
            owner_type = ts_row['owner_type']
            resolution = ts_row.get('resolution', '')

            ts_data = load_time_series_data(db_path, ts_row['time_series_uuid'])
            if ts_data.empty:
                continue

            # For yearly data (P1Y resolution), use first value directly
            # For other resolutions, use mean if varying, first if constant
            if resolution == 'P1Y':
                # Yearly data - use first year's value (or could use specific year)
                heatrate_val = ts_data.iloc[0]
            elif is_constant_series(ts_data):
                heatrate_val = ts_data.iloc[0]
            else:
                heatrate_val = ts_data.mean()

            heatrates[owner_type] = heatrate_val

    unit_data = []
    for _, row in gu.iterrows():
        unit_name = row['name']
        prime_mover = row['prime_mover']
        fuel = row.get('fuel')  # May be None in this database
        prime_mover_desc = get_prime_mover_description(prime_mover)

        # Look up the time series owner_type using the mapping (pass unit_name for fuel hints)
        ts_owner = get_ts_owner_for_unit(prime_mover, fuel, unit_name)

        # Calculate efficiency from heatrate if available
        efficiency = 1.0
        if ts_owner and ts_owner in heatrates:
            heatrate_val = heatrates[ts_owner]
            if heatrate_val and heatrate_val > 0:
                efficiency = HEATRATE_CONVERSION_FACTOR / heatrate_val
        elif ts_owner and ts_owner in DEFAULT_EFFICIENCIES:
            # Use default efficiency from config when heatrate not available
            efficiency = DEFAULT_EFFICIENCIES[ts_owner]

        unit_data.append({
            'name': row['name'],
            'description': f"Prime mover: {prime_mover} ({prime_mover_desc})",
            'availability': None,
            'conversion_method': 'constant_efficiency',
            'discount_rate': None,
            'efficiency': efficiency,
            'investment_method': None,
            'payback_time': None,
            'units_existing': 1.0,  # Each GridDB unit is a single unit
        })

    df = pd.DataFrame(unit_data)
    if df.empty:
        return pd.DataFrame(columns=['availability', 'conversion_method', 'discount_rate',
                                     'efficiency', 'investment_method', 'payback_time',
                                     'units_existing']).rename_axis('unit')

    df = df.set_index('name')
    df.index.name = 'unit'
    # Drop description from columns (it's metadata only)
    if 'description' in df.columns:
        df = df.drop(columns=['description'])
    return df


def transform_unit_to_node(griddb: Dict[str, pd.DataFrame],
                           db_path: str,
                           warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Transform generation_unit connections to CESM unit_to_node ports.

    Creates ports from generation_units to their balancing_topology.
    Port name follows pattern: unit_name.balance_name

    Also populates costs from time series:
    - capcost -> investment_cost
    - fom -> fixed_cost
    - vom -> other_operational_cost
    """
    if 'generation_units' not in griddb or 'balancing_topologies' not in griddb:
        warnings.add_missing("No generation_units or balancing_topologies for unit_to_node")
        return pd.DataFrame(columns=['availability', 'capacity', 'constraint_flow_coefficient',
                                     'fixed_cost', 'investment_cost', 'other_operational_cost',
                                     'profile_limit_lower', 'profile_limit_upper']).set_index(
            pd.MultiIndex.from_tuples([], names=['name', 'source', 'sink']))

    gu = griddb['generation_units']
    bt = griddb['balancing_topologies']

    # Create mapping from balancing_topology id to name
    bt_id_to_name = dict(zip(bt['id'], bt['name']))

    # Load cost time series indexed by owner_type
    # capcost -> investment_cost, fom -> fixed_cost, vom -> other_operational_cost
    capcosts = {}
    fom_costs = {}
    vom_costs = {}

    if 'time_series_associations' in griddb:
        tsa = griddb['time_series_associations']

        for cost_name, cost_dict in [('capcost', capcosts), ('fom', fom_costs), ('vom', vom_costs)]:
            cost_ts = tsa[tsa['name'] == cost_name]
            for _, ts_row in cost_ts.iterrows():
                owner_type = ts_row['owner_type']
                resolution = ts_row.get('resolution', '')

                ts_data = load_time_series_data(db_path, ts_row['time_series_uuid'])
                if ts_data.empty:
                    continue

                # For yearly data (P1Y), use first year's value directly
                if resolution == 'P1Y':
                    cost_val = float(ts_data.iloc[0])
                elif is_constant_series(ts_data):
                    cost_val = float(ts_data.iloc[0])
                else:
                    cost_val = float(ts_data.mean())

                cost_dict[owner_type] = cost_val

    port_data = []
    for _, row in gu.iterrows():
        unit_name = row['name']
        prime_mover = row['prime_mover']
        fuel = row.get('fuel')
        bt_id = row['balancing_topology']
        bt_name = bt_id_to_name.get(bt_id, f"zone_{bt_id}")

        port_name = f"{unit_name}.{bt_name}"
        capacity = row.get('rating', row.get('base_power', 1.0))

        # Look up costs using the mapping (pass unit_name for fuel hints)
        ts_owner = get_ts_owner_for_unit(prime_mover, fuel, unit_name)

        investment_cost = capcosts.get(ts_owner) if ts_owner else None
        fixed_cost = fom_costs.get(ts_owner) if ts_owner else None
        other_operational_cost = vom_costs.get(ts_owner) if ts_owner else None

        port_data.append({
            'name': port_name,
            'source': unit_name,
            'sink': bt_name,
            'availability': None,
            'capacity': capacity,
            'constraint_flow_coefficient': None,
            'fixed_cost': fixed_cost,
            'investment_cost': investment_cost,
            'other_operational_cost': other_operational_cost,
            'profile_limit_lower': None,
            'profile_limit_upper': None,
        })

    df = pd.DataFrame(port_data)
    if df.empty:
        return pd.DataFrame(columns=['availability', 'capacity', 'constraint_flow_coefficient',
                                     'fixed_cost', 'investment_cost', 'other_operational_cost',
                                     'profile_limit_lower', 'profile_limit_upper']).set_index(
            pd.MultiIndex.from_tuples([], names=['name', 'source', 'sink']))

    df = df.set_index(['name', 'source', 'sink'])
    return df


def transform_node_to_unit(griddb: Dict[str, pd.DataFrame],
                           fuel_commodities: pd.DataFrame,
                           warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Transform fuel connections to CESM node_to_unit ports.

    Creates ports from fuel commodities to generation_units using the
    PRIME_MOVER_FUEL_TO_TS_OWNER mapping to determine which fuel connects
    to which unit type.
    """
    if 'generation_units' not in griddb:
        warnings.add_missing("No generation_units for node_to_unit")
        return pd.DataFrame(columns=['availability', 'capacity', 'constraint_flow_coefficient',
                                     'fixed_cost', 'investment_cost', 'other_operational_cost',
                                     'profile_limit_lower', 'profile_limit_upper']).set_index(
            pd.MultiIndex.from_tuples([], names=['name', 'source', 'sink']))

    gu = griddb['generation_units']

    # Get available fuel commodity names (lowercase)
    available_fuels = set(fuel_commodities.index) if not fuel_commodities.empty else set()

    port_data = []
    for _, row in gu.iterrows():
        unit_name = row['name']
        prime_mover = row['prime_mover']
        fuel = row.get('fuel')  # May be None

        # Use the mapping to find the fuel type for this unit (pass unit_name for fuel hints)
        ts_owner = get_ts_owner_for_unit(prime_mover, fuel, unit_name)

        if ts_owner:
            # Convert to lowercase for CESM commodity name
            cesm_fuel = ts_owner.lower()

            # Only create connection if fuel commodity exists
            if cesm_fuel in available_fuels:
                port_name = f"{cesm_fuel}.{unit_name}"

                port_data.append({
                    'name': port_name,
                    'source': cesm_fuel,
                    'sink': unit_name,
                    'availability': None,
                    'capacity': None,
                    'constraint_flow_coefficient': None,
                    'fixed_cost': None,
                    'investment_cost': None,
                    'other_operational_cost': None,
                    'profile_limit_lower': None,
                    'profile_limit_upper': None,
                })

    df = pd.DataFrame(port_data)
    if df.empty:
        return pd.DataFrame(columns=['availability', 'capacity', 'constraint_flow_coefficient',
                                     'fixed_cost', 'investment_cost', 'other_operational_cost',
                                     'profile_limit_lower', 'profile_limit_upper']).set_index(
            pd.MultiIndex.from_tuples([], names=['name', 'source', 'sink']))

    df = df.set_index(['name', 'source', 'sink'])
    return df


def transform_unit_to_node_profile(griddb: Dict[str, pd.DataFrame],
                                   db_path: str,
                                   timeline: pd.DataFrame,
                                   warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Transform renewable max_active_power to unit_to_node.ts.profile_limit_upper.

    Capacity factors for renewables become profile_limit_upper time series.
    """
    if 'generation_units' not in griddb or 'time_series_associations' not in griddb:
        warnings.add_missing("No generation_units or time_series_associations for profiles")
        return pd.DataFrame(index=timeline.index).rename_axis('datetime')

    gu = griddb['generation_units']
    bt = griddb['balancing_topologies']
    tsa = griddb['time_series_associations']

    # Create mapping from balancing_topology id to name
    bt_id_to_name = dict(zip(bt['id'], bt['name']))

    # Find renewable time series
    renewable_ts = tsa[
        (tsa['owner_type'].str.contains('Renewable', case=False, na=False)) &
        (tsa['name'] == 'max_active_power')
    ]

    if renewable_ts.empty:
        warnings.add_warning("No renewable time series found for profile_limit_upper")
        return pd.DataFrame(index=timeline.index).rename_axis('datetime')

    # Build profile DataFrame with MultiIndex columns
    profile_data = {}
    column_tuples = []

    for _, ts_row in renewable_ts.iterrows():
        owner_id = ts_row['owner_id']

        # Find the generation unit
        unit_row = gu[gu['id'] == owner_id]
        if unit_row.empty:
            continue

        unit_row = unit_row.iloc[0]
        unit_name = unit_row['name']
        bt_id = unit_row['balancing_topology']
        bt_name = bt_id_to_name.get(bt_id, f"zone_{bt_id}")

        # Load time series data
        ts_data = load_time_series_data(db_path, ts_row['time_series_uuid'])
        if ts_data.empty:
            continue

        # Normalize to capacity factor (0-1) based on rating
        rating = unit_row.get('rating', unit_row.get('base_power', 1.0))
        if rating > 0:
            cf_values = ts_data.values / rating
        else:
            cf_values = ts_data.values

        # Column tuple: (port_name, unit_name, balance_name)
        port_name = f"{unit_name}.{bt_name}"
        col_key = (port_name, unit_name, bt_name)
        column_tuples.append(col_key)
        profile_data[col_key] = cf_values

    if not profile_data:
        return pd.DataFrame(index=timeline.index).rename_axis('datetime')

    # Build DataFrame
    df = pd.DataFrame(profile_data, index=timeline.index[:len(next(iter(profile_data.values())))])
    df.columns = pd.MultiIndex.from_tuples(column_tuples, names=['name', 'source', 'sink'])
    df.index.name = 'datetime'
    return df


def transform_storage(griddb: Dict[str, pd.DataFrame],
                      warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Transform storage_units to CESM storage entities.
    """
    if 'storage_units' not in griddb:
        warnings.add_missing("No storage_units table")
        return pd.DataFrame(columns=['availability', 'discount_rate', 'fixed_cost',
                                     'flow_annual', 'flow_scaling_method', 'investment_cost',
                                     'investment_method', 'payback_time', 'penalty_downward',
                                     'penalty_upward', 'storage_capacity', 'storages_existing'
                                     ]).rename_axis('storage')

    su = griddb['storage_units']

    storage_data = []
    for _, row in su.iterrows():
        prime_mover = row.get('prime_mover', 'ES')
        prime_mover_desc = get_prime_mover_description(prime_mover)

        storage_data.append({
            'name': row['name'],
            'description': f"Prime mover: {prime_mover} ({prime_mover_desc})",
            'availability': None,
            'discount_rate': None,
            'fixed_cost': None,
            'flow_annual': None,
            'flow_scaling_method': None,
            'investment_cost': None,
            'investment_method': None,
            'payback_time': None,
            'penalty_downward': None,
            'penalty_upward': None,
            'storage_capacity': row.get('max_capacity', 0.0),
            'storages_existing': 1.0,
        })

    df = pd.DataFrame(storage_data)
    if df.empty:
        return pd.DataFrame(columns=['availability', 'discount_rate', 'fixed_cost',
                                     'flow_annual', 'flow_scaling_method', 'investment_cost',
                                     'investment_method', 'payback_time', 'penalty_downward',
                                     'penalty_upward', 'storage_capacity', 'storages_existing'
                                     ]).rename_axis('storage')

    df = df.set_index('name')
    df.index.name = 'storage'
    if 'description' in df.columns:
        df = df.drop(columns=['description'])
    return df


def transform_link(griddb: Dict[str, pd.DataFrame],
                   warnings: TransformationWarnings) -> pd.DataFrame:
    """
    Transform transmission_lines + arcs and storage connections to CESM link.

    - transmission_lines with arcs become links between balance nodes
    - storage_units connected to balancing_topologies become storage<->balance links
    """
    if 'balancing_topologies' not in griddb:
        warnings.add_missing("No balancing_topologies for link transformation")
        return pd.DataFrame(columns=['availability', 'capacity', 'discount_rate',
                                     'efficiency', 'fixed_cost', 'investment_cost',
                                     'investment_method', 'links_existing',
                                     'operational_cost', 'payback_time', 'transfer_method'
                                     ]).set_index(pd.MultiIndex.from_tuples([], names=['name', 'node_A', 'node_B']))

    bt = griddb['balancing_topologies']
    bt_id_to_name = dict(zip(bt['id'], bt['name']))

    link_data = []

    # 1. Transform transmission_lines with arcs
    if 'transmission_lines' in griddb and 'arcs' in griddb:
        tl = griddb['transmission_lines']
        arcs = griddb['arcs']

        arc_id_to_nodes = {}
        for _, arc_row in arcs.iterrows():
            from_id = arc_row['from_id']
            to_id = arc_row['to_id']
            from_name = bt_id_to_name.get(from_id)
            to_name = bt_id_to_name.get(to_id)
            if from_name and to_name:
                arc_id_to_nodes[arc_row['id']] = (from_name, to_name)

        for _, row in tl.iterrows():
            arc_id = row.get('arc_id')
            if arc_id not in arc_id_to_nodes:
                continue

            node_a, node_b = arc_id_to_nodes[arc_id]
            # Use transmission line name if available, otherwise generate from nodes
            tl_name = row.get('name')
            if tl_name:
                link_name = tl_name
            else:
                link_name = f"{node_a}__{node_b}"

            link_data.append({
                'name': link_name,
                'node_A': node_a,
                'node_B': node_b,
                'availability': None,
                'capacity': row.get('continuous_rating', 1.0),
                'discount_rate': None,
                'efficiency': 1.0,
                'fixed_cost': None,
                'investment_cost': None,
                'investment_method': None,
                'links_existing': 1.0,
                'operational_cost': None,
                'payback_time': None,
                'transfer_method': 'regular_linear',
            })

    # 2. Transform storage connections to links
    if 'storage_units' in griddb:
        su = griddb['storage_units']

        for _, row in su.iterrows():
            storage_name = row['name']
            bt_id = row.get('balancing_topology')
            bt_name = bt_id_to_name.get(bt_id)

            if not bt_name:
                continue

            link_name = f"{storage_name}__{bt_name}"
            efficiency = row.get('efficiency_up', 1.0) or 1.0

            link_data.append({
                'name': link_name,
                'node_A': storage_name,
                'node_B': bt_name,
                'availability': None,
                'capacity': row.get('rating', row.get('base_power', 1.0)),
                'discount_rate': None,
                'efficiency': efficiency,
                'fixed_cost': None,
                'investment_cost': None,
                'investment_method': None,
                'links_existing': 1.0,
                'operational_cost': None,
                'payback_time': None,
                'transfer_method': 'regular_linear',
            })

    df = pd.DataFrame(link_data)
    if df.empty:
        return pd.DataFrame(columns=['availability', 'capacity', 'discount_rate',
                                     'efficiency', 'fixed_cost', 'investment_cost',
                                     'investment_method', 'links_existing',
                                     'operational_cost', 'payback_time', 'transfer_method'
                                     ]).set_index(pd.MultiIndex.from_tuples([], names=['name', 'node_A', 'node_B']))

    df = df.set_index(['name', 'node_A', 'node_B'])
    return df


def transform_system(warnings: TransformationWarnings) -> pd.DataFrame:
    """Create default system entity."""
    system_data = [{
        'name': 'default',
        'inflation_rate': 0.02,
        'solve_order': None,
    }]
    df = pd.DataFrame(system_data)
    df = df.set_index('name')
    df.index.name = 'system'
    return df


def transform_solve_pattern(warnings: TransformationWarnings) -> pd.DataFrame:
    """Create default solve_pattern entity."""
    return pd.DataFrame(columns=['contains_solve', 'duration', 'next_solve',
                                 'periods_additional_horizon', 'solve_mode',
                                 'start_time', 'time_resolution']).rename_axis('solve_pattern')


def transform_constraint(warnings: TransformationWarnings) -> pd.DataFrame:
    """Create empty constraint table."""
    return pd.DataFrame(columns=['constant', 'sense']).rename_axis('constraint')


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def to_cesm(db_path: str) -> Dict[str, pd.DataFrame]:
    """
    Main transformation function: GridDB SQLite -> CESM DataFrames.

    Args:
        db_path: Path to GridDB SQLite database

    Returns:
        Dictionary of CESM-format DataFrames matching yaml_to_df() output
    """
    warnings = TransformationWarnings()
    cesm = {}

    try:
        # Load GridDB tables
        print("Loading GridDB tables...")
        griddb = load_griddb_tables(db_path)

        # Phase 1: Timeline and periods
        print("Phase 1: Extracting timeline and periods...")
        cesm['timeline'] = transform_timeline(griddb, db_path, warnings)
        cesm['period'] = transform_periods(griddb, db_path, warnings)

        # Phase 2: Core topology
        print("Phase 2: Transforming core topology...")
        cesm['balance'] = transform_balance(griddb, warnings)
        cesm['commodity'] = transform_commodity(griddb, db_path, warnings)
        cesm['group'], cesm['group_entity'] = transform_groups(griddb, warnings)

        # Phase 3: Units
        print("Phase 3: Transforming units...")
        cesm['unit'] = transform_unit(griddb, db_path, warnings)
        cesm['unit_to_node'] = transform_unit_to_node(griddb, db_path, warnings)
        cesm['node_to_unit'] = transform_node_to_unit(griddb, cesm['commodity'], warnings)

        # Phase 4: Storage and links
        print("Phase 4: Transforming storage and links...")
        cesm['storage'] = transform_storage(griddb, warnings)
        cesm['link'] = transform_link(griddb, warnings)

        # Phase 5: Time series
        print("Phase 5: Transforming time series...")
        cesm['balance.ts.flow_profile'] = transform_balance_flow_profile(
            griddb, db_path, cesm['timeline'], warnings)
        cesm['unit_to_node.ts.profile_limit_upper'] = transform_unit_to_node_profile(
            griddb, db_path, cesm['timeline'], warnings)

        # Phase 6: System and solve entities
        print("Phase 6: Creating system entities...")
        cesm['system'] = transform_system(warnings)
        cesm['solve_pattern'] = transform_solve_pattern(warnings)
        cesm['constraint'] = transform_constraint(warnings)

        # Report warnings
        warnings.report()

        print(f"\nTransformation complete. Created {len(cesm)} CESM DataFrames.")

    except Exception as e:
        print(f"Critical error during transformation: {e}")
        import traceback
        traceback.print_exc()

    return cesm
