"""
Transformer for converting source energy system data to target grid database format.

This module converts dict of source dataframes to target dataframes matching SQL schema.
Uses vectorized operations and multi-index alignment for efficiency.

v0.3.0 Changes:
- generation_units table SPLIT into thermal_generators, renewable_generators, hydro_generators
- Column rename: prime_mover -> prime_mover_type in all tables
- storage_units restructured with JSON fields (efficiency, storage_level_limits, etc.)
- New storage_technology_types lookup table
- hydro_reservoir -> hydro_reservoirs (plural) with expanded schema
- supply_technologies: area and balancing_topology changed from TEXT to INTEGER FK
- operational_data is now a VIEW (not a table) - do NOT insert into it
- entity_types has new is_topology column
- arcs.from_id and arcs.to_id are now NOT NULL
- Entity table entity_table field uses new table names (thermal_generators, etc.)
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from linkml_runtime.utils.schemaview import SchemaView

# =============================================================================
# v0.3.0 MAPPINGS
# =============================================================================

# Map CESM fuel commodity names to standardized fuel names
FUEL_NAME_MAP = {
    'coal': 'COAL',
    'natural_gas': 'NATURAL_GAS',
    'gas': 'GAS',
    'oil': 'OIL',
    'nuclear': 'NUCLEAR',
    'hydro': 'HYDRO',
    'wind': 'WIND',
    'solar': 'SOLAR',
    'biomass': 'BIOMASS',
    'geothermal': 'GEOTHERMAL',
    'waste': 'WASTE',
    'distillate_fuel_oil': 'DISTILLATE_FUEL_OIL',
    'residual_fuel_oil': 'RESIDUAL_FUEL_OIL',
    'other': 'OTHER',
}

# EIA Prime Mover codes for v0.2.0
EIA_PRIME_MOVERS = {
    'BA': 'Battery Energy Storage',
    'BT': 'Binary Cycle Turbine',
    'CA': 'Compressed Air Energy Storage',
    'CC': 'Combined Cycle',
    'CE': 'Reciprocating Engine',
    'CP': 'Concentrated Solar Power',
    'CS': 'Combined Cycle Steam',
    'CT': 'Combustion (Gas) Turbine',
    'ES': 'Energy Storage',
    'FC': 'Fuel Cell',
    'FW': 'Flywheel Energy Storage',
    'GT': 'Gas Turbine',
    'HA': 'Hydro Francis',
    'HB': 'Hydro Bulb',
    'HK': 'Hydro Kaplan',
    'HY': 'Hydro',
    'IC': 'Internal Combustion Engine',
    'OT': 'Other',
    'PS': 'Pumped Storage',
    'PVe': 'Photovoltaic',
    'ST': 'Steam Turbine',
    'WS': 'Wind Offshore',
    'WT': 'Wind Onshore',
}

# =============================================================================
# v0.3.0 PRIME MOVER CLASSIFICATION SETS
# =============================================================================

THERMAL_PRIME_MOVERS = {'ST', 'CT', 'CC', 'CS', 'IC', 'GT', 'FC', 'CE', 'BT', 'CP'}
RENEWABLE_PRIME_MOVERS = {'PVe', 'WT', 'WS', 'OT'}
HYDRO_PRIME_MOVERS = {'HY', 'HA', 'HB', 'HK'}

# Storage technology type mapping from prime mover code
STORAGE_TECHNOLOGY_MAP = {
    'BA': 'Battery',
    'PS': 'Pumped Hydro',
    'CA': 'Compressed Air',
    'FW': 'Flywheel',
    'ES': 'Other',
}

# =============================================================================
# CLASSIFICATION THRESHOLDS (v0.2.0 - parameter-based decision tree)
# =============================================================================

# Storage classification thresholds
BATTERY_MIN_EFFICIENCY = 0.85      # Efficiency >= 0.85 -> Battery or Flywheel
FLYWHEEL_MAX_DURATION_HOURS = 0.5  # Duration < 30 minutes -> Flywheel
PUMPED_HYDRO_MIN_EFFICIENCY = 0.70 # Efficiency 0.70-0.85 -> Pumped Storage

# Thermal classification thresholds (gas-fired)
CCGT_CERTAIN_MIN_EFFICIENCY = 0.54   # efficiency >= 0.54 -> CC with certainty
CCGT_UNCERTAIN_MIN_EFFICIENCY = 0.50 # efficiency 0.50-0.54 -> CC (uncertain, could be IC)
CT_MAX_EFFICIENCY = 0.45             # efficiency < 0.45 -> CT

# Oil-fired thresholds
IC_MIN_EFFICIENCY = 0.45  # efficiency >= 0.45 for oil -> IC

# Solar detection threshold (FFT-based)
SOLAR_DAILY_PATTERN_THRESHOLD = 0.5  # daily_pattern_ratio > 0.5 -> Solar

# =============================================================================
# v0.3.0 DEFAULT OPERATION COST JSON STRUCTURES
# =============================================================================

DEFAULT_THERMAL_OPERATION_COST = {
    "cost_type": "THERMAL",
    "fixed": 0,
    "shut_down": 0,
    "start_up": 0,
    "variable": {
        "variable_cost_type": "COST",
        "power_units": "NATURAL_UNITS",
        "value_curve": {
            "curve_type": "INPUT_OUTPUT",
            "function_data": {
                "function_type": "LINEAR",
                "proportional_term": 0,
                "constant_term": 0,
            },
        },
        "vom_cost": {
            "curve_type": "INPUT_OUTPUT",
            "function_data": {
                "function_type": "LINEAR",
                "proportional_term": 0,
                "constant_term": 0,
            },
        },
    },
}

DEFAULT_RENEWABLE_OPERATION_COST = {
    "cost_type": "RENEWABLE",
    "fixed": 0,
    "variable": {
        "variable_cost_type": "COST",
        "power_units": "NATURAL_UNITS",
        "value_curve": {
            "curve_type": "INPUT_OUTPUT",
            "function_data": {
                "function_type": "LINEAR",
                "proportional_term": 0,
                "constant_term": 0,
            },
        },
        "vom_cost": {
            "curve_type": "INPUT_OUTPUT",
            "function_data": {
                "function_type": "LINEAR",
                "proportional_term": 0,
                "constant_term": 0,
            },
        },
    },
    "curtailment_cost": {
        "variable_cost_type": "COST",
        "power_units": "NATURAL_UNITS",
        "value_curve": {
            "curve_type": "INPUT_OUTPUT",
            "function_data": {
                "function_type": "LINEAR",
                "proportional_term": 0,
                "constant_term": 0,
            },
        },
        "vom_cost": {
            "curve_type": "INPUT_OUTPUT",
            "function_data": {
                "function_type": "LINEAR",
                "proportional_term": 0,
                "constant_term": 0,
            },
        },
    },
}

DEFAULT_HYDRO_OPERATION_COST = {
    "cost_type": "HYDRO_GEN",
    "fixed": 0.0,
    "variable": {
        "variable_cost_type": "COST",
        "power_units": "NATURAL_UNITS",
        "value_curve": {
            "curve_type": "INPUT_OUTPUT",
            "function_data": {
                "function_type": "LINEAR",
                "proportional_term": 0,
                "constant_term": 0,
            },
        },
        "vom_cost": {
            "curve_type": "INPUT_OUTPUT",
            "function_data": {
                "function_type": "LINEAR",
                "proportional_term": 0,
                "constant_term": 0,
            },
        },
    },
}


@dataclass
class ClassificationResult:
    """Result of a technology classification."""
    eia_code: str
    is_uncertain: bool
    reason: str


@dataclass
class ClassificationReport:
    """Collector for classification uncertainties."""
    storage_classifications: List[Tuple[str, ClassificationResult]] = field(default_factory=list)
    thermal_classifications: List[Tuple[str, ClassificationResult]] = field(default_factory=list)
    renewable_classifications: List[Tuple[str, ClassificationResult]] = field(default_factory=list)

    def add_storage(self, name: str, result: ClassificationResult):
        self.storage_classifications.append((name, result))

    def add_thermal(self, name: str, result: ClassificationResult):
        self.thermal_classifications.append((name, result))

    def add_renewable(self, name: str, result: ClassificationResult):
        self.renewable_classifications.append((name, result))

    def report(self):
        """Print classification uncertainty report."""
        uncertain_storage = [(n, r) for n, r in self.storage_classifications if r.is_uncertain]
        uncertain_thermal = [(n, r) for n, r in self.thermal_classifications if r.is_uncertain]
        uncertain_renewable = [(n, r) for n, r in self.renewable_classifications if r.is_uncertain]

        if not (uncertain_storage or uncertain_thermal or uncertain_renewable):
            print("\n" + "="*80)
            print("CLASSIFICATION REPORT: All classifications certain")
            print("="*80)
            return

        print("\n" + "="*80)
        print("CLASSIFICATION UNCERTAINTY REPORT")
        print("="*80)

        if uncertain_storage:
            print("\nSTORAGE UNITS:")
            for name, result in uncertain_storage:
                print(f"  * {name}: {result.eia_code} - {result.reason}")

        if uncertain_thermal:
            print("\nTHERMAL UNITS:")
            for name, result in uncertain_thermal:
                print(f"  * {name}: {result.eia_code} - {result.reason}")

        if uncertain_renewable:
            print("\nRENEWABLE UNITS:")
            for name, result in uncertain_renewable:
                print(f"  * {name}: {result.eia_code} - {result.reason}")

    def summary(self) -> str:
        """Return summary counts."""
        total = (len(self.storage_classifications) +
                 len(self.thermal_classifications) +
                 len(self.renewable_classifications))
        uncertain = (sum(1 for _, r in self.storage_classifications if r.is_uncertain) +
                     sum(1 for _, r in self.thermal_classifications if r.is_uncertain) +
                     sum(1 for _, r in self.renewable_classifications if r.is_uncertain))
        return f"Classifications: {total} total, {uncertain} uncertain"


def analyze_daily_pattern(profile_series: pd.Series) -> float:
    """
    Analyze time series for daily (24-hour) pattern using FFT.

    Solar profiles have a strong 24-hour cycle (zero at night, peak at noon).
    Wind profiles are more stochastic without a strong daily cycle.

    Args:
        profile_series: Time series of capacity factors

    Returns:
        daily_pattern_ratio: Ratio of 24-hour component magnitude to total variance.
                            Higher values (>0.5) indicate solar-like daily pattern.
    """
    if profile_series is None or len(profile_series) < 24:
        return 0.0

    try:
        # Convert to numpy array and remove NaN
        values = profile_series.dropna().values.astype(float)
        if len(values) < 24:
            return 0.0

        # Compute FFT
        fft_result = np.fft.fft(values)
        magnitudes = np.abs(fft_result)

        # Find the frequency bin corresponding to 24-hour period
        # Assuming hourly data: frequency index = len(values) / 24
        n = len(values)
        freq_24h_idx = round(n / 24)

        if freq_24h_idx == 0 or freq_24h_idx >= n // 2:
            return 0.0

        # Get magnitude at 24-hour frequency (and its harmonic)
        mag_24h = magnitudes[freq_24h_idx]
        # Also check 12-hour harmonic (common in solar)
        freq_12h_idx = round(n / 12)
        mag_12h = magnitudes[freq_12h_idx] if freq_12h_idx < n // 2 else 0.0

        # Total variance (excluding DC component)
        total_variance = np.sum(magnitudes[1:n//2])

        if total_variance == 0:
            return 0.0

        # Ratio of daily pattern to total
        daily_pattern_ratio = (mag_24h + mag_12h * 0.5) / total_variance

        return float(daily_pattern_ratio)

    except Exception:
        return 0.0


def check_has_inflow(source: Dict[str, pd.DataFrame], balance_name: str) -> bool:
    """
    Check if a balance node has natural inflow (positive sum of flow_profile).

    In CESM, natural inflow (like water for hydro) is modeled as positive values
    in the balance.flow_profile time series.

    Args:
        source: Dictionary of source dataframes
        balance_name: Name of the balance node to check

    Returns:
        True if the balance node has positive total inflow
    """
    # Check for flow_profile time series
    flow_profile_key = 'balance.ts.flow_profile'
    if flow_profile_key not in source:
        return False

    flow_df = source[flow_profile_key]

    # Check if this balance node has a column
    if balance_name not in flow_df.columns:
        return False

    # Sum the flow profile (positive = inflow, negative = outflow)
    total_flow = flow_df[balance_name].sum()

    return total_flow > 0


def classify_storage_prime_mover(
    storage_capacity: float,
    link_capacity: float,
    link_efficiency: float,
    has_inflow: bool
) -> ClassificationResult:
    """
    Classify storage technology based on technical parameters.

    Decision tree:
    1. Duration < 30 min + efficiency >= 0.85 -> Flywheel (FW)
    2. Efficiency >= 0.85 -> Battery (BA)
    3. Efficiency 0.70-0.85 -> Pumped Storage (PS)
    4. Efficiency < 0.70 -> Generic Energy Storage (ES)

    Args:
        storage_capacity: Energy capacity in MWh
        link_capacity: Power rating in MW
        link_efficiency: Round-trip efficiency (0-1)
        has_inflow: Whether connected balance node has natural inflow

    Returns:
        ClassificationResult with EIA code, uncertainty flag, and reason
    """
    # Calculate storage duration (hours)
    if link_capacity and link_capacity > 0:
        duration_hours = storage_capacity / link_capacity
    else:
        duration_hours = None

    # Decision tree
    if duration_hours is not None and duration_hours < FLYWHEEL_MAX_DURATION_HOURS:
        if link_efficiency >= BATTERY_MIN_EFFICIENCY:
            return ClassificationResult(
                'FW', False,
                f"duration={duration_hours:.2f}h < 0.5h, efficiency={link_efficiency:.2f} >= 0.85"
            )
        else:
            return ClassificationResult(
                'BA', True,
                f"short duration ({duration_hours:.2f}h) but low efficiency ({link_efficiency:.2f})"
            )

    if link_efficiency >= BATTERY_MIN_EFFICIENCY:
        return ClassificationResult('BA', False, f"efficiency={link_efficiency:.2f} >= 0.85")

    if link_efficiency >= PUMPED_HYDRO_MIN_EFFICIENCY:
        if has_inflow:
            return ClassificationResult('PS', False, f"efficiency={link_efficiency:.2f} in 0.70-0.85 range, has inflow")
        else:
            return ClassificationResult(
                'PS', True,
                f"efficiency={link_efficiency:.2f} in 0.70-0.85 range, no inflow detected"
            )

    # Low efficiency - generic storage
    return ClassificationResult('ES', True, f"low efficiency={link_efficiency:.2f} < 0.70")


def classify_thermal_prime_mover(fuel_type: str, efficiency: float) -> ClassificationResult:
    """
    Classify thermal generation technology based on fuel type and efficiency.

    Decision tree:
    - Coal/nuclear/biomass/waste -> Steam Turbine (ST)
    - Gas: efficiency >= 0.54 -> Combined Cycle (CC)
    - Gas: efficiency 0.50-0.54 -> CC (uncertain)
    - Gas: efficiency < 0.45 -> Combustion Turbine (CT)
    - Oil: efficiency >= 0.45 -> Internal Combustion (IC)
    - Oil: efficiency < 0.45 -> Combustion Turbine (CT)

    Args:
        fuel_type: Fuel commodity name (lowercase)
        efficiency: Unit efficiency (0-1)

    Returns:
        ClassificationResult with EIA code, uncertainty flag, and reason
    """
    fuel_lower = fuel_type.lower() if fuel_type else ''

    # Steam turbine fuels
    if any(f in fuel_lower for f in ['coal', 'nuclear', 'uranium']):
        return ClassificationResult('ST', False, f"fuel={fuel_type} -> Steam Turbine")

    if any(f in fuel_lower for f in ['biomass', 'waste', 'wood']):
        return ClassificationResult('ST', False, f"fuel={fuel_type} -> Steam Turbine")

    # Gas-fired classification based on efficiency
    if any(f in fuel_lower for f in ['gas', 'natural_gas', 'lng', 'methane']):
        if efficiency >= CCGT_CERTAIN_MIN_EFFICIENCY:
            return ClassificationResult('CC', False, f"gas-fired, efficiency={efficiency:.2f} >= 0.54 -> Combined Cycle")
        elif efficiency >= CCGT_UNCERTAIN_MIN_EFFICIENCY:
            return ClassificationResult(
                'CS', True,
                f"gas-fired, efficiency={efficiency:.2f} in 0.50-0.54 range (could be IC)"
            )
        elif efficiency < CT_MAX_EFFICIENCY:
            return ClassificationResult(
                'CT', False,
                f"gas-fired, efficiency={efficiency:.2f} < 0.45 -> Combustion Turbine"
            )
        else:
            return ClassificationResult('CT', True, f"gas-fired, efficiency={efficiency:.2f} in boundary 0.45-0.50")

    # Oil-fired classification
    if any(f in fuel_lower for f in ['oil', 'diesel', 'distillate', 'residual', 'petroleum']):
        if efficiency >= IC_MIN_EFFICIENCY:
            return ClassificationResult(
                'IC', False,
                f"oil-fired, efficiency={efficiency:.2f} >= 0.45 -> Internal Combustion"
            )
        else:
            return ClassificationResult(
                'CT', False,
                f"oil-fired, efficiency={efficiency:.2f} < 0.45 -> Combustion Turbine"
            )

    # Unknown fuel - classify by efficiency
    if efficiency >= CCGT_CERTAIN_MIN_EFFICIENCY:
        return ClassificationResult('CC', True, f"unknown fuel '{fuel_type}', efficiency={efficiency:.2f} >= 0.54")
    elif efficiency < CT_MAX_EFFICIENCY:
        return ClassificationResult('CT', True, f"unknown fuel '{fuel_type}', efficiency={efficiency:.2f} < 0.45")
    else:
        return ClassificationResult(
            'ST', True,
            f"unknown fuel '{fuel_type}', efficiency={efficiency:.2f} -> default Steam Turbine"
        )


def classify_renewable_prime_mover(
    has_profile: bool,
    has_inflow: bool,
    profile_data: pd.Series = None
) -> ClassificationResult:
    """
    Classify renewable generation technology based on parameters.

    Decision tree:
    - Has inflow -> Hydro (HY)
    - Has profile with strong daily pattern -> Photovoltaic (PVe)
    - Has profile without daily pattern -> Wind (WT)
    - No profile, no inflow -> Other (OT)

    Args:
        has_profile: Whether unit has profile_limit_upper time series
        has_inflow: Whether connected balance node has natural inflow
        profile_data: Time series data for pattern analysis

    Returns:
        ClassificationResult with EIA code, uncertainty flag, and reason
    """
    # Hydro: has natural water inflow
    if has_inflow:
        return ClassificationResult('HY', False, "has natural inflow -> Hydro")

    # Variable renewables with capacity factor profiles
    if has_profile:
        if profile_data is not None and len(profile_data) >= 24:
            daily_ratio = analyze_daily_pattern(profile_data)

            if daily_ratio > SOLAR_DAILY_PATTERN_THRESHOLD:
                return ClassificationResult('PVe', False, f"daily pattern ratio={daily_ratio:.2f} > 0.5 -> Solar")
            elif daily_ratio > SOLAR_DAILY_PATTERN_THRESHOLD * 0.8:  # Near threshold
                return ClassificationResult(
                    'WT', True,
                    f"daily pattern ratio={daily_ratio:.2f} near threshold -> Wind (uncertain)"
                )
            else:
                return ClassificationResult('WT', False, f"daily pattern ratio={daily_ratio:.2f} < 0.5 -> Wind")
        else:
            # Has profile but no data for analysis - assume wind (more common)
            return ClassificationResult('WT', True, "has profile but insufficient data for pattern analysis")

    # No profile, no inflow - could be geothermal or other baseload renewable
    return ClassificationResult('OT', True, "no profile, no inflow -> Other (geothermal? baseload?)")


def map_fuel_name(cesm_fuel_name: str) -> str:
    """Map CESM fuel name to standardized fuel name."""
    if cesm_fuel_name is None:
        return None
    lower_name = cesm_fuel_name.lower().replace(' ', '_').replace('-', '_')
    return FUEL_NAME_MAP.get(lower_name, cesm_fuel_name.upper())


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
        Value at df.loc[idx, col] or default. Returns default for NA/NaN values.
    """
    if col in df.columns:
        val = df.loc[idx, col]
        if pd.isna(val):
            return default
        return val
    return default


@dataclass
class TransformationErrors:
    """Collector for transformation errors and warnings.

    Args:
        strict: If True, report all errors and warnings. If False, suppress
                warnings about missing optional data (relaxed mode for incremental updates).
    """
    errors: List[str] = field(default_factory=list)
    missing_transformations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    strict: bool = field(default=True)

    def add_error(self, message: str):
        self.errors.append(message)

    def add_missing(self, message: str):
        self.missing_transformations.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def report(self):
        """Print all collected errors and warnings.

        In non-strict mode, warnings and missing transformations are suppressed
        since missing data may already exist in the target database.
        """
        if self.errors:
            print("\n" + "="*80)
            print("TRANSFORMATION ERRORS:")
            print("="*80)
            for err in self.errors:
                print(f"  * {err}")

        # In non-strict mode, suppress warnings about missing data
        if self.strict and self.warnings:
            print("\n" + "="*80)
            print("TRANSFORMATION WARNINGS:")
            print("="*80)
            for warn in self.warnings:
                print(f"  * {warn}")
        elif not self.strict and self.warnings:
            print(f"\n(Suppressed {len(self.warnings)} warnings in relaxed validation mode)")

        if self.strict and self.missing_transformations:
            print("\n" + "="*80)
            print("MISSING TRANSFORMATIONS (source->target):")
            print("="*80)
            for miss in self.missing_transformations:
                print(f"  * {miss}")
        elif not self.strict and self.missing_transformations:
            print(
                f"(Suppressed {len(self.missing_transformations)}"
                " missing transformation notices in relaxed validation mode)"
            )


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


def get_generator_table_for_prime_mover(prime_mover: str) -> str:
    """
    Determine the target generator table based on prime mover code.

    v0.3.0: generation_units is split into three tables.

    Args:
        prime_mover: EIA prime mover code

    Returns:
        Target table name ('thermal_generators', 'renewable_generators', or 'hydro_generators')
    """
    if prime_mover in THERMAL_PRIME_MOVERS:
        return 'thermal_generators'
    elif prime_mover in RENEWABLE_PRIME_MOVERS:
        return 'renewable_generators'
    elif prime_mover in HYDRO_PRIME_MOVERS:
        return 'hydro_generators'
    else:
        # Default to thermal for unknown prime movers
        return 'thermal_generators'


def get_entity_type_for_prime_mover(prime_mover: str) -> str:
    """
    Determine the entity type based on prime mover code.

    v0.3.0: Maps prime mover to appropriate entity type.

    Args:
        prime_mover: EIA prime mover code

    Returns:
        Entity type string
    """
    if prime_mover in THERMAL_PRIME_MOVERS:
        return 'ThermalStandard'
    elif prime_mover in RENEWABLE_PRIME_MOVERS:
        return 'RenewableDispatch'
    elif prime_mover in HYDRO_PRIME_MOVERS:
        return 'HydroDispatch'
    else:
        return 'ThermalStandard'


def determine_prime_mover(
    source: Dict[str, pd.DataFrame],
    unit_name: str,
    classification_report: ClassificationReport = None
) -> str:
    """
    Determine prime mover type for a unit using EIA codes based on technical parameters.

    v0.2.0: Uses parameter-based decision tree instead of name-based heuristics.

    Classification approach:
    1. Check if unit has fuel input -> Thermal classification (by fuel type + efficiency)
    2. Check for natural inflow -> Hydro
    3. Check for capacity factor profile -> Solar (FFT daily pattern) or Wind
    4. Default -> Other

    Args:
        source: Dictionary of source dataframes
        unit_name: Name of the unit to classify
        classification_report: Optional report to track uncertain classifications

    Returns:
        EIA prime mover code (e.g., 'ST', 'CT', 'CC', 'PVe', 'WT', 'HY')
    """
    fuel_type = None
    efficiency = 1.0  # Default efficiency

    # Get unit efficiency
    if 'unit' in source:
        units = source['unit']
        if unit_name in units.index:
            efficiency = safe_get(units, unit_name, 'efficiency', 1.0)
            if efficiency is None:
                efficiency = 1.0

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
                    fuel_type = input_node
                    break

    # THERMAL: Has fuel input - use efficiency-based classification
    if fuel_type:
        result = classify_thermal_prime_mover(fuel_type, efficiency)
        if classification_report:
            classification_report.add_thermal(unit_name, result)
        return result.eia_code

    # THERMAL WITHOUT FUEL: Low efficiency (< 95%) without fuel input indicates
    # a heat engine (thermal unit) whose fuel isn't explicitly modeled.
    # Renewables have ~100% conversion efficiency since their "fuel" isn't tracked.
    # Threshold 95% allows for small losses in renewable systems.
    THERMAL_EFFICIENCY_THRESHOLD = 95.0  # percentage
    if efficiency < THERMAL_EFFICIENCY_THRESHOLD:
        result = ClassificationResult(
            'ST', False,
            f"no fuel input but efficiency={efficiency:.1f}% < {THERMAL_EFFICIENCY_THRESHOLD}% "
            f"indicates thermal (heat engine)"
        )
        if classification_report:
            classification_report.add_thermal(unit_name, result)
        return result.eia_code

    # RENEWABLE: Check for inflow and profile
    # Find connected balance node(s) via unit_to_node
    connected_balance_nodes = []
    if 'unit_to_node' in source:
        unit_to_node = source['unit_to_node']
        unit_outputs = unit_to_node[unit_to_node.index.get_level_values('source') == unit_name]
        for idx in unit_outputs.index:
            if isinstance(idx, tuple) and len(idx) >= 3:
                _, _, sink_node = idx
                connected_balance_nodes.append(sink_node)

    # Check if any connected node has inflow
    has_inflow = False
    for balance_name in connected_balance_nodes:
        if check_has_inflow(source, balance_name):
            has_inflow = True
            break

    # Check if unit has profile_limit_upper time series
    has_profile = False
    profile_data = None
    for key in source.keys():
        if key.startswith('unit_to_node.ts.profile_limit_'):
            ts_df = source[key]
            if hasattr(ts_df, 'columns'):
                if isinstance(ts_df.columns, pd.MultiIndex):
                    # MultiIndex: check source level
                    if unit_name in ts_df.columns.get_level_values(1):
                        has_profile = True
                        # Extract the profile data for this unit
                        try:
                            # Get all columns where source == unit_name
                            mask = ts_df.columns.get_level_values(1) == unit_name
                            profile_data = ts_df.iloc[:, mask].iloc[:, 0]
                        except Exception:
                            profile_data = None
                        break
                else:
                    # Simple index
                    if unit_name in ts_df.columns:
                        has_profile = True
                        profile_data = ts_df[unit_name]
                        break

    # Classify renewable using decision tree
    result = classify_renewable_prime_mover(has_profile, has_inflow, profile_data)
    if classification_report:
        classification_report.add_renewable(unit_name, result)
    return result.eia_code


def determine_storage_prime_mover(
    source: Dict[str, pd.DataFrame],
    storage_name: str,
    classification_report: ClassificationReport = None
) -> str:
    """
    Determine prime mover type for a storage unit using EIA codes based on technical parameters.

    v0.2.0: Uses parameter-based decision tree instead of name-based heuristics.

    Classification approach:
    1. Duration < 30 min + high efficiency -> Flywheel
    2. High efficiency (>=0.85) -> Battery
    3. Medium efficiency (0.70-0.85) -> Pumped Storage
    4. Low efficiency -> Generic Energy Storage

    Args:
        source: Dictionary of source dataframes
        storage_name: Name of the storage unit to classify
        classification_report: Optional report to track uncertain classifications

    Returns:
        EIA prime mover code (e.g., 'BA', 'PS', 'CA', 'FW', 'ES')
    """
    storage_capacity = 0.0
    link_capacity = 0.0
    link_efficiency = 0.9  # Default

    # Get storage capacity from storage table
    if 'storage' in source:
        storages = source['storage']
        if storage_name in storages.index:
            storage_capacity = safe_get(storages, storage_name, 'storage_capacity', 0.0)
            if storage_capacity is None:
                storage_capacity = 0.0

    # Find link connecting storage to balance node and get capacity/efficiency
    if 'link' in source:
        links = source['link']
        for link_idx in links.index:
            if isinstance(link_idx, tuple) and len(link_idx) == 3:
                _, node_a, node_b = link_idx
                if node_a == storage_name or node_b == storage_name:
                    link_capacity = safe_get(links, link_idx, 'capacity', 0.0)
                    link_efficiency = safe_get(links, link_idx, 'efficiency', 0.9)
                    if link_capacity is None:
                        link_capacity = 0.0
                    if link_efficiency is None:
                        link_efficiency = 0.9
                    break

    # Check for inflow to connected balance node
    has_inflow = False
    if 'link' in source:
        links = source['link']
        for link_idx in links.index:
            if isinstance(link_idx, tuple) and len(link_idx) == 3:
                _, node_a, node_b = link_idx
                if node_a == storage_name:
                    has_inflow = check_has_inflow(source, node_b)
                elif node_b == storage_name:
                    has_inflow = check_has_inflow(source, node_a)
                if has_inflow:
                    break

    # Classify using decision tree
    result = classify_storage_prime_mover(storage_capacity, link_capacity, link_efficiency, has_inflow)
    if classification_report:
        classification_report.add_storage(storage_name, result)
    return result.eia_code


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
                                errors: TransformationErrors,
                                classification_report: ClassificationReport = None) -> pd.DataFrame:
    """Create entities table and populate ID mappings.

    v0.3.0: Routes generators to thermal_generators, renewable_generators, or
    hydro_generators based on prime mover classification. Uses is_topology-aware
    entity types.
    """
    entities_data = []

    # Helper to add entity
    def add_entity(source_table: str, entity_type: str, key: str, name: str = None):
        # Store name and entity_type in IDGenerator for later TEXT FK lookups and type resolution
        entity_name = name or create_composite_name(key)
        entity_id = id_gen.get_or_create(source_table, key, entity_name, entity_type)
        entities_data.append({
            'id': entity_id,
            'entity_table': source_table,
            'entity_type': entity_type
        })
        return entity_id

    # 1. Prime mover types (will be created by transform_prime_mover_types)

    # 2. Fuels from commodities
    if 'commodity' in source:
        commodities = source['commodity']
        fuels = commodities[commodities['commodity_type'] == 'fuel']
        for commodity_name in fuels.index:
            add_entity('fuels', 'ThermalStandard', commodity_name, commodity_name)

    # 3. Planning regions from groups with type 'node'
    #    v0.3.0: entity_type = 'Area' with is_topology=TRUE
    if 'group' in source:
        groups = source['group']
        planning_groups = groups[groups['group_type'] == 'node']
        for group_name in planning_groups.index:
            add_entity('planning_regions', 'Area', group_name, group_name)

    # 4. Balancing topologies from balance nodes in power_grid
    #    v0.3.0: entity_type = 'LoadZone' with is_topology=TRUE
    power_grid_nodes = identify_power_grid_nodes(source, errors)
    if 'balance' in source:
        for balance_name in source['balance'].index:
            if balance_name in power_grid_nodes:
                add_entity('balancing_topologies', 'LoadZone', balance_name, balance_name)

    # 5. Generation units from units with units_existing
    #    v0.3.0: Route to thermal_generators, renewable_generators, or hydro_generators
    if 'unit' in source:
        units = source['unit']
        existing_units = units[safe_filter(units, 'units_existing') & (units['units_existing'] > 0)]
        for unit_name in existing_units.index:
            # Determine prime_mover classification
            prime_mover = determine_prime_mover(source, unit_name)
            # Route to appropriate table based on prime mover
            gen_table = get_generator_table_for_prime_mover(prime_mover)
            entity_type = get_entity_type_for_prime_mover(prime_mover)
            add_entity(gen_table, entity_type, unit_name, unit_name)

    # 6. Supply technologies (investment candidates) from units with investment data
    if 'unit' in source:
        units = source['unit']
        investment_units = units[
            safe_filter(units, 'investment_cost') &
            safe_filter(units, 'discount_rate')
        ]
        for unit_name in investment_units.index:
            key = f"supply_tech_{unit_name}_candidate"
            candidate_name = f"{unit_name}_candidate"
            prime_mover = determine_prime_mover(source, unit_name)
            entity_type = get_entity_type_for_prime_mover(prime_mover)
            add_entity('supply_technologies', entity_type, key, candidate_name)

    # 7. Arcs from links
    #    Only create arc entities when BOTH endpoints are power_grid nodes
    if 'link' in source:
        links = source['link']
        arc_pairs = set()
        for idx in links.index:
            if isinstance(idx, tuple) and len(idx) == 3:
                _, node_a, node_b = idx
            else:
                errors.add_warning(f"Unexpected link index format (expected 3-tuple): {idx}")
                continue

            # Only create arc if both endpoints are power_grid nodes
            if node_a in power_grid_nodes and node_b in power_grid_nodes:
                if node_b and (node_a, node_b) not in arc_pairs and (node_b, node_a) not in arc_pairs:
                    arc_pairs.add((node_a, node_b))
                    arc_key = f"{node_a}_{node_b}"
                    add_entity('arcs', 'Arc', arc_key, arc_key)

    # 8. Transmission lines from links with links_existing
    #    Only create transmission line entities when a valid arc exists
    if 'link' in source:
        links = source['link']
        existing_links = links[safe_filter(links, 'links_existing') & (links['links_existing'] > 0)]
        for idx in existing_links.index:
            if isinstance(idx, tuple) and len(idx) == 3:
                _, node_a, node_b = idx
                # Check if arc exists for this link
                arc_key_fwd = f"{node_a}_{node_b}"
                arc_key_rev = f"{node_b}_{node_a}"
                if (('arcs', arc_key_fwd) not in id_gen.entity_map and
                        ('arcs', arc_key_rev) not in id_gen.entity_map):
                    continue  # Skip - no valid arc
            link_key = create_composite_name(idx)
            add_entity('transmission_lines', 'Line', link_key, link_key)

    # 8b. Transport technologies (investment candidates) from links with investment data
    #     Only create transport technology entities when a valid arc exists
    if 'link' in source:
        links = source['link']
        investment_links = links[
            safe_filter(links, 'investment_cost') &
            safe_filter(links, 'discount_rate')
        ]
        for idx in investment_links.index:
            if isinstance(idx, tuple) and len(idx) == 3:
                _, node_a, node_b = idx
                # Check if arc exists for this link
                arc_key_fwd = f"{node_a}_{node_b}"
                arc_key_rev = f"{node_b}_{node_a}"
                if (('arcs', arc_key_fwd) not in id_gen.entity_map and
                        ('arcs', arc_key_rev) not in id_gen.entity_map):
                    continue  # Skip - no valid arc
            link_key = create_composite_name(idx)
            candidate_key = f"{link_key}_candidate"
            add_entity('transport_technologies', 'Line', candidate_key, candidate_key)

    # 9. Storage units from storage with storages_existing
    if 'storage' in source:
        storages = source['storage']
        existing_storages = storages[safe_filter(storages, 'storages_existing') & (storages['storages_existing'] > 0)]
        for storage_name in existing_storages.index:
            add_entity('storage_units', 'EnergyReservoirStorage', storage_name, storage_name)

    # 10. Storage technologies removed in v0.2.0 (table not in target schema)

    # 11. Transmission interchanges from groups with link members
    if 'group' in source and 'group_entity' in source:
        group_entities = source['group_entity']
        for group_name in group_entities.index.get_level_values('group').unique():
            try:
                group_entities.xs(group_name, level='group')
                key = f"interchange_{group_name}"
                add_entity('transmission_interchanges', 'AreaInterchange', key, key)
            except KeyError:
                continue

    if not entities_data:
        errors.add_error("No entities created from source data")
        return pd.DataFrame()

    return pd.DataFrame(entities_data)


def transform_entity_types(id_gen: IDGenerator, errors: TransformationErrors) -> pd.DataFrame:
    """
    Create entity_types table.

    v0.3.0: Includes is_topology column. Topology types (LoadZone, Area, ACBus)
    have is_topology=TRUE.
    """
    # Collect all unique entity types used
    entity_types_used = set()
    for entity_id, entity_type in id_gen.type_map.items():
        entity_types_used.add(entity_type)

    # Define which entity types are topology types
    TOPOLOGY_TYPES = {'LoadZone', 'Area', 'ACBus'}

    entity_types_data = []
    for et in sorted(entity_types_used):
        entity_types_data.append({
            'name': et,
            'is_topology': et in TOPOLOGY_TYPES
        })

    return pd.DataFrame(entity_types_data)


def transform_prime_mover_types(id_gen: IDGenerator, errors: TransformationErrors) -> pd.DataFrame:
    """
    Create prime_mover_types table with EIA prime mover classifications.

    Populates with all EIA prime mover codes from the EIA_PRIME_MOVERS dict.
    """
    data = []
    for i, (code, description) in enumerate(sorted(EIA_PRIME_MOVERS.items()), start=1):
        data.append({'id': i, 'name': code, 'description': description})
    return pd.DataFrame(data)


def transform_fuels(source: Dict[str, pd.DataFrame],
                     id_gen: IDGenerator,
                     errors: TransformationErrors) -> pd.DataFrame:
    """Create fuels table.

    v0.2.0: Uses standardized fuel names (COAL, NATURAL_GAS, etc.)
    """
    if 'commodity' not in source:
        return pd.DataFrame(columns=['id', 'name', 'description'])

    commodities = source['commodity']
    fuels = commodities[commodities['commodity_type'] == 'fuel']

    fuels_data = []
    mapped_names = set()
    for commodity_name in fuels.index:
        fuel_id = id_gen.get('fuels', commodity_name)
        mapped_name = map_fuel_name(commodity_name)
        mapped_names.add(mapped_name)
        fuels_data.append({
            'id': fuel_id,
            'name': mapped_name,
            'description': f"Fuel: {commodity_name}"
        })

    # Ensure 'OTHER' fuel always exists (used as default for thermal units without fuel)
    if 'OTHER' not in mapped_names:
        other_id = id_gen.get_or_create('fuels', '_other_default', 'OTHER', 'ThermalStandard')
        fuels_data.append({
            'id': other_id,
            'name': 'OTHER',
            'description': 'Default fuel for units without explicit fuel commodity'
        })

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
            'name': region_name,
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
                        entities_data = group_entities.xs(group_name, level='group')
                        entities = entities_data.index.get_level_values('entity').tolist()
                        if balance_name in entities:
                            area_id = id_gen.get('planning_regions', group_name)
                            break
                    except KeyError:
                        continue

        topologies_data.append({
            'id': topo_id,
            'name': topo_name,
            'area': area_id,
            'description': f"Balancing topology: {balance_name}"
        })

    return pd.DataFrame(topologies_data)


def transform_arcs(source: Dict[str, pd.DataFrame],
                   id_gen: IDGenerator,
                   errors: TransformationErrors) -> pd.DataFrame:
    """Create arcs table from links.

    v0.3.0: from_id and to_id are NOT NULL.
    """
    if 'link' not in source:
        return pd.DataFrame(columns=['id', 'from_id', 'to_id'])

    links = source['link']
    arc_pairs = set()
    arcs_data = []

    for idx in links.index:
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

        # Skip if arc entity was not registered (endpoints not both in power_grid)
        if ('arcs', arc_key) not in id_gen.entity_map:
            continue

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

        # Skip if transmission line entity was not registered (no valid arc)
        if ('transmission_lines', link_key) not in id_gen.entity_map:
            continue

        line_id = id_gen.get('transmission_lines', link_key)
        line_name = id_gen.get_name('transmission_lines', link_key)

        # Get arc_id
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
            'name': line_name,
            'arc_id': arc_id,
            'continuous_rating': continuous_rating,
            'ste_rating': None,
            'lte_rating': None,
            'line_length': None
        })

    errors.add_missing("link transmission parameters (ste_rating, lte_rating, line_length) not mapped")

    return pd.DataFrame(lines_data)


def _find_unit_balancing_topology_and_rating(
    source: Dict[str, pd.DataFrame],
    unit_name: str,
    units_existing: float,
    id_gen: IDGenerator,
    errors: TransformationErrors,
    power_grid_nodes: set
) -> Optional[Tuple[int, float, float]]:
    """
    Find balancing topology ID and calculate rating/base_power for a generation unit.

    Returns:
        Tuple of (balancing_topology_id, rating, base_power) or None if not found.
    """
    if 'unit_to_node' not in source:
        return None

    unit_to_node = source['unit_to_node']
    unit_to_nodes = unit_to_node[unit_to_node.index.get_level_values('source') == unit_name]

    if len(unit_to_nodes) == 0:
        errors.add_warning(f"Unit {unit_name} has no unit_to_node connections")
        return None

    # Filter to power_grid nodes only
    power_grid_connections = []
    for idx in unit_to_nodes.index:
        if isinstance(idx, tuple) and len(idx) == 3:
            _, source_node, sink_node = idx
        else:
            errors.add_warning(f"unit_to_node index {idx} is not a 3-tuple, skipping")
            continue
        if sink_node in power_grid_nodes:
            power_grid_connections.append((idx, sink_node))

    if len(power_grid_connections) == 0:
        errors.add_warning(f"Unit {unit_name} not connected to power_grid balance nodes")
        return None

    # Use first power_grid connection
    idx, balance_node = power_grid_connections[0]
    capacity = safe_get(unit_to_nodes, idx, 'capacity', 1.0)

    rating = base_power = units_existing * capacity

    try:
        balancing_topology = id_gen.get('balancing_topologies', balance_node)
    except KeyError:
        errors.add_error(f"Balancing topology {balance_node} not found for unit {unit_name}")
        return None

    return (balancing_topology, rating, base_power)


def _find_unit_fuel(source: Dict[str, pd.DataFrame], unit_name: str, id_gen: IDGenerator) -> Optional[str]:
    """
    Find fuel name for a unit via node_to_unit connections.

    Returns:
        Mapped standardized fuel name, or None if not found.
    """
    if 'node_to_unit' not in source or 'commodity' not in source:
        return None

    node_to_unit = source['node_to_unit']
    commodities = source['commodity']

    unit_inputs = node_to_unit[node_to_unit.index.get_level_values('sink') == unit_name]
    for input_idx in unit_inputs.index:
        _, input_node, _ = input_idx
        if input_node in commodities.index:
            if safe_get(commodities, input_node, 'commodity_type') == 'fuel':
                try:
                    fuel_name = id_gen.get_name('fuels', input_node)
                    return map_fuel_name(fuel_name)
                except KeyError:
                    pass

    return None


def get_fuel_cost_for_unit(source: Dict[str, pd.DataFrame], unit_name: str) -> Optional[float]:
    """
    Get the fuel cost for a unit by looking up its input commodity via node_to_unit.

    Returns the commodity's price_per_unit, or None if not found.
    """
    if 'node_to_unit' not in source or 'commodity' not in source:
        return None

    node_to_unit = source['node_to_unit']
    commodities = source['commodity']

    # Find commodities connected to this unit (unit is the sink)
    for idx in node_to_unit.index:
        if isinstance(idx, tuple) and len(idx) == 3:
            _, source_node, sink_node = idx
        else:
            continue

        if sink_node != unit_name:
            continue

        # Check if the source node is a fuel commodity
        if source_node in commodities.index:
            commodity_type = safe_get(commodities, source_node, 'commodity_type')
            if commodity_type == 'fuel':
                price = safe_get(commodities, source_node, 'price_per_unit')
                if price is not None:
                    return float(price)

    return None


def _build_thermal_operation_cost(fuel_price: Optional[float], efficiency: float) -> str:
    """
    Build the operation_cost JSON for a thermal generator.

    If fuel_price and efficiency are available, computes the proportional_term
    as fuel_price / (efficiency/100) (cost per MWh of electricity output).

    CESM stores efficiency as percentage (0-100), so we divide by 100 to get
    the fractional efficiency for the cost calculation.

    Returns:
        JSON string of the operation cost structure.
    """
    cost = dict(DEFAULT_THERMAL_OPERATION_COST)  # shallow copy
    cost = json.loads(json.dumps(cost))  # deep copy

    if fuel_price is not None and efficiency is not None and efficiency > 0:
        # CESM efficiency is percentage (e.g., 38.0 = 38%), convert to fraction
        efficiency_fraction = efficiency / 100.0
        proportional_term = fuel_price / efficiency_fraction
        cost['variable']['value_curve']['function_data']['proportional_term'] = proportional_term

    return json.dumps(cost)


def transform_thermal_generators(source: Dict[str, pd.DataFrame],
                                   id_gen: IDGenerator,
                                   errors: TransformationErrors,
                                   classification_report: ClassificationReport = None) -> pd.DataFrame:
    """Create thermal_generators table.

    v0.3.0: Split from generation_units. Contains units with thermal prime movers
    (ST, CT, CC, CS, IC, GT, FC, CE, BT, CP).
    """
    if 'unit' not in source or 'unit_to_node' not in source:
        return pd.DataFrame(columns=[
            'id', 'name', 'prime_mover_type', 'fuel', 'balancing_topology',
            'rating', 'base_power', 'active_power_limits', 'must_run', 'operation_cost'
        ])

    units = source['unit']
    existing_units = units[safe_filter(units, 'units_existing') & (units['units_existing'] > 0)].copy()

    power_grid_nodes = identify_power_grid_nodes(source, errors)

    thermal_data = []

    for unit_name in existing_units.index:
        prime_mover = determine_prime_mover(source, unit_name, classification_report)

        # Only process thermal prime movers
        if prime_mover not in THERMAL_PRIME_MOVERS:
            continue

        units_existing = safe_get(existing_units, unit_name, 'units_existing', 1.0)

        result = _find_unit_balancing_topology_and_rating(
            source, unit_name, units_existing, id_gen, errors, power_grid_nodes
        )
        if result is None:
            continue

        balancing_topology, rating, base_power = result

        try:
            unit_id = id_gen.get('thermal_generators', unit_name)
        except KeyError:
            errors.add_warning(f"Thermal generator entity not found for {unit_name}")
            continue

        unit_name_str = id_gen.get_name('thermal_generators', unit_name)

        # Find fuel
        mapped_fuel_name = _find_unit_fuel(source, unit_name, id_gen)
        if mapped_fuel_name is None:
            mapped_fuel_name = 'OTHER'

        # Build active_power_limits JSON
        active_power_limits = json.dumps({"min": 0, "max": rating})

        # Build operation_cost from fuel_price / efficiency
        efficiency = safe_get(existing_units, unit_name, 'efficiency', 1.0)
        if efficiency is None:
            efficiency = 1.0
        fuel_price = get_fuel_cost_for_unit(source, unit_name)
        operation_cost = _build_thermal_operation_cost(fuel_price, efficiency)

        thermal_data.append({
            'id': unit_id,
            'name': unit_name_str,
            'prime_mover_type': prime_mover,
            'fuel': mapped_fuel_name,
            'balancing_topology': balancing_topology,
            'rating': rating,
            'base_power': base_power,
            'active_power_limits': active_power_limits,
            'must_run': False,
            'operation_cost': operation_cost,
        })

    return pd.DataFrame(thermal_data)


def transform_renewable_generators(source: Dict[str, pd.DataFrame],
                                     id_gen: IDGenerator,
                                     errors: TransformationErrors,
                                     classification_report: ClassificationReport = None) -> pd.DataFrame:
    """Create renewable_generators table.

    v0.3.0: Split from generation_units. Contains units with renewable prime movers
    (PVe, WT, WS, OT).
    """
    if 'unit' not in source or 'unit_to_node' not in source:
        return pd.DataFrame(columns=[
            'id', 'name', 'prime_mover_type', 'balancing_topology',
            'rating', 'base_power'
        ])

    units = source['unit']
    existing_units = units[safe_filter(units, 'units_existing') & (units['units_existing'] > 0)].copy()

    power_grid_nodes = identify_power_grid_nodes(source, errors)

    renewable_data = []

    for unit_name in existing_units.index:
        prime_mover = determine_prime_mover(source, unit_name, classification_report)

        # Only process renewable prime movers
        if prime_mover not in RENEWABLE_PRIME_MOVERS:
            continue

        units_existing = safe_get(existing_units, unit_name, 'units_existing', 1.0)

        result = _find_unit_balancing_topology_and_rating(
            source, unit_name, units_existing, id_gen, errors, power_grid_nodes
        )
        if result is None:
            continue

        balancing_topology, rating, base_power = result

        try:
            unit_id = id_gen.get('renewable_generators', unit_name)
        except KeyError:
            errors.add_warning(f"Renewable generator entity not found for {unit_name}")
            continue

        unit_name_str = id_gen.get_name('renewable_generators', unit_name)

        renewable_data.append({
            'id': unit_id,
            'name': unit_name_str,
            'prime_mover_type': prime_mover,
            'balancing_topology': balancing_topology,
            'rating': rating,
            'base_power': base_power,
        })

    return pd.DataFrame(renewable_data)


def transform_hydro_generators(source: Dict[str, pd.DataFrame],
                                 id_gen: IDGenerator,
                                 errors: TransformationErrors,
                                 classification_report: ClassificationReport = None) -> pd.DataFrame:
    """Create hydro_generators table.

    v0.3.0: Split from generation_units. Contains units with hydro prime movers
    (HY, HA, HB, HK).
    """
    if 'unit' not in source or 'unit_to_node' not in source:
        return pd.DataFrame(columns=[
            'id', 'name', 'prime_mover_type', 'balancing_topology',
            'rating', 'base_power', 'active_power_limits'
        ])

    units = source['unit']
    existing_units = units[safe_filter(units, 'units_existing') & (units['units_existing'] > 0)].copy()

    power_grid_nodes = identify_power_grid_nodes(source, errors)

    hydro_data = []

    for unit_name in existing_units.index:
        prime_mover = determine_prime_mover(source, unit_name, classification_report)

        # Only process hydro prime movers
        if prime_mover not in HYDRO_PRIME_MOVERS:
            continue

        units_existing = safe_get(existing_units, unit_name, 'units_existing', 1.0)

        result = _find_unit_balancing_topology_and_rating(
            source, unit_name, units_existing, id_gen, errors, power_grid_nodes
        )
        if result is None:
            continue

        balancing_topology, rating, base_power = result

        try:
            unit_id = id_gen.get('hydro_generators', unit_name)
        except KeyError:
            errors.add_warning(f"Hydro generator entity not found for {unit_name}")
            continue

        unit_name_str = id_gen.get_name('hydro_generators', unit_name)

        # Build active_power_limits JSON
        active_power_limits = json.dumps({"min": 0, "max": rating})

        hydro_data.append({
            'id': unit_id,
            'name': unit_name_str,
            'prime_mover_type': prime_mover,
            'balancing_topology': balancing_topology,
            'rating': rating,
            'base_power': base_power,
            'active_power_limits': active_power_limits,
        })

    return pd.DataFrame(hydro_data)


def transform_storage_units(source: Dict[str, pd.DataFrame],
                              id_gen: IDGenerator,
                              errors: TransformationErrors,
                              classification_report: ClassificationReport = None) -> pd.DataFrame:
    """Create storage_units table.

    v0.3.0: Restructured with JSON fields for efficiency, storage_level_limits,
    input/output_active_power_limits, plus new columns storage_technology_type,
    storage_capacity, initial_storage_capacity_level, etc.
    """
    if 'storage' not in source or 'link' not in source:
        return pd.DataFrame(columns=[
            'id', 'name', 'prime_mover_type', 'storage_technology_type',
            'balancing_topology', 'rating', 'base_power',
            'storage_capacity', 'storage_level_limits', 'initial_storage_capacity_level',
            'input_active_power_limits', 'output_active_power_limits',
            'efficiency', 'available', 'conversion_factor', 'storage_target', 'cycle_limits'
        ])

    storages = source['storage']
    existing_storages = storages[
        safe_filter(storages, 'storages_existing') & (storages['storages_existing'] > 0)
    ].copy()
    links = source['link']

    power_grid_nodes = identify_power_grid_nodes(source, errors)

    storage_units_data = []

    for storage_name in existing_storages.index:
        storage_id = id_gen.get('storage_units', storage_name)
        storage_name_str = id_gen.get_name('storage_units', storage_name)
        storage_capacity = safe_get(existing_storages, storage_name, 'storage_capacity', 1.0)
        if storage_capacity is None:
            storage_capacity = 1.0

        # Find link connecting this storage to a balance node
        storage_links = []
        for link_idx in links.index:
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
        link_efficiency = safe_get(links, link_idx, 'efficiency', 1.0)
        if link_efficiency is None:
            link_efficiency = 1.0

        rating = base_power = capacity

        try:
            balancing_topology = id_gen.get('balancing_topologies', balance_node)
        except KeyError:
            errors.add_error(f"Balancing topology {balance_node} not found for storage {storage_name}")
            continue

        # v0.3.0: Use parameter-based decision tree for storage classification
        storage_prime_mover = determine_storage_prime_mover(source, storage_name, classification_report)

        # v0.3.0: Map prime mover to storage technology type
        storage_tech_type = STORAGE_TECHNOLOGY_MAP.get(storage_prime_mover, 'Other')

        # v0.3.0: Build JSON fields
        # CESM efficiency is percentage (e.g., 90.0 = 90%), convert to fraction for GridDB
        efficiency_fraction = link_efficiency / 100.0
        efficiency_json = json.dumps({"in": efficiency_fraction, "out": efficiency_fraction})
        storage_level_limits_json = json.dumps({"min": 0, "max": storage_capacity})
        initial_storage_capacity_level = storage_capacity * 0.5
        input_active_power_limits_json = json.dumps({"min": 0, "max": rating})
        output_active_power_limits_json = json.dumps({"min": 0, "max": rating})

        storage_units_data.append({
            'id': storage_id,
            'name': storage_name_str,
            'prime_mover_type': storage_prime_mover,
            'storage_technology_type': storage_tech_type,
            'balancing_topology': balancing_topology,
            'rating': rating,
            'base_power': base_power,
            'storage_capacity': storage_capacity,
            'storage_level_limits': storage_level_limits_json,
            'initial_storage_capacity_level': initial_storage_capacity_level,
            'input_active_power_limits': input_active_power_limits_json,
            'output_active_power_limits': output_active_power_limits_json,
            'efficiency': efficiency_json,
            'available': True,
            'conversion_factor': 1.0,
            'storage_target': 0.0,
            'cycle_limits': 10000,
        })

    errors.add_missing("storage efficiency parameters not fully mapped (using link efficiency as proxy)")

    return pd.DataFrame(storage_units_data)


def transform_storage_technology_types(source: Dict[str, pd.DataFrame],
                                         id_gen: IDGenerator,
                                         errors: TransformationErrors,
                                         classification_report: ClassificationReport = None) -> pd.DataFrame:
    """Create storage_technology_types lookup table.

    v0.3.0: New table mapping storage technology type names.
    Populated from storage classifications used during transformation.
    """
    # Collect all storage technology types from the storage classifications
    tech_types_used = set()

    if 'storage' in source and 'link' in source:
        storages = source['storage']
        existing_storages = storages[
            safe_filter(storages, 'storages_existing') & (storages['storages_existing'] > 0)
        ]
        for storage_name in existing_storages.index:
            pm = determine_storage_prime_mover(source, storage_name)
            tech_type = STORAGE_TECHNOLOGY_MAP.get(pm, 'Other')
            tech_types_used.add(tech_type)

    # If no storages found, add at least common types
    if not tech_types_used:
        tech_types_used = {'Battery', 'Pumped Hydro', 'Other'}

    tech_types_data = []
    for idx, tech_name in enumerate(sorted(tech_types_used), start=1):
        tech_types_data.append({
            'id': idx,
            'name': tech_name,
            'description': f"Storage technology: {tech_name}"
        })

    return pd.DataFrame(tech_types_data)


def transform_supply_technologies(source: Dict[str, pd.DataFrame],
                                    id_gen: IDGenerator,
                                    errors: TransformationErrors,
                                    classification_report: ClassificationReport = None) -> pd.DataFrame:
    """Create supply_technologies table.

    v0.3.0: prime_mover -> prime_mover_type, area and balancing_topology changed
    from TEXT to INTEGER FK.
    """
    if 'unit' not in source:
        return pd.DataFrame(columns=['id', 'prime_mover_type', 'fuel', 'area',
                                     'balancing_topology', 'scenario'])

    units = source['unit']
    investment_units = units[
        safe_filter(units, 'investment_cost') &
        safe_filter(units, 'discount_rate')
    ].copy()

    supply_tech_data = []

    for unit_name in investment_units.index:
        key = f"supply_tech_{unit_name}_candidate"
        tech_id = id_gen.get('supply_technologies', key)

        # v0.3.0: Determine prime mover type using parameter-based decision tree
        prime_mover = determine_prime_mover(source, unit_name, classification_report)

        # Find fuel (same logic as generator tables)
        mapped_fuel_name = _find_unit_fuel(source, unit_name, id_gen)

        # v0.3.0: Find balancing topology as INTEGER id (not TEXT name)
        balancing_topology_id = None
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
                        balancing_topology_id = id_gen.get('balancing_topologies', sink_node)
                        break
                    except KeyError:
                        pass

        # v0.3.0: Find area as INTEGER id (not TEXT name)
        area_id = None
        if balancing_topology_id is not None and 'group_entity' in source and 'group' in source:
            group_entities = source['group_entity']
            groups = source['group']
            # Find which planning region contains the balance node
            for group_name in groups[groups['group_type'] == 'node'].index:
                if group_name in group_entities.index.get_level_values('group'):
                    try:
                        entities_data = group_entities.xs(group_name, level='group')
                        entity_names = entities_data.index.get_level_values('entity').tolist()
                        # Check if the balance node (sink_node) is in this planning region
                        if 'unit_to_node' in source:
                            unit_to_nodes_inner = source['unit_to_node']
                            unit_outputs = unit_to_nodes_inner[
                                unit_to_nodes_inner.index.get_level_values('source') == unit_name
                            ]
                            for ut_idx in unit_outputs.index:
                                if isinstance(ut_idx, tuple) and len(ut_idx) == 3:
                                    _, _, sn = ut_idx
                                    if sn in entity_names and sn in power_grid_nodes:
                                        area_id = id_gen.get('planning_regions', group_name)
                                        break
                        if area_id is not None:
                            break
                    except KeyError:
                        continue

        supply_tech_data.append({
            'id': tech_id,
            'prime_mover_type': prime_mover,
            'fuel': mapped_fuel_name,
            'area': area_id,
            'balancing_topology': balancing_topology_id,
            'scenario': None
        })

    return pd.DataFrame(supply_tech_data)


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
    - Units with investment data -> supply_technologies attributes
    - Links with investment data -> transport_technologies attributes
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
                'arc_id': None,
                'max_flow_from': max_flow_from if max_flow_from is not None else 0.0,
                'max_flow_to': max_flow_to if max_flow_to is not None else 0.0
            })

    return pd.DataFrame(interchanges_data)


def get_parameter_unit(param_name: str, schema_path: str = 'model/cesm_v0.1.0.yaml') -> str:
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
    Get owner_type for a given entity_id.

    v0.3.0: Returns entity type names without SiennaOpenAPIModels prefix.

    Uses the entity_type stored in IDGenerator, which accounts for prime_mover
    classification.

    Args:
        entity_id: The entity ID to look up
        id_gen: IDGenerator instance containing entity mappings

    Returns:
        Component type string (e.g., 'ThermalStandard')
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
        'transmission_interchanges': 'AreaInterchange',
        'loads': 'PowerLoad',
        'fuels': 'ThermalStandard',
        'arcs': 'Arc',
        'storage_units': 'EnergyReservoirStorage',
        'thermal_generators': 'ThermalStandard',
        'renewable_generators': 'RenewableDispatch',
        'hydro_generators': 'HydroDispatch',
        'supply_technologies': 'ThermalStandard',
    }

    # Reverse lookup in id_gen.entity_map to find source_table for this entity_id
    for (source_table, key), eid in id_gen.entity_map.items():
        if eid == entity_id:
            return TABLE_TO_OWNER_TYPE.get(source_table, 'ThermalStandard')

    return 'ThermalStandard'  # Fallback for unknown entities


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

    Each supplemental attribute is registered as an entity via id_gen so that it
    has a valid entry in the entities table (required by FK constraints).

    Returns:
        Tuple of (supplemental_attributes DataFrame, associations DataFrame)
    """
    attributes_data = []
    associations_data = []

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
                    attr_key = f"suppl_attr_eff_{link_key}"
                    attr_id = id_gen.get_or_create(
                        'supplemental_attributes', attr_key, attr_key, 'SupplementalAttribute')
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
                    attr_key = f"suppl_attr_dr_{link_key}"
                    attr_id = id_gen.get_or_create(
                        'supplemental_attributes', attr_key, attr_key, 'SupplementalAttribute')
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
                    attr_key = f"suppl_attr_crp_{link_key}"
                    attr_id = id_gen.get_or_create(
                        'supplemental_attributes', attr_key, attr_key, 'SupplementalAttribute')
                    attributes_data.append({
                        'id': attr_id,
                        'TYPE': 'capital_recovery_period',
                        'value': encode_attribute_value(capital_recovery_period, "float", "years")
                    })
                    associations_data.append({
                        'attribute_id': attr_id,
                        'entity_id': entity_id
                    })

    return pd.DataFrame(attributes_data), pd.DataFrame(associations_data)


def transform_time_series(source: Dict[str, pd.DataFrame],
                           id_gen: IDGenerator,
                           errors: TransformationErrors) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create time_series_associations and static_time_series tables.

    v0.3.0: Updated to look up generation units from thermal_generators,
    renewable_generators, and hydro_generators instead of single generation_units table.
    """
    associations_data = []
    static_data = []
    ts_assoc_id = 0
    static_row_id = 0

    # v0.3.0: Generator tables to search for owner lookups
    GENERATOR_TABLES = ('thermal_generators', 'renewable_generators', 'hydro_generators')

    def _find_generator_owner_id(unit_name: str) -> Optional[int]:
        """Try to find generator entity ID across all three generator tables."""
        for table in GENERATOR_TABLES:
            try:
                return id_gen.get(table, unit_name)
            except KeyError:
                continue
        return None

    # Process profile data from unit_to_node.ts.profile_limit_*
    for key in source.keys():
        if key.startswith('unit_to_node.ts.profile_limit_'):
            ts_df = source[key]
            param_name = key.split('unit_to_node.ts.')[1]

            # Validate datetime index (required for initial_timestamp and resolution)
            if not isinstance(ts_df.index, pd.DatetimeIndex):
                errors.add_error(
                    f"Time series '{key}' missing datetime index"
                    " (required for initial_timestamp/resolution)"
                )
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

                    # Try to find corresponding generation unit across all generator tables
                    unit_name = col_tuple[0] if isinstance(col_tuple, tuple) else col_tuple
                    owner_id = _find_generator_owner_id(unit_name)
                    if owner_id is None:
                        errors.add_warning(f"Cannot find generator for profile {unit_name}")
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
                        'owner_type': owner_type,
                        'owner_category': 'Component',
                        'features': '',
                        'scaling_factor_multiplier': None,
                        'metadata_uuid': metadata_uuid,
                        'units': unit_str
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

                    owner_id = _find_generator_owner_id(col)
                    if owner_id is None:
                        errors.add_warning(f"Cannot find generator for profile {col}")
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
                        'owner_type': owner_type,
                        'owner_category': 'Component',
                        'features': '',
                        'scaling_factor_multiplier': None,
                        'metadata_uuid': metadata_uuid,
                        'units': unit_str
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
                    'owner_type': owner_type,
                    'owner_category': 'Component',
                    'features': '',
                    'scaling_factor_multiplier': None,
                    'metadata_uuid': metadata_uuid,
                    'units': unit_str
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
                    'owner_type': owner_type,
                    'owner_category': 'Component',
                    'features': '',
                    'scaling_factor_multiplier': None,
                    'metadata_uuid': metadata_uuid,
                    'units': unit_str
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


def to_griddb(source: Dict[str, pd.DataFrame], strict: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Main transformation function.

    v0.3.0: Splits generation_units into thermal_generators, renewable_generators,
    hydro_generators. Removes operational_data (now a VIEW). Adds
    storage_technology_types. Renames hydro_reservoir to hydro_reservoirs.

    Args:
        source: Dictionary of source dataframes
        strict: If True, report all errors and warnings. If False, suppress
                warnings about missing optional data (e.g., timeline data that
                may already exist in the target database). Default: True.

    Returns:
        Dictionary of target dataframes matching SQL schema
    """
    errors = TransformationErrors(strict=strict)
    id_gen = IDGenerator()
    classification_report = ClassificationReport()
    target = {}

    try:
        # Phase 1: Create entities and populate ID mappings
        print("Phase 1: Creating entities and ID mappings...")
        target['entities'] = transform_entities_and_ids(source, id_gen, errors, classification_report)

        # Phase 1b: Create entity_types table (v0.3.0: includes is_topology)
        target['entity_types'] = transform_entity_types(id_gen, errors)

        # Phase 2: Create all other tables
        print("Phase 2: Creating target tables...")
        target['prime_mover_types'] = transform_prime_mover_types(id_gen, errors)
        target['fuels'] = transform_fuels(source, id_gen, errors)
        target['planning_regions'] = transform_planning_regions(source, id_gen, errors)
        target['balancing_topologies'] = transform_balancing_topologies(source, id_gen, errors)
        target['arcs'] = transform_arcs(source, id_gen, errors)
        target['transmission_lines'] = transform_transmission_lines(source, id_gen, errors)
        target['transmission_interchanges'] = transform_transmission_interchanges(source, id_gen, errors)

        # v0.3.0: Three generator tables instead of one generation_units table
        target['thermal_generators'] = transform_thermal_generators(
            source, id_gen, errors, classification_report
        )
        target['renewable_generators'] = transform_renewable_generators(
            source, id_gen, errors, classification_report
        )
        target['hydro_generators'] = transform_hydro_generators(
            source, id_gen, errors, classification_report
        )

        target['storage_units'] = transform_storage_units(source, id_gen, errors, classification_report)

        # v0.3.0: New storage_technology_types lookup table
        target['storage_technology_types'] = transform_storage_technology_types(
            source, id_gen, errors, classification_report
        )

        target['supply_technologies'] = transform_supply_technologies(
            source, id_gen, errors, classification_report
        )
        target['transport_technologies'] = transform_transport_technologies(source, id_gen, errors)

        # v0.3.0: operational_data is now a VIEW - do NOT insert into it

        # Reconcile: add any entities created during Phase 2 that weren't in Phase 1
        existing_entity_ids = set(target['entities']['id'].tolist())
        new_entities = []
        for (source_table, key), entity_id in id_gen.entity_map.items():
            if entity_id not in existing_entity_ids:
                entity_type = id_gen.get_type(entity_id) or 'Unknown'
                new_entities.append({
                    'id': entity_id,
                    'entity_table': source_table,
                    'entity_type': entity_type
                })
        if new_entities:
            target['entities'] = pd.concat(
                [target['entities'], pd.DataFrame(new_entities)], ignore_index=True
            )

        # Phase 3: Time series
        print("Phase 3: Creating time series data...")
        target['time_series_associations'], target['static_time_series'] = \
            transform_time_series(source, id_gen, errors)

        # Phase 4: Attributes and supplemental attributes
        print("Phase 4: Creating attributes and supplemental attributes...")
        target['attributes'] = transform_attributes(source, id_gen, errors)
        target['supplemental_attributes'], target['supplemental_attributes_association'] = \
            transform_supplemental_attributes(source, id_gen, errors)

        # Add supplemental_attribute entities to entities table
        if not target['supplemental_attributes'].empty:
            new_entities = []
            for _, row in target['supplemental_attributes'].iterrows():
                new_entities.append({
                    'id': row['id'],
                    'entity_table': 'supplemental_attributes',
                    'entity_type': 'SupplementalAttribute'
                })
            if new_entities:
                new_entities_df = pd.DataFrame(new_entities)
                target['entities'] = pd.concat(
                    [target['entities'], new_entities_df], ignore_index=True
                )
                # Ensure SupplementalAttribute entity_type exists in entity_types
                if not target['entity_types'].empty:
                    existing_types = set(target['entity_types']['name'].tolist())
                    if 'SupplementalAttribute' not in existing_types:
                        new_type = pd.DataFrame([{
                            'name': 'SupplementalAttribute',
                            'is_topology': False
                        }])
                        target['entity_types'] = pd.concat(
                            [target['entity_types'], new_type], ignore_index=True
                        )

        # Phase 5: Empty tables for completeness
        # v0.3.0: hydro_reservoir -> hydro_reservoirs (plural)
        target['hydro_reservoirs'] = pd.DataFrame(columns=[
            'id', 'name', 'available', 'storage_level_limits', 'initial_level',
            'spillage_limits', 'inflow', 'outflow', 'level_targets',
            'intake_elevation', 'head_to_volume_factor', 'operation_cost',
            'level_data_type'
        ])
        target['hydro_reservoir_connections'] = pd.DataFrame(columns=['source_id', 'sink_id'])
        target['loads'] = pd.DataFrame(columns=['id', 'name', 'balancing_topology', 'base_power'])

        errors.add_missing("hydro_reservoirs and hydro_reservoir_connections - not processed")
        errors.add_missing("loads - cannot generate from flexible source units (no clear base_power)")

    except Exception as e:
        errors.add_error(f"Critical transformation error: {str(e)}")
        import traceback
        errors.add_error(traceback.format_exc())

    # Report all errors and warnings
    errors.report()

    # Report classification uncertainties
    classification_report.report()
    print(f"\n{classification_report.summary()}")

    return target
