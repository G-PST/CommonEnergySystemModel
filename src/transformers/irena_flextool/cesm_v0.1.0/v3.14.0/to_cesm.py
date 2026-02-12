"""
FlexTool to CESM transformation functions.

This module contains Python functions for transformations that cannot be
expressed in the YAML configuration (from_flextool.yaml), specifically:
- time_from_spine: Reconstructs timeline, solve_pattern, and period data

Most transformations are now handled by the YAML configuration:
- Entity copies (unit, commodity)
- Link entities with node_A/node_B (using value: [N] dimension extraction)
- Port entities with source/sink (using value: [N] dimension extraction)
- Balance/storage splitting (using if_parameter/if_not_parameter)
- Parameter transformations

These functions are the reverse of time_to_spine in cesm_to_flextool.py.
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta


def _parse_datetime_string(dt_str: str) -> datetime:
    """
    Parse a datetime string in ISO 8601 format.

    Handles formats like '2023-01-01T00:00:00' (no timezone).
    """
    return datetime.fromisoformat(dt_str)


def _get_timestep_minutes(flextool: Dict[str, pd.DataFrame]) -> Optional[int]:
    """
    Extract timestep duration in minutes from timeline.str.timestep_duration.

    Args:
        flextool: Dictionary of FlexTool DataFrames

    Returns:
        Timestep duration in minutes, or None if not found
    """
    ts_df_name = 'timeline.str.timestep_duration'
    if ts_df_name not in flextool:
        return None

    ts_df = flextool[ts_df_name]
    if ts_df.empty:
        return None

    # Get the first non-null value (duration in hours)
    for col in ts_df.columns:
        values = ts_df[col].dropna()
        if len(values) > 0:
            hours = float(values.iloc[0])
            return int(hours * 60)

    return None


def _convert_str_index_to_datetime(
    index: pd.Index,
    start_time: datetime,
    timestep_minutes: int
) -> pd.DatetimeIndex:
    """
    Convert string-based timestep index to DatetimeIndex.

    Handles indexes like 't0001', 't0002', etc. by extracting the numeric
    part and calculating datetime based on start_time and timestep_minutes.

    Args:
        index: Index with string values (e.g., 't0001', 't0002')
        start_time: Datetime of the first timestep
        timestep_minutes: Duration of each timestep in minutes

    Returns:
        DatetimeIndex
    """
    new_index = []
    for idx_val in index:
        if isinstance(idx_val, str):
            # Extract numeric part (e.g., 't0001' -> 1, 'step_5' -> 5)
            numeric_part = ''.join(c for c in idx_val if c.isdigit())
            if numeric_part:
                step_num = int(numeric_part) - 1  # Convert to 0-based
                dt = start_time + timedelta(minutes=step_num * timestep_minutes)
                new_index.append(dt)
            else:
                # Can't parse, try as datetime string
                try:
                    new_index.append(pd.to_datetime(idx_val))
                except:
                    raise ValueError(f"Cannot convert index value '{idx_val}' to datetime")
        else:
            # Already datetime or other, try to convert
            new_index.append(pd.to_datetime(idx_val))

    return pd.DatetimeIndex(new_index)


def time_from_spine(flextool: Dict[str, pd.DataFrame],
                    cesm: Dict[str, pd.DataFrame],
                    start_time: datetime) -> tuple:
    """
    Extract temporal data from FlexTool format and add to CESM.

    This is the reverse of time_to_spine in cesm_to_flextool.py.

    Creates/updates:
    - cesm['timeline']: DataFrame with DatetimeIndex
    - cesm['solve_pattern']: DataFrame with start_time, duration, solve_mode
    - cesm['period']: DataFrame with years_represented

    Reads from flextool:
    - timeline.str.timestep_duration: Map of datetime → duration hours
    - timeset.str.timeset_duration: Map of start_time → duration (timesteps)
    - solve.str.period_timeset: Map of period → timeset for each solve
    - solve.str.years_represented: Map of period → years for each solve
    - solve: DataFrame with solve_mode parameter

    Args:
        flextool: Dictionary of FlexTool DataFrames
        cesm: Dictionary of CESM DataFrames (will be modified)
        start_time: Start datetime for the timeline (required for string index conversion)

    Returns:
        Tuple of (cesm, str_to_datetime_lookup) where str_to_datetime_lookup maps
        string timestamps to datetime objects for use by other transformations
    """
    # Get timestep duration from the data
    timestep_minutes = _get_timestep_minutes(flextool)

    # --- Create timeline from timestep_duration ---
    # Initialize lookup table for string-to-datetime conversion (used by timeset processing)
    str_to_datetime_lookup = {}

    if 'timeline.str.timestep_duration' in flextool:
        timestep_df = flextool['timeline.str.timestep_duration']

        # Check if index needs conversion from string timesteps (e.g., 't0001')
        if timestep_df.index.dtype == 'object' and len(timestep_df.index) > 0:
            first_val = str(timestep_df.index[0])
            # Check if it looks like a timestep string (contains digits but not ISO datetime format)
            if any(c.isdigit() for c in first_val) and 'T' not in first_val and '-' not in first_val[:4]:
                # String timesteps like 't0001' - need conversion
                if timestep_minutes is None:
                    raise ValueError("Cannot convert string index to datetime: timestep_duration not found")
                datetime_index = _convert_str_index_to_datetime(
                    timestep_df.index, start_time, timestep_minutes
                )
                # Build lookup table for string-to-datetime conversion
                str_to_datetime_lookup = dict(zip(timestep_df.index.astype(str), datetime_index))
            else:
                # Assume datetime strings, convert directly
                datetime_index = pd.to_datetime(timestep_df.index)
        else:
            # Already datetime or can be converted directly
            datetime_index = pd.to_datetime(timestep_df.index)

        # Create timeline DataFrame with just the DatetimeIndex
        # CESM timeline is stored as the index of this DataFrame
        cesm['timeline'] = pd.DataFrame(index=datetime_index)
        cesm['timeline'].index.name = 'datetime'

    # --- Create solve_pattern from timeset and solve data ---
    solve_patterns = {}
    multi_row_timesets = {}  # To collect multi-row timeset data

    # Get start_time and duration from timeset_duration
    if 'timeset.str.timeset_duration' in flextool:
        timeset_duration_df = flextool['timeset.str.timeset_duration']

        # Each column is a timeset, index is start_time, values are duration
        for timeset_name in timeset_duration_df.columns:
            col_data = timeset_duration_df[timeset_name].dropna()

            if len(col_data) == 1:
                # Single-row timeset: existing logic for solve_patterns
                solve_start_str = str(col_data.index[0])
                duration = int(col_data.iloc[0])

                # Convert string timestep to datetime if needed
                if timestep_minutes is not None and any(c.isdigit() for c in solve_start_str):
                    if 'T' not in solve_start_str and '-' not in solve_start_str[:4]:
                        # String like 't0001' - convert to datetime
                        solve_start_dt = _convert_str_index_to_datetime(
                            pd.Index([solve_start_str]), start_time, timestep_minutes
                        )[0]
                    else:
                        solve_start_dt = pd.to_datetime(solve_start_str)
                else:
                    solve_start_dt = pd.to_datetime(solve_start_str)

                # Find which solve patterns use this timeset
                if 'solve.str.period_timeset' in flextool:
                    period_timeset_df = flextool['solve.str.period_timeset']
                    for solve_name in period_timeset_df.columns:
                        if timeset_name in period_timeset_df[solve_name].values:
                            if solve_name not in solve_patterns:
                                solve_patterns[solve_name] = {
                                    'start_time': solve_start_dt,
                                    'duration': duration
                                }

            elif len(col_data) > 1:
                # Multi-row timeset: convert index to datetime and store separately
                converted_index = []
                for str_ts in col_data.index.astype(str):
                    if str_to_datetime_lookup and str_ts in str_to_datetime_lookup:
                        converted_index.append(str_to_datetime_lookup[str_ts])
                    else:
                        # Try direct conversion if no lookup available
                        try:
                            converted_index.append(pd.to_datetime(str_ts))
                        except Exception:
                            raise ValueError(
                                f"Timestamp '{str_ts}' not found in lookup table and "
                                "cannot be parsed as datetime"
                            )

                # Create a timeset for each solve using the timeset in original data
                if 'solve.str.period_timeset' in flextool:
                    period_timeset_df = flextool['solve.str.period_timeset']
                    for solve_name in period_timeset_df.columns:
                        if timeset_name in period_timeset_df[solve_name].values:
                            multi_row_timesets[solve_name] = pd.Series(
                                col_data.values,
                                index=pd.DatetimeIndex(converted_index)
                            )

    # Create combined dataframe for multi-row timesets
    if multi_row_timesets:
        cesm['solve_pattern.ts.start_time_durations'] = pd.DataFrame(multi_row_timesets)

    # Create solve_pattern DataFrame
    if solve_patterns:
        if 'solve_pattern' not in cesm:
            cesm['solve_pattern'] = pd.DataFrame.from_dict(solve_patterns, orient='index')
        else:
            temp_df = pd.DataFrame.from_dict(solve_patterns, orient='index')
            cesm['solve_pattern'] = cesm['solve_pattern'].join(temp_df, how='outer', rsuffix='_cesm')


    # --- Extract periods from period_timeset and years_represented ---
    periods = {}

    if 'solve.str.period_timeset' in flextool:
        period_timeset_df = flextool['solve.str.period_timeset']
        # Collect all unique periods
        for solve_name in period_timeset_df.columns:
            for period_name in period_timeset_df.index:
                if pd.notna(period_timeset_df.loc[period_name, solve_name]):
                    if period_name not in periods:
                        periods[period_name] = {}

    # Get years_represented for each period
    if 'solve.str.years_represented' in flextool:
        years_df = flextool['solve.str.years_represented']
        for solve_name in years_df.columns:
            for period_name in years_df.index:
                value = years_df.loc[period_name, solve_name]
                if pd.notna(value):
                    if period_name not in periods:
                        periods[period_name] = {}
                    periods[period_name]['years_represented'] = value

    # Create period DataFrame
    if periods:
        cesm['period'] = pd.DataFrame.from_dict(periods, orient='index')
        cesm['period'].index.name = 'name'

    return cesm, str_to_datetime_lookup


def _get_unit_to_node_mapping(flextool: Dict[str, pd.DataFrame]) -> Dict[str, list]:
    """
    Build a mapping from unit names to their unit.outputNode entity names.

    Args:
        flextool: Dictionary of FlexTool DataFrames

    Returns:
        Dictionary mapping unit names to list of unit.outputNode entity names
    """
    mapping = {}

    if 'unit.outputNode' not in flextool:
        return mapping

    output_node_df = flextool['unit.outputNode']

    # The index is MultiIndex with (entity_name, unit, node)
    # Level 1 contains the unit name
    for entity_name in output_node_df.index:
        if isinstance(entity_name, tuple):
            # MultiIndex: (entity_name, unit, node)
            unit_name = entity_name[1]
            entity_key = entity_name[0]
        else:
            # Single index - try to extract unit name from entity name
            # Entity names are like "coal_plant.west"
            unit_name = entity_name.split('.')[0] if '.' in str(entity_name) else entity_name
            entity_key = entity_name

        if unit_name not in mapping:
            mapping[unit_name] = []
        mapping[unit_name].append(entity_key)

    return mapping


def _ensure_dataframe_exists(cesm: Dict[str, pd.DataFrame],
                             df_name: str,
                             index_source: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Ensure a DataFrame exists in cesm dictionary. Create if missing.

    Args:
        cesm: Dictionary of CESM DataFrames
        df_name: Name of the DataFrame to ensure exists
        index_source: Optional DataFrame to copy index from

    Returns:
        The existing or newly created DataFrame
    """
    if df_name not in cesm:
        if index_source is not None:
            cesm[df_name] = pd.DataFrame(index=index_source.index)
        else:
            cesm[df_name] = pd.DataFrame()
    return cesm[df_name]


def capacities_from_spine(flextool: Dict[str, pd.DataFrame],
                          cesm: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Transform unit capacity data from FlexTool format to CESM.

    This is the reverse of the YAML transformations that aggregate
    unit_to_node capacity data to unit virtual_unitsize.

    Logic per unit:
    - If virtual_unitsize exists:
        - virtual_unitsize → capacity (unit_to_node) for ALL unit.outputNode
        - existing → units_existing (unit)
    - If virtual_unitsize does NOT exist:
        - existing → capacity (unit_to_node) for ALL unit.outputNode
        - units_existing = 1 (unit)

    Also transforms costs (FIRST unit_to_node only):
    - invest_cost → investment_cost
    - fixed_cost → fixed_cost
    - salvage_value → salvage_value

    Args:
        flextool: Dictionary of FlexTool DataFrames (source)
        cesm: Dictionary of CESM DataFrames (will be modified)

    Returns:
        Updated cesm dictionary with capacity transformations applied
    """
    # Build unit → unit.outputNode mapping
    unit_to_output_nodes = _get_unit_to_node_mapping(flextool)

    if not unit_to_output_nodes:
        # No unit.outputNode relationships found
        return cesm

    # Get source DataFrames
    unit_df = flextool.get('unit', pd.DataFrame())
    unit_str_dfs = {k: v for k, v in flextool.items() if k.startswith('unit.str.')}

    # Check which columns exist
    has_virtual_unitsize_col = 'virtual_unitsize' in unit_df.columns
    has_existing_col = 'existing' in unit_df.columns
    has_virtual_unitsize_str = 'unit.str.virtual_unitsize' in unit_str_dfs
    has_existing_str = 'unit.str.existing' in unit_str_dfs

    # Ensure target DataFrames exist
    _ensure_dataframe_exists(cesm, 'unit_to_node', flextool.get('unit.outputNode'))
    _ensure_dataframe_exists(cesm, 'unit', flextool.get('unit'))

    # --- Transform capacity and units_existing per unit ---
    for unit_name, output_nodes in unit_to_output_nodes.items():
        if unit_name not in unit_df.index:
            continue

        # Check if this unit has virtual_unitsize defined (constant)
        has_virtual_unitsize = (
            has_virtual_unitsize_col and
            pd.notna(unit_df.loc[unit_name, 'virtual_unitsize'])
        )

        # Check if this unit has virtual_unitsize defined (period-indexed)
        has_virtual_unitsize_period = (
            has_virtual_unitsize_str and
            unit_name in unit_str_dfs['unit.str.virtual_unitsize'].columns and
            not unit_str_dfs['unit.str.virtual_unitsize'][unit_name].dropna().empty
        )

        if has_virtual_unitsize or has_virtual_unitsize_period:
            # Case 1: virtual_unitsize exists
            # Use virtual_unitsize as capacity, existing as units_existing

            # Constant capacity from virtual_unitsize
            if has_virtual_unitsize:
                capacity_value = unit_df.loc[unit_name, 'virtual_unitsize']
                for entity_name in output_nodes:
                    if entity_name in cesm['unit_to_node'].index:
                        cesm['unit_to_node'].loc[entity_name, 'capacity'] = capacity_value

            # Period-indexed capacity from virtual_unitsize
            if has_virtual_unitsize_period:
                _ensure_dataframe_exists(cesm, 'unit_to_node.str.capacity')
                series = unit_str_dfs['unit.str.virtual_unitsize'][unit_name].dropna()
                for entity_name in output_nodes:
                    cesm['unit_to_node.str.capacity'][entity_name] = series

            # Constant units_existing from existing
            if has_existing_col and pd.notna(unit_df.loc[unit_name, 'existing']):
                if unit_name in cesm['unit'].index:
                    cesm['unit'].loc[unit_name, 'units_existing'] = unit_df.loc[unit_name, 'existing']

            # Period-indexed units_existing from existing
            if has_existing_str and unit_name in unit_str_dfs['unit.str.existing'].columns:
                _ensure_dataframe_exists(cesm, 'unit.str.units_existing')
                series = unit_str_dfs['unit.str.existing'][unit_name].dropna()
                if not series.empty:
                    cesm['unit.str.units_existing'][unit_name] = series

        else:
            # Case 2: virtual_unitsize does NOT exist
            # Use existing as capacity, set units_existing = 1

            # Constant capacity from existing
            if has_existing_col and pd.notna(unit_df.loc[unit_name, 'existing']):
                capacity_value = unit_df.loc[unit_name, 'existing']
                for entity_name in output_nodes:
                    if entity_name in cesm['unit_to_node'].index:
                        cesm['unit_to_node'].loc[entity_name, 'capacity'] = capacity_value

            # Period-indexed capacity from existing
            if has_existing_str and unit_name in unit_str_dfs['unit.str.existing'].columns:
                _ensure_dataframe_exists(cesm, 'unit_to_node.str.capacity')
                series = unit_str_dfs['unit.str.existing'][unit_name].dropna()
                if not series.empty:
                    for entity_name in output_nodes:
                        cesm['unit_to_node.str.capacity'][entity_name] = series

            # Set units_existing = 1
            if unit_name in cesm['unit'].index:
                cesm['unit'].loc[unit_name, 'units_existing'] = 1.0

    # --- Transform costs (FIRST unit_to_node only) ---
    # Parameter name mapping: FlexTool → CESM
    cost_params = {
        'invest_cost': 'investment_cost',
        'fixed_cost': 'fixed_cost',
        'salvage_value': 'salvage_value'
    }

    for ft_param, cesm_param in cost_params.items():
        # Constant values
        if ft_param in unit_df.columns:
            for unit_name, output_nodes in unit_to_output_nodes.items():
                if unit_name in unit_df.index and output_nodes:
                    value = unit_df.loc[unit_name, ft_param]
                    if pd.notna(value):
                        # Only the FIRST unit_to_node gets the cost
                        first_entity = output_nodes[0]
                        if first_entity in cesm['unit_to_node'].index:
                            cesm['unit_to_node'].loc[first_entity, cesm_param] = value

        # Period-indexed values
        str_df_name = f'unit.str.{ft_param}'
        if str_df_name in unit_str_dfs:
            src_df = unit_str_dfs[str_df_name]
            cesm_str_name = f'unit_to_node.str.{cesm_param}'
            _ensure_dataframe_exists(cesm, cesm_str_name)

            for unit_name, output_nodes in unit_to_output_nodes.items():
                if unit_name in src_df.columns and output_nodes:
                    series = src_df[unit_name].dropna()
                    if not series.empty:
                        # Only the FIRST unit_to_node gets the cost
                        first_entity = output_nodes[0]
                        cesm[cesm_str_name][first_entity] = series

    return cesm


def _process_profile_to_cesm(flextool: Dict[str, pd.DataFrame],
                              cesm: Dict[str, pd.DataFrame],
                              connection_df_name: str,
                              target_entity_name: str,
                              target_ts_prefix: str,
                              swap_dimensions: bool = False,
                              str_to_datetime_lookup: Optional[Dict[str, datetime]] = None) -> Dict[str, pd.DataFrame]:
    """
    Process profile connections from FlexTool to CESM format.

    Reads from FlexTool:
    - unit.node.profile: MultiIndex (unit, node, profile) with profile_method column
    - connection_df_name: unit.outputNode or unit.inputNode - for filtering
    - profile['profile']: constant profile values
    - profile.ts.profile: timeseries profile values

    Writes to CESM:
    - target_entity_name: unit_to_node or node_to_unit with profile_limit_* columns
    - target_ts_prefix.profile_limit_*: timeseries data

    Args:
        flextool: Source FlexTool dataframe dictionary
        cesm: Target CESM dataframe dictionary
        connection_df_name: 'unit.outputNode' or 'unit.inputNode'
        target_entity_name: 'unit_to_node' or 'node_to_unit'
        target_ts_prefix: 'unit_to_node.ts' or 'node_to_unit.ts'
        swap_dimensions: For node_to_unit, source=node and sink=unit (swapped)
        str_to_datetime_lookup: Mapping from string timestamps to datetime objects

    Returns:
        Updated cesm dictionary
    """
    # 1. Check if required dataframes exist - return early if not
    if 'unit.node.profile' not in flextool or connection_df_name not in flextool:
        return cesm

    unit_node_profile = flextool['unit.node.profile']
    connection_df = flextool[connection_df_name]

    # 2. Create (unit, node) index from connection_df for filtering
    #    connection_df has MultiIndex (name, unit, node) - drop 'name' to get (unit, node)
    connection_index = connection_df.index.droplevel('name')  # Drop 'name', keep (unit, node)

    # 3. Filter unit.node.profile to rows where (unit, node) is in connections
    #    unit.node.profile has MultiIndex (unit, node, profile)
    unit_node_profile_index = unit_node_profile.index.droplevel('name')
    filtered_profiles = unit_node_profile[unit_node_profile_index.droplevel('profile').isin(connection_index)]

    # 4. Initialize data collectors for constants and timeseries
    constant_data = {'upper': {}, 'lower': {}}
    ts_data = {'upper': {}, 'lower': {}}

    # 5. Loop through filtered rows
    for idx, row in filtered_profiles.iterrows():
        name, unit, node, profile_name = idx  # Unpack MultiIndex

        # Check profile_method
        profile_method = row.get('profile_method')
        if profile_method not in ['upper_limit', 'lower_limit']:
            continue

        target_param = 'upper' if profile_method == 'upper_limit' else 'lower'

        # Build target entity key (name, source, sink)
        entity_name = f"{unit}.{node}"
        if swap_dimensions:
            entity_key = (entity_name, node, unit)  # node_to_unit: source=node, sink=unit
        else:
            entity_key = (entity_name, unit, node)  # unit_to_node: source=unit, sink=node

        # 6. Look up profile data (timeseries first, then constant)
        if 'profile.str.profile' in flextool and profile_name in flextool['profile.str.profile'].columns:
            # Timeseries data
            ts_data[target_param][entity_key] = flextool['profile.str.profile'][profile_name]
        elif 'profile' in flextool and 'profile' in flextool['profile'].columns:
            # Constant data
            if profile_name in flextool['profile'].index:
                constant_data[target_param][entity_key] = flextool['profile'].loc[profile_name, 'profile']

    # 7. Write constant data to target entity dataframe
    for param_type, data in constant_data.items():
        if data:
            param_name = f'profile_limit_{param_type}'
            if target_entity_name not in cesm:
                cesm[target_entity_name] = pd.DataFrame()
            for entity_key, value in data.items():
                cesm[target_entity_name].loc[entity_key, param_name] = value
            # Ensure proper MultiIndex naming
            if isinstance(cesm[target_entity_name].index, pd.MultiIndex):
                cesm[target_entity_name].index.names = ['name', 'source', 'sink']

    # 8. Write timeseries data to target ts dataframes
    for param_type, data in ts_data.items():
        if data:
            df_name = f'{target_ts_prefix}.profile_limit_{param_type}'
            ts_df = pd.DataFrame(data)
            ts_df.columns = pd.MultiIndex.from_tuples(
                ts_df.columns.tolist(),
                names=['name', 'source', 'sink']
            )

            # Convert string index to datetime using lookup if available
            if str_to_datetime_lookup and ts_df.index.dtype == 'object':
                # Use lookup to convert string timestamps to datetime
                new_index = []
                for idx_val in ts_df.index:
                    str_val = str(idx_val)
                    if str_val in str_to_datetime_lookup:
                        new_index.append(str_to_datetime_lookup[str_val])
                    else:
                        # Fallback: try direct conversion
                        new_index.append(pd.to_datetime(idx_val))
                ts_df.index = pd.DatetimeIndex(new_index)
            elif ts_df.index.dtype == 'object':
                # No lookup, try direct conversion
                ts_df.index = pd.to_datetime(ts_df.index)

            ts_df.index.name = 'datetime'
            cesm[df_name] = ts_df

    return cesm


def profile_to_cesm(flextool: Dict[str, pd.DataFrame],
                    cesm: Dict[str, pd.DataFrame],
                    str_to_datetime_lookup: Dict[str, datetime]) -> Dict[str, pd.DataFrame]:
    """
    Transform profile data from FlexTool to CESM format.

    Reads from FlexTool:
    - unit.node.profile: MultiIndex (unit, node, profile) with profile_method column
    - unit.outputNode: connection entities for filtering output connections
    - unit.inputNode: connection entities for filtering input connections
    - profile['profile']: constant profile values
    - profile.ts.profile: timeseries profile values

    Writes to CESM:
    - unit_to_node: profile_limit_upper/profile_limit_lower columns (constants)
    - node_to_unit: profile_limit_upper/profile_limit_lower columns (constants)
    - unit_to_node.ts.profile_limit_upper/lower: timeseries data
    - node_to_unit.ts.profile_limit_upper/lower: timeseries data

    Args:
        flextool: Source FlexTool dataframe dictionary
        cesm: Target CESM dataframe dictionary
        str_to_datetime_lookup: Mapping from string timestamps to datetime objects

    Returns:
        Updated cesm dictionary
    """
    # Process output nodes (unit.outputNode → unit_to_node)
    cesm = _process_profile_to_cesm(
        flextool, cesm,
        connection_df_name='unit.outputNode',
        target_entity_name='unit_to_node',
        target_ts_prefix='unit_to_node.ts',
        swap_dimensions=False,  # source=unit, sink=node
        str_to_datetime_lookup=str_to_datetime_lookup
    )

    # Process input nodes (unit.inputNode → node_to_unit)
    cesm = _process_profile_to_cesm(
        flextool, cesm,
        connection_df_name='unit.inputNode',
        target_entity_name='node_to_unit',
        target_ts_prefix='node_to_unit.ts',
        swap_dimensions=True,  # source=node, sink=unit (swapped)
        str_to_datetime_lookup=str_to_datetime_lookup
    )

    return cesm



def efficiency_to_cesm(flextool: Dict[str, pd.DataFrame],
                        cesm: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Transform efficiency parameters from FlexTool to CESM conversion_rates format.

    Reads from FlexTool:
    - unit/connection DataFrames with columns:
      - efficiency (mandatory) - unitless ratio (e.g., 0.4 = 40%)
      - min_load (optional) - unitless ratio (e.g., 0.5 = 50%)
      - efficiency_at_min_load (optional) - unitless ratio

    Writes to CESM:
    - For constant efficiency (no min_load): single float in conversion_rates column
    - For variable efficiency (min_load present): multiarray DataFrame with
      operating_point and conversion_rate rows

    Args:
        flextool: Dictionary of FlexTool DataFrames (source)
        cesm: Dictionary of CESM DataFrames (will be modified)

    Returns:
        Updated cesm dictionary with efficiency transformations applied
    """
    for entity_type in [('unit', 'unit'), ('connection', 'link')]:
        if entity_type[0] not in flextool:
            continue

        source_df = flextool[entity_type[0]]

        if 'efficiency' not in source_df.columns:
            continue

        # Ensure target DataFrame exists
        if entity_type[1] not in cesm:
            cesm[entity_type[1]] = pd.DataFrame(index=source_df.index)

        map_names = {}
        if isinstance(cesm[entity_type[1]].index, pd.MultiIndex):
            for full_name in cesm[entity_type[1]].index:
                map_names[full_name[0].split('.')[0]] = full_name
                #full_names[entity_name] = [(name, node_A, node_B) for name, node_A, node_B in index_tuples if stripped_name == name.split('.')[0]]
        else:
            for name in cesm[entity_type[1]].index:
                map_names[name] = name

        # Separate entities into constant vs variable efficiency
        constant_entities = []
        variable_entities = []

        for entity_name in source_df.index:
            efficiency = source_df.loc[entity_name, 'efficiency']
            if pd.isna(efficiency):
                continue

            has_min_load = (
                'min_load' in source_df.columns and
                pd.notna(source_df.loc[entity_name, 'min_load'])
            )
            has_eff_at_min = (
                'efficiency_at_min_load' in source_df.columns and
                pd.notna(source_df.loc[entity_name, 'efficiency_at_min_load'])
            )

            if has_min_load or has_eff_at_min:
                variable_entities.append(entity_name)
            else:
                constant_entities.append(entity_name)

        # Process constant efficiency entities
        for source_entity_name, target_entity_name in map_names.items():
            if source_entity_name in constant_entities:
                efficiency = source_df.loc[source_entity_name, 'efficiency']
                # Convert ratio to percentage
                cesm[entity_type[1]].loc[target_entity_name, 'conversion_rates'] = efficiency * 100

        # Process variable efficiency entities (multiarray format)
        if variable_entities:
            multiarray_data = {}

            for entity_name in variable_entities:
                efficiency = source_df.loc[entity_name, 'efficiency']
                min_load = source_df.loc[entity_name, 'min_load'] if 'min_load' in source_df.columns else None
                eff_at_min = source_df.loc[entity_name, 'efficiency_at_min_load'] if 'efficiency_at_min_load' in source_df.columns else None

                # Use efficiency_at_min_load if available, otherwise fall back to efficiency
                if pd.isna(eff_at_min):
                    eff_at_min = efficiency

                # Use min_load if available, otherwise default to some value (shouldn't happen if in variable list)
                if pd.isna(min_load):
                    min_load = 0.0  # Default fallback

                # Build data for this entity
                # Row 0: 100% operating point with efficiency at full load
                # Row 1: min_load operating point with efficiency at min load
                multiarray_data[map_names[entity_name]] = {
                    (0, 'operating_point'): 100.0,
                    (0, 'conversion_rate'): efficiency * 100,
                    (1, 'operating_point'): min_load * 100,
                    (1, 'conversion_rate'): eff_at_min * 100,
                }

            # Create multiarray DataFrame
            if multiarray_data:
                # Convert to DataFrame with MultiIndex
                df_data = pd.DataFrame(multiarray_data)
                df_data.index = pd.MultiIndex.from_tuples(
                    df_data.index.tolist(),
                    names=['index', 'parameter']
                )
                cesm[f'{entity_type[1]}.multiarray.conversion_rates'] = df_data

    return cesm


def transform_to_cesm(flextool: Dict[str, pd.DataFrame],
                      cesm: Dict[str, pd.DataFrame],
                      start_time: datetime) -> Dict[str, pd.DataFrame]:
    """
    Apply all Python-based transformations from FlexTool to CESM.

    This is the main entry point called after the YAML transformer.

    Args:
        flextool: Dictionary of FlexTool DataFrames (source)
        cesm: Dictionary of CESM DataFrames (partially transformed by YAML)
        start_time: Start datetime for the timeline (required)

    Returns:
        Updated cesm dictionary with all Python transformations applied
    """
    # Apply time-related transformations (returns lookup for datetime conversion)
    cesm, str_to_datetime_lookup = time_from_spine(flextool, cesm, start_time)

    # Apply capacity-related transformations
    cesm = capacities_from_spine(flextool, cesm)

    # Apply profile-related transformations
    cesm = profile_to_cesm(flextool, cesm, str_to_datetime_lookup)

    # Apply efficiency-related transformations
    cesm = efficiency_to_cesm(flextool, cesm)

    return cesm
