"""
Read CESM data from DuckDB and write to FlexTool Spine database.

This script reads CESM DataFrames from a DuckDB file, transforms them
to FlexTool format, and writes to a Spine database.
"""

import argparse
from pathlib import Path
import pandas as pd
from core.transform_parameters import transform_data
from writers.to_spine_db import dataframes_to_spine
from readers.from_duckdb import dataframes_from_duckdb


REQUIRED_TABLES = [
    'solve_pattern',
]

# At least one of these period arrays must be present
REQUIRED_PERIOD_TABLES = [
#    'solve_pattern.array.periods_realise_operations',
#    'solve_pattern.array.periods_realise_investments',
]


def validate_cesm_data(cesm: dict) -> None:
    """Validate that required CESM tables exist and are non-empty.

    Raises:
        SystemExit: If required tables are missing or empty.
    """
    missing = []
    empty = []

    for table in REQUIRED_TABLES:
        if table not in cesm:
            missing.append(table)
        elif cesm[table].empty:
            empty.append(table)

    # Check that at least one period table exists and is non-empty
    has_period_table = False
    for table in REQUIRED_PERIOD_TABLES:
        if table in cesm and not cesm[table].empty:
            has_period_table = True
            break

    if missing or empty or not has_period_table:
        print("\nError: Required CESM data is missing or incomplete.")
        print("=" * 60)
        if missing:
            print("\nMissing tables:")
            for t in missing:
                print(f"  - {t}")
        if empty:
            print("\nEmpty tables (no data):")
            for t in empty:
                print(f"  - {t}")
        if not has_period_table:
            print("\nMissing period data - at least one of these must be present:")
            for t in REQUIRED_PERIOD_TABLES:
                print(f"  - {t}")
        print("\nThe 'solve_pattern' data defines which periods to solve and")
        print("is required for FlexTool transformation. Please ensure your")
        print("input data includes 'solve_pattern' with at least one entry")
        print("defining 'periods_realise_operations' or")
        print("'periods_realise_investments'.")
        print("=" * 60)
        raise SystemExit(1)


def path_to_sqlite_url(path_or_url: str) -> str:
    """Convert file path to SQLite URL if needed.

    Passes through URLs that are already in a valid format (sqlite:, http:, https:).
    Converts plain file paths to sqlite:/// URLs.
    """
    if path_or_url.startswith(("sqlite:", "http:", "https:")):
        return path_or_url
    return f"sqlite:///{Path(path_or_url).resolve()}"


def to_utc_string(dt) -> str:
    """
    Convert a datetime-like value to ISO 8601 UTC time string.

    Handles pandas Timestamp, datetime, and string inputs.
    If timezone-aware, converts to UTC first.
    If timezone-naive, assumes UTC.
    Returns format: '2023-01-01T00:00:00' (no 'Z' suffix for FlexTool compatibility)
    """
    if isinstance(dt, str):
        # Parse string to datetime first
        dt = pd.to_datetime(dt)

    # Convert to pandas Timestamp if needed
    if not isinstance(dt, pd.Timestamp):
        dt = pd.Timestamp(dt)

    # If timezone-aware, convert to UTC
    if dt.tz is not None:
        dt = dt.tz_convert('UTC')

    # Format as ISO 8601 without Z suffix (FlexTool uses plain strings)
    return dt.strftime('%Y-%m-%dT%H:%M:%S')


def timedelta_to_hours(td) -> float:
    """Convert pandas Timedelta to hours."""
    if td is None or pd.isna(td):
        return None
    if isinstance(td, pd.Timedelta):
        return td.total_seconds() / 3600
    # Fallback for numeric values (assume already in hours)
    return float(td)


def time_to_spine(flextool, cesm):
    """
    Add temporal data from cesm to flextool.

    Handles the following CESM solve_pattern attributes:
    - solve_mode → solve.str.solve_mode
    - periods_realise_operations → solve.array.realized_periods
    - periods_realise_investments → solve.array.realized_invest_periods
    - periods_pass_storage_data → solve.array.fix_storage_periods
    - periods_additional_horizon + periods_realise_investments → solve.array.invest_periods
    - rolling_jump → solve.str.rolling_solve_jump (hours)
    - rolling_jump + rolling_additional_horizon → solve.str.rolling_solve_horizon (hours)
    - duration → solve.str.rolling_duration (hours, for rolling solves)
    - contains_solve_pattern → solve.array.contains_solves
    """
    # Create dataframe for timeline entity
    flextool['timeline'] = pd.DataFrame(index=['cesm_timeline'])

    # Create time resolution dataframe
    dt_series = cesm["timeline"].index
    time_diffs = -dt_series.diff(-1).total_seconds() / 3600
    time_diffs = pd.Series(time_diffs)
    time_diffs.iloc[-1] = time_diffs.iloc[-2]

    # Convert timeline index to Zulu time strings for consistency
    timeline_zulu = pd.Index([to_utc_string(dt) for dt in dt_series])

    flextool['timeline.str.timestep_duration'] = pd.DataFrame({
        'cesm_timeline': time_diffs.values},
        index=timeline_zulu,
    )
    flextool['timeline.str.timestep_duration'].index.name = 'datetime'

    # Create timeset dataframe - one timeset per solve_pattern
    solve_pattern_names = list(cesm['solve_pattern'].index)
    flextool['timeset'] = pd.DataFrame({
        'timeline': ['cesm_timeline'] * len(solve_pattern_names)},
        index=solve_pattern_names
    )

    # Create timeset_duration parameter from solve_pattern.map.start_time_durations
    # or construct from solve_pattern scalar columns (start_time, duration)
    solve_pattern_df = cesm['solve_pattern']

    if 'solve_pattern.ts.start_time_durations' in cesm:
        durations_df = cesm['solve_pattern.ts.start_time_durations']
    elif 'start_time' in solve_pattern_df.columns and 'duration' in solve_pattern_df.columns:
        # Construct from scalar columns
        durations_data = {}
        for solve_name in solve_pattern_df.index:
            start_time = solve_pattern_df.loc[solve_name, 'start_time']
            duration = solve_pattern_df.loc[solve_name, 'duration']
            if pd.notna(start_time) and pd.notna(duration):
                if start_time not in durations_data:
                    durations_data[start_time] = {}
                durations_data[start_time][solve_name] = duration
        durations_df = pd.DataFrame(durations_data).T
        durations_df.index = pd.to_datetime(durations_df.index)
    else:
        raise ValueError("solve_pattern must have 'start_time_durations' or 'start_time' and 'duration' columns")

    # Calculate timestep duration from timeline for converting Timedelta to timestep count
    timestep_hours = time_diffs.iloc[0]

    # Convert Timedelta durations to timestep counts
    timeset_duration_data = {}
    for start_time in durations_df.index:
        start_time_str = to_utc_string(start_time)
        timeset_duration_data[start_time_str] = {}
        for solve_name in durations_df.columns:
            duration_td = durations_df.loc[start_time, solve_name]
            if pd.notna(duration_td):
                if isinstance(duration_td, pd.Timedelta):
                    duration_hours = duration_td.total_seconds() / 3600
                    duration = int(duration_hours / timestep_hours)
                else:
                    duration = int(float(duration_td))
                timeset_duration_data[start_time_str][solve_name] = duration

    # Create DataFrame with start_time as index, solve patterns as columns
    flextool['timeset.str.timeset_duration'] = pd.DataFrame(timeset_duration_data).T
    flextool['timeset.str.timeset_duration'].index.name = 'datetime'

    # Collect all periods from all solve patterns
    solve_periods = {}
    all_periods = []
    for index, solve_pattern in cesm['solve_pattern'].iterrows():
        periods_ops = []
        periods_inv = []
        periods_add = []
        periods_add_inv = []  # additional investment horizon periods
        periods_add_ops = []  # additional operations horizon periods

        if 'solve_pattern.array.periods_realise_operations' in cesm:
            if index in cesm['solve_pattern.array.periods_realise_operations'].columns:
                periods_ops = list(cesm['solve_pattern.array.periods_realise_operations'][index].dropna())

        if 'solve_pattern.array.periods_realise_investments' in cesm:
            if index in cesm['solve_pattern.array.periods_realise_investments'].columns:
                periods_inv = list(cesm['solve_pattern.array.periods_realise_investments'][index].dropna())

        if 'solve_pattern.array.periods_additional_investments_horizon' in cesm:
            if index in cesm['solve_pattern.array.periods_additional_investments_horizon'].columns:
                periods_add_inv = list(cesm['solve_pattern.array.periods_additional_investments_horizon'][index].dropna())

        if 'solve_pattern.array.periods_additional_operations_horizon' in cesm:
            if index in cesm['solve_pattern.array.periods_additional_operations_horizon'].columns:
                periods_add_ops = list(cesm['solve_pattern.array.periods_additional_operations_horizon'][index].dropna())

        solve_periods[index] = {
            'operations': periods_ops,
            'investments': periods_inv,
            'additional_investments': periods_add_inv,
            'additional_operations': periods_add_ops,
        }
        all_periods.extend(periods_ops + periods_inv + periods_add + periods_add_inv + periods_add_ops)

    # Create solve.str.period_timeset dataframe
    # Only include periods that are actually used in each solve
    period_timeset = pd.DataFrame()
    for solve_name, periods_dict in solve_periods.items():
        # Combine all period lists and remove duplicates while preserving order
        all_periods_list = (
            periods_dict['operations'] +
            periods_dict['investments'] +
            periods_dict['additional_investments'] +
            periods_dict['additional_operations']
        )
        all_solve_periods = list(dict.fromkeys(all_periods_list))  # Dedupe, preserve order
        for period in all_solve_periods:
            period_timeset.loc[period, solve_name] = solve_name
    flextool['solve.str.period_timeset'] = period_timeset

    # Create solve.str.years_represented
    # Only include periods that are actually used in each solve
    solve_years_represented = pd.DataFrame()
    for solve_name, periods_dict in solve_periods.items():
        # Combine all period lists and remove duplicates while preserving order
        all_periods_list = (
            periods_dict['operations'] +
            periods_dict['investments'] +
            periods_dict['additional_investments'] +
            periods_dict['additional_operations']
        )
        all_solve_periods = list(dict.fromkeys(all_periods_list))  # Dedupe, preserve order
        for period in all_solve_periods:
            if period in cesm['period']['years_represented'].index:
                years = cesm['period']['years_represented'].loc[period]
                solve_years_represented.loc[period, solve_name] = years
    flextool['solve.str.years_represented'] = solve_years_represented

    # Create solve entity dataframe with single-valued parameters
    solve_data = {}
    for index, row in solve_pattern_df.iterrows():
        solve_data[index] = {}

        # solve_mode
        if 'solve_mode' in solve_pattern_df.columns and pd.notna(row.get('solve_mode')):
            if row['solve_mode'] == 'rolling_solve':
                solve_data[index]['solve_mode'] = 'rolling_window'    
            else:
                solve_data[index]['solve_mode'] = row['solve_mode']

        # rolling_solve_jump (from rolling_jump)
        if 'rolling_jump' in solve_pattern_df.columns and pd.notna(row.get('rolling_jump')):
            jump_hours = timedelta_to_hours(row['rolling_jump'])
            if jump_hours is not None:
                solve_data[index]['rolling_solve_jump'] = jump_hours

                # rolling_solve_horizon = rolling_jump + rolling_additional_horizon
                horizon_hours = jump_hours
                if 'rolling_additional_horizon' in solve_pattern_df.columns and pd.notna(row.get('rolling_additional_horizon')):
                    add_horizon = timedelta_to_hours(row['rolling_additional_horizon'])
                    if add_horizon is not None:
                        horizon_hours += add_horizon
                solve_data[index]['rolling_solve_horizon'] = horizon_hours

        # rolling_duration (from duration array, for rolling solves)
        if row.get('solve_mode') == 'rolling_solve' and 'solve_pattern.map.start_time_durations' in cesm:
            if index in cesm['solve_pattern.map.start_time_durations'].columns:
                duration_vals = cesm['solve_pattern.map.start_time_durations'][index].dropna()
                if len(duration_vals) > 0:
                    dur_hours = timedelta_to_hours(duration_vals.iloc[0])
                    if dur_hours is not None:
                        solve_data[index]['rolling_duration'] = dur_hours

    # Create the solve entity dataframe
    if solve_data:
        flextool['solve'] = pd.DataFrame.from_dict(solve_data, orient='index')
        flextool['solve'].index.name = 'solve'

    # Create solve.array.realized_periods (from periods_realise_operations)
    if 'solve_pattern.array.periods_realise_operations' in cesm:
        flextool['solve.array.realized_periods'] = cesm['solve_pattern.array.periods_realise_operations'].copy()

    # Create solve.array.realized_invest_periods (from periods_realise_investments)
    if 'solve_pattern.array.periods_realise_investments' in cesm:
        flextool['solve.array.realized_invest_periods'] = cesm['solve_pattern.array.periods_realise_investments'].copy()

    # Create solve.array.invest_periods (investments + additional_investments)
    invest_periods_data = {}
    for solve_name, periods_dict in solve_periods.items():
        # Dedupe while preserving order
        combined_list = periods_dict['investments'] + periods_dict['additional_investments']
        combined = list(dict.fromkeys(combined_list))
        if combined:
            invest_periods_data[solve_name] = combined
    if invest_periods_data:
        max_len = max(len(v) for v in invest_periods_data.values())
        invest_periods_df = pd.DataFrame(index=range(max_len))
        for solve_name, periods in invest_periods_data.items():
            invest_periods_df[solve_name] = pd.Series(periods)
        flextool['solve.array.invest_periods'] = invest_periods_df

    # Create solve.array.fix_storage_periods (from periods_pass_storage_data)
    if 'solve_pattern.array.periods_pass_storage_data' in cesm:
        flextool['solve.array.fix_storage_periods'] = cesm['solve_pattern.array.periods_pass_storage_data'].copy()

    # Create solve.array.contains_solves (from contains_solve_pattern)
    if 'contains_solve_pattern' in solve_pattern_df.columns:
        contains_solves_data = {}
        for index, row in solve_pattern_df.iterrows():
            if pd.notna(row.get('contains_solve_pattern')):
                val = row['contains_solve_pattern']
                # Handle both single value and list
                if isinstance(val, list):
                    contains_solves_data[index] = val
                else:
                    contains_solves_data[index] = [val]
        if contains_solves_data:
            max_len = max(len(v) for v in contains_solves_data.values())
            contains_df = pd.DataFrame(index=range(max_len))
            for solve_name, solves in contains_solves_data.items():
                contains_df[solve_name] = pd.Series(solves)
            flextool['solve.array.contains_solves'] = contains_df

    return flextool


def conversion_rates_to_spine(flextool: dict, cesm: dict) -> dict:
    """
    Convert CESM conversion_rates back to FlexTool efficiency parameters.

    This is the reverse of efficiency_to_cesm in to_cesm.py.

    Reads from CESM:
    - cesm[entity_type[0]]['conversion_rates'] - constant efficiency as percentage
    - cesm[f'{entity_type[0]}.multiarray.conversion_rates'] - variable efficiency DataFrame
      with MultiIndex (index, parameter) where parameter is 'operating_point' or 'conversion_rate'

    Writes to FlexTool:
    - flextool[entity_type[1]]['efficiency'] = conversion_rate at 100% / 100
    - flextool[entity_type[1]]['min_load'] = operating_point at index 1 / 100
    - flextool[entity_type[1]]['efficiency_at_min_load'] = conversion_rate at index 1 / 100

    Args:
        flextool: Dictionary of FlexTool DataFrames (will be modified)
        cesm: Dictionary of CESM DataFrames (source)

    Returns:
        Updated flextool dictionary with efficiency transformations applied
    """
    for entity_type in [('unit', 'unit'), ('link', 'connection')]:
        multiarray_key = f'{entity_type[0]}.multiarray.conversion_rates'

        # Process multiarray (variable efficiency) first
        if multiarray_key in cesm:
            multiarray_df = cesm[multiarray_key]

            # Ensure target DataFrame exists
            if entity_type[0] not in flextool:
                flextool[entity_type[0]] = pd.DataFrame()

            for entity_name in multiarray_df.columns:
                entity_data = multiarray_df[entity_name].dropna()

                # Get all unique indices (first level of MultiIndex) after dropping NaNs
                indices = entity_data.index.get_level_values('index').unique()
                last_idx = indices.max()

                # Extract values for index 0 (100% operating point)
                efficiency_pct = entity_data.loc[(0, 'conversion_rate')]

                # Extract values for the last index (min_load operating point)
                min_load_pct = entity_data.loc[(last_idx, 'operating_point')]
                eff_at_min_pct = entity_data.loc[(last_idx, 'conversion_rate')]

                # Convert percentages to ratios and write to flextool
                flextool[entity_type[1]].loc[entity_name, 'efficiency'] = efficiency_pct / 100
                flextool[entity_type[1]].loc[entity_name, 'min_load'] = min_load_pct / 100
                flextool[entity_type[1]].loc[entity_name, 'efficiency_at_min_load'] = eff_at_min_pct / 100

        # Process constant efficiency
        if entity_type[0] in cesm and 'conversion_rates' in cesm[entity_type[0]].columns:
            source_df = cesm[entity_type[0]]

            # Ensure target DataFrame exists
            if entity_type[1] not in flextool:
                flextool[entity_type[1]] = pd.DataFrame()

            for entity_name in source_df.index:
                conversion_rate = source_df.loc[entity_name, 'conversion_rates']
                if isinstance(entity_name, tuple):
                    target_entity_name = entity_name[0]
                else:
                    target_entity_name = entity_name
                if pd.notna(conversion_rate):
                    # Skip entities already processed via multiarray
                    if target_entity_name not in flextool[entity_type[1]].index or \
                       'efficiency' not in flextool[entity_type[1]].columns or \
                       pd.isna(flextool[entity_type[1]].loc[target_entity_name, 'efficiency']):
                        # Convert percentage to ratio
                        flextool[entity_type[1]].loc[target_entity_name, 'efficiency'] = conversion_rate / 100

    return flextool


def main():
    """Main function to convert CESM DuckDB data to FlexTool Spine database."""
    parser = argparse.ArgumentParser(
        description="Convert CESM DuckDB data to FlexTool Spine database format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input DuckDB file path"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output Spine database file path or URL (e.g., output.sqlite or sqlite:///output.sqlite)"
    )
    parser.add_argument(
        "--transformer",
        "-t",
        type=str,
        default="src/transformers/irena_flextool/cesm_v0.1.0/v3.14.0/to_flextool.yaml",
        help="Transformer configuration file path (default: src/transformers/irena_flextool/cesm_v0.1.0/v3.14.0/to_flextool.yaml)"
    )

    args = parser.parse_args()

    # Load CESM data from DuckDB
    print(f"Loading CESM data from {args.input}...")
    cesm = dataframes_from_duckdb(args.input)

    # Validate required tables exist
    #validate_cesm_data(cesm)

    # Transform from CESM to FlexTool (using configuration file)
    print("Transforming to FlexTool format...")
    flextool = transform_data(cesm, args.transformer)

    # Process time parameters separately
    flextool = time_to_spine(flextool, cesm)

    # Convert conversion_rates to efficiency parameters
    flextool = conversion_rates_to_spine(flextool, cesm)

    # Write FlexTool dataset to Spine DB (FlexTool format)
    output_url = path_to_sqlite_url(args.output)
    print(f"Writing to Spine database: {output_url}...")
    dataframes_to_spine(flextool, output_url)

    print(f"Data successfully written to {args.output}")


if __name__ == "__main__":
    main()
