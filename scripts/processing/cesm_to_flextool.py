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
    'solve_pattern.array.periods_realise_operations',
    'solve_pattern.array.periods_realise_investments',
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

    if missing or empty:
        print("\nError: Required CESM data is missing or incomplete.")
        print("=" * 60)
        if missing:
            print(f"\nMissing tables:")
            for t in missing:
                print(f"  - {t}")
        if empty:
            print(f"\nEmpty tables (no data):")
            for t in empty:
                print(f"  - {t}")
        print("\nThe 'solve_pattern' data defines which periods to solve and")
        print("is required for FlexTool transformation. Please ensure your")
        print("input data includes 'solve_pattern' with at least one entry")
        print("defining 'periods_realise_operations' and")
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


def time_to_spine(flextool, cesm):
    """
    Add temporal data from cesm to flextool.
    """
    # Create dataframe for timeline entity
    flextool['timeline'] = pd.DataFrame(index = ['cesm_timeline'])

    # Create time resolution dataframe
    dt_series = cesm["timeline"].index
    time_diffs = -dt_series.diff(-1).total_seconds() / 3600
    time_diffs = pd.Series(time_diffs)
    time_diffs.iloc[-1] = time_diffs.iloc[-2]

    # Convert timeline index to Zulu time strings for consistency
    timeline_zulu = pd.Index([to_utc_string(dt) for dt in dt_series])

    flextool['timeline.str.timestep_duration'] = pd.DataFrame({
        'cesm_timeline': time_diffs.values},
        index = timeline_zulu,
    )
    flextool['timeline.str.timestep_duration'].index.name = 'datetime'

    # Create timeset dataframe
    flextool['timeset'] = pd.DataFrame({
        'timeline': ['cesm_timeline']},
        index = ['cesm_timeset']
    )

    # Create timeset_duration parameter from solve_pattern
    # Get start_time and duration from solve_pattern
    solve_pattern_df = cesm['solve_pattern']

    # Use first solve_pattern entry (CESM currently supports only one)
    first_solve = solve_pattern_df.iloc[0]

    # Get start_time: from solve_pattern or fall back to first timeline entry
    if 'start_time' in solve_pattern_df.columns and pd.notna(first_solve.get('start_time')):
        start_time = to_utc_string(first_solve['start_time'])
    else:
        # Fall back to first timestep of timeline
        start_time = to_utc_string(dt_series[0])

    # Get duration and ensure it's an integer
    if 'duration' not in solve_pattern_df.columns or pd.isna(first_solve.get('duration')):
        raise ValueError("solve_pattern must have a 'duration' column with a valid value")

    duration_value = first_solve['duration']
    try:
        # Convert to float first (handles strings like "10.0" and ints like 10)
        duration_float = float(duration_value)
        duration = int(duration_float)
        # Ensure no fractional part was lost (10.0 is ok, 10.5 is not)
        if duration != duration_float:
            raise ValueError(f"duration must be a whole number, got {duration_value}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"duration must be convertible to integer, got {duration_value}: {e}")

    # Create timeset_duration map with one row: {start_time: duration}
    flextool['timeset.str.timeset_duration'] = pd.DataFrame({
        'cesm_timeset': [duration]},
        index = [start_time],
    )
    flextool['timeset.str.timeset_duration'].index.name = 'datetime'

    # Create solve.str.period_timeset dataframe
    solve_periods = {}
    all_periods = []
    for index, solve_pattern in cesm['solve_pattern'].iterrows():
        periods = list(cesm["solve_pattern.array.periods_realise_operations"][index]) \
                  + list(cesm["solve_pattern.array.periods_realise_investments"][index])
        solve_periods[index] = list(set(periods))
        for period in periods:
            all_periods.append(period)
    all_periods_unique = list(set(all_periods))

    period_timeset = pd.DataFrame(index=all_periods_unique)
    for solve_name, periods in solve_periods.items():
        for period in periods:
            period_timeset.loc[period, solve_name] = 'cesm_timeset'
    flextool['solve.str.period_timeset'] = period_timeset

    solve_years_represented = pd.DataFrame(index=all_periods_unique)
    for solve_name, periods in solve_periods.items():
        for period in periods:
            for cesm_period, years in cesm['period']['years_represented'].items():
                solve_years_represented.loc[cesm_period, solve_name] = years
    flextool['solve.str.years_represented'] = solve_years_represented


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
    validate_cesm_data(cesm)

    # Transform from CESM to FlexTool (using configuration file)
    print("Transforming to FlexTool format...")
    flextool = transform_data(cesm, args.transformer)

    # Process time parameters separately
    flextool = time_to_spine(flextool, cesm)

    # Write FlexTool dataset to Spine DB (FlexTool format)
    output_url = path_to_sqlite_url(args.output)
    print(f"Writing to Spine database: {output_url}...")
    dataframes_to_spine(flextool, output_url)

    print(f"Data successfully written to {args.output}")


if __name__ == "__main__":
    main()
