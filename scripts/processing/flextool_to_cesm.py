"""
Read FlexTool data from Spine database and convert to CESM DuckDB format.

This script reads FlexTool DataFrames from a Spine database using a
specified scenario filter, transforms them to CESM format, and writes
to a DuckDB file.
"""

import argparse
import importlib.util
from pathlib import Path
from datetime import datetime
from readers.from_spine_db import spine_to_dataframes, list_scenarios
from core.transform_parameters import transform_data
from writers.to_duckdb import dataframes_to_duckdb


def path_to_sqlite_url(path_or_url: str) -> str:
    """Convert file path to SQLite URL if needed.

    Passes through URLs that are already in a valid format (sqlite:, http:, https:).
    Converts plain file paths to sqlite:/// URLs.
    """
    if path_or_url.startswith(("sqlite:", "http:", "https:")):
        return path_or_url
    return f"sqlite:///{Path(path_or_url).resolve()}"


def load_transformer_module(cesm_version: str, flextool_version: str):
    """
    Dynamically load the Python transformer module for the specified versions.

    Args:
        cesm_version: CESM version (e.g., 'cesm_v0.1.0')
        flextool_version: FlexTool version (e.g., 'v3.14.0')

    Returns:
        The loaded module containing transform_to_cesm function
    """
    module_path = Path(f"src/transformers/irena_flextool/{cesm_version}/{flextool_version}/to_cesm.py")

    if not module_path.exists():
        raise FileNotFoundError(
            f"Transformer module not found: {module_path}\n"
            f"Ensure the path exists for versions {cesm_version}/{flextool_version}"
        )

    spec = importlib.util.spec_from_file_location("to_cesm", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_dataframe_summary(dataframes: dict, title: str = "DATAFRAME SUMMARY") -> None:
    """Print a summary of the loaded dataframes."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    # Categorize dataframes
    entity_dfs = {}
    ts_dfs = {}
    str_dfs = {}
    array_dfs = {}

    for name, df in dataframes.items():
        if '.ts.' in name:
            ts_dfs[name] = df
        elif '.str.' in name:
            str_dfs[name] = df
        elif '.array.' in name:
            array_dfs[name] = df
        else:
            entity_dfs[name] = df

    # Print entity dataframes
    if entity_dfs:
        print(f"\nEntity dataframes ({len(entity_dfs)}):")
        print("-" * 40)
        for name in sorted(entity_dfs.keys()):
            df = entity_dfs[name]
            print(f"  {name}")
            print(f"    Shape: {df.shape[0]} entities x {df.shape[1]} parameters")
            if df.shape[1] > 0:
                print(f"    Parameters: {', '.join(str(c) for c in df.columns[:5])}" +
                      (f"... (+{df.shape[1]-5} more)" if df.shape[1] > 5 else ""))

    # Print time series dataframes
    if ts_dfs:
        print(f"\nTime series dataframes ({len(ts_dfs)}):")
        print("-" * 40)
        for name in sorted(ts_dfs.keys()):
            df = ts_dfs[name]
            print(f"  {name}")
            print(f"    Shape: {df.shape[0]} timesteps x {df.shape[1]} entities")

    # Print map (str) dataframes
    if str_dfs:
        print(f"\nMap dataframes ({len(str_dfs)}):")
        print("-" * 40)
        for name in sorted(str_dfs.keys()):
            df = str_dfs[name]
            print(f"  {name}")
            print(f"    Shape: {df.shape[0]} entries x {df.shape[1]} entities")

    # Print array dataframes
    if array_dfs:
        print(f"\nArray dataframes ({len(array_dfs)}):")
        print("-" * 40)
        for name in sorted(array_dfs.keys()):
            df = array_dfs[name]
            print(f"  {name}")
            print(f"    Shape: {df.shape[0]} elements x {df.shape[1]} entities")

    print("\n" + "=" * 60)
    print(f"Total: {len(dataframes)} dataframes")
    print("=" * 60)


def main():
    """Main function to convert FlexTool Spine database to CESM DuckDB."""
    parser = argparse.ArgumentParser(
        description="Convert FlexTool Spine database to CESM DuckDB format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input Spine database file path or URL (e.g., input.sqlite or sqlite:///input.sqlite)"
    )
    parser.add_argument(
        "scenario",
        type=str,
        help="Scenario name to read from the database (required)"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output DuckDB file path"
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios in the database and exit"
    )
    parser.add_argument(
        "--cesm-version",
        "-c",
        type=str,
        default="cesm_v0.1.0",
        help="CESM version (default: cesm_v0.1.0)"
    )
    parser.add_argument(
        "--flextool-version",
        "-f",
        type=str,
        default="v3.14.0",
        help="FlexTool version (default: v3.14.0)"
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Start datetime of first timestep (ISO format, e.g., '2023-01-01T00:00:00'). "
             "Required if database uses non-datetime indexes like 't0001'."
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print detailed summary of dataframes at each stage"
    )

    args = parser.parse_args()

    # Convert path to URL
    input_url = path_to_sqlite_url(args.input)

    # List scenarios if requested
    if args.list_scenarios:
        print(f"Available scenarios in {args.input}:")
        for scenario in list_scenarios(input_url):
            print(f"  - {scenario}")
        return

    # Parse start time if provided
    start_time = None
    if args.start_time:
        try:
            start_time = datetime.fromisoformat(args.start_time)
        except ValueError:
            print("Error: Invalid start-time format. Use ISO format (e.g., '2023-01-01T00:00:00')")
            raise SystemExit(1)
    else:
        # Provide default (no leap year)
        start_time = datetime.fromisoformat('2025-01-01T00:00:00')

    # Determine transformer paths
    transformer_yaml = f"src/transformers/irena_flextool/{args.cesm_version}/{args.flextool_version}/from_flextool.yaml"

    # Validate transformer files exist
    if not Path(transformer_yaml).exists():
        print(f"Error: Transformer configuration not found: {transformer_yaml}")
        print(f"Ensure the path exists for versions {args.cesm_version}/{args.flextool_version}")
        raise SystemExit(1)

    # Load FlexTool data from Spine database
    print(f"Loading FlexTool data from {args.input}...")
    print(f"Using scenario: {args.scenario}")
    print(f"Transformer versions: {args.cesm_version} / {args.flextool_version}")
    if start_time:
        print(f"Start time: {start_time}")

    try:
        flextool = spine_to_dataframes(
            input_url,
            args.scenario
        )
    except Exception as e:
        print(f"\nError reading database: {e}")
        print("\nAvailable scenarios:")
        try:
            for scenario in list_scenarios(input_url):
                print(f"  - {scenario}")
        except Exception:
            print("  Could not list scenarios")
        raise SystemExit(1)

    if args.summary:
        print_dataframe_summary(flextool, "FLEXTOOL INPUT DATA")

    # Transform from FlexTool to CESM format using YAML transformer
    print("\nTransforming to CESM format (YAML transformer)...")
    cesm = transform_data(flextool, transformer_yaml, start_time=start_time)

    # Apply specific Python transformations (time related)
    print("\nApplying Python transformations...")
    try:
        transformer_module = load_transformer_module(args.cesm_version, args.flextool_version)
        cesm = transformer_module.transform_to_cesm(flextool, cesm, start_time)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping Python transformations.")
    except Exception as e:
        print(f"Error in Python transformer: {e}")
        raise SystemExit(1)

    if args.summary:
        print_dataframe_summary(cesm, "CESM OUTPUT DATA")

    # Write CESM dataset to DuckDB
    print(f"\nWriting to DuckDB: {args.output}...")
    dataframes_to_duckdb(cesm, args.output)

    print(f"\nData successfully written to {args.output}")
    print(f"Total: {len(cesm)} CESM dataframes")


if __name__ == "__main__":
    main()
