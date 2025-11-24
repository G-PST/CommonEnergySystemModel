import argparse
import pandas as pd
from generated.cesm_pydantic import Dataset  # This is the generated class
from linkml_runtime.loaders import yaml_loader
from core.linkml_to_dataframes import yaml_to_df
from core.transform_parameters import transform_data
from writers.to_spine_db import dataframes_to_spine
from spinedb_api import DatabaseMapping
from spinedb_api.exception import NothingToCommit

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
    
    flextool['timeline.str.timestep_duration'] = pd.DataFrame({
        'cesm_timeline': time_diffs.values},
        index = cesm["timeline"].index,
    )
    flextool['timeline.str.timestep_duration'].index.name = 'datetime'

    # Create timeset dataframe
    flextool['timeset'] = pd.DataFrame({
        'timeline': ['cesm_timeline']},
        index = ['cesm_timeset']
    )
    
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
    """Main function to convert CESM YAML data to FlexTool Spine database."""
    parser = argparse.ArgumentParser(
        description="Convert CESM YAML data to FlexTool Spine database format"
    )
    parser.add_argument(
        "--database",
        "-d",
        type=str,
        default="sqlite:///input_data.sqlite",
        help="Database file url (default: sqlite:///input_data.sqlite)"
    )

    args = parser.parse_args()

    # Load your data
    dataset = yaml_loader.load("data/samples/cesm-sample.yaml", target_class=Dataset)

    # Extract all DataFrames
    cesm = yaml_to_df(dataset, schema_path="model/cesm.yaml")

    # Transform from CESM to FlexTool (using configuration file)
    flextool = transform_data(cesm,
        "src/transformers/irena_flextool/cesm_v0.1.0/v3.14.0/to_flextool.yaml")

    # Process time parameters separately
    flextool = time_to_spine(flextool, cesm)

    # Write FlexTool dataset to Spine DB (FlexTool format)
    dataframes_to_spine(flextool, args.database)

    print(f"Data successfully written to {args.database}")


if __name__ == "__main__":
    main()
