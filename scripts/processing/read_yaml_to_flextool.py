import pandas as pd
from generated.cesm import Database  # This is the generated class
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
    # Create time resolution dataframe
    dt_series = pd.to_datetime(cesm["timeline"])
    time_diffs = -dt_series.diff(-1).total_seconds() / 3600
    time_diffs = pd.Series(time_diffs)
    time_diffs.iloc[-1] = time_diffs.iloc[-2]
    
    timeline = pd.DataFrame({
        'datetime': cesm["timeline"],
        'cesm_timeline': time_diffs
    })

    flextool['timeline'] = pd.DataFrame({
        'name': ['cesm_timeline']
    })
    flextool['timeline.table.timestep_duration'] = timeline

    # Create solve.table.period_timeset dataframe
    periods = set()
    periods.update(cesm['solve_pattern']['periods_realise_investments'].iloc[0])
    periods.update(cesm['solve_pattern']['periods_realise_operations'].iloc[0])
    periods.update(cesm['solve_pattern']['periods_additional_horizon'].iloc[0])

    flextool['timeset'] = pd.DataFrame({
        'name': ['cesm_timeset'],
        'timeline': ['cesm_timeline']
    })
    period_timeset = pd.DataFrame({'period': list(periods)})
    for solve_name in flextool['solve']['name']:
        period_timeset[solve_name] = 'cesm_timeset'
    
    flextool['solve.table.period_timeset'] = period_timeset

    return flextool

# Load your data
database = yaml_loader.load("data/samples/cesm-sample.yaml", target_class=Database)

# Extract all DataFrames
cesm = yaml_to_df(database, schema_path="model/cesm.yaml")

# Transform from CESM to FlexTool (using configuration file)
flextool = transform_data(cesm, 
    "transformers/irena_flextool/cesm_v0.1.0/v3.14.0/to_flextool.yaml")

# Process time parameters separately
flextool = time_to_spine(flextool, cesm)

# Write FlexTool dataset to Spine DB (FlexTool format)
dataframes_to_spine(flextool, "sqlite:///input_data_test.sqlite")

print("foo")