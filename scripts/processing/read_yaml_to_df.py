import pandas as pd
from generated.cesm import Database  # This is the generated class
from linkml_runtime.loaders import yaml_loader
from core.linkml_to_dataframes import yaml_to_df
from core.transform_parameters import transform_data
from writers.to_spine_db import dataframes_to_spine

# Load your data
database = yaml_loader.load("data/samples/cesm-sample.yaml", target_class=Database)

# Extract all DataFrames
dfs = yaml_to_df(database, schema_path="model/cesm.yaml")

# Access specific DataFrames
balances_df = dfs['balance']
flow_profile_ts = dfs['balance.ts.flow_profile']

# debug_cesm_structure(database)
# debug_cesm_entities(database)
foo = transform_data(dfs, "transformers/irena_flextool/cesm_v0.1.0/v3.14.0/parameters_to_flextool.yaml")

timeline = pd.DataFrame({
    'datetime': pd.date_range('2023-01-01', periods=10, freq='H')
})

dataframes_to_spine(foo, "sqlite:///input_data_test.sqlite", timeline)

print("foo")