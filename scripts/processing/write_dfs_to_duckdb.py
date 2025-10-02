from linkml_runtime.loaders import yaml_loader
from generated.cesm import Database
from core.linkml_to_dataframes import yaml_to_df
from core.dataframes_to_duckdb import write_dataframes_to_duckdb

# Extract DataFrames from LinkML
database = yaml_loader.load("data/samples/cesm-sample.yaml", target_class=Database)
dfs = yaml_to_df(database, schema_path="model/cesm.yaml")

# Write to DuckDB
conn = write_dataframes_to_duckdb(dfs, db_path="cesm.duckdb")

# Query examples
conn.execute("SELECT * FROM balances").fetchdf()
conn.execute("SELECT * FROM balances_flow_profile_wide").fetchdf()