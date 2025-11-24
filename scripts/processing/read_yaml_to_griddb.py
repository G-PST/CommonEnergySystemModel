import pandas as pd
from generated.cesm_pydantic import Dataset  # This is the generated class
from linkml_runtime.loaders import yaml_loader
from linkml_runtime.utils.introspection import package_schemaview
from core.linkml_to_dataframes import yaml_to_df
from core.transform_parameters import transform_data
from transformers.griddb import to_griddb
from transformers.griddb.to_sqlite import write_to_sqlite
from os import path
import sys
sys.path.append(path.abspath('scripts'))
from helpers.print_dataframe_structures import print_dataframes_structure

schema_path="model/cesm.yaml"
griddb_sql_schema_path="src/transformers/griddb/cesm_v0.1.0/v0.1.0/schema.sql"
output_db_path="griddb.sqlite"

# Load your data
dataset = yaml_loader.load("data/samples/cesm-sample.yaml", target_class=Dataset)

# Extract all DataFrames
cesm_dfs = yaml_to_df(dataset, schema_path=schema_path)
print_dataframes_structure(cesm_dfs, output_file='data_structure_cesm.txt')

# Process the transformation
griddb_dfs = to_griddb("cesm_v0.1.0", "v0.1.0", cesm_dfs)
print_dataframes_structure(griddb_dfs, output_file='data_structure_griddb.txt')

# Write FlexTool dataset to SQLite (GridDB format)
# succeed = dataframes_to_sqlite(griddb, "sqlite:///griddb.sqlite")
write_to_sqlite(griddb_sql_schema_path, griddb_dfs, output_db_path)


print("Done")