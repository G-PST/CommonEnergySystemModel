import pandas as pd
import generated.cesm as cesm  # This is the generated class
from linkml_runtime.loaders import yaml_loader
from linkml_runtime.utils.introspection import package_schemaview
from core.linkml_to_dataframes import yaml_to_df
from core.transform_parameters import transform_data
from transformers.griddb import to_griddb
from transformers.griddb.to_sqlite import write_to_sqlite

schema_path="model/cesm.yaml"
griddb_sql_schema_path="src/transformers/griddb/cesm_v0.1.0/v0.1.0/schema.sql"
output_db_path="griddb.sqlite"

# Load your data
database = yaml_loader.load("data/samples/cesm-sample.yaml", target_class=cesm.Database)

# Extract all DataFrames
cesm_dfs = yaml_to_df(database, schema_path=schema_path)

# Transform from CESM to something closer to Griddb (using configuration file)
# griddb_partial = transform_data(cesm, 
#    "transformers/griddb/cesm_v0.1.0/v0.1.0/to_griddb.yaml")

# Process the rest of it
griddb_dfs = to_griddb("cesm_v0.1.0", "v0.1.0", cesm_dfs)

# Write FlexTool dataset to SQLite (GridDB format)
# succeed = dataframes_to_sqlite(griddb, "sqlite:///griddb.sqlite")
write_to_sqlite(griddb_sql_schema_path, griddb_dfs, output_db_path)


print("Done")