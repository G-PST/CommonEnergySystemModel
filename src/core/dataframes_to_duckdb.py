import duckdb
import pandas as pd
from typing import Dict, List
import re

def write_dataframes_to_duckdb(dfs: Dict[str, pd.DataFrame], db_path: str = None) -> duckdb.DuckDBPyConnection:
    """
    Write LinkML-extracted DataFrames to DuckDB database.
    
    Args:
        dfs: Dictionary of DataFrames from extract_dataframes_from_database()
        db_path: Path to DuckDB file (None for in-memory)
    
    Returns:
        DuckDB connection object
    """
    
    # Create DuckDB connection
    if db_path:
        conn = duckdb.connect(db_path)
    else:
        conn = duckdb.connect()  # In-memory database
    
    # Separate single-dimensional and time-series DataFrames
    single_dim_dfs = {}
    timeseries_dfs = {}
    
    for name, df in dfs.items():
        if name.endswith('_df'):
            single_dim_dfs[name] = df
        elif name.endswith('_ts'):
            timeseries_dfs[name] = df
    
    # Create and populate single-dimensional tables
    for table_name, df in single_dim_dfs.items():
        # Remove '_df' suffix for table name
        clean_table_name = table_name[:-3]
        _create_and_populate_table(conn, clean_table_name, df)
    
    # Create and populate time-series tables
    for table_name, df in timeseries_dfs.items():
        # Convert 'balances.flow_profile_ts' to 'balances_flow_profile'
        clean_table_name = table_name.replace('.', '_').replace('_ts', '')
        _create_and_populate_timeseries_table(conn, clean_table_name, df)
    
    return conn

def _create_and_populate_table(conn: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame):
    """Create and populate a single-dimensional table."""
    
    # Generate CREATE TABLE statement
    col_definitions = []
    for col in df.columns:
        col_type = _infer_duckdb_type(df[col])
        # Handle special identifier columns
        if col in ['name', 'id']:
            col_definitions.append(f"{col} {col_type} NOT NULL")
        else:
            col_definitions.append(f"{col} {col_type}")
    
    # Add primary key if name or id exists
    primary_key = ""
    if 'name' in df.columns:
        primary_key = ", PRIMARY KEY (name)"
    elif 'id' in df.columns:
        primary_key = ", PRIMARY KEY (id)"
    
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join(col_definitions)}{primary_key}
    )
    """
    
    conn.execute(create_sql)
    
    # Insert data
    conn.register('temp_df', df)
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
    conn.unregister('temp_df')
    
    print(f"Created table '{table_name}' with {len(df)} rows")

def _create_and_populate_timeseries_table(conn: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame):
    """Create and populate a time-series table with dynamic columns."""
    
    # Convert wide format to long format for normalized storage
    if 'datetime' not in df.columns:
        raise ValueError(f"Time-series DataFrame {table_name} must have 'datetime' column")
    
    # Melt the DataFrame to long format
    entity_columns = [col for col in df.columns if col != 'datetime']
    
    if not entity_columns:
        print(f"Warning: No entity columns found in {table_name}")
        return
    
    melted_df = df.melt(
        id_vars=['datetime'], 
        value_vars=entity_columns,
        var_name='entity_name',
        value_name='value'
    )
    
    # Remove rows with null values
    melted_df = melted_df.dropna(subset=['value'])
    
    # Create normalized time-series table
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        datetime TIMESTAMP NOT NULL,
        entity_name VARCHAR NOT NULL,
        value DOUBLE,
        PRIMARY KEY (datetime, entity_name)
    )
    """
    
    conn.execute(create_sql)
    
    # Insert data
    conn.register('temp_ts_df', melted_df)
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_ts_df")
    conn.unregister('temp_ts_df')
    
    print(f"Created time-series table '{table_name}' with {len(melted_df)} rows")
    
    # Also create a view for wide format access
    _create_timeseries_view(conn, table_name, entity_columns)

def _create_timeseries_view(conn: duckdb.DuckDBPyConnection, table_name: str, entity_columns: List[str]):
    """Create a view that pivots time-series data back to wide format."""
    
    view_name = f"{table_name}_wide"
    
    # Generate pivot columns
    pivot_cols = [f"'{entity}'" for entity in entity_columns]
    
    pivot_sql = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    SELECT datetime,
           {', '.join([f"MAX(CASE WHEN entity_name = '{entity}' THEN value END) AS {entity}" 
                      for entity in entity_columns])}
    FROM {table_name}
    GROUP BY datetime
    ORDER BY datetime
    """
    
    conn.execute(pivot_sql)
    print(f"Created wide-format view '{view_name}'")

def _infer_duckdb_type(series: pd.Series) -> str:
    """Infer DuckDB column type from pandas Series."""
    
    if series.dtype == 'object':
        # Check if it's a string column
        if series.dropna().apply(lambda x: isinstance(x, str)).all():
            return 'VARCHAR'
        else:
            return 'VARCHAR'  # Default for mixed types
    elif series.dtype in ['int64', 'int32']:
        return 'INTEGER'
    elif series.dtype in ['float64', 'float32']:
        return 'DOUBLE'
    elif series.dtype == 'bool':
        return 'BOOLEAN'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'TIMESTAMP'
    else:
        return 'VARCHAR'  # Default fallback

def query_examples(conn: duckdb.DuckDBPyConnection):
    """Example queries for the created database."""
    
    print("\n=== Example Queries ===")
    
    # List all tables
    print("\n1. Available tables:")
    tables = conn.execute("SHOW TABLES").fetchdf()
    print(tables)
    
    # Query single-dimensional data
    print("\n2. Balances data:")
    try:
        balances = conn.execute("SELECT * FROM balances LIMIT 5").fetchdf()
        print(balances)
    except:
        print("No balances table found")
    
    # Query time-series data (normalized)
    print("\n3. Flow profile time-series (normalized):")
    try:
        flow_ts = conn.execute("""
            SELECT datetime, entity_name, value 
            FROM balances_flow_profile 
            WHERE datetime <= '2023-01-01T02:00'
            ORDER BY datetime, entity_name
        """).fetchdf()
        print(flow_ts)
    except:
        print("No balances_flow_profile table found")
    
    # Query time-series data (wide format view)
    print("\n4. Flow profile time-series (wide format):")
    try:
        flow_wide = conn.execute("""
            SELECT * FROM balances_flow_profile_wide 
            WHERE datetime <= '2023-01-01T02:00'
            ORDER BY datetime
        """).fetchdf()
        print(flow_wide)
    except:
        print("No balances_flow_profile_wide view found")

# Example usage
def example_usage():
    """Complete example of extracting from LinkML and writing to DuckDB."""
    
    # Assuming you have the dfs from the previous extractor function
    # dfs = extract_dataframes_from_database(database, schema_path="cesm.yaml")
    
    # Example DataFrame structure (replace with your actual dfs)
    example_dfs = {
        'balances_df': pd.DataFrame({
            'name': ['west', 'east', 'heat'],
            'id': [1, 2, 3],
            'flow_annual': [100000.0, 80000.0, None],
            'flow_scaling_method': ['scale_to_annual', 'scale_to_annual', 'use_profile_directly'],
            'penalty_upward': [10000.0, 10000.0, 10000.0]
        }),
        'balances.flow_profile_ts': pd.DataFrame({
            'datetime': pd.to_datetime(['2023-01-01T00:00', '2023-01-01T01:00', '2023-01-01T02:00']),
            'west': [-1002.1, -980.7, -968.0],
            'east': [-1002.1, -980.7, -968.0],
            'heat': [-30.0, -40.0, -50.0]
        })
    }
    
    # Write to DuckDB
    conn = write_dataframes_to_duckdb(example_dfs, db_path="cesm_data.duckdb")
    
    # Run example queries
    query_examples(conn)
    
    # Close connection
    conn.close()
    
    print("\nDatabase created successfully!")

if __name__ == "__main__":
    example_usage()