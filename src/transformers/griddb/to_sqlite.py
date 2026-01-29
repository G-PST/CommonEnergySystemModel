"""
SQLite dumper for target dataframes.

This module creates and populates an SQLite database from the target dataframes,
using the provided SQL schema for initialization.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict


def _insert_or_replace(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    """
    Insert or replace rows in an existing table.

    Uses INSERT OR REPLACE to update existing rows (matched by primary key)
    or insert new ones. This is used for incremental updates.

    Args:
        conn: SQLite connection
        table_name: Name of the table to insert into
        df: DataFrame with data to insert/replace
    """
    if df.empty:
        return

    columns = df.columns.tolist()
    placeholders = ', '.join(['?' for _ in columns])
    column_names = ', '.join([f'"{col}"' for col in columns])

    sql = f'INSERT OR REPLACE INTO "{table_name}" ({column_names}) VALUES ({placeholders})'

    # Convert DataFrame to list of tuples for executemany
    # Handle None/NaN conversion
    data = []
    for row in df.itertuples(index=False):
        row_data = []
        for val in row:
            if pd.isna(val):
                row_data.append(None)
            else:
                row_data.append(val)
        data.append(tuple(row_data))

    cursor = conn.cursor()
    cursor.executemany(sql, data)
    conn.commit()


def write_to_sqlite(schema_path: str,
                   target_dataframes: Dict[str, pd.DataFrame],
                   output_db_path: str,
                   clear_existing: bool = True) -> None:
    """
    Create SQLite database from schema and populate with dataframes.

    Args:
        schema_path: Path to schema.sql file
        target_dataframes: Dictionary of dataframes matching schema tables
        output_db_path: Path for output SQLite database file
        clear_existing: If True, delete existing database and create fresh.
                       If False, add/replace data in existing database.
                       Default: True.

    Raises:
        FileNotFoundError: If schema file doesn't exist
        sqlite3.Error: If database operations fail
    """

    # Validate schema file exists
    schema_file = Path(schema_path)
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    # Read schema SQL
    print(f"Reading schema from: {schema_path}")
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()

    # Create/overwrite database based on clear_existing flag
    output_path = Path(output_db_path)
    if clear_existing:
        if output_path.exists():
            print(f"Removing existing database: {output_db_path}")
            output_path.unlink()
        print(f"Creating database: {output_db_path}")
    else:
        if output_path.exists():
            print(f"Updating existing database: {output_db_path}")
        else:
            print(f"Creating new database: {output_db_path}")
            # If database doesn't exist, we need to create schema
            clear_existing = True  # Force schema creation for new database
    
    # Connect and initialize schema
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()

    try:
        if clear_existing:
            # Execute schema using executescript
            # This handles multiple statements including DROP, CREATE, and INSERT
            print("Initializing schema...")
            cursor.executescript(schema_sql)
            conn.commit()
            print("Schema initialized successfully")
        else:
            print("Skipping schema initialization (incremental update mode)")
        
        # Define table insertion order (respecting foreign keys)
        # Tables must be inserted in dependency order
        table_order = [
            # entity_types already populated by schema
            'entities',
            'prime_mover_types',
            'fuels',
            'planning_regions',
            'balancing_topologies',
            'arcs',
            'transmission_lines',
            'transmission_interchanges',
            'generation_units',
            'storage_units',
            'hydro_reservoir',
            'hydro_reservoir_connections',
            'supply_technologies',
            'storage_technologies',
            'transport_technologies',
            'operational_data',
            'attributes',
            'supplemental_attributes',
            'supplemental_attributes_association',
            'time_series_associations',
            'static_time_series_data',
            'deterministic_forecast_data',
            'loads'
        ]
        
        # Insert data for each table
        print("\nInserting data into tables...")
        for table_name in table_order:
            if table_name in target_dataframes:
                df = target_dataframes[table_name]

                if df is not None and not df.empty:
                    # Convert DataFrame to SQL
                    # Note: Generated columns (like operational_cost_type, json_type)
                    # should not be in dataframes - they're computed by SQLite
                    try:
                        # Replace NaN with None for proper NULL insertion
                        df = df.where(pd.notna(df), None)

                        if clear_existing:
                            # Fresh database: use append mode
                            df.to_sql(
                                table_name,
                                conn,
                                if_exists='append',
                                index=False,
                                method='multi',  # Faster bulk insert
                                chunksize=1000   # Process in chunks for large datasets
                            )
                        else:
                            # Incremental update: use INSERT OR REPLACE
                            _insert_or_replace(conn, table_name, df)

                        print(f"  ✓ {table_name}: {len(df)} rows")
                    except Exception as e:
                        print(f"  ✗ {table_name}: Error - {str(e)}")
                        # Continue with other tables even if one fails
                else:
                    print(f"  ○ {table_name}: Empty (skipped)")
            else:
                print(f"  ○ {table_name}: Not in dataframes (skipped)")
        
        # Commit all changes
        conn.commit()
        print(f"\n✓ Database created successfully: {output_db_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("DATABASE SUMMARY")
        print("="*60)
        
        for table_name in table_order:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"  {table_name:40s} {count:6d} rows")
            except sqlite3.OperationalError:
                # Table might not exist
                pass
        
        print("="*60)
        
    except sqlite3.Error as e:
        print(f"\n✗ Database error: {str(e)}")
        conn.rollback()
        raise
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        conn.rollback()
        raise
    
    finally:
        conn.close()
        print(f"\nDatabase connection closed")


def verify_database(db_path: str) -> Dict[str, int]:
    """
    Verify database integrity and return row counts.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dictionary mapping table names to row counts
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    row_counts = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_counts[table] = cursor.fetchone()[0]
    
    conn.close()
    return row_counts


def export_table_to_csv(db_path: str, table_name: str, output_csv: str) -> None:
    """
    Export a single table from SQLite to CSV.
    
    Args:
        db_path: Path to SQLite database
        table_name: Name of table to export
        output_csv: Output CSV file path
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    df.to_csv(output_csv, index=False)
    conn.close()
    print(f"Exported {len(df)} rows from {table_name} to {output_csv}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python dumper.py <schema.sql> <output.db> <verify_only>")
        print("  verify_only: 'true' to only verify existing database, 'false' to create new")
        sys.exit(1)
    
    schema_path = sys.argv[1]
    db_path = sys.argv[2]
    verify_only = sys.argv[3].lower() == 'true'
    
    if verify_only:
        print(f"Verifying database: {db_path}")
        row_counts = verify_database(db_path)
        print("\nTable row counts:")
        for table, count in sorted(row_counts.items()):
            print(f"  {table:40s} {count:6d} rows")
    else:
        print("Error: This example requires target dataframes")
        print("Use dump_to_sqlite() function in your code")