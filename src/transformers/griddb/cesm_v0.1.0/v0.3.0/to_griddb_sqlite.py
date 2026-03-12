"""
SQLite writer for GridDB v0.3.0 schema.

Wraps the generic sqlite_writer with the v0.3.0 table insertion order.
"""

import importlib.util
import os
from typing import Dict

import pandas as pd

# Import the generic SQLite writer using importlib so that this module works
# regardless of how/where it is invoked (no package install required).
_writer_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', 'writers', 'sqlite_writer.py'
)
_spec = importlib.util.spec_from_file_location(
    'sqlite_writer', os.path.abspath(_writer_path)
)
_writer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_writer)

# v0.3.0 table insertion order (respecting foreign keys)
TABLE_ORDER = [
    'entity_types',
    'entities',
    'prime_mover_types',
    'fuels',
    'storage_technology_types',
    'planning_regions',
    'balancing_topologies',
    'arcs',
    'transmission_lines',
    'transmission_interchanges',
    'thermal_generators',
    'renewable_generators',
    'hydro_generators',
    'storage_units',
    'hydro_reservoirs',
    'hydro_reservoir_connections',
    'supply_technologies',
    'transport_technologies',
    'attributes',
    'supplemental_attributes',
    'supplemental_attributes_association',
    'time_series_associations',
    'static_time_series',
    'loads',
]


def write_to_sqlite(schema_path: str,
                    target_dataframes: Dict[str, pd.DataFrame],
                    output_db_path: str,
                    clear_existing: bool = True) -> None:
    """
    Create v0.3.0 GridDB SQLite database from schema and populate with dataframes.

    This is a thin wrapper around the generic writer that supplies the
    v0.3.0 table insertion order.

    Args:
        schema_path: Path to the v0.3.0 schema.sql file
        target_dataframes: Dictionary of dataframes matching schema tables
        output_db_path: Path for output SQLite database file
        clear_existing: If True, delete existing database and create fresh.
                       If False, add/replace data in existing database.
    """
    _writer.write_to_sqlite(
        schema_path=schema_path,
        target_dataframes=target_dataframes,
        output_db_path=output_db_path,
        table_order=TABLE_ORDER,
        clear_existing=clear_existing,
    )


# Re-export utility functions for convenience
verify_database = _writer.verify_database
export_table_to_csv = _writer.export_table_to_csv
