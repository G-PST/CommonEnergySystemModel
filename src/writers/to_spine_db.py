"""
Convert FlexTool dataframes to Spine Toolbox database.

This module provides functions to write pandas DataFrames to a Spine database,
handling entity classes, entities, parameters, and time series data.
"""

from typing import Dict
import pandas as pd
from spinedb_api import DatabaseMapping
from spinedb_api.exception import NothingToCommit
from spinedb_api.parameter_value import to_database


def dataframes_to_spine(
    dataframes: Dict[str, pd.DataFrame],
    db_url: str,
    timeline_df: pd.DataFrame = None,
    import_datetime: str = None,
    purge_before_import: bool = True
):
    """
    Write dataframes to Spine database.
    
    Args:
        dataframes: Dict mapping dataframe names to DataFrames
        db_url: Database URL (e.g., "sqlite:///path/to/database.sqlite")
        timeline_df: DataFrame with 'datetime' column defining the timeline
        import_datetime: Datetime string for alternative name (format: yyyy-mm-dd_hh-mm)
                        If None, uses current datetime
        purge_before_import: If True, purge parameter values, entities, and alternatives 
                            before import (default: True)
    """
    from datetime import datetime
    
    # Generate alternative name with datetime
    if import_datetime is None:
        import_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M')
    alternative_name = f'import_{import_datetime}'
    
    with DatabaseMapping(db_url) as db_map:
        # Phase -1: Purge if requested
        if purge_before_import:
            print("Phase -1: Purging database...")
            #db_map.purge_items('parameter_value')
            #db_map.purge_items('entity')
            db_map.purge_items('alternative')
            db_map.refresh_session()
            db_map.commit_session("Purged parameter values, entities and alternatives")
            print("  Purged parameter values, entities, and alternatives")
        
        # Separate dataframes by type
        entity_dfs = {}
        ts_dfs = {}
        table_dfs = {}

        for name, df in dataframes.items():
            if '.ts.' in name or '.table.' in name:
                # Replace dot multi-dim naming convention with double underscore
                for col_name, col in df.items():
                    if '.' in col_name:
                        df.rename(columns = {col_name: '__'.join(col_name.split('.'))}, inplace = True)
            if '.ts.' in name:
                ts_dfs[name] = df
            elif '.table.' in name:
                table_dfs[name] = df
            else:
                entity_dfs[name] = df
                # Replace dot multi-dim naming convention with double underscore
                if not '.' in name:
                    for (index, row) in df.iterrows():
                        if '.' in row['name']:
                            row['name'] = ('__'.join(row['name'].split('.')))

        
        # Phase 0: Add alternative
        print(f"Phase 0: Adding alternative '{alternative_name}'...")
        try:
            db_map.add_alternative(name=alternative_name)
            db_map.commit_session(f"Added alternative {alternative_name}")
            print(f"  Added alternative: {alternative_name}")
        except Exception as e:
            print(f"  Alternative {alternative_name} already exists or error: {e}")

       # Phase 1: Add entity classes and entities
        print("Phase 1: Adding entity classes and entities...")
        _add_entity_classes_and_entities(db_map, entity_dfs)
        try:
            db_map.commit_session("Added entity classes and entities")
        except NothingToCommit:
            print("No entities to commit")        
        
        # Phase 2: Add parameter definitions and values
        print("Phase 2: Adding parameter definitions and values...")
        _add_parameters(db_map, entity_dfs, alternative_name)
        try:
            db_map.commit_session("Added parameter definitions and values")
        except NothingToCommit:
            print("No parameters (constants) to commit")        

        # Phase 3: Add time series parameters
        if ts_dfs:
            print("Phase 3: Adding time series parameters...")
            _add_time_series(db_map, ts_dfs, timeline_df, alternative_name)
            try:
                db_map.commit_session("Added time series parameters")
            except NothingToCommit:
                print("No time series parameters to commit")
        
        # Phase 4: Add table (map) parameters
        if table_dfs:
            print("Phase 4: Adding table (map) parameters...")
            _add_tables(db_map, table_dfs, alternative_name)
            try:
                db_map.commit_session("Added table parameters")
            except NothingToCommit:
                print("No time series parameters to commit")
        
        print("Done!")

def _add_entity_classes_and_entities(db_map: DatabaseMapping, entity_dfs: Dict[str, pd.DataFrame]):
    """Add entity classes and their entities."""
    # Sort: single-dimensional classes first (no dots), then multi-dimensional
    sorted_classes = sorted(entity_dfs.keys(), key=lambda x: ('.' in x, x))
    
    for class_name in sorted_classes:
        df = entity_dfs[class_name]
        
        # Determine if multi-dimensional
        if '.' in class_name:
            dimensions = class_name.split('.')
            dimension_name_list = tuple(df.keys().tolist()[:len(dimensions)])
            class_name = '__'.join(dimensions)
        else:
            dimension_name_list = None
        
        # Add entity class
        try:
            db_map.add_entity_class(
                name=class_name,
                dimension_name_list=dimension_name_list
            )
            print(f"  Added entity class: {class_name}")
        except Exception as e:
            print(f"  Entity class {class_name} already exists or error: {e}")
        
        # Add entities
        if 'name' not in df.columns:
            # Multi-dimensional: first N columns are dimensions
            if dimension_name_list:
                dimension_cols = list(dimension_name_list)
                for _, row in df.iterrows():
                    element_name_list = tuple(str(row[dim]) for dim in dimension_cols)
                    try:
                        db_map.add_entity(
                            entity_class_name=class_name,
                            element_name_list=element_name_list
                        )
                    except Exception as e:
                        pass  # Entity might already exist
        else:
            # Single-dimensional: 'name' column contains entity names
            for entity_name in df['name'].unique():
                try:
                    db_map.add_entity(
                        entity_class_name=class_name,
                        name=str(entity_name)
                    )
                except Exception as e:
                    pass  # Entity might already exist

def _add_parameters(db_map: DatabaseMapping, entity_dfs: Dict[str, pd.DataFrame], alternative_name: str):
    """Add parameter definitions and values."""
    for class_name, df in entity_dfs.items():
        # Determine dimension columns
        if '.' in class_name:
            dimensions = class_name.split('.')
            dimension_cols = list(df.keys().tolist()[:len(dimensions)])
            class_name = '__'.join(dimensions)
        else:
            dimension_cols = ['name']
        
        # Parameter columns are all columns except dimension/name columns
        param_cols = [col for col in df.columns if col not in dimension_cols]
        
        # Add parameter definitions
        for param_name in param_cols:
            try:
                db_map.add_parameter_definition(
                    entity_class_name=class_name,
                    name=param_name
                )
                print(f"  Added parameter definition: {class_name}.{param_name}")
            except Exception as e:
                pass  # Parameter might already exist
        
        # Add parameter values
        for _, row in df.iterrows():
            # Build entity_byname
            if '__' in class_name:
                entity_byname = tuple(str(row[dim]) for dim in dimension_cols)
            else:
                entity_byname = (str(row['name']),)
            
            # Add values for each parameter
            for param_name in param_cols:
                value = row[param_name]
                
                # Skip None/NaN values
                if pd.isna(value):
                    continue
                
                # Convert value to Spine format
                if isinstance(value, (int, float)):
                    parsed_value = float(value)
                elif isinstance(value, str):
                    parsed_value = value
                else:
                    parsed_value = value
                
                try:
                    db_map.add_parameter_value(
                        entity_class_name=class_name,
                        parameter_definition_name=param_name,
                        entity_byname=entity_byname,
                        alternative_name=alternative_name,
                        parsed_value=parsed_value
                    )
                except Exception as e:
                    print(f"  Warning: Could not add value for {class_name}.{param_name}: {e}")


def _add_time_series(
    db_map: DatabaseMapping,
    ts_dfs: Dict[str, pd.DataFrame],
    timeline_df: pd.DataFrame,
    alternative_name: str
):
    """Add time series parameter values."""
    # Extract start time from timeline
    if timeline_df is not None and 'datetime' in timeline_df.columns:
        start_time = pd.to_datetime(timeline_df['datetime'].iloc[0]).isoformat()
    else:
        start_time = None
    
    for ts_name, df in ts_dfs.items():
        # Parse name: class_name.ts.parameter_name
        parts = ts_name.split('.ts.')
        if len(parts) != 2:
            print(f"  Warning: Invalid time series name format: {ts_name}")
            continue
        
        class_name = parts[0]
        param_name = parts[1]
        
        # Add parameter definition if needed
        try:
            db_map.add_parameter_definition(
                entity_class_name=class_name,
                name=param_name
            )
        except Exception:
            pass  # Already exists
        
        # Entity names are columns (except 'datetime')
        entity_cols = [col for col in df.columns if col != 'datetime']
        
        for entity_name in entity_cols:
            # Extract time series data
            values = df[entity_name].tolist()
            
            # Build time series in Spine format
            if start_time:
                ts_value = {
                    "type": "time_series",
                    "data": values,
                    "index": {
                        "start": start_time,
                        "resolution": "1h"
                    }
                }
            else:
                # Use datetime column if available
                if 'datetime' in df.columns:
                    timestamps = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
                    ts_value = {
                        "type": "time_series",
                        "data": [[ts, val] for ts, val in zip(timestamps, values)]
                    }
                else:
                    # Fallback to array without timestamps
                    ts_value = {
                        "type": "time_series",
                        "data": values
                    }
            
            # Convert to database format
            db_value, value_type = to_database(ts_value)
            
            try:
                db_map.add_parameter_value(
                    entity_class_name=class_name,
                    parameter_definition_name=param_name,
                    entity_byname=(entity_name,),
                    alternative_name=alternative_name,
                    value=db_value,
                    type=value_type
                )
                print(f"  Added time series: {class_name}.{param_name} for {entity_name}")
            except Exception as e:
                print(f"  Warning: Could not add time series for {entity_name}: {e}")


def _add_tables(db_map: DatabaseMapping, table_dfs: Dict[str, pd.DataFrame], alternative_name: str):
    """Add table (map) parameter values."""
    from spinedb_api.parameter_value import Map, DateTime
    
    for table_name, df in table_dfs.items():
        # Parse name: class_name.table.parameter_name
        parts = table_name.split('.table.')
        if len(parts) != 2:
            print(f"  Warning: Invalid table name format: {table_name}")
            continue
        
        class_name = parts[0]
        param_name = parts[1]
        
        # Add parameter definition if needed
        try:
            db_map.add_parameter_definition(
                entity_class_name=class_name,
                name=param_name
            )
        except Exception:
            pass  # Already exists
        
        # Entity names are columns (except 'datetime')
        entity_cols = [col for col in df.columns if col != 'datetime']
        
        # Determine index type from datetime column
        if 'datetime' in df.columns:
            # Convert datetime strings to DateTime objects for indexes
            indexes = df['datetime'].tolist()
            index_name = 'time'
        else:
            print(f"  Warning: No datetime column found in {table_name}")
            continue
        
        for entity_name in entity_cols:
            # Extract values for this entity
            values = df[entity_name].tolist()
            
            # Create Map object
            map_value = Map(
                indexes=indexes,
                values=values,
                index_name=index_name
            )
            
            # Convert to database format
            db_value, value_type = to_database(map_value)
            
            try:
                db_map.add_parameter_value(
                    entity_class_name=class_name,
                    parameter_definition_name=param_name,
                    entity_byname=(entity_name,),
                    alternative_name=alternative_name,
                    value=db_value,
                    type=value_type
                )
                print(f"  Added table map: {class_name}.{param_name} for {entity_name}")
            except Exception as e:
                print(f"  Warning: Could not add table map for {entity_name}: {e}")


# Example usage
if __name__ == "__main__":
    # Example: Create sample dataframes
    sample_dfs = {
        'node': pd.DataFrame({
            'name': ['west', 'east', 'heat'],
            'annual_flow': [100000.0, 80000.0, None],
            'penalty_up': [10000.0, 10000.0, 10000.0]
        }),
        'connection': pd.DataFrame({
            'name': ['charger', 'pony1'],
            'efficiency': [0.90, 0.98],
            'capacity': [750.0, 500.0]
        }),
        'unit.outputNode': pd.DataFrame({
            'unit': ['coal_plant', 'gas_plant'],
            'outputNode': ['west', 'east'],
            'capacity': [100.0, 50.0],
            'efficiency': [0.9, 0.95]
        }),
        'node.table.inflow': pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=10, freq='H'),
            'west': [-1002.1, -980.7, -968, -969.1, -971.9, -957.8, -975.2, -975.1, -973.2, -800],
            'east': [-1002.1, -980.7, -968, -969.1, -971.9, -957.8, -975.2, -975.1, -973.2, -800],
            'heat': [-30, -40, -50, -60, -50, -50, -50, -50, -50, -50]
        })
    }
    
    timeline = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=8760, freq='H')
    })
    
    # Write to database
    # dataframes_to_spine(sample_dfs, "sqlite:///test_flextool.sqlite", timeline, import_datetime='2025-10-02_15-30')