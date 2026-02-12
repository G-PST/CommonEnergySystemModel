# Transformer Developer Guide

This guide covers the full data transformation pipeline for creating transformers that convert between energy system model formats using CESM (Central Energy System Model) as the interchange format.

## Table of Contents

1. [Overview](#1-overview)
2. [DataFrame Conventions](#2-dataframe-conventions)
3. [YAML Transformer Syntax](#3-yaml-transformer-syntax)
4. [Python Transformer Functions](#4-python-transformer-functions)
5. [DuckDB Storage Format](#5-duckdb-storage-format)
6. [Readers and Writers](#6-readers-and-writers)
7. [Creating a New Transformer](#7-creating-a-new-transformer)

---

## 1. Overview

### Purpose

The transformation pipeline enables bidirectional data conversion between different energy system modeling tools. Rather than creating direct converters between every pair of formats (N×N converters), the system uses CESM as a central interchange format, requiring only N import + N export transformers.

### Full Data Flow from model to model

```
Source Model (e.g., FlexTool Spine DB)
    ↓ Reader (from_spine_db.py)
Source DataFrames
    ↓ YAML Transformer (from_flextool.yaml)
    ↓ Python Transformer (to_cesm.py)
CESM DataFrames
    ↓ Writer (to_duckdb.py)
DuckDB Storage
    ↓ Reader (from_duckdb.py)
CESM DataFrames
    ↓ YAML Transformer (to_flextool.yaml)
    ↓ Python Transformer (to_flextool.py)
Target DataFrames
    ↓ Writer (to_spine_db.py, to_yaml.py, etc.)
Target Model Format
```

### CESM as Central Interchange Format

CESM (stored in DuckDB and defined by [LinkML schema](model/cesm.yaml)) serves as the canonical data representation:

- **Normalization**: All timestamps are UTC, all entity relationships are explicit
- **Versioning**: CESM schema versions enable forward compatibility
- **Validation**: Central format enables schema validation
- **Storage**: DuckDB provides efficient columnar storage with metadata preservation

---

## 2. DataFrame Conventions

The pipeline uses pandas DataFrames to perform data transformations with specific conventions for entity data, time series, and indexed maps.

### Entity DataFrames

Entity DataFrames store entities and their scalar parameter values.

**Structure:**
- **Index**: Entity names (single Index or MultiIndex for dimensional entities)
- **Columns**: Parameter values (scalars)
- **Naming**: `{class_name}` (e.g., `unit`, `balance`, `unit_to_node`)

**Example: Single-dimensional entity**
```python
# DataFrame name: 'unit'
#   Index: unit names
#   Columns: scalar parameters

unit_df = pd.DataFrame({
    'efficiency': [0.45, 0.55, 0.38],
    'units_existing': [2, 1, 3],
    'investment_method': ['not_allowed', 'no_limits', 'not_allowed']
}, index=pd.Index(['coal_plant', 'gas_turbine', 'wind_farm'], name='unit'))
```

**Example: Multi-dimensional entity**
```python
# DataFrame name: 'unit_to_node'
#   Index: MultiIndex with (name, source, sink)
#   Columns: scalar parameters

unit_to_node_df = pd.DataFrame({
    'capacity': [500.0, 200.0, 100.0],
    'other_operational_cost': [25.0, 15.0, 0.0]
}, index=pd.MultiIndex.from_tuples([
    ('coal_plant.west', 'coal_plant', 'west'),
    ('gas_turbine.east', 'gas_turbine', 'east'),
    ('wind_farm.west', 'wind_farm', 'west')
], names=['name', 'source', 'sink']))
```

### Time Series / Map / Array DataFrames

These DataFrames store indexed parameter values where the index is datetime, string keys, or array positions.

**Structure:**
- **Index**: datetime (for time series), string keys (for maps), or position (for arrays)
- **Columns**: Entity names (critical for transformations!)
- **Naming patterns**:
  - `{class}.ts.{parameter}` - time series with datetime index
  - `{class}.str.{parameter}` - string-indexed maps
  - `{class}.array.{parameter}` - position-indexed arrays

**Example: Time series**
```python
# DataFrame name: 'unit.ts.availability'
#   Index: DatetimeIndex
#   Columns: entity names

availability_ts = pd.DataFrame({
    'coal_plant': [1.0, 1.0, 0.9, 0.95, ...],
    'gas_turbine': [1.0, 1.0, 1.0, 1.0, ...],
    'wind_farm': [0.3, 0.45, 0.6, 0.2, ...]
}, index=pd.date_range('2023-01-01', periods=8760, freq='h', name='datetime'))
```

**Example: String-indexed map (period data)**
```python
# DataFrame name: 'unit.str.virtual_unitsize'
#   Index: period names (strings)
#   Columns: entity names

period_capacity = pd.DataFrame({
    'coal_plant': [500.0, 500.0, 400.0],
    'gas_turbine': [200.0, 250.0, 300.0]
}, index=pd.Index(['y2025', 'y2030', 'y2035'], name='period'))
```

**Example: Array**
```python
# DataFrame name: 'system.array.solve_order'
#   Index: numeric position
#   Columns: entity names

solve_order = pd.DataFrame({
    'main_system': ['y2025', 'y2030', 'y2035']
})
```

### MultiIndex Column Convention

For multi-dimensional entity classes (e.g., `unit__node`, `unit_to_node`), time series/map/array DataFrames use MultiIndex columns:

**Structure:**
- **Column tuples**: `(entity_name, dimension1, dimension2, ...)`
- **Level names**: `['name', 'source', 'sink']` etc.

**Example:**
```python
# DataFrame name: 'unit_to_node.ts.profile_limit_upper'
#   Index: DatetimeIndex
#   Columns: MultiIndex with (name, source, sink)

profile_ts = pd.DataFrame({
    ('coal_plant.west', 'coal_plant', 'west'): [1.0, 0.95, 0.9, ...],
    ('gas_turbine.east', 'gas_turbine', 'east'): [1.0, 1.0, 1.0, ...]
}, index=pd.date_range('2023-01-01', periods=8760, freq='h', name='datetime'))

profile_ts.columns = pd.MultiIndex.from_tuples(
    profile_ts.columns.tolist(),
    names=['name', 'source', 'sink']
)
```

This convention enables dimension-based transformations (reordering, aggregation) in the YAML transformer.

---

## 3. YAML Transformer Syntax

The YAML transformer handles most data transformations declaratively. Each operation in the YAML file defines a transformation from source to target.

### Basic Structure

```yaml
operation-name:
- source_specification
- target_specification
- operations_list (optional)
```

### Entity Operations

**Simple entity copy:**
```yaml
# Copy all entities from 'unit' class to 'unit' class
unit-entities:
- unit
- unit
```

**Entity copy with dimension specification:**
```yaml
# Copy unit_to_node entities, specifying dimension names
unit-to-node-entities:
- unit_to_node
- unit.outputNode:
    order: [[0], [1], [2]]
- dimensions: [unit, node]
```

**Conditional entity copy with `if_parameter`:**
```yaml
# Only copy nodes that have the 'has_balance' parameter set
balance-entities:
- node
- balance:
    if_parameter: has_balance
```

**Conditional entity copy with `if_not_parameter`:**
```yaml
# Copy nodes that have has_balance but NOT has_storage
balance-entities:
- node
- balance
- - if_parameter: has_balance
  - if_not_parameter: has_storage
```

**Multiple condition parameters:**
```yaml
# Copy if any of the listed parameters exist
profile-entities:
- unit_to_node
- profile:
    order: [[0]]
    if_parameter: [profile_limit_upper, profile_limit_lower]
```

### Dimension Reordering

The `order` specification controls how source dimensions map to target dimensions.

**Syntax:** `order: [[source_dims_for_target_0], [source_dims_for_target_1], ...]`

- Index `0` refers to the source entity name
- Index `1, 2, ...` refer to source dimension elements

**Examples:**

```yaml
# Direct 1:1 mapping (entity_name, dim1, dim2) -> (name, source, sink)
unit_to_node-entities:
- unit.outputNode
- unit_to_node:
    order: [[0], [1], [2]]
- dimensions: [source, sink]

# Swap dimensions: (name, dim1, dim2) -> (name, dim2, dim1)
node_to_unit-entities:
- unit.inputNode
- node_to_unit:
    order: [[0], [2], [1]]
- dimensions: [source, sink]

# Extract only entity name (drop dimensions)
link efficiency:
- connection: efficiency
- link: efficiency
- order: [[0]]

# Combine dimensions into name
link-entities:
- connection.node.node
- link:
    order: [[1], [2], [3]]
- dimensions: [node_A, node_B]
```

### Parameter Transformations

**Simple parameter copy:**
```yaml
unit_efficiency:
- unit: efficiency
- unit: efficiency
```

**Parameter rename:**
```yaml
units_existing:
- unit: existing
- unit: units_existing
```

**Parameter with value rename mapping:**
```yaml
unit_investment_method:
- unit: invest_method
- unit: investment_method
- rename:
    not_allowed: not_allowed
    invest_no_limit: no_limits
```

### Data Type Conversions

Use list notation `[parameter, [type]]` for indexed parameters:

**Time series (`ts`) - datetime indexed:**
```yaml
balance_inflow:
- node: [inflow, [str]]        # Source: string-indexed map
- balance: [flow_profile, [ts]] # Target: datetime-indexed time series
- - if_parameter: has_balance
  - if_not_parameter: has_storage
```

**String-indexed map (`str`):**
```yaml
unit_to_node profile_limit_upper:
- unit_to_node: [profile_limit_upper, [ts]]  # Source: time series
- profile: [profile, [str]]                   # Target: string-indexed map
- - order: [[0]]
```

**Array:**
```yaml
solve_order:
- system: [solve_order, [array]]
- model: [solves, [array]]
```

### Mathematical Operations

**Multiply/Divide/Add/Subtract with constant:**
```yaml
storage_investment_cost:
- node: invest_cost
- storage: investment_cost
- - if_parameter: has_storage
  - operation: multiply
    with: 1000

storage fixed-cost:
- storage: fixed_cost
- node: fixed_cost
- operation:
  - multiply:
      with: 0.001
```

**Algebra between multiple sources:**
```yaml
# Multiply values from two different sources
# Formula: source1 * source2 where match defines dimension alignment
test-multiply:
- node_to_unit: other_operational_cost
  unit: efficiency
- unit.inputNode: test
- - algebra: "1*2"
    match: [[2], [1]]  # Match dim 2 of source 1 with dim 1 of source 2
  - order: [[2], [1]]
```

### Aggregation

When reducing dimensions, specify how to aggregate:

```yaml
unit_to_node investment_cost:
- unit_to_node: investment_cost
- unit: invest_cost
- - order: [[1]]
    aggregate: sum  # Options: sum, average, max, min, first
```

### Creating Parameters from Entities

Use `value` to create parameters based on entity existence:

**Constant value:**
```yaml
has_balance:
- [balance, storage]
- node: has_balance
- value: 'yes'
```

**Value from dimension:**
```yaml
# Extract dimension value as parameter
# value: [N] extracts the Nth dimension
link_node_A:
- connection.node.node
- link: node_A
- - order: [[0]]
  - value: [2]  # Extract dimension 2 as the parameter value
```

---

## 4. Python Transformer Functions

Use Python transformers for complex logic that cannot be expressed in YAML.

### When to Use Python vs YAML

**Use YAML for:**
- Direct entity/parameter copies
- Simple renames and value mappings
- Dimension reordering
- Basic mathematical operations with constants
- Conditional filtering

**Use Python for:**
- Complex conditional logic spanning multiple DataFrames
- Temporal data reconstruction (timelines, solve patterns)
- Lookups across different entity classes
- Business logic requiring iteration or state

### Python Transformer Structure

```python
"""
Module docstring explaining the transformations.
"""

import pandas as pd
from typing import Dict
from datetime import datetime


def transform_to_cesm(source: Dict[str, pd.DataFrame],
                      cesm: Dict[str, pd.DataFrame],
                      start_time: datetime) -> Dict[str, pd.DataFrame]:
    """
    Main entry point called after the YAML transformer.

    Args:
        source: Dictionary of source DataFrames
        cesm: Dictionary of CESM DataFrames (partially transformed by YAML)
        start_time: Start datetime for the timeline

    Returns:
        Updated cesm dictionary with all Python transformations applied
    """
    # Apply transformations
    cesm = my_custom_transform(source, cesm)
    cesm = another_transform(source, cesm, start_time)

    return cesm


def my_custom_transform(source: Dict[str, pd.DataFrame],
                        cesm: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Individual transformation function.

    Keep functions focused on a single transformation concern.
    """
    # Implementation
    return cesm
```

### Available Helpers in transform_parameters.py

The `transform_parameters.py` module provides utilities:

```python
from src.core.transform_parameters import (
    transform_data,      # Main YAML transformer function
    load_config,         # Load YAML configuration
    is_timeseries,       # Check if DataFrame is time series
    get_entity_index,    # Get entity index (handles pivoted data)
    set_entity_index,    # Set entity index
    index_to_names,      # Convert index to list of name lists
    list_of_lists_to_index,  # Convert name lists back to index
    reorder_dimensions,  # Reorder DataFrame dimensions
    apply_rename,        # Apply value rename mapping
)
```

### Example: Time Data Transformation

```python
def time_from_spine(flextool: Dict[str, pd.DataFrame],
                    cesm: Dict[str, pd.DataFrame],
                    start_time: datetime) -> Dict[str, pd.DataFrame]:
    """
    Extract temporal data from FlexTool format and add to CESM.

    Creates/updates:
    - cesm['timeline']: DataFrame with DatetimeIndex
    - cesm['solve_pattern']: DataFrame with start_time, duration
    - cesm['period']: DataFrame with years_represented
    """
    # Get timestep duration from source data
    timestep_minutes = _get_timestep_minutes(flextool)

    # Create timeline from timestep_duration
    if 'timeline.str.timestep_duration' in flextool:
        timestep_df = flextool['timeline.str.timestep_duration']

        # Convert string index to datetime if needed
        datetime_index = _convert_str_index_to_datetime(
            timestep_df.index, start_time, timestep_minutes
        )

        cesm['timeline'] = pd.DataFrame(index=datetime_index)
        cesm['timeline'].index.name = 'datetime'

    return cesm
```

---

## 5. DuckDB Storage Format

DuckDB serves as the persistent storage for CESM data, with metadata to reconstruct exact DataFrame structures.

### Column Encoding

MultiIndex columns are encoded using `::` separator:

```
("region", "north", "heat") -> "region::north::heat"
```

The `::` separator is used because entity names may contain dots (e.g., `source.sink`).

### Metadata Table

Each DuckDB file contains a `_dataframe_metadata` table with:

| Column | Type | Description |
|--------|------|-------------|
| `table_name` | string | Original DataFrame name (e.g., `unit.ts.availability`) |
| `sql_table_name` | string | SQL-safe table name (dots replaced with underscores) |
| `index_type` | string | `'single'`, `'datetime'`, or `'multi'` |
| `index_count` | int | Number of index columns (stored as first N columns) |
| `columns_multiindex` | bool | Whether columns are MultiIndex |
| `columns_levels` | JSON | Array of level names if MultiIndex columns |

### Storage Layout

```
DuckDB File
├── _dataframe_metadata (table)
├── unit (table) - entity DataFrame
├── unit_to_node (table) - entity DataFrame with MultiIndex
├── unit_ts_availability (table) - time series (original: unit.ts.availability)
└── ...
```

### Verifying with DBeaver

To inspect stored data:

1. Open DuckDB file in DBeaver
2. Query metadata: `SELECT * FROM "_dataframe_metadata"`
3. Query data tables: `SELECT * FROM "unit_ts_availability"`
4. Note that index columns are the first N columns based on `index_count`

---

## 6. Readers and Writers

### Available Readers

**from_spine_db.py** - Read Spine Toolbox databases:
```python
from src.readers.from_spine_db import spine_to_dataframes

dfs = spine_to_dataframes(
    db_url="sqlite:///path/to/database.sqlite",
    scenario="base"
)
```

**from_duckdb.py** - Read DuckDB CESM storage:
```python
from src.readers.from_duckdb import dataframes_from_duckdb

dfs = dataframes_from_duckdb("path/to/cesm.duckdb")
```

### Available Writers

**to_duckdb.py** - Write to DuckDB CESM storage:
```python
from src.writers.to_duckdb import dataframes_to_duckdb

dataframes_to_duckdb(
    dataframes=dfs,
    db_path="output/cesm.duckdb",
    overwrite=True
)
```

**to_spine_db.py** - Write to Spine Toolbox databases:
```python
from src.writers.to_spine_db import dataframes_to_spine

dataframes_to_spine(
    dataframes=dfs,
    db_url="sqlite:///path/to/output.sqlite",
    import_datetime="2025-01-15_10-30",
    purge_before_import=True
)
```

### How Readers/Writers Preserve Structure

**Entity DataFrames:**
- Index stored as first column(s)
- MultiIndex levels preserved via metadata
- Column names preserved as-is

**Time Series/Map/Array DataFrames:**
- DatetimeIndex converted to/from ISO 8601 strings
- MultiIndex columns encoded/decoded with `::` separator
- Level names preserved in metadata

---

## 7. Creating a New Transformer

### Directory Structure

Transformers are organized by source format, CESM version, and source version:

```
src/transformers/
└── {source_format}/
    └── cesm_{cesm_version}/
        └── {source_version}/
            ├── from_{source}.yaml      # Import: source → CESM
            ├── to_cesm.py              # Import: complex transformations (optional)
            ├── to_{target}.yaml        # Export: CESM → target
            └── from_cesm.py            # Export: complex transformations (optional)
```

**Example:**
```
src/transformers/
└── irena_flextool/
    └── cesm_v0.1.0/
        └── v3.14.0/
            ├── from_flextool.yaml
            ├── to_cesm.py
            ├── to_flextool.yaml
            └── from_cesm.py
```

### Step-by-Step: Creating an Import Transformer

1. **Create directory structure:**
   ```bash
   mkdir -p src/transformers/{source}/cesm_v0.1.0/{version}
   ```

2. **Analyze source data format:**
   - Identify entity classes and their relationships
   - Map source parameters to CESM equivalents
   - Note indexed data (time series, maps, arrays)

3. **Create `from_{source}.yaml`:**
   ```yaml
   # Entity mappings
   source-entity-entities:
   - source_class
   - cesm_class
   
   # Parameter mappings
   source_param:
   - source_class: source_param_name
   - cesm_class: cesm_param_name
   ```

4. **Create `to_cesm.py` (if needed):**
   ```python
   def transform_to_cesm(source, cesm, start_time):
       # Complex transformations
       return cesm
   ```

5. **Test the transformer:**
   ```python
   from src.readers.from_spine_db import spine_to_dataframes
   from src.core.transform_parameters import transform_data
   
   source_dfs = spine_to_dataframes(db_url, scenario)
   cesm_dfs = transform_data(source_dfs, 'from_source.yaml')
   ```

### Step-by-Step: Creating an Export Transformer

1. **Create `to_{target}.yaml`:**
   ```yaml
   # Entity mappings (reverse of import)
   cesm-entity-entities:
   - cesm_class
   - target_class
   
   # Parameter mappings
   cesm_param:
   - cesm_class: cesm_param_name
   - target_class: target_param_name
   ```

2. **Create `from_cesm.py` (if needed):**
   ```python
   def transform_from_cesm(cesm, target, timeline):
       # Complex reverse transformations
       return target
   ```

3. **Test the full round-trip:**
   ```python
   # Import
   source_dfs = reader(source_path)
   cesm_dfs = transform_data(source_dfs, 'from_source.yaml')
   
   # Export
   target_dfs = transform_data(cesm_dfs, 'to_target.yaml')
   writer(target_dfs, target_path)
   ```

### Checklist: Import Transformer

- [ ] Create `from_{source}.yaml` with entity mappings
- [ ] Map all scalar parameters
- [ ] Handle indexed data (time series, maps, arrays) with type annotations
- [ ] Add dimension specifications for multi-dimensional entities
- [ ] Implement conditional logic with `if_parameter`/`if_not_parameter`
- [ ] Create `to_cesm.py` for complex transformations (if needed)
- [ ] Test with sample data
- [ ] Verify round-trip consistency

### Checklist: Export Transformer

- [ ] Create `to_{target}.yaml` with entity mappings
- [ ] Map all CESM parameters to target format
- [ ] Handle data type conversions (`[ts]` ↔ `[str]`)
- [ ] Implement dimension reordering with `order`
- [ ] Add aggregation where dimensions collapse
- [ ] Create `from_cesm.py` for complex transformations (if needed)
- [ ] Test with sample CESM data
- [ ] Verify output matches target format specification

---

## Appendix: Quick Reference

### YAML Operation Types

| Pattern | Type | Description |
|---------|------|-------------|
| `- source_class` + `- target_class` | Entity copy | Copy entities between classes |
| `- source: param` + `- target: param` | Parameter transform | Copy/transform parameter values |
| `- source` + `- target: param` + `- value: X` | Create parameter | Create parameter with fixed value |

### Index Type Indicators

| Suffix | Index Type | Example |
|--------|------------|---------|
| (none) | Entity names | `unit`, `balance` |
| `.ts.` | DatetimeIndex | `unit.ts.availability` |
| `.str.` | String keys | `solve.str.period_timeset` |
| `.array.` | Numeric position | `system.array.solve_order` |

### Common Order Patterns

| Pattern | Effect |
|---------|--------|
| `[[0]]` | Keep only entity name |
| `[[0], [1], [2]]` | Keep all dimensions in order |
| `[[0], [2], [1]]` | Swap dimensions 1 and 2 |
| `[[1], [2], [3]]` | Drop entity name, use dimensions |
| `[[0, 1]]` | Combine entity name with dimension 1 |
