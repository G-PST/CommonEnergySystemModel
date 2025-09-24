from readers.linkml_yaml import LinkMLYAMLReader
from core.data_handler import DataHandler
from core.linkml_to_xarray import linkml_to_xarray
import xarray as xr

# Read the data
reader = LinkMLYAMLReader()
data = reader.read("data/samples/ines-sample.yaml")  # Returns Database object

# Direct access (your preferred approach)
units = data.unit  # Dict of Unit objects
balances = data.balances  # Dict of Balance objects

# Optional utilities via DataHandler
handler = DataHandler(data)
print("Entity counts:", handler.count_entities())
errors = handler.validate_database()
if errors:
    print("Validation errors:", errors)

# Convert to xarray
print("\nConverting to xarray...")
dataarrays = linkml_to_xarray(data)

# Show available DataArrays
print(f"\nAvailable DataArrays: {list(dataarrays.keys())}")

# Group DataArrays by type for easier exploration
base_arrays = {k: v for k, v in dataarrays.items() if not any(multi in k for multi in ['unit_to_node', 'node_to_unit', 'link'])}
unit_to_node_arrays = {k: v for k, v in dataarrays.items() if 'unit_to_node' in k}
node_to_unit_arrays = {k: v for k, v in dataarrays.items() if 'node_to_unit' in k}
link_arrays = {k: v for k, v in dataarrays.items() if 'link' in k}

print(f"\nBase DataArrays: {list(base_arrays.keys())}")
print(f"Unit-to-node DataArrays: {list(unit_to_node_arrays.keys())}")
print(f"Node-to-unit DataArrays: {list(node_to_unit_arrays.keys())}")
print(f"Link DataArrays: {list(link_arrays.keys())}")

# Explore some base DataArrays
print("\n--- Base DataArray Examples ---")

if 'unit_efficiency' in dataarrays:
    print("\nUnit efficiencies:")
    print(dataarrays['unit_efficiency'])

if 'balances_flow_profile' in dataarrays:
    print("\nBalance flow profiles (first few time steps):")
    flow_profiles = dataarrays['balances_flow_profile']
    print(flow_profiles.isel(time=slice(0, 3)))  # First 3 time steps

# Explore multi-dimensional DataArrays
print("\n--- Multi-dimensional DataArray Examples ---")

if 'unit_to_node_capacity' in dataarrays:
    print("\nUnit-to-node capacity matrix:")
    capacity_array = dataarrays['unit_to_node_capacity']
    print(capacity_array)

    # Sum capacity by unit (works with proper MultiIndex!)
    print("\nTotal capacity per unit:")
    total_cap_per_unit = capacity_array.groupby('unit').sum()
    print(total_cap_per_unit)
    
    # Sum capacity by node
    print("\nTotal capacity per node:")
    total_cap_per_node = capacity_array.groupby('node').sum()
    print(total_cap_per_node)

if 'link_efficiency' in dataarrays:
    print("\nLink efficiencies:")
    print(dataarrays['link_efficiency'])

if 'link_capacity' in dataarrays:
    print("\nLink capacities:")
    print(dataarrays['link_capacity'])

# Example operations
print("\n--- Example Operations ---")

# 1. Time-based operations
if 'balances_flow_profile' in dataarrays:
    print("\nTime-based analysis:")
    flow_profiles = dataarrays['balances_flow_profile']

    # Average flow over time for each balance
    avg_flows = flow_profiles.mean(dim='time')
    print("Average flows:", avg_flows)

    # Peak flows
    peak_flows = flow_profiles.max(dim='time')
    print("Peak flows:", peak_flows)

# 2. Multi-dimensional time series operations
if 'unit_to_node_profile_limit_upper' in dataarrays:
    print("\nUnit-to-node profile limits (time series):")
    profiles = dataarrays['unit_to_node_profile_limit_upper']
    print(profiles)
    
    # Average profile over time by unit
    print("\nAverage profile limits by unit:")
    avg_by_unit = profiles.groupby('unit').mean()
    print(avg_by_unit)

# 3. Simple transformations
print("\n--- Simple Transformations ---")

# Scale all efficiencies by 100 to get percentages
if 'unit_efficiency' in dataarrays:
    efficiency_pct = dataarrays['unit_efficiency'] * 100
    print("Unit efficiencies as percentages:")
    print(efficiency_pct)

# 4. Combining DataArrays into custom Datasets when needed
print("\n--- Custom Dataset Creation ---")

# Create a custom dataset with related unit parameters
unit_data_vars = {k: v for k, v in dataarrays.items() if k.startswith('unit_') and 'unit_to_node' not in k}
if unit_data_vars:
    unit_dataset = xr.Dataset(unit_data_vars)
    print(f"\nCustom unit dataset with {len(unit_data_vars)} variables:")
    print(unit_dataset)

# Create a dataset with balance-related data
balance_data_vars = {k: v for k, v in dataarrays.items() if k.startswith('balances_')}
if balance_data_vars:
    balance_dataset = xr.Dataset(balance_data_vars)
    print(f"\nCustom balance dataset with {len(balance_data_vars)} variables:")
    print(balance_dataset)

# 5. Sparse data advantages
print("\n--- Sparse Data Advantages ---")
if unit_to_node_arrays:
    print("Multi-dimensional DataArrays preserve sparseness:")
    print("- Only actual connections are stored (no NaN-filled dense matrices)")
    print("- Efficient memory usage for sparse connectivity")
    print("- Operations like groupby work correctly on MultiIndex coordinates")
    print("- Each DataArray can have different sparse patterns")

# 6. Cross-array operations
print("\n--- Cross-Array Operations ---")

# Find units that appear in both efficiency and capacity data
if 'unit_efficiency' in dataarrays and 'unit_to_node_capacity' in dataarrays:
    efficiency_units = set(dataarrays['unit_efficiency'].coords['unit'].values)
    capacity_units = set(dataarrays['unit_to_node_capacity'].coords['unit'].values)
    common_units = efficiency_units.intersection(capacity_units)
    print(f"\nUnits with both efficiency and capacity data: {common_units}")

print("\n--- DataArrays available for analysis ---")
print("Individual DataArrays preserve sparseness and allow flexible operations")
print("Create custom Datasets by grouping related DataArrays as needed")
print("Each DataArray maintains optimal coordinate structure for its data pattern")