from readers.linkml_yaml import LinkMLYAMLReader
from core.data_handler import DataHandler
from core.linkml_to_xarray import linkml_to_xarray

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
datasets = linkml_to_xarray(data)

# Show available datasets
print(f"\nAvailable datasets: {list(datasets.keys())}")

# Get individual datasets
base_ds = datasets['base']
unit_to_node_ds = datasets.get('unit_to_node')
node_to_unit_ds = datasets.get('node_to_unit')
link_ds = datasets.get('link')

# Explore the base dataset
print("\nBase Dataset overview:")
print(base_ds)

print("\nBase Dataset coordinates:")
for coord_name, coord_values in base_ds.coords.items():
    print(f"  {coord_name}: {len(coord_values)} items - {list(coord_values.values)}")

print("\nBase Dataset data variables:")
for var_name in base_ds.data_vars:
    var = base_ds[var_name]
    print(f"  {var_name}: {var.dims} - shape {var.shape}")

# Explore multi-dimensional datasets
if unit_to_node_ds is not None:
    print("\nUnit-to-node Dataset overview:")
    print(unit_to_node_ds)
    
    print("\nUnit-to-node data variables:")
    for var_name in unit_to_node_ds.data_vars:
        var = unit_to_node_ds[var_name]
        print(f"  {var_name}: {var.dims} - shape {var.shape}")

if link_ds is not None:
    print("\nLink Dataset overview:")
    print(link_ds)

# Example operations
print("\n--- Example Operations ---")

# 1. Access unit efficiency from base dataset
if 'unit_efficiency' in base_ds:
    print("\nUnit efficiencies:")
    print(base_ds['unit_efficiency'])

# 2. Look at flow profiles over time from base dataset
if 'balances_flow_profile' in base_ds:
    print("\nBalance flow profiles (first few time steps):")
    flow_profiles = base_ds['balances_flow_profile']
    print(flow_profiles.isel(time=slice(0, 3)))  # First 3 time steps

# 3. Multi-dimensional data (unit-to-node capacities)
if unit_to_node_ds is not None and 'unit_to_node_capacity' in unit_to_node_ds:
    print("\nUnit-to-node capacity matrix:")
    capacity_matrix = unit_to_node_ds['unit_to_node_capacity']
    print(capacity_matrix)

    # Sum capacity by unit (now works with proper MultiIndex!)
    print("\nTotal capacity per unit:")
    total_cap_per_unit = capacity_matrix.groupby('unit').sum()
    print(total_cap_per_unit)
    
    # Sum capacity by node
    print("\nTotal capacity per node:")
    total_cap_per_node = capacity_matrix.groupby('node').sum()
    print(total_cap_per_node)

# 4. Time-based operations from base dataset
if 'balances_flow_profile' in base_ds:
    print("\nTime-based analysis:")
    flow_profiles = base_ds['balances_flow_profile']

    # Average flow over time for each balance
    avg_flows = flow_profiles.mean(dim='time')
    print("Average flows:", avg_flows)

    # Peak flows
    peak_flows = flow_profiles.max(dim='time')
    print("Peak flows:", peak_flows)

# 5. Multi-dimensional time series operations
if unit_to_node_ds is not None and 'unit_to_node_profile_limit_upper' in unit_to_node_ds:
    print("\nUnit-to-node profile limits (time series):")
    profiles = unit_to_node_ds['unit_to_node_profile_limit_upper']
    print(profiles)
    
    # Average profile over time by unit
    print("\nAverage profile limits by unit:")
    avg_by_unit = profiles.groupby('unit').mean()
    print(avg_by_unit)

# 6. Simple transformations
print("\n--- Simple Transformations ---")

# Scale all efficiencies by 100 to get percentages
if 'unit_efficiency' in base_ds:
    efficiency_pct = base_ds['unit_efficiency'] * 100
    print("Unit efficiencies as percentages:")
    print(efficiency_pct)

# 7. Cross-dataset operations (when coordinates align)
print("\n--- Cross-Dataset Operations ---")
if unit_to_node_ds is not None and 'unit_to_node_capacity' in unit_to_node_ds:
    print("Multi-dimensional datasets enable focused operations:")
    print("- Each dataset has proper coordinate structure")
    print("- Operations like groupby work correctly")
    print("- No NaN-filled dense arrays")
    print("- Separate coordinate spaces avoid conflicts")

print("\n--- Datasets saved for further analysis ---")
print("You can now use standard xarray operations for analysis and visualization")
print("Each dataset maintains proper coordinate alignment for its data type")