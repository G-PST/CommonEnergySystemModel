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
dataset = linkml_to_xarray(data)

# Explore the dataset
print("\nDataset overview:")
print(dataset)

print("\nCoordinates:")
for coord_name, coord_values in dataset.coords.items():
    print(f"  {coord_name}: {len(coord_values)} items - {list(coord_values.values)}")

print("\nData variables:")
for var_name in dataset.data_vars:
    var = dataset[var_name]
    print(f"  {var_name}: {var.dims} - shape {var.shape}")

# Example operations
print("\n--- Example Operations ---")

# 1. Access unit efficiency
if 'unit_efficiency' in dataset:
    print("\nUnit efficiencies:")
    print(dataset['unit_efficiency'])

# 2. Look at flow profiles over time
if 'balances_flow_profile' in dataset:
    print("\nBalance flow profiles (first few time steps):")
    flow_profiles = dataset['balances_flow_profile']
    print(flow_profiles.isel(time=slice(0, 3)))  # First 3 time steps

# 3. Multi-dimensional data (unit-to-node capacities)
if 'unit_to_node_capacity' in dataset:
    print("\nUnit-to-node capacity matrix:")
    capacity_matrix = dataset['unit_to_node_capacity']
    print(capacity_matrix)

    # Sum capacity by unit
    print("\nTotal capacity per unit:")
    total_cap_per_unit = capacity_matrix.sum(dim='node', skipna=True)
    print(total_cap_per_unit)

# 4. Time-based operations
if 'balances_flow_profile' in dataset:
    print("\nTime-based analysis:")
    flow_profiles = dataset['balances_flow_profile']

    # Average flow over time for each balance
    avg_flows = flow_profiles.mean(dim='time')
    print("Average flows:", avg_flows)

    # Peak flows
    peak_flows = flow_profiles.max(dim='time')
    print("Peak flows:", peak_flows)

# 5. Simple transformations
print("\n--- Simple Transformations ---")

# Scale all efficiencies by 100 to get percentages
if 'unit_efficiency' in dataset:
    efficiency_pct = dataset['unit_efficiency'] * 100
    print("Unit efficiencies as percentages:")
    print(efficiency_pct)

# Calculate power output (efficiency * capacity where both exist)
if 'unit_efficiency' in dataset and 'unit_to_node_capacity' in dataset:
    # This would require aligning dimensions, more complex example
    print("\nNote: Complex multi-dimensional operations would require dimension alignment")

print("\n--- Dataset saved for further analysis ---")
print("You can now use standard xarray operations for analysis and visualization")