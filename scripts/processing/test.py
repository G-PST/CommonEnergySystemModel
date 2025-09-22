import xarray as xr
import pandas as pd
import numpy as np

# Create individual coordinates first
unit_coord = xr.DataArray(['gas', 'coal'], dims='unit_dim', name='unit')
node_coord = xr.DataArray(['west', 'east'], dims='node_dim', name='node')

# Define sparse data points
data = [50, 100, 70]
sparse_units = ['gas', 'coal', 'coal'] 
sparse_nodes = ['west', 'west', 'east']

# Create MultiIndex from the existing coordinate values
multi_idx = pd.MultiIndex.from_arrays(
    [sparse_units, sparse_nodes], 
    names=['unit', 'node']
)

# Create DataArray with MultiIndex
capacity = xr.DataArray(data, coords=[multi_idx], dims=['plant'])

# Create Dataset
ds = xr.Dataset({
    'capacity': capacity,
    'unit': unit_coord,
    'node': node_coord
})

# Define sparse data points
data2 = [20, 10, np.nan]
sparse_units2 = ['gas', 'coal'] 
sparse_nodes2 = ['west', 'west']

# Create MultiIndex from the existing coordinate values
multi_idx2 = pd.MultiIndex.from_arrays(
    [sparse_units, sparse_nodes], 
    names=['unit', 'node']
)

# Create DataArray with MultiIndex
nominal = xr.DataArray(data2, coords=[multi_idx], dims=['plant'])

ds['nominal'] = nominal

print(ds)