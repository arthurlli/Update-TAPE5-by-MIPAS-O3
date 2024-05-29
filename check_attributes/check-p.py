import netCDF4 as nc
import numpy as np

# Replace 'MIPAS-E_IMK.201201.V8R_O3_761.nc' with your actual file path
#file_path = 'MIPAS-E_IMK.201201.V8R_O3_761.nc' # 761: 7 stands for noctilucent cloud
file_path = 'MIPAS-E_IMK.201201.V8R_O3_561.nc'

# Open the netCDF file
dataset = nc.Dataset(file_path, 'r')

# Extract pressure levels variable
pressure_levels = dataset.variables['pressure']

# Display information about pressure levels
print("Pressure Levels Information:")
print("--------------------")
print(f"Variable Name: {pressure_levels.name}")
print(f"Shape: {pressure_levels.shape}")
print(f"Data Type: {pressure_levels.dtype}")
print(f"Attributes: {pressure_levels.ncattrs()}")
if 'units' in pressure_levels.ncattrs():
    print(f"Units: {pressure_levels.units}")

# Display the actual pressure levels data
print("Pressure Levels Data:")
print(np.array(pressure_levels[:]))

# Close the netCDF file
dataset.close()
