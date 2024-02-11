import netCDF4 as nc

# Replace 'MIPAS-E_IMK.201201.V8R_O3_761.nc' with your actual file path
file_path = 'MIPAS-E_IMK.201201.V8R_O3_761.nc'

# Open the netCDF file
dataset = nc.Dataset(file_path, 'r')

# Extract time variable and its attributes
time_variable = dataset.variables['time']
time_values = time_variable[:]
time_attributes = time_variable.ncattrs()

# Display information about the time variable
print("Time Variable Information:")
print("--------------------")
print(f"Variable Name: time")
print(f"Shape: {time_values.shape}")
print(f"Data Type: {time_values.dtype}")
print(f"Attributes: {time_attributes}")

# Display the actual time values
print("\nTime Values:")
print(time_values)

# Close the netCDF file
dataset.close()
