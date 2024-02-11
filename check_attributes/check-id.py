import netCDF4 as nc
import numpy as np

# Replace 'MIPAS-E_IMK.201201.V8R_O3_761.nc' with your actual file path
file_path = 'MIPAS-E_IMK.201201.V8R_O3_761.nc'

# Open the netCDF file
dataset = nc.Dataset(file_path, 'r')

# Extract sub_id and geo_id variables
sub_id_values = dataset.variables['sub_id'][:]
geo_id_values = dataset.variables['geo_id'][:]

# Display information about sub_id variable
print("sub_id Variable Information:")
print("--------------------")
print(f"Variable Name: sub_id")
print(f"Shape: {sub_id_values.shape}")
print(f"Data Type: {sub_id_values.dtype}")
print(f"Attributes: {dataset.variables['sub_id'].ncattrs()}")
print("Sample Data:")
print(sub_id_values)

# Display information about geo_id variable
print("\ngeo_id Variable Information:")
print("--------------------")
print(f"Variable Name: geo_id")
print(f"Shape: {geo_id_values.shape}")
print(f"Data Type: {geo_id_values.dtype}")
print(f"Attributes: {dataset.variables['geo_id'].ncattrs()}")
print("Sample Data:")
print(geo_id_values)

variable_name = 'geo_id'
variable = dataset.variables[variable_name]

# Display attributes of the variable
print(f"\nAttributes of {variable_name}:")
print("--------------------")

# Loop through attributes and display values
for attr_name in variable.ncattrs():
    attr_value = variable.getncattr(attr_name)
    print(f"{attr_name}: {attr_value}")


# Close the netCDF file
dataset.close()
