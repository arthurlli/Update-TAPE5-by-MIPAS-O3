# SZA: solar zenith angle

import netCDF4 as nc

# Replace 'MIPAS-E_IMK.201201.V8R_O3_761.nc' with your actual file path
#file_path = 'MIPAS-E_IMK.201201.V8R_O3_761.nc' # 761: 7 stands for noctilucent cloud
file_path = 'MIPAS-E_IMK.201201.V8R_O3_561.nc'

# Open the netCDF file
dataset = nc.Dataset(file_path, 'r')

# Extract sza variable and its attributes
sza_variable = dataset.variables['sza']
sza_values = sza_variable[:]
sza_attributes = sza_variable.ncattrs()

# Display information about the sza variable
print("SZA Variable Information:")
print("--------------------")
print(f"Variable Name: sza")
print(f"Shape: {sza_values.shape}")
print(f"Data Type: {sza_values.dtype}")
print(f"Attributes: {sza_attributes}")

# Display the actual sza values
print("\nSZA Values:")
print(sza_values)

print("\nSZA Values single:")
print(sza_values[40])



variable_name = 'sza'
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


# Close the netCDF file
#dataset.close()
