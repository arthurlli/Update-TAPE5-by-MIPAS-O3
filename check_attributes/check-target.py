import netCDF4 as nc

# Replace 'MIPAS-E_IMK.201201.V8R_O3_761.nc' with your actual file path
#file_path = 'MIPAS-E_IMK.201201.V8R_O3_761.nc' # 761: 7 stands for noctilucent cloud
file_path = 'MIPAS-E_IMK.201201.V8R_O3_561.nc'

# Open the netCDF file
dataset = nc.Dataset(file_path, 'r')

# Extract target variable and its attributes
target_variable = dataset.variables['target']
target_values = target_variable[:]
target_attributes = target_variable.ncattrs()

# Display information about the target variable
print("Target Variable Information:")
print("--------------------")
print(f"Variable Name: target")
print(f"Shape: {target_values.shape}")
print(f"Data Type: {target_values.dtype}")
print(f"Attributes: {target_attributes}")

# Display the actual target values
print("\nTarget Values:")
print(target_values)

variable_name = 'target'
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
