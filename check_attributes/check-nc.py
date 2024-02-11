import netCDF4 as nc

# read netCDF4
file_path = 'MIPAS-E_IMK.201201.V8R_O3_761.nc'

# Open the netCDF file
dataset = nc.Dataset(file_path, 'r')

# Display basic information about the dataset
print("Dataset Information:")
print("--------------------")
print("File Path:", file_path)
print("File Format:", dataset.file_format)
print("Dimensions:", dataset.dimensions.keys())
print("Variables:")
for var_name, var in dataset.variables.items():
    print(f"  {var_name}:")
    print(f"    Shape: {var.shape}")
    print(f"    Data Type: {var.dtype}")
    print(f"    Attributes: {var.ncattrs()}")

# Close the netCDF file
dataset.close()
