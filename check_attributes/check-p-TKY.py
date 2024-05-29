import netCDF4 as nc
import numpy as np
from scipy.spatial.distance import cdist

def find_nearest_lat_lon(latitude, longitude, latitudes, longitudes):
    # Use Haversine formula to calculate distances
    distances = cdist([(latitude, longitude)], list(zip(latitudes, longitudes)))
    # Find the indices of the minimum distance
    min_idx = np.argmin(distances)
    return min_idx

# Replace 'MIPAS-E_IMK.201201.V8R_O3_761.nc' with your actual file path
#file_path = 'MIPAS-E_IMK.201201.V8R_O3_761.nc' # 761: 7 stands for noctilucent cloud
file_path = 'MIPAS-E_IMK.201201.V8R_O3_561.nc'

# Open the netCDF file
dataset = nc.Dataset(file_path, 'r')

# Tokyo coordinates (replace with the actual coordinates)
tokyo_latitude = 35.6895
tokyo_longitude = 139.6917

# Extract latitude and longitude variables
latitude_values = dataset.variables['latitude'][:]
longitude_values = dataset.variables['longitude'][:]

# Find the indices of the nearest latitude and longitude to Tokyo
tokyo_lat_idx = find_nearest_lat_lon(tokyo_latitude, tokyo_longitude, latitude_values, longitude_values)
tokyo_lon_idx = find_nearest_lat_lon(tokyo_latitude, tokyo_longitude, latitude_values, longitude_values)
print(f"TKY lat lon index: {tokyo_lat_idx},{tokyo_lon_idx}")
print(f"TKY lat lon values: {latitude_values[tokyo_lat_idx]},{longitude_values[tokyo_lon_idx]}")

# Extract pressure levels at the specified latitude and longitude
#pressure_levels = dataset.variables['pressure'][:, tokyo_lat_idx, tokyo_lon_idx]
pressure_levels = dataset.variables['pressure']
print(f"Pressure levels of {tokyo_lat_idx},{tokyo_lon_idx} is")
print(pressure_levels[:,tokyo_lat_idx])

# Display information about pressure levels at Tokyo
print("Pressure Levels Information at Tokyo:")
print("--------------------")
print(f"Latitude at Tokyo: {latitude_values[tokyo_lat_idx]}")
print(f"Longitude at Tokyo: {longitude_values[tokyo_lon_idx]}")
print(f"Pressure Levels Shape: {pressure_levels.shape}")
print(f"Pressure Levels Data:")
print(np.array(pressure_levels))

# Close the netCDF file
dataset.close()
