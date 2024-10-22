import xarray as xr
import rasterio
from rasterio.transform import from_origin
import os
import pandas as pd

# Define input and output directories
input_dir = 'era5'
output_dir = 'era5_processed'

# Run for each of the variables, commenting out as you go to run only one variable at a time

# Dewpoint Mean
# input_name = '2m_dewpoint_temperature.daily_mean'
# output_name = 'dewpoint_2m_temperature'
# band = 'd2m'

# Temperature Maximum
# input_name = '2m_temperature.daily_maximum'
# output_name = 'maximum_2m_air_temperature'
# band = 't2m'

# Temperature Minimum
# input_name = '2m_temperature.daily_minimum'
# output_name = 'minimum_2m_air_temperature'
# band = 't2m'

# Temperature Mean
# input_name = '2m_temperature.daily_mean'
# output_name = 'mean_2m_air_temperature'
# band = 't2m'

# Wind U Component
# input_name = '10m_u_component_of_wind.daily_mean'
# output_name = 'u_component_of_wind_10m'
# band = 'u10'

# Wind V Component
# input_name = '10m_v_component_of_wind.daily_mean'
# output_name = 'v_component_of_wind_10m'
# band = 'v10'

# Surface Pressure
# input_name = 'surface_pressure.daily_mean'
# output_name = 'surface_pressure'
# band = 'sp'

# Total Precip
input_name = 'total_precipitation.daily_mean'
output_name = 'total_precipitation_mean'
band = 'tp'
    

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate list of dates from January 2020 to June 2024
start_date = '2019-01'
end_date = '2023-12'
dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Use 'MS' for month start

# Loop through each month and process the corresponding NetCDF file
for date in dates:
    month_str = date.strftime('%Y-%m')
    input_file = f'{input_name}.{month_str}.nc'
    nc_path = os.path.join(input_dir, input_file)
    
    if not os.path.exists(nc_path):
        print(f"File not found: {nc_path}")
        continue
    
    # Load the NetCDF file
    ds = xr.open_dataset(nc_path)

    # Get the coordinates and time values
    lats = ds['lat'].values
    lons = ds['lon'].values
    times = ds['time'].values

    # Define the spatial resolution and transformation
    resolution_x = (lons[-1] - lons[0]) / (len(lons) - 1)
    resolution_y = (lats[-1] - lats[0]) / (len(lats) - 1)
    transform = from_origin(lons[0] - resolution_x / 2, lats[0] + resolution_y / 2, resolution_x, -resolution_y)

    # Loop through each time step and save as TIFF
    for i, time in enumerate(times):
        # Extract the data for the current day
        data = ds[band].isel(time=i).values
        
        # Check if data is valid
        if data is None or data.size == 0:
            print(f"No data for date: {time}")
            continue
        
        # Create the output file name
        date_str = str(time)[:10]
        output_file = f'{date_str}.{output_name}.tif'
        output_path = os.path.join(output_dir, output_file)
        
        # Define metadata for the TIFF file
        metadata = {
            'driver': 'GTiff',
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'dtype': str(data.dtype),  # Ensure dtype is a string
            'crs': 'EPSG:4326',  # Assuming the data is in geographic coordinates
            'transform': transform,
        }
        
        # Write data to a TIFF file
        try:
            with rasterio.open(output_path, 'w', **metadata) as dst:
                dst.write(data, 1)
        except Exception as e:
            print(f"Error writing file {output_path}: {e}")

print("Conversion complete.")
