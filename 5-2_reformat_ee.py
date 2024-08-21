import os
import shutil
import numpy as np
import rasterio
import glob

def calculate_temperature_range(max_temp_path, min_temp_path, output_path):
    with rasterio.open(max_temp_path) as max_src, rasterio.open(min_temp_path) as min_src:
        max_temp = max_src.read(1)
        min_temp = min_src.read(1)
        temp_range = max_temp - min_temp
        
        profile = max_src.profile
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(temp_range, 1)

def calculate_relative_humidity(temp_path, dewpoint_path, output_path):
    with rasterio.open(temp_path) as temp_src, rasterio.open(dewpoint_path) as dewpoint_src:
        temp_k = temp_src.read(1)
        dewpoint_k = dewpoint_src.read(1)
        
        # Convert temperatures from Kelvin to Celsius
        temp_c = temp_k - 273.15
        dewpoint_c = dewpoint_k - 273.15
        
        # Calculate relative humidity using the correct formula
        rh = 100 * (np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c)) / 
                    np.exp((17.625 * temp_c) / (243.04 + temp_c)))
        
        # Clip values to 0-100 range
        rh = np.clip(rh, 0, 100)
        
        # Update profile for the output raster
        profile = temp_src.profile
        profile.update(dtype=rh.dtype)
        
        # Write the relative humidity to the output file
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(rh, 1)

def calculate_wind_speed(u_path, v_path, output_path):
    with rasterio.open(u_path) as u_src, rasterio.open(v_path) as v_src:
        u = u_src.read(1)
        v = v_src.read(1)
        wind_speed = np.sqrt(u**2 + v**2)
        
        profile = u_src.profile
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(wind_speed, 1)

def move_files(file_paths, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    for file_path in file_paths:
        shutil.move(file_path, os.path.join(destination_folder, os.path.basename(file_path)))

def process_files(input_folder, old_folder):
    os.makedirs(old_folder, exist_ok=True)
    
    max_temp_files = glob.glob(os.path.join(input_folder, '*maximum_2m_temperature.tif'))
    min_temp_files = glob.glob(os.path.join(input_folder, '*minimum_2m_temperature.tif'))
    mean_temp_files = glob.glob(os.path.join(input_folder, '*mean_2m_temperature.tif'))
    dewpoint_files = glob.glob(os.path.join(input_folder, '*dewpoint_2m_temperature.tif'))
    u_wind_files = glob.glob(os.path.join(input_folder, '*u_component_of_wind_10m.tif'))
    v_wind_files = glob.glob(os.path.join(input_folder, '*v_component_of_wind_10m.tif'))
    # mslp_files = glob.glob(os.path.join(input_folder, '*mean_sea_level_pressure.tif'))
    precip_files = glob.glob(os.path.join(input_folder, '*total_precipitation.tif'))
    surface_pressure_files = glob.glob(os.path.join(input_folder, '*surface_pressure.tif'))

    # Process temperature range
    for max_temp_path, min_temp_path in zip(max_temp_files, min_temp_files):
        date = os.path.basename(max_temp_path).split('.')[0]
        range_output_path = os.path.join(input_folder, f'{date}.range_2m_temperature.tif')
        calculate_temperature_range(max_temp_path, min_temp_path, range_output_path)
    
    move_files(max_temp_files + min_temp_files, old_folder)

    # Move mean sea level pressure to old folder
    # move_files(mslp_files, old_folder)
    
    # Process relative humidity
    for mean_temp_path, dewpoint_path in zip(mean_temp_files, dewpoint_files):
        date = os.path.basename(mean_temp_path).split('.')[0]
        rh_output_path = os.path.join(input_folder, f'{date}.relative_humidity.tif')
        calculate_relative_humidity(mean_temp_path, dewpoint_path, rh_output_path)
    
    move_files(dewpoint_files, old_folder)

    # Process wind speed
    for u_path, v_path in zip(u_wind_files, v_wind_files):
        date = os.path.basename(u_path).split('.')[0]
        wind_output_path = os.path.join(input_folder, f'{date}.wind_10m.tif')
        calculate_wind_speed(u_path, v_path, wind_output_path)
    
    move_files(u_wind_files + v_wind_files, old_folder)

    # Identify remaining files to move
    remaining_files = glob.glob(os.path.join(input_folder, '*'))
    used_files = mean_temp_files + precip_files + surface_pressure_files + \
                 glob.glob(os.path.join(input_folder, '*range_2m_air_temperature.tif')) + \
                 glob.glob(os.path.join(input_folder, '*relative_humidity.tif')) + \
                 glob.glob(os.path.join(input_folder, '*wind_10m.tif'))
    files_to_move = [f for f in remaining_files if f not in used_files]

    # Move remaining unused files to old folder
    move_files(files_to_move, old_folder)

input_folder = 'era5_processed/'
old_folder = 'era5_unused'

process_files(input_folder, old_folder)
