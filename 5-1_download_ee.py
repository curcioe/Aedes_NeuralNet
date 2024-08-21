import ee
import geemap
from datetime import datetime, timedelta
import os
from multiprocessing import Pool
import glob
import rasterio
from rasterio.merge import merge

# Initialize the Earth Engine module
service_account = 'kai-xu@kaicast.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'kaicast-006e967d36db.json')
ee.Initialize(credentials)

# Define the start and end dates
start_date = '2020-01-01'
end_date = '2020-12-31'

# Create a date range
start_dt = datetime.strptime(start_date, '%Y-%m-%d')
end_dt = datetime.strptime(end_date, '%Y-%m-%d')
date_list = [(start_dt + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_dt - start_dt).days + 1)]

# Select the ERA5 Daily Aggregates dataset
dataset = ee.ImageCollection('ECMWF/ERA5/DAILY')

# Select the bands you want to download
bands = [
    'mean_2m_air_temperature',
    'minimum_2m_air_temperature',
    'maximum_2m_air_temperature',
    'dewpoint_2m_temperature',
    'total_precipitation',
    'surface_pressure',
    'u_component_of_wind_10m',
    'v_component_of_wind_10m'
]

# Define the global region of interest
region1 = ee.Geometry.Polygon([
    [[-180, -90], [-180, 90], [0, 90], [0, -90], [-180, -90]]
])
region2 = ee.Geometry.Polygon([
    [[0, -90], [0, 90], [180, 90], [180, -90], [0, -90]]
])

# Local directory to save the files
out_dir = 'era5/'
os.makedirs(out_dir, exist_ok=True)

# Function to download and save each day's data
def download_day_data(date):
    try:
        era5_data = dataset.filter(ee.Filter.date(date, (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')))
        mosaic = era5_data.select(bands).mosaic()
        filename1 = f'half1.{date}.tif'
        filename2 = f'half2.{date}.tif'
        file_path1 = os.path.join(out_dir, filename1)
        file_path2 = os.path.join(out_dir, filename2)

        # Export first half image
        geemap.ee_export_image(
            mosaic,
            filename=file_path1,
            scale=27830,
            region=region1,
            crs='EPSG:4326',
            file_per_band=True
        )

        # Export second half image
        geemap.ee_export_image(
            mosaic,
            filename=file_path2,
            scale=27830,
            region=region2,
            crs='EPSG:4326',
            file_per_band=True
        )

        print(f'Files downloaded and saved locally as {file_path1} and {file_path2}')
    except Exception as e:
        print(f"Error downloading data for {date}: {e}")

# Function to download data using multiprocessing
def download_data_in_parallel(dates):
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(download_day_data, dates)

# Function to merge downloaded halves
def merge_halves():
    # Get all half1 files
    half1_files = glob.glob(os.path.join(out_dir, 'half1.*.tif'))

    for half1_file in half1_files:
        # Extract the unique ID
        unique_id = os.path.basename(half1_file).split('.', 1)[-1]
        
        # Construct the half2 filename
        half2_file = os.path.join(out_dir, f'half2.{unique_id}')
        
        # Check if the corresponding half2 file exists
        if os.path.exists(half2_file):
            print(f"Found matching half2 file for: {half1_file}")
            
            # Open both halves
            with rasterio.open(half1_file) as src1, rasterio.open(half2_file) as src2:
                # Merge the two halves
                mosaic, out_transform = merge([src1, src2])
                
                # Copy the metadata from the first file
                out_meta = src1.meta.copy()
                
                # Update the metadata
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform
                })
                
                # Save the merged image
                new_filename = os.path.join(out_dir, f'{unique_id}')
                with rasterio.open(new_filename, "w", **out_meta) as dest:
                    dest.write(mosaic)
            
            # Delete the half files
            os.remove(half1_file)
            os.remove(half2_file)
            
            print(f"Merged and saved: {new_filename}")
        else:
            print(f"Corresponding half2 file not found for: {half1_file}")

# Main entry point
if __name__ == '__main__':
    download_data_in_parallel(date_list)
    print("All files downloaded successfully.")
    merge_halves()
    print("Merging completed.")
