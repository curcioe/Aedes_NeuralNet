import os
import rasterio
import shutil

# Path to the directory containing the TIFF files
input_directory = 'era5_processed'
output_directory = 'era5_processed'
old_files_directory = 'precipitation_old'

# Create the old files directory if it doesn't exist
os.makedirs(old_files_directory, exist_ok=True)

# List all files in the input directory
all_files = os.listdir(input_directory)

# Process each precipitation file
for file_name in all_files:
    if 'total_precipitation' in file_name:
        date_str = file_name.split('.')[0]
        precip_file = os.path.join(input_directory, file_name)
        output_file = os.path.join(output_directory, f'{date_str}.total_precip_scaled.tif')

        # Read the precipitation data
        with rasterio.open(precip_file) as precip_src:
            precipitation = precip_src.read(1)
            profile = precip_src.profile

        # Multiply precipitation values by 24
        precipitation_24h = precipitation * 24

        # Update the profile for the output file
        profile.update(dtype=rasterio.float32, count=1)

        # Write the modified precipitation to the output file
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(precipitation_24h.astype(rasterio.float32), 1)

        # Move the original file to the old files directory
        shutil.move(precip_file, os.path.join(old_files_directory, file_name))

        print(f'Processed {date_str}')
