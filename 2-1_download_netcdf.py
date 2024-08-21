import cdsapi
import requests
from multiprocessing import Pool
import os

# Create the 'era5' directory if it doesn't exist
download_folder = "era5"
os.makedirs(download_folder, exist_ok=True)

c = cdsapi.Client(timeout=300)

years = [str(year) for year in range(2019, 2020)]
months = ['{:02d}'.format(month) for month in range(4, 5)]
current_year = '2024'
current_month = '08'

# Variables and their corresponding daily statistics to download
variables_stats = [
    ("2m_temperature", ["daily_mean", "daily_minimum", "daily_maximum"]),
    ("10m_u_component_of_wind", ["daily_mean"]),
    ("10m_v_component_of_wind", ["daily_mean"]),
    ("2m_dewpoint_temperature", ["daily_mean"]),
    ("surface_pressure", ["daily_mean"]),
    ("total_precipitation", ["daily_mean"])
]

# Function to download data
def download_data(params):
    yr, mn, var, stat = params
    c = cdsapi.Client(timeout=300)  # Create a new client instance for each process
    file_name = os.path.join(download_folder, f"{var}.{stat}.{yr}-{mn}.nc")
    temp_file_name = os.path.join(download_folder, f"tmp.{var}.{stat}.{yr}-{mn}.nc")

    # Check if file already exists
    if os.path.exists(file_name):
        print(f"File {file_name} already exists. Skipping download.")
        return
    
    # Check if temporary file exists (indicating a previous incomplete download)
    if os.path.exists(temp_file_name):
        print(f"Temporary file {temp_file_name} found. Resuming download.")
        os.remove(temp_file_name)  # Remove incomplete download to restart
    
    result = c.service(
        "tool.toolbox.orchestrator.workflow",
        params={
            "realm": "user-apps",
            "project": "app-c3s-daily-era5-statistics",
            "version": "master",
            "kwargs": {
                "dataset": "reanalysis-era5-single-levels",
                "product_type": "reanalysis",
                "variable": var,
                "statistic": stat,
                "year": yr,
                "month": mn,
                "time_zone": "UTC+00:00",
                "frequency": "1-hourly"
            },
            "workflow_name": "application"
        }
    )
    
    location = result[0]['location']
    res = requests.get(location, stream=True)
    print(f"Writing data to {temp_file_name}")
    
    with open(temp_file_name, 'wb') as fh:
        for r in res.iter_content(chunk_size=1024):
            fh.write(r)
    
    # Rename the temporary file to the final file name
    os.rename(temp_file_name, file_name)
    print(f"Download completed. Renamed {temp_file_name} to {file_name}")

# Prepare parameters for multiprocessing
params = []
for yr in years:
    for mn in months:
        if yr == current_year and mn > current_month:
            break
        for var, stats in variables_stats:
            for stat in stats:
                file_name = os.path.join(download_folder, f"{var}.{stat}.{yr}-{mn}.nc")
                if not os.path.exists(file_name):
                    params.append((yr, mn, var, stat))

# Use multiprocessing to download data
if __name__ == '__main__':
    with Pool(processes=8) as pool:  # Adjust the number of processes as needed
        pool.map(download_data, params)
