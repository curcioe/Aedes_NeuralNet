import cdsapi
import os

# Define the directory where you want to store the downloaded files
download_folder = "D:/Evan Curcio/era5"  # <-- Modify this path

# Define the variables and corresponding daily statistics to download
variables_stats = [
    ("2m_temperature", ["daily_mean", "daily_minimum", "daily_maximum"]),
    ("10m_u_component_of_wind", ["daily_mean"]),
    ("10m_v_component_of_wind", ["daily_mean"]),
    ("2m_dewpoint_temperature", ["daily_mean"]),
    ("surface_pressure", ["daily_mean"]),
    ("total_precipitation", ["daily_mean"])
]

# General parameters
dataset = "derived-era5-single-levels-daily-statistics"
years = ["2020", "2021"]  # Modify to handle multiple years
months = [
    "01", "02", "03", "04", "05", "06",
    "07", "08", "09", "10", "11", "12"
]
days = [
    "01", "02", "03", "04", "05", "06",
    "07", "08", "09", "10", "11", "12",
    "13", "14", "15", "16", "17", "18",
    "19", "20", "21", "22", "23", "24",
    "25", "26", "27", "28", "29", "30", "31"
]
time_zone = "utc+00:00"
frequency = "1_hourly"

# Ensure the directory exists
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Initialize the API client
client = cdsapi.Client()

# Function to download data
def download_data(year, month, variable, stat):
    # Construct file and temp file names
    file_name = os.path.join(download_folder, f"{variable}.{stat}.{year}-{month}.nc")
    temp_file_name = os.path.join(download_folder, f"tmp.{variable}.{stat}.{year}-{month}.nc")

    # Check if file already exists
    if os.path.exists(file_name):
        print(f"File {file_name} already exists. Skipping download.")
        return

    # Check if temporary file exists (indicating a previous incomplete download)
    if os.path.exists(temp_file_name):
        print(f"Temporary file {temp_file_name} found. Resuming download.")
        os.remove(temp_file_name)  # Remove incomplete download to restart

    # Construct the API request for each variable, statistic, year, and month
    request = {
        "product_type": "reanalysis",
        "variable": [variable],  # List format as per API requirements
        "year": [year],          # Single year for this request
        "month": [month],        # Single month for this request
        "day": days,             # All days in the month
        "daily_statistic": stat,
        "time_zone": time_zone,
        "frequency": frequency
    }

    # Log the request (optional)
    print(f"Requesting {variable} with statistic {stat} for {year}-{month}...")

    # Download the data
    try:
        # Using the temp file for intermediate download
        client.retrieve(dataset, request).download(temp_file_name)

        # Once download is successful, rename the temp file to the final file name
        os.rename(temp_file_name, file_name)
        print(f"Download complete: {file_name}")

    except Exception as e:
        print(f"Failed to download {file_name}: {e}")
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)  # Clean up temp file on failure

# Loop through each variable and their respective statistics
for variable, stats in variables_stats:
    for stat in stats:
        # Loop through each year
        for year in years:
            # Loop through each month
            for month in months:
                # Download the data for each combination of variable, statistic, year, and month
                download_data(year, month, variable, stat)
