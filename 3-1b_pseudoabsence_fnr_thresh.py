import ee
import datetime
import pandas as pd
import multiprocessing
import os
import time
import random
import math
import numpy as np
import csv
import rasterio
from global_land_mask import globe

# Authenticate and initialize the Earth Engine module with a service account.
service_account = 'kai-xu@kaicast.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'kaicast-006e967d36db.json')
ee.Initialize(credentials)

def get_era5_data_from_tiff(latitude, longitude, date_str, bands):
    try:
        properties = {'date': date_str}
        for band in bands:
            file_path = f"C:/Users/curcioej/era5_processed/{date_str}.{band}.tif"
            if os.path.exists(file_path):
                with rasterio.open(file_path) as src:
                    val = next(src.sample([(longitude, latitude)]))
                    properties[band] = float(val[0])
            else:
                print(f"TIFF file not found: {file_path}")
                properties[band] = None
        return properties
    except Exception as e:
        print(f"Error fetching data from TIFF for {date_str} at {latitude, {longitude}}: {e}")
        return None

def get_era5_data_from_ee(latitude, longitude, date_str, bands):
    try:
        point = ee.Geometry.Point([longitude, latitude])
        start_date = ee.Date(date_str)
        end_date = start_date.advance(1, 'day')
        
        era5 = ee.ImageCollection("ECMWF/ERA5/DAILY").filterDate(start_date, end_date).filterBounds(point)
        
        def extract_data(image):
            first_dict = image.reduceRegion(ee.Reducer.first(), point, 27830)
            return ee.Feature(None, first_dict).set('date', image.date().format('YYYY-MM-dd'))

        data_reduced = era5.map(extract_data)
        data_list = data_reduced.getInfo()
        
        if data_list['features']:
            properties = data_list['features'][0]['properties']
            filtered_properties = {k: v for k, v in properties.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
            filtered_properties['date'] = date_str
            return filtered_properties
        else:
            return None
    except ee.ee_exception.EEException as e:
        print(f"Error fetching data from Earth Engine for {date_str} at {latitude, {longitude}}: {e}")
        return None

def get_era5_data(latitude, longitude, date_str, num_days, bands):
    input_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    end_date = input_date - datetime.timedelta(days=1)
    
    result = []
    
    for day in range(num_days):
        current_date = end_date - datetime.timedelta(days=day)
        current_date_str = current_date.strftime("%Y-%m-%d")
        
        if current_date.year >= 2019: # hard coded to always get data from tif, and never from earth engine
            data = get_era5_data_from_tiff(latitude, longitude, current_date_str, bands)
        else:
            data = get_era5_data_from_ee(latitude, longitude, current_date_str, bands)
        
        if data:
            result.append(data)
        else:
            result.append({'date': current_date_str, **{band: None for band in bands}})
    
    return result

def process_row(row, num_days, bands):
    date_str = row['observed_on']
    latitude = row['latitude']
    longitude = row['longitude']
    data = get_era5_data(latitude, longitude, date_str, num_days, bands)
    if not data:
        return None

    result = {'observed_on': date_str, 'latitude': latitude, 'longitude': longitude}
    for day, item in enumerate(data):
        for band in bands:
            result[f"-{day + 1}.{band}"] = item.get(band, None)

    if any(value is not None for key, value in result.items() if key not in ['observed_on', 'latitude', 'longitude']):
        return result
    return None

def fetch_and_save_weather_data(input_csv, output_csv, num_days=365, bands=None, batch_size=10):
    if bands is None:
        bands = ['dewpoint_2m_temperature', 'maximum_2m_air_temperature', 'mean_2m_air_temperature', 
                 'minimum_2m_air_temperature', 'surface_pressure', 'total_precipitation', 'u_component_of_wind_10m', 'v_component_of_wind_10m']

    df = pd.read_csv(input_csv)
    num_rows = len(df)
    num_batches = (num_rows + batch_size - 1) // batch_size
    all_results = []
    start_time = time.time()

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, num_rows)
        batch_df = df.iloc[batch_start:batch_end]

        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            batch_results = pool.starmap(process_row, [(row, num_days, bands) for _, row in batch_df.iterrows()])

        all_results.extend([item for item in batch_results if item is not None])

        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time / (batch_num + 1) * (num_batches - batch_num - 1)
        print(f"Batch {batch_num + 1}/{num_batches} processed. Estimated time remaining: {remaining_time / 60:.2f} minutes.")

    if all_results:
        columns = ['observed_on', 'latitude', 'longitude'] + [f"-{day}.{band}" for day in range(365, 0, -1) for band in bands]
        pd.DataFrame(all_results)[columns].to_csv(output_csv, index=False)

def generate_random_points(n, date_range):
    points = []
    while len(points) < n:
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        weight = math.cos(math.radians(lat))
        if random.random() < weight:
            random_date = random.choice(date_range)
            points.append({'latitude': lat, 'longitude': lon, 'observed_on': random_date})
    return points

import pandas as pd
import time
import rasterio
import numpy as np

def generate_pseudo_absences(input_file, output_file, absence_ratio, fnr_input_file, fnr_threshold, num_days=365, bands=None, batch_size=10):
    if bands is None:
        bands = ['dewpoint_2m_temperature', 'maximum_2m_air_temperature', 'mean_2m_air_temperature', 
                 'minimum_2m_air_temperature', 'surface_pressure', 'total_precipitation', 'u_component_of_wind_10m', 'v_component_of_wind_10m']

    # Load false negative rate GeoTiff file
    with rasterio.open(fnr_input_file) as fnr_src:
        fnr_data = fnr_src.read(1)
        fnr_transform = fnr_src.transform
        fnr_nodata = fnr_src.nodata

    observations = pd.read_csv(input_file)
    num_samples = len(observations) * absence_ratio
    original_points = set((row['latitude'], row['longitude']) for _, row in observations.iterrows())

    earliest_date = observations['observed_on'].min()
    latest_date = observations['observed_on'].max()
    date_range = pd.date_range(start=earliest_date, end=latest_date).strftime("%Y-%m-%d").tolist()

    print(f"Date range: {earliest_date} to {latest_date}")

    pseudo_absences = []
    start_time = time.time()
    saved_absences = 0

    while len(pseudo_absences) < num_samples:
        points = generate_random_points(batch_size, date_range)
        
        for point in points:
            latitude = point['latitude']
            longitude = point['longitude']
            date_str = point['observed_on']
            
            if (latitude, longitude) not in original_points and globe.is_land(latitude, longitude):
                # Get the false negative rate at this point
                row, col = ~fnr_transform * (longitude, latitude)
                row, col = int(row), int(col)

                # Check if the point is within the bounds of the raster
                if 0 <= row < fnr_data.shape[0] and 0 <= col < fnr_data.shape[1]:
                    fnr_value = fnr_data[row, col]
                    
                    # Check if the false negative rate is below the threshold and not a nodata value
                    if fnr_value != fnr_nodata and fnr_value <= fnr_threshold:
                        data = get_era5_data(latitude, longitude, date_str, num_days, bands)
                        if data:
                            result = {'observed_on': date_str, 'latitude': latitude, 'longitude': longitude}
                            for day, item in enumerate(data):
                                for band in bands:
                                    result[f"-{day + 1}.{band}"] = item.get(band, None)
                            
                            if any(value is not None for key, value in result.items() if key not in ['observed_on', 'latitude', 'longitude']):
                                pseudo_absences.append(result)

        if len(pseudo_absences) > num_samples:
            pseudo_absences = pseudo_absences[:num_samples]

        if len(pseudo_absences) - saved_absences >= 10:
            pd.DataFrame(pseudo_absences[saved_absences:len(pseudo_absences)]).to_csv(output_file, mode='a', header=(saved_absences == 0), index=False)
            saved_absences = len(pseudo_absences)

        elapsed_time = time.time() - start_time
        progress_ratio = len(pseudo_absences) / num_samples
        remaining_time = elapsed_time / progress_ratio * (1 - progress_ratio)
        print(f"Processed {len(pseudo_absences)} absences. Estimated time remaining: {remaining_time / 60:.2f} minutes.")

    if len(pseudo_absences) > saved_absences:
        pd.DataFrame(pseudo_absences[saved_absences:len(pseudo_absences)]).to_csv(output_file, mode='a', header=(saved_absences == 0), index=False)

def create_training_set(real_file, absence_file, output_file):
    real_df = pd.read_csv(real_file, low_memory=False)
    absence_df = pd.read_csv(absence_file, low_memory=False)
    
    print("Real dataframe 'observed_on' dtype:", real_df['observed_on'].dtype)
    print("Real dataframe 'observed_on' unique values:", real_df['observed_on'].unique()[:5])
    print("Absence dataframe 'observed_on' dtype:", absence_df['observed_on'].dtype)
    print("Absence dataframe 'observed_on' unique values:", absence_df['observed_on'].unique()[:5])
    
    def safe_parse_date(date_str):
        try:
            return pd.to_datetime(date_str, errors='raise')
        except ValueError:
            return pd.NaT

    real_df['observed_on'] = real_df['observed_on'].apply(safe_parse_date)
    absence_df['observed_on'] = absence_df['observed_on'].apply(safe_parse_date)
    
    real_df = real_df.dropna(subset=['observed_on'])
    absence_df = absence_df.dropna(subset=['observed_on'])
    
    real_df['real_observation'] = 1
    absence_df['real_observation'] = 0
    
    combined_df = pd.concat([real_df, absence_df], ignore_index=True)
    
    combined_df = combined_df.sort_values(by=['real_observation', 'observed_on'], ascending=[False, True])
    
    bands = ['dewpoint_2m_temperature', 'maximum_2m_air_temperature', 'mean_2m_air_temperature', 
             'minimum_2m_air_temperature', 'surface_pressure', 'total_precipitation', 'u_component_of_wind_10m', 'v_component_of_wind_10m']
    
    columns = ['real_observation', 'observed_on', 'latitude', 'longitude'] + [f"-{day}.{band}" for day in range(365, 0, -1) for band in bands]
    combined_df = combined_df[columns]
    
    combined_df.to_csv(output_file, index=False)
    
    print(f"Training set created and saved to {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Rows from real file: {len(real_df)}")
    print(f"Rows from absence file: {len(absence_df)}")

if __name__ == "__main__":
    input_csv = 'C:/Users/curcioej/datasets/aedes_aegypti.csv'
    output_real_csv = 'C:/Users/curcioej/datasets/aedes_aegypti_real_2iter.csv'
    fnr_input_file = 'C:/Users/curcioej/predictions_nn_fnr_365/2021-01-01_fnr.tif'
    fnr_thresh = 0.1
    batch_size_pseudo = 10

    # Create real observations; comment out code below if we already have this data
    fetch_and_save_weather_data(input_csv, output_real_csv, num_days=365)
    
    # Create pseudo-absences; comment out code below if we already have this data
    absence_ratio = 20
    pseudo_absence_output_csv = 'C:/Users/curcioej/datasets/aedes_aegypti_absence_2iter.csv'
    generate_pseudo_absences(input_file=output_real_csv, 
                             output_file=pseudo_absence_output_csv, 
                             absence_ratio=absence_ratio, 
                             fnr_input_file=fnr_input_file,
                             fnr_threshold=fnr_thresh,
                             num_days=365, 
                             bands=None, 
                             batch_size=batch_size_pseudo, 
                             )

    # Merge real and pseudo-absences into one file
    training_output_csv = 'C:/Users/curcioej/datasets/aedes_aegypti_dataset_raw_2iter.csv'
    create_training_set(output_real_csv, pseudo_absence_output_csv, training_output_csv)
