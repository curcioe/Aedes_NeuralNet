import pandas as pd
import numpy as np

def calculate_relative_humidity(dewpoint, temp):
    dewpoint_celsius = dewpoint - 273.15
    temp_celsius = temp - 273.15
    rh = 100 * (np.exp((17.625 * dewpoint_celsius) / (243.04 + dewpoint_celsius)) / 
                np.exp((17.625 * temp_celsius) / (243.04 + temp_celsius)))
    return rh

def process_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Process each day from -365 to -1
    for day in range(-365, 0):
        day_str = f'{day}'

        # Create range_2m_air_temperature
        df[f'{day_str}.range_2m_air_temperature'] = (
            df[f'{day_str}.maximum_2m_air_temperature'] - 
            df[f'{day_str}.minimum_2m_air_temperature']
        )

        # Create relative_humidity
        df[f'{day_str}.relative_humidity'] = calculate_relative_humidity(
            df[f'{day_str}.dewpoint_2m_temperature'],
            df[f'{day_str}.mean_2m_air_temperature']
        )

        # Combine u and v components of wind
        df[f'{day_str}.wind_10m'] = np.sqrt(
            df[f'{day_str}.u_component_of_wind_10m']**2 + 
            df[f'{day_str}.v_component_of_wind_10m']**2
        )

        # Drop unnecessary columns
        columns_to_drop = [
            f'{day_str}.minimum_2m_air_temperature',
            f'{day_str}.maximum_2m_air_temperature',
            f'{day_str}.dewpoint_2m_temperature',
            f'{day_str}.u_component_of_wind_10m',
            f'{day_str}.v_component_of_wind_10m'
        ]
        df = df.drop(columns=columns_to_drop)

    # Reorder columns
    first_columns = ['real_observation', 'observed_on', 'latitude', 'longitude']
    other_columns = [col for col in df.columns if col not in first_columns]
    
    # Sort other columns by day (from -365 to -1) and then alphabetically
    other_columns = sorted(other_columns, 
                           key=lambda x: (int(x.split('.')[0]) if '.' in x else 0, x))
    
    df = df[first_columns + other_columns]

    # Save the processed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Usage
input_file = '../datasets/aedes_aegypti_dataset_raw.csv'  # Replace with your input file name
output_file = '../datasets/aedes_aegypti_dataset.csv'  # Replace with your desired output file name

process_csv(input_file, output_file)
