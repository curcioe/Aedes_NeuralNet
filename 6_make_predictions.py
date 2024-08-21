import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import rasterio
from datetime import datetime, timedelta

# File paths and folders
final_model_file = 'predictions/final_model_state_dict.pth'  # Replace with the actual path to the model file
era5_directory = 'era5_processed'  # Replace with the actual path to your ERA5 data directory
output_folder = 'predictions_final'  # Replace with the desired output folder

# Date range for predictions (string format)
start_date_str = "2020-12-30"  # Replace with your desired start date
end_date_str = "2020-12-31"  # Replace with your desired end date
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

# Sequence length (number of days)
sequence_length = 365  # Days range from -365 to -1

# Weather variable types
weather_var_types = ['mean_2m_temperature', 'range_2m_air_temperature', 'relative_humidity', 'surface_pressure', 'total_precip_scaled', 'wind_10m']

# Load the saved model
class FeedforwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Calculate input size
input_size = sequence_length * len(weather_var_types)

# Initialize the model
final_model = FeedforwardNN(input_size)

# Load the model state dict
final_model.load_state_dict(torch.load(final_model_file))
final_model.eval()

# Function to load and process the ERA5 files for multiple bands
def load_era5_data_for_date_range(observed_date, sequence_length):
    data_list = []
    for i in range(sequence_length):
        day = observed_date - timedelta(days=(sequence_length - 1 - i))
        day_data = []
        for var in weather_var_types:
            file_name = f"{day.strftime('%Y-%m-%d')}.{var}.tif"
            file_path = os.path.join(era5_directory, file_name)
            if os.path.exists(file_path):
                with rasterio.open(file_path) as src:
                    data = src.read(1)  # Read the first band
                    day_data.append(data.flatten())
            else:
                print(f"File not found: {file_path}")
                return None
        if day_data:
            data_list.append(np.stack(day_data, axis=1))  # Stack the data for each variable
    era5_data = np.stack(data_list, axis=0)  # Stack the data for each day
    return era5_data

# Function to save predictions to a GeoTIFF file
def save_predictions_to_geotiff(probabilities, reference_file, output_file):
    with rasterio.open(reference_file) as src:
        meta = src.meta
        meta.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(probabilities.reshape(src.shape).astype(rasterio.float32), 1)

# Generate predictions for each day in the specified range
current_date = start_date
while current_date <= end_date:
    # Load and process the data
    era5_data = load_era5_data_for_date_range(current_date, sequence_length)
    
    if era5_data is not None:
        # Reshape data for standardization
        era5_data = era5_data.reshape(-1, len(weather_var_types))  # Reshape to (num_samples, 6)
        
        # Apply scalers (assuming scalers were saved earlier or can be reapplied)
        scaled_data = []
        for i, var in enumerate(weather_var_types):
            scaler = StandardScaler()
            scaled_data.append(scaler.fit_transform(era5_data[:, [i]]).flatten())
        
        # Stack the scaled data back together
        era5_data_scaled = np.stack(scaled_data, axis=1).reshape(-1, sequence_length * len(weather_var_types)).astype(np.float32)

        # Convert to tensor for prediction
        era5_data_tensor = torch.tensor(era5_data_scaled, dtype=torch.float32)
        
        # Make predictions using the final model
        with torch.no_grad():
            outputs = final_model(era5_data_tensor)
            probabilities = torch.sigmoid(outputs).numpy().flatten()
        
        # Save probabilities to a GeoTIFF file
        reference_file = os.path.join(era5_directory, f"{(current_date - timedelta(days=1)).strftime('%Y-%m-%d')}.{weather_var_types[0]}.tif")
        output_file = os.path.join(output_folder, f"{current_date.strftime('%Y-%m-%d')}.tif")
        save_predictions_to_geotiff(probabilities, reference_file, output_file)
        
        print(f"Saved predictions for {current_date.strftime('%Y-%m-%d')}")
    else:
        print(f"No data found for {current_date.strftime('%Y-%m-%d')}")
    
    # Move to the next day
    current_date += timedelta(days=1)

print("Finished generating predictions for all dates in the specified range.")
