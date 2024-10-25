import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from datetime import datetime, timedelta
import rasterio
from scipy import interpolate
import itertools

# Hyperparameters and configuration
absence_ratios = [0.2]
kfold_patience = 10
final_model_patience = 100
batch_size = 32
learning_rate = 0.001
k_folds = 5
sequence_length = 30  # Change this value to adjust the number of days used for input features

# Default selected features (modify this list as needed)
selected_features = [
    "mean_2m_air_temperature", 
    "range_2m_air_temperature", 
    "relative_humidity", 
    "surface_pressure", 
    "total_precipitation", 
    "wind_10m"
]

# File paths and folders
data_file = 'C:/Users/curcioej/datasets/aedes_aegypti_dataset.csv'
era5_directory = 'C:/Users/curcioej/era5_processed'  # Replace with the actual path
output_folder = 'C:/Users/curcioej/predictions_nn_fnr_loo'  # Replace with the desired output folder
auc_csv_file = os.path.join(output_folder, 'auc_values.csv')
auc_plot_file = os.path.join(output_folder, 'auc_vs_absence_ratio.png')
shap_image_file = os.path.join(output_folder, 'shap_feature_importance.png')
roc_data_file = os.path.join(output_folder, 'roc_data.csv')
roc_plot_file = os.path.join(output_folder, 'roc_curve.png')
test_predictions_file = os.path.join(output_folder, 'test_predictions.csv')
test_statistics_file = os.path.join(output_folder, 'test_statistics.csv')
shap_values_file = os.path.join(output_folder, 'shap_values.csv')
final_model_file = os.path.join(output_folder, 'final_model_state_dict.pth')
loo_results_file = os.path.join(output_folder, 'loo_feature_importance.csv')
                                
# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Date range for predictions (string format)
start_date_str = "2021-01-01"  # Replace with your desired start date
end_date_str = "2021-01-01"  # Replace with your desired end date
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

# Define the neural network
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

# Load the data
data = pd.read_csv(data_file)

# Filter columns based on sequence_length
columns_to_keep = ['observed_on', 'latitude', 'longitude', 'real_observation']
for day in range(-sequence_length, 0):
    for feature in selected_features:
        columns_to_keep.extend([col for col in data.columns if col.startswith(f'{day}.{feature}')])

# Filter the dataframe to keep only the relevant columns
data = data[columns_to_keep]

# Identify the columns that represent the selected weather variables
weather_columns = [col for col in data.columns[4:] if any(var in col for var in selected_features)]
weather_var_types = sorted(set([col.split('.')[1] for col in weather_columns]))  # Extract selected weather variable types

# Initialize scalers for each selected weather variable
scalers = {var: StandardScaler() for var in selected_features}

# Scale the data for each weather variable
for var in weather_var_types:
    cols_to_standardize = [col for col in weather_columns if var in col]
    scalers[var].fit(data[cols_to_standardize])
    data[cols_to_standardize] = scalers[var].transform(data[cols_to_standardize])

# Separate the data into presence (1) and absence (0) observations
presence_data = data[data['real_observation'] == 1]
absence_data = data[data['real_observation'] == 0]

# Create test set with 20% of presence observations and equal number of absences
presence_train, presence_test = train_test_split(presence_data, test_size=0.2, random_state=42)
absence_test = absence_data.sample(n=len(presence_test), random_state=42)
test_set = pd.concat([presence_test, absence_test]).reset_index(drop=True)

# Remaining data for training and validation
remaining_presence = presence_train
remaining_absence = absence_data.drop(absence_test.index)

# Function to create balanced dataset for validation
def create_balanced_dataset(presence, absence, n_samples):
    balanced_presence = presence.sample(n=n_samples, replace=False, random_state=42)
    balanced_absence = absence.sample(n=n_samples, replace=False, random_state=42)
    return pd.concat([balanced_presence, balanced_absence]).reset_index(drop=True)

def prepare_ffnn_data_optimized(df, sequence_length, weather_var_types):
    num_samples = df.shape[0]
    
    # Create the list of column names based on sequence_length and weather_var_types
    columns = []
    for day in range(-sequence_length, 0):
        for var in weather_var_types:
            columns.append(f'{day}.{var}')
    
    # Ensure that the columns exist in the DataFrame
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in the DataFrame: {missing_columns}")
    
    # Extract the relevant data
    weather_data = df[columns].values  # Select the relevant columns
    
    # Calculate the number of features based on the length of columns selected
    num_features_total = sequence_length * len(weather_var_types)
    
    # Ensure that the total size of the data matches the expected shape
    expected_size = num_samples * num_features_total
    actual_size = weather_data.size
    if actual_size != expected_size:
        raise ValueError(f"Cannot reshape array of size {actual_size} into shape ({num_samples},{num_features_total}). "
                         f"Check the sequence_length and selected_features.")
    
    # Reshape into (num_samples, sequence_length * num_features_per_day)
    X = weather_data.reshape(num_samples, num_features_total)
    y = df['real_observation'].values
    
    return X, y

# Store AUC values for different absence ratios
auc_values = []
best_overall_auc = 0
best_overall_model = None
best_overall_input_size = 0
best_absence_ratio = None

# Define feedforward neural network hyperparameters
input_size = sequence_length * len(weather_var_types)

# Sampling the absence data once per fold, rather than for each iteration
absence_sample_cache = {}

for absence_ratio in absence_ratios:
    fold_aucs = []
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(remaining_presence)):
        # Create balanced validation set
        val_presence = remaining_presence.iloc[val_idx]
        val_set = create_balanced_dataset(val_presence, remaining_absence, len(val_presence))
        
        # Create training set with specified absence ratio
        train_presence = remaining_presence.iloc[train_idx]
        n_train_absence = int(len(train_presence) * absence_ratio)
        
        # Check if we already have a sample for this absence ratio
        if absence_ratio not in absence_sample_cache:
            absence_sample_cache[absence_ratio] = remaining_absence.sample(n=n_train_absence, replace=False, random_state=42)
        
        train_absence = absence_sample_cache[absence_ratio]
        train_set = pd.concat([train_presence, train_absence]).reset_index(drop=True)
        
        # Prepare features and target for feedforward neural network
        X_train, y_train = prepare_ffnn_data_optimized(train_set, sequence_length, weather_var_types)
        X_val, y_val = prepare_ffnn_data_optimized(val_set, sequence_length, weather_var_types)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        
        # Create DataLoader for training and validation sets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize the feedforward neural network model, loss function, and optimizer
        model = FeedforwardNN(input_size)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training the model with early stopping
        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(1000):  # Set a high number, we'll use early stopping
            model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_probs = torch.sigmoid(val_outputs).numpy().flatten()
                
                fpr, tpr, thresholds = roc_curve(y_val, val_probs)
                val_auc = auc(fpr, tpr)
            
            # Ensure train_loss is a float before formatting
            train_loss_per_batch = train_loss / len(train_loader)
            
            print(f'Absence Ratio: {absence_ratio}, Fold {fold+1}, Epoch [{epoch+1}], '
                  f'Train Loss: {train_loss_per_batch:.4f}, '
                  f'Val AUC: {val_auc:.4f}')
        
            # Save the model if validation AUC improves
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check if we should stop early
            if patience_counter >= kfold_patience:
                print("Early stopping")
                break
        
        # Load the best model state
        model.load_state_dict(best_model_state)
        fold_aucs.append(best_val_auc)
        
        # Update the best overall model if this fold has the highest AUC
        if best_val_auc > best_overall_auc:
            best_overall_auc = best_val_auc
            best_overall_model = model
            best_overall_input_size = input_size
            best_absence_ratio = absence_ratio
    
    # Store the mean AUC for this absence ratio
    mean_auc = np.mean(fold_aucs)
    auc_values.append(mean_auc)

# Save AUC values to a file
auc_df = pd.DataFrame({'absence_ratio': absence_ratios, 'auc': auc_values})
auc_df.to_csv(auc_csv_file, index=False)

# Plot AUC vs Absence Ratio
plt.figure()
plt.plot(absence_ratios, auc_values, marker='o')
plt.xlabel('Absence Ratio')
plt.ylabel('AUC')
plt.title('AUC vs Absence Ratio')
plt.grid(True)
plt.savefig(auc_plot_file)
plt.close()

# Train the final model using the best hyperparameters
final_train_set = pd.concat([remaining_presence, remaining_absence]).reset_index(drop=True)
X_final, y_final = prepare_ffnn_data_optimized(final_train_set, sequence_length, weather_var_types)

# Perform an 80/20 split for training and validation sets
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_final_tensor = torch.tensor(X_train_final, dtype=torch.float32)
y_train_final_tensor = torch.tensor(y_train_final, dtype=torch.float32).view(-1, 1)
X_val_final_tensor = torch.tensor(X_val_final, dtype=torch.float32)
y_val_final_tensor = torch.tensor(y_val_final, dtype=torch.float32).view(-1, 1)

# Create DataLoaders for final training and validation sets
final_train_dataset = TensorDataset(X_train_final_tensor, y_train_final_tensor)
final_val_dataset = TensorDataset(X_val_final_tensor, y_val_final_tensor)
final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)
final_val_loader = DataLoader(final_val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the final model
final_model = FeedforwardNN(best_overall_input_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)

# Train the final model with early stopping
best_final_val_auc = 0.0
best_final_model_state = None
final_patience_counter = 0

for epoch in range(1000):  # Set a high number, we'll use early stopping
    final_model.train()
    train_loss = 0
    
    for X_batch, y_batch in final_train_loader:
        optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    final_model.eval()
    with torch.no_grad():
        val_outputs = final_model(X_val_final_tensor)
        val_probs = torch.sigmoid(val_outputs).numpy().flatten()
        
        fpr, tpr, thresholds = roc_curve(y_val_final, val_probs)
        val_auc = auc(fpr, tpr)
        
    # Ensure train_loss is a float before formatting
    train_loss_per_batch = train_loss / len(final_train_loader)
    
    print(f'Final Model Training - Epoch [{epoch+1}], Train Loss: {train_loss_per_batch:.4f}, '
          f'Val AUC: {val_auc:.4f}')
    
    # Save the model if validation AUC improves
    if val_auc > best_final_val_auc:
        best_final_val_auc = val_auc
        best_final_model_state = final_model.state_dict()
        final_patience_counter = 0
    else:
        final_patience_counter += 1
    
    # Check if we should stop early
    if final_patience_counter >= final_model_patience:
        print("Early stopping for final model")
        break

# Load the best final model state
final_model.load_state_dict(best_final_model_state)

# Save the final model state dict
torch.save(final_model.state_dict(), final_model_file)

# Evaluate the final model on the test set
X_test, y_test = prepare_ffnn_data_optimized(test_set, sequence_length, weather_var_types)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

final_model.eval()
with torch.no_grad():
    test_outputs = final_model(X_test_tensor)
    test_probs = torch.sigmoid(test_outputs).numpy().flatten()
    
    fpr, tpr, thresholds = roc_curve(y_test, test_probs)
    test_auc = auc(fpr, tpr)
    
    # Calculate FNR for each threshold
    fnr = 1 - tpr

    # Create an interpolation function to map probabilities to FNRs
    fnr_interpolator = interpolate.interp1d(thresholds, fnr, bounds_error=False, fill_value=(1, 0))
    
    test_preds = (test_probs > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    test_tpr = tp / (tp + fn)
    test_tnr = tn / (tn + fp)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)

# Save test predictions to a file
test_predictions_df = pd.DataFrame({'y_true': y_test, 'y_pred': test_preds, 'y_prob': test_probs})
test_predictions_df.to_csv(test_predictions_file, index=False)

# Save test statistics to a file
test_statistics_df = pd.DataFrame({
    'test_loss': [criterion(torch.tensor(test_preds, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)).item()],
    'test_accuracy': [test_accuracy],
    'test_precision': [test_precision],
    'test_recall': [test_recall],
    'test_tpr': [test_tpr],
    'test_tnr': [test_tnr],
    'test_auc': [test_auc],
    'test_f1': [test_f1]
})
test_statistics_df.to_csv(test_statistics_file, index=False)

print(f"Final Model Test Results:")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test TPR: {test_tpr:.4f}")
print(f"Test TNR: {test_tnr:.4f}")

# Save ROC data to a file
roc_data_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
roc_data_df.to_csv(roc_data_file, index=False)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.savefig(roc_plot_file)
plt.close()

# Define the save_probabilities_to_geotiff function
def save_probabilities_to_geotiff(probabilities, reference_file, output_file):
    with rasterio.open(reference_file) as src:
        meta = src.meta.copy()
    
    meta.update(dtype=rasterio.float32, count=1)
    
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(probabilities.astype(rasterio.float32), 1)

# Add this function definition right before the prediction loop

def load_era5_data_for_date_range(current_date, sequence_length, era5_directory, selected_features):
    data = []
    for i in range(sequence_length):
        date = current_date - timedelta(days=sequence_length-1-i)
        date_str = date.strftime('%Y-%m-%d')
        
        day_data = []
        for feature in selected_features:
            file_path = os.path.join(era5_directory, f"{date_str}.{feature}.tif")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
            
            with rasterio.open(file_path) as src:
                day_data.append(src.read(1))
        
        data.append(np.stack(day_data))
    
    return np.stack(data)

# Generate TIFF file predictions for each day in the specified range
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

current_date = start_date
while current_date <= end_date:
    era5_data = load_era5_data_for_date_range(current_date, sequence_length, era5_directory, selected_features)
    
    if era5_data is not None:
        print(f"ERA5 data shape: {era5_data.shape}")
        
        original_shape = era5_data.shape[2:]  # (lat, lon)
        num_features = len(selected_features)
        
        # Reshape to (lat * lon, sequence_length, num_features)
        era5_data = era5_data.transpose(2, 3, 0, 1).reshape(-1, sequence_length, num_features)
        
        # Scale the data using the pre-fitted scalers
        scaled_data = np.zeros_like(era5_data)
        for i, var in enumerate(selected_features):
            feature_data = era5_data[:, :, i].reshape(-1, sequence_length)
            scaled_data[:, :, i] = scalers[var].transform(feature_data).reshape(-1, sequence_length)
        
        # Reshape for model input (lat * lon, sequence_length * num_features)
        era5_data_scaled = scaled_data.reshape(-1, sequence_length * num_features)
        
        era5_data_tensor = torch.tensor(era5_data_scaled, dtype=torch.float32)
        
        final_model.eval()
        with torch.no_grad():
            outputs = final_model(era5_data_tensor)
            probabilities = torch.sigmoid(outputs).numpy().flatten()
        
        print(f"Probabilities shape: {probabilities.shape}, min: {probabilities.min()}, max: {probabilities.max()}")
        
        # Map probabilities to FNRs
        fnr_values = fnr_interpolator(probabilities)
        
        fnr_reshaped = fnr_values.reshape(original_shape)
        probs_reshaped = probabilities.reshape(original_shape)
        
        print(f"FNR shape: {fnr_reshaped.shape}, min: {fnr_reshaped.min()}, max: {fnr_reshaped.max()}")

        # Scatter plot of FNR vs probabilities flattened
        plt.figure(figsize=(8, 6))
        plt.scatter(probabilities, fnr_values, alpha=0.5, color='green')
        plt.title(f"FNR vs Probabilities")
        plt.xlabel('Probabilities')
        plt.ylabel('FNR')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{current_date.strftime('%Y-%m-%d')}_fnr_nn_scatter_flattened.png"))

        # Scatter plot of FNR vs probabilities shaped
        plt.figure(figsize=(8, 6))
        plt.scatter(probs_reshaped, fnr_reshaped, alpha=0.5, color='green')
        plt.title(f"FNR vs Probabilities")
        plt.xlabel('Probabilities')
        plt.ylabel('FNR')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{current_date.strftime('%Y-%m-%d')}_fnr_nn_scatter_shaped.png"))
        
        reference_file = os.path.join(era5_directory, f"{current_date.strftime('%Y-%m-%d')}.{selected_features[0]}.tif")
        fnr_output_file = os.path.join(output_folder, f"{current_date.strftime('%Y-%m-%d')}_fnr.tif")
        prob_output_file = os.path.join(output_folder, f"{current_date.strftime('%Y-%m-%d')}_prob.tif")
        save_probabilities_to_geotiff(fnr_reshaped, reference_file, fnr_output_file)
        save_probabilities_to_geotiff(probs_reshaped, reference_file, prob_output_file)
        
        print(f"Saved FNR predictions for {current_date.strftime('%Y-%m-%d')}")
    else:
        print(f"No data found for {current_date.strftime('%Y-%m-%d')}")
    
    current_date += timedelta(days=1)

print("Finished generating FNR predictions for all dates in the specified range.")

# Function to remove one feature at a time and evaluate performance
print("Starting Leave-One-Out Feature Importance (LOO-FI) analysis...")

# Track the AUC for the full model and leave-one-out models
model_results = []

# Get the baseline performance (AUC on the test set with all features)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

final_model.eval()
with torch.no_grad():
    test_outputs = final_model(X_test_tensor)
    test_probs = torch.sigmoid(test_outputs).numpy().flatten()
    
fpr, tpr, thresholds = roc_curve(y_test, test_probs)
baseline_auc = auc(fpr, tpr)

# Save the result for the full model
model_results.append({"Model Type": "Full Model", "AUC": baseline_auc})

# Iterate through each weather variable type (leave-one-out for each feature)
for i, weather_var in enumerate(weather_var_types):
    # Create a copy of the test set where we will remove one feature at a time
    X_test_loo = X_test.copy()
    
    # Find the indices of the columns corresponding to this feature in the flattened input
    feature_indices = [j for j, col in enumerate(weather_columns) if weather_var in col]
    
    # Set the feature's columns to zero (i.e., leave out the feature)
    X_test_loo[:, feature_indices] = 0
    
    # Convert to tensor
    X_test_loo_tensor = torch.tensor(X_test_loo, dtype=torch.float32)
    
    # Make predictions with the feature left out
    final_model.eval()
    with torch.no_grad():
        test_outputs_loo = final_model(X_test_loo_tensor)
        test_probs_loo = torch.sigmoid(test_outputs_loo).numpy().flatten()
    
    # Compute new AUC
    fpr_loo, tpr_loo, thresholds_loo = roc_curve(y_test, test_probs_loo)
    auc_loo = auc(fpr_loo, tpr_loo)
    
    # Save the result for the leave-one-out model
    model_results.append({"Model Type": f"Leave out {weather_var}", "AUC": auc_loo})

# Leave-One-Out for pairs of features
for pair in itertools.combinations(weather_var_types, 2):
    weather_var1, weather_var2 = pair
    
    # Create a copy of the test set where we will remove two features at a time
    X_test_loo = X_test.copy()
    
    # Find the indices of the columns corresponding to both features
    feature_indices_1 = [j for j, col in enumerate(weather_columns) if weather_var1 in col]
    feature_indices_2 = [j for j, col in enumerate(weather_columns) if weather_var2 in col]
    
    # Set the columns corresponding to both features to zero
    X_test_loo[:, feature_indices_1] = 0
    X_test_loo[:, feature_indices_2] = 0
    
    # Convert to tensor
    X_test_loo_tensor = torch.tensor(X_test_loo, dtype=torch.float32)
    
    # Make predictions with both features left out
    final_model.eval()
    with torch.no_grad():
        test_outputs_loo = final_model(X_test_loo_tensor)
        test_probs_loo = torch.sigmoid(test_outputs_loo).numpy().flatten()
    
    # Compute new AUC
    fpr_loo, tpr_loo, thresholds_loo = roc_curve(y_test, test_probs_loo)
    auc_loo = auc(fpr_loo, tpr_loo)
    
    # Save the result for the leave-two-out model
    model_results.append({"Model Type": f"Leave out {weather_var1} and {weather_var2}", "AUC": auc_loo})


# Convert the results to a DataFrame
df_results = pd.DataFrame(model_results)

# Save the DataFrame to CSV
df_results.to_csv(loo_results_file, index=False)

print("Leave-One-Out Feature Importance results saved to CSV.")