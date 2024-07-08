import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
from google.colab import files
uploaded = files.upload()

# Find the uploaded file name
data = list(uploaded.keys())[0]

# Read the data file
data = pd.read_csv(data)

# Define your tolerance level for accuracy calculation
tolerance = 0.10  # 10% tolerance

# Define the features and target
X = data[['Azimuth', 'Dis_Serving', 'RSRQ']]  # Added 'RSRQ' as another feature
y = data['RSRP']


#For relation effect features on RSRP
features = ['Azimuth', 'Dis_Serving', 'RSRQ', 'RS SINR Carrier 1']
target = 'RSRP' # Also define the target column for clarity
df = data # Create a shorter alias for the DataFrame

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns # Import the seaborn module

# Set plot style
sns.set(style="whitegrid")

# Plot each feature against RSRP
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=feature, y=target, hue=target, palette='viridis', legend=None)
    plt.title(f'The Effect of {feature} on RSRP')
    plt.xlabel(feature)
    plt.ylabel('RSRP')
    plt.show()


    # Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Assuming the uploaded file is a CSV, you can read it as follows
# Replace 'your_dataset.csv' with the actual filename from the upload
Data = list(uploaded.keys())[0]
data = pd.read_csv(Data)

# Step 2: Preprocess the data
# Select the features and the target variable
features = ['RS SINR Carrier 1', 'RSRQ', 'Dis_Serving', 'Azimuth']
target = 'RSRP'

# Check if these columns exist in the dataset
if not all(column in data.columns for column in features + [target]):
    raise ValueError("Dataset does not contain all the required columns.")

X = data[features]
y = data[target]

# Handle missing values if any (this is a simple approach, you might need more complex handling)
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Step 3: Train a RandomForest model
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor()

# Train the model
model.fit(X_train, y_train)

# Step 4: Get feature importances
importances = model.feature_importances_

# Create a dataframe for easier plotting
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Sort the dataframe by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Step 5: Plot the feature importance
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance on RSRP')
plt.xticks(rotation=90)
plt.show()
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Function to plot the loss
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Mean Squared Error')
    plt.legend()
    plt.show()

# Function to calculate and print the accuracy
def calculate_accuracy(model, X_test, y_test, title):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    range_of_target = y_test.max() - y_test.min()
    accuracy = 100 * (1 - mae / range_of_target)
    print(f'{title} - MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%')

# Function to train and evaluate the model
def train_and_evaluate(data, features, target, sample_size, test_size=200, epochs=300):
    # Select the sample
    data_sample = data.sample(n=sample_size, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data_sample[features])
    y = data_sample[target].values

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = Sequential([
        Dense(2, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(4, activation='sigmoid'),
        Dense(8, activation='sigmoid'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # Output layer with linear activation for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Plot the loss
    plot_loss(history, f'Loss / Mean Squared Error with {sample_size} Samples')

    # Evaluate on test data
    X_test = scaler.transform(data.sample(test_size, random_state=42)[features])
    y_test = data.sample(test_size, random_state=42)[target].values
    calculate_accuracy(model, X_test, y_test, f'Model with {sample_size} Samples')

# Example DataFrame definition (replace this with your actual data loading)
df = pd.DataFrame({
    'RSRQ': np.random.randn(15000) * 5 - 10,
    'Dis_Serving': np.random.rand(15000) * 1000,
    'RSRP': np.random.randn(15000) * 10 - 80,
})

# Define features and target
features = ['RSRQ', 'Dis_Serving']
target = 'RSRP'

# Evaluate with 6,000 samples
train_and_evaluate(df, features, target, sample_size=6000)

# Input Layer:
# Number of Input Features: 2 (based on input_dim specified during model building)

# Hidden Layers:
# Layer 1: Dense layer with 2 neurons, ReLU activation function
# Layer 2: Dense layer with 4 neurons, Sigmoid activation function
# Layer 3: Dense layer with 8 neurons, Sigmoid activation function
# Layer 4: Dense layer with 16 neurons, ReLU activation function

# Output Layer:
# Dense layer with 1 neuron, Linear activation function (for regression)

# Model Training and Evaluation:
# Data Splitting:
# Total Samples: 6,000
# Training Set: 4,200 samples (70%)
# Validation Set: 1,200 samples (20% of the remaining after training split)
# Test Set: 600 samples (10%)

# Normalization:
# Features (RSRQ and Dis_Serving) are normalized using StandardScaler.


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define the neural network architecture
def build_model(input_dim):
    model = Sequential([
        Dense(2, activation='relu', input_shape=(input_dim,)),
        Dense(4, activation='sigmoid'),
        Dense(8, activation='sigmoid'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to plot the loss
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Mean Squared Error')
    plt.legend()
    plt.show()

# Function to calculate and print the accuracy
def calculate_accuracy(model, X_test, y_test, title):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    range_of_target = y_test.max() - y_test.min()
    accuracy = 100 * (1 - mae / range_of_target)
    print(f'{title} - MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%')

# Function to train and evaluate the model
def train_and_evaluate(data, features, target, sample_size, test_size=200, epochs=300):
    # Select the sample
    data_sample = data.sample(n=sample_size, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data_sample[features])
    y = data_sample[target].values

    # Split the data into 70% training and 30% combined validation and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Further split the combined 30% into 200 samples for testing and the rest for validation
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=len(y_temp) - test_size, random_state=42)

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Plot the loss
    plot_loss(history, f'Loss / Mean Squared Error with {sample_size} Samples')

    # Calculate accuracy and MAE
    calculate_accuracy(model, X_test, y_test, f'Model with {sample_size} Samples')

# Define features and target
features = ['RSRQ', 'Dis_Serving']
target = 'RSRP'

# Ensure df is defined for testing purposes
# Example DataFrame definition (replace this with actual data loading)
df = pd.DataFrame({
    'RSRQ': np.random.randn(15000) * 5 - 10,
    'Dis_Serving': np.random.rand(15000) * 1000,
    'RSRP': np.random.randn(15000) * 10 - 80,
})

# Evaluate with 6,000 samples
train_and_evaluate(df, features, target, sample_size=6000)


# Input Layer:
# Number of Input Features: 2 (based on input_dim specified during model building)

# Hidden Layers:
# Layer 1: Dense layer with 64 neurons, ReLU activation function
# Layer 2: Dense layer with 64 neurons, Sigmoid activation function
# Layer 3: Dense layer with 64 neurons, Sigmoid activation function
# Layer 4: Dense layer with 64 neurons, ReLU activation function

# Output Layer:
# Dense layer with 1 neuron, Linear activation function (for regression)

# Model Training and Evaluation:
# Data Splitting:
# Total Samples: 6,000
# Training Set: 4,200 samples (70%)
# Validation Set: 1,200 samples (20% of the remaining after training split)
# Test Set: 600 samples (10%)

# Normalization:
# Features (RSRQ and Dis_Serving) are normalized using StandardScaler.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define the neural network architecture
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to plot the loss
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Mean Squared Error')
    plt.legend()
    plt.show()

# Function to calculate and print the accuracy
def calculate_accuracy(model, X_test, y_test, title):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    range_of_target = y_test.max() - y_test.min()
    accuracy = 100 * (1 - mae / range_of_target)
    print(f'{title} - MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%')

# Function to train and evaluate the model
def train_and_evaluate(data, features, target, sample_size, test_size=200, epochs=300):
    # Select the sample
    data_sample = data.sample(n=sample_size, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data_sample[features])
    y = data_sample[target].values

    # Split the data into 70% training and 30% combined validation and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Further split the combined 30% into 200 samples for testing and the rest for validation
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=len(y_temp) - test_size, random_state=42)

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Plot the loss
    plot_loss(history, f'Loss / Mean Squared Error with {sample_size} Samples')

    # Calculate accuracy and MAE
    calculate_accuracy(model, X_test, y_test, f'Model with {sample_size} Samples')

# Define features and target
features = ['RSRQ', 'Dis_Serving']
target = 'RSRP'

# Ensure df is defined for testing purposes
# Example DataFrame definition (replace this with actual data loading)
df = pd.DataFrame({
    'RSRQ': np.random.randn(15000) * 5 - 10,
    'Dis_Serving': np.random.rand(15000) * 1000,
    'RSRP': np.random.randn(15000) * 10 - 80,
})

# Evaluate with 6,000 samples
train_and_evaluate(df, features, target, sample_size=6000)


# Input Layer:
# Number of Input Features: 2 (based on input_dim specified during model building)

# Hidden Layers:
# Layer 1: Dense layer with 32 neurons, Sigmoid activation function
# Layer 2: Dense layer with 32 neurons, ReLU activation function
# Layer 3: Dense layer with 32 neurons, Sigmoid activation function
# Layer 4: Dense layer with 32 neurons, ReLU activation function

# Output Layer:
# Dense layer with 1 neuron, Linear activation function (for regression)

# Model Training and Evaluation:
# Data Splitting:
# Total Samples: 4,000
# Training Set: 2,800 samples (70%)
# Validation Set: 600 samples (20% of the remaining after training split)
# Test Set: 200 samples (10%)

# Normalization:
# Features (RSRQ and Dis_Serving) are normalized using StandardScaler.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define the neural network architecture
def build_model(input_dim):
    model = Sequential([
        Dense(32, activation='sigmoid', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(32, activation='sigmoid'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to plot the loss
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Mean Squared Error')
    plt.legend()
    plt.show()

# Function to calculate and print the accuracy
def calculate_accuracy(model, X_test, y_test, title):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    range_of_target = y_test.max() - y_test.min()
    accuracy = 100 * (1 - mae / range_of_target)
    print(f'{title} - MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%')

# Function to train and evaluate the model
def train_and_evaluate(data, features, target, sample_size, test_size=200, epochs=300):
    # Select the sample
    data_sample = data.sample(n=sample_size, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data_sample[features])
    y = data_sample[target].values

    # Split the data into 70% training and 30% combined validation and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Further split the combined 30% into 200 samples for testing and the rest for validation
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=len(y_temp) - test_size, random_state=42)

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Plot the loss
    plot_loss(history, f'Loss / Mean Squared Error with {sample_size} Samples')

    # Calculate accuracy and MAE
    calculate_accuracy(model, X_test, y_test, f'Model with {sample_size} Samples')

# Define features and target
features = ['RSRQ', 'Dis_Serving']
target = 'RSRP'

# Ensure df is defined for testing purposes
# Example DataFrame definition (replace this with actual data loading)
df = pd.DataFrame({
    'RSRQ': np.random.randn(15000) * 5 - 10,
    'Dis_Serving': np.random.rand(15000) * 1000,
    'RSRP': np.random.randn(15000) * 10 - 80,
})

# Evaluate with 4,000 samples
train_and_evaluate(df, features, target, sample_size=4000)


# Input Layer:
# Number of Input Features: 2 (based on input_dim specified during model building)

# Hidden Layers:
# Layer 1: Dense layer with 2 neurons, ReLU activation function
# Layer 2: Dense layer with 32 neurons, ReLU activation function
# Layer 3: Dense layer with 16 neurons, Sigmoid activation function
# Layer 4: Dense layer with 32 neurons, ReLU activation function

# Output Layer:
# Dense layer with 1 neuron, Linear activation function (for regression)

# Model Training and Evaluation:
# Data Splitting:
# Total Samples: 8,000
# Training Set: 5,600 samples (70%)
# Validation Set: 1,200 samples (20% of the remaining after training split)
# Test Set: 200 samples (10%)

# Normalization:
# Features (RSRQ and Dis_Serving) are normalized using StandardScaler.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define the neural network architecture
def build_model(input_dim):
    model = Sequential([
        Dense(2, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='sigmoid'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to plot the loss
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Mean Squared Error')
    plt.legend()
    plt.show()

# Function to calculate and print the accuracy
def calculate_accuracy(model, X_test, y_test, title):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    range_of_target = y_test.max() - y_test.min()
    accuracy = 100 * (1 - mae / range_of_target)
    print(f'{title} - MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%')

# Function to train and evaluate the model
def train_and_evaluate(data, features, target, sample_size, test_size=200, epochs=300):
    # Select the sample
    data_sample = data.sample(n=sample_size, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data_sample[features])
    y = data_sample[target].values

    # Split the data into 70% training and 30% combined validation and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Further split the combined 30% into 200 samples for testing and the rest for validation
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=len(y_temp) - test_size, random_state=42)

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Plot the loss
    plot_loss(history, f'Loss / Mean Squared Error with {sample_size} Samples')

    # Calculate accuracy and MAE
    calculate_accuracy(model, X_test, y_test, f'Model with {sample_size} Samples')

# Define features and target
features = ['RSRQ', 'Dis_Serving']
target = 'RSRP'

# Ensure df is defined for testing purposes
# Example DataFrame definition (replace this with actual data loading)
df = pd.DataFrame({
    'RSRQ': np.random.randn(15000) * 5 - 10,
    'Dis_Serving': np.random.rand(15000) * 1000,
    'RSRP': np.random.randn(15000) * 10 - 80,
})

# Evaluate with 8,000 samples
train_and_evaluate(df, features, target, sample_size=8000)


# Input Layer:
# Number of Input Features: 2 (based on input_dim specified during model building)

# Hidden Layers:
# Layer 1: Dense layer with 64 neurons, ReLU activation function
# Layer 2: Dense layer with 64 neurons, Sigmoid activation function
# Layer 3: Dense layer with 64 neurons, Sigmoid activation function
# Layer 4: Dense layer with 64 neurons, ReLU activation function

# Output Layer:
# Dense layer with 1 neuron, Linear activation function (for regression)

# Model Training and Evaluation:
# Data Splitting:
# Total Samples: 6,000
# Training Set: 3,600 samples (60%)
# Validation Set: 2,400 samples (40% of the remaining after training split)
# Test Set: 100 samples (fixed size)

# Normalization:
# Features (RSRQ and Dis_Serving) are normalized using StandardScaler.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define the neural network architecture
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to plot the loss
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Mean Squared Error')
    plt.legend()
    plt.show()

# Function to calculate and print the accuracy
def calculate_accuracy(model, X_test, y_test, title):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    range_of_target = y_test.max() - y_test.min()
    accuracy = 100 * (1 - mae / range_of_target)
    print(f'{title} - MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%')

# Function to train and evaluate the model
def train_and_evaluate(data, features, target, sample_size, test_samples=100, epochs=300):
    # Select the sample
    data_sample = data.sample(n=sample_size, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data_sample[features])
    y = data_sample[target].values

    # Split the data into 60% training and 40% combined validation and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # Further split the combined 40% into 100 samples for testing and the rest for validation
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=(len(y_temp) - test_samples) / len(y_temp), random_state=42)

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Plot the loss
    plot_loss(history, f'Loss / Mean Squared Error with {sample_size} Samples')

    # Calculate accuracy and MAE
    calculate_accuracy(model, X_test, y_test, f'Model with {sample_size} Samples')

# Define features and target
features = ['RSRQ', 'Dis_Serving']
target = 'RSRP'

# Ensure df is defined for testing purposes
# Example DataFrame definition (replace this with actual data loading)
df = pd.DataFrame({
    'RSRQ': np.random.randn(15000) * 5 - 10,
    'Dis_Serving': np.random.rand(15000) * 1000,
    'RSRP': np.random.randn(15000) * 10 - 80,
})

# Evaluate with 6,000 samples
train_and_evaluate(df, features, target, sample_size=6000)



# Input Layer:
# Number of Input Features: 2 (based on input_dim specified during model building)

# Hidden Layers:
# Layer 1: Dense layer with 64 neurons, ReLU activation function
# Layer 2: Dense layer with 64 neurons, Sigmoid activation function
# Layer 3: Dense layer with 64 neurons, Sigmoid activation function
# Layer 4: Dense layer with 64 neurons, ReLU activation function

# Output Layer:
# Dense layer with 1 neuron, Linear activation function (for regression)

# Model Training and Evaluation:
# Data Splitting:
# Total Samples: 6,000
# Training Set: 3,600 samples (60%)
# Validation Set: 2,400 samples (40% of the remaining after training split)
# Test Set: 900 samples (fixed size)

# Normalization:
# Features (RSRQ and Dis_Serving) are normalized using StandardScaler.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define the neural network architecture
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to plot the loss
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Mean Squared Error')
    plt.legend()
    plt.show()

# Function to calculate and print the accuracy
def calculate_accuracy(model, X_test, y_test, title):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    range_of_target = y_test.max() - y_test.min()
    accuracy = 100 * (1 - mae / range_of_target)
    print(f'{title} - MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%')

# Function to train and evaluate the model
def train_and_evaluate(data, features, target, sample_size, test_samples=900, epochs=300):
    # Select the sample
    data_sample = data.sample(n=sample_size, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data_sample[features])
    y = data_sample[target].values

    # Split the data into 60% training and 40% combined validation and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # Further split the combined 40% into 900 samples for testing and the rest for validation
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=(len(y_temp) - test_samples) / len(y_temp), random_state=42)

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Plot the loss
    plot_loss(history, f'Loss / Mean Squared Error with {sample_size} Samples')

    # Calculate accuracy and MAE
    calculate_accuracy(model, X_test, y_test, f'Model with {sample_size} Samples')

# Define features and target
features = ['RSRQ', 'Dis_Serving']
target = 'RSRP'

# Ensure df is defined for testing purposes
df = pd.DataFrame({
    'RSRQ': np.random.randn(15000) * 5 - 10,
    'Dis_Serving': np.random.rand(15000) * 1000,
    'RSRP': np.random.randn(15000) * 10 - 80,
})

# Evaluate with 6,000 samples
train_and_evaluate(df, features, target, sample_size=6000)

# Input Layer:
# Number of Input Features: 2 (based on input_dim specified during model building)

# Hidden Layers:
# Layer 1: Dense layer with 2 neurons, ReLU activation function
# Layer 2: Dense layer with 4 neurons, ReLU activation function
# Layer 3: Dense layer with 6 neurons, ReLU activation function
# Layer 4: Dense layer with 6 neurons, ReLU activation function
# Layer 5: Dense layer with 6 neurons, ReLU activation function

# Output Layer:
# Dense layer with 1 neuron, Linear activation function (for regression)

# Model Training and Evaluation:
# Data Splitting:
# Total Samples: 15,000
# Training Set: 6,000 samples (40%)
# Test Set: 9,000 samples (60%)
# Features (RSRQ and Dis_Serving) are normalized using StandardScaler.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define the neural network architecture
def build_model(input_dim):
    model = Sequential([
        Dense(2, activation='relu', input_shape=(input_dim,)),
        Dense(4, activation='relu'),
        Dense(6, activation='relu'),
        Dense(6, activation='relu'),
        Dense(6, activation='relu'),
        Dense(1,  activation='linear'),  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to plot the loss
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Mean Squared Error')
    plt.legend()
    plt.show()

# Function to calculate and print the accuracy
def calculate_accuracy(model, X_test, y_test, title):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    range_of_target = y_test.max() - y_test.min()
    accuracy = 100 * (1 - mae / range_of_target)
    print(f'{title} - MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%')


# Function to train and evaluate the model
def train_and_evaluate(data, features, target, sample_size, test_size=0.3, epochs=300):
    # Select the sample
    data_sample = data.sample(n=sample_size, random_state=42)




    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data_sample[features])
    y = data_sample[target].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Plot the loss
    plot_loss(history, f'Loss / Mean Squared Error with {sample_size} Samples')

    # Calculate accuracy and MAE
    calculate_accuracy(model, X_test, y_test, f'Model with {sample_size} Samples')



# Define features and target
features = ['RSRQ', 'Dis_Serving']
target = 'RSRP'

# Ensure df is defined for testing purposes
df = pd.DataFrame({

    'RSRQ': np.random.randn(15000) * 5 - 10,
    'Dis_Serving': np.random.rand(15000) * 1000,
    'RSRP': np.random.randn(15000) * 10 - 80,
    'Azimuth': np.random.rand(15000) * 360,
    'RS SINR Carrier 1': np.random.randn(15000),
})

# Evaluate with 6,000 samples
train_and_evaluate(df, features, target, sample_size=6000)



# Input Layer:
# Number of Input Features: 4 (based on input_dim specified during model building)

# Hidden Layers:
# Layer 1: Dense layer with 2 neurons, ReLU activation function
# Layer 2: Dense layer with 16 neurons, Sigmoid activation function
# Layer 3: Dense layer with 16 neurons, Sigmoid activation function
# Layer 4: Dense layer with 32 neurons, ReLU activation function

# Output Layer:
# Dense layer with 1 neuron, Linear activation function (for regression)

# Model Training and Evaluation:
# Data Splitting:
# Total Samples: 15,000
# Training Set: 14,000 samples (70%)
# Test Set: 1,000 samples (30%)
# Features (RSRQ, Dis_Serving, Azimuth, RS SINR Carrier 1) are normalized using StandardScaler.


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define the neural network architecture
def build_model(input_dim):
    model = Sequential([
        Dense(2, activation='relu', input_shape=(input_dim,)),
        Dense(16, activation='sigmoid'),
        Dense(16, activation='sigmoid'),
        Dense(32, activation='relu'),
        Dense(1,  activation='linear'),  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to plot the loss
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Mean Squared Error')
    plt.legend()
    plt.show()

# Function to calculate and print the accuracy
def calculate_accuracy(model, X_test, y_test, title):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    range_of_target = y_test.max() - y_test.min()
    accuracy = 100 * (1 - mae / range_of_target)
    print(f'{title} - MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%')


# Function to train and evaluate the model
def train_and_evaluate(data, features, target, sample_size, test_size=0.3, epochs=300):
    # Select the sample
    data_sample = data.sample(n=sample_size, random_state=42)




    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data_sample[features])
    y = data_sample[target].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Plot the loss
    plot_loss(history, f'Loss / Mean Squared Error with {sample_size} Samples')

    # Calculate accuracy and MAE
    calculate_accuracy(model, X_test, y_test, f'Model with {sample_size} Samples')



# Define features and target
features = ['RSRQ', 'Dis_Serving', 'Azimuth', 'RS SINR Carrier 1']
target = 'RSRP'

# Ensure df is defined for testing purposes
df = pd.DataFrame({

    'RSRQ': np.random.randn(15000) * 5 - 10,
    'Dis_Serving': np.random.rand(15000) * 1000,
    'RSRP': np.random.randn(15000) * 10 - 80,
    'Azimuth': np.random.rand(15000) * 360,
    'RS SINR Carrier 1': np.random.randn(15000),
})

# Evaluate with 14,000 samples
train_and_evaluate(df, features, target, sample_size=14000)


# Neural Network Model Summary:
# -----------------------------

# Input Layer:
# ------------
# - Number of neurons: 2
# - Activation function: ReLU
# - Input shape: Determined by the input dimension of the data

# Hidden Layers:
# --------------
# 1. Dense Layer 1:
#    - Number of neurons: 16
#    - Activation function: Sigmoid

# 2. Dense Layer 2:
#    - Number of neurons: 16
#    - Activation function: Sigmoid

# 3. Dense Layer 3:
#    - Number of neurons: 32
#    - Activation function: ReLU

# Output Layer:
# -------------
# - Number of neurons: 1
# - Activation function: Linear (for regression)

# Compilation:
# ------------
# - Optimizer: Adam
# - Loss function: Mean Squared Error (MSE)
# - Metrics: Mean Absolute Error (MAE)


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define the neural network architecture
def build_model(input_dim):
    model = Sequential([
        Dense(2, activation='relu', input_shape=(input_dim,)),
        Dense(16, activation='sigmoid'),
        Dense(16, activation='sigmoid'),
        Dense(32, activation='relu'),
        Dense(1,  activation='linear'),  # Output layer with linear activation for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to plot the loss
def plot_loss(history, title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Mean Squared Error')
    plt.legend()
    plt.show()

# Function to calculate and print the accuracy
def calculate_accuracy(model, X_test, y_test, title):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    range_of_target = y_test.max() - y_test.min()
    accuracy = 100 * (1 - mae / range_of_target)
    print(f'{title} - MAE: {mae:.4f}, Accuracy: {accuracy:.2f}%')


# Function to train and evaluate the model
def train_and_evaluate(data, features, target, sample_size, test_size=0.3, epochs=300):
    # Select the sample
    data_sample = data.sample(n=sample_size, random_state=42)




    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(data_sample[features])
    y = data_sample[target].values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Plot the loss
    plot_loss(history, f'Loss / Mean Squared Error with {sample_size} Samples')

    # Calculate accuracy and MAE
    calculate_accuracy(model, X_test, y_test, f'Model with {sample_size} Samples')



# Define features and target
features = ['RSRQ', 'Dis_Serving']
target = 'RSRP'

# Ensure df is defined for testing purposes
df = pd.DataFrame({

    'RSRQ': np.random.randn(15000) * 5 - 10,
    'Dis_Serving': np.random.rand(15000) * 1000,
    'RSRP': np.random.randn(15000) * 10 - 80,
    'Azimuth': np.random.rand(15000) * 360,
    'RS SINR Carrier 1': np.random.randn(15000),
})

# Evaluate with 6,000 samples
train_and_evaluate(df, features, target, sample_size=6000)

# Evaluate with 9,000 samples
train_and_evaluate(df, features, target, sample_size=8000)


