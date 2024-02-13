import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

sequence_length = None
normalisation = None
dataframe_2 = pd.read_csv('./data/creditcard.csv')

def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)
        return np.array(sequences)


def data_preparation(sequence_length = sequence_length, normalization = normalisation):

    x = dataframe_2.drop('Class', axis=1)

    x_anomalies = x[dataframe_2['Class'] == 1]

    X = x[dataframe_2['Class'] == 0]

    # Starting date
    start_date = pd.to_datetime('1970-01-01')
    offset = pd.DateOffset(years=43, months=8)
    new_date = start_date + offset
    seconds_data =  x['Time']
    new_dates = [new_date + timedelta(seconds=seconds) for seconds in seconds_data]
    dataframe_2['Time_in_years'] = new_dates

    features_for_pca = x.drop('Time', axis=1)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(features_for_pca)

    plt.figure(figsize=(10, 5))
    plt.scatter(new_dates, reduced_data[:, 1], c=np.where(dataframe_2['Class'] == 1, 'red', 'blue'), marker='o', alpha=0.5)
    plt.title('PCA: 2D Plot of Time against Principal Component 2 with Anomalies Highlighted')
    plt.xlabel('Time')
    plt.ylabel('Component 2')
    plt.show()

    #sequence_length = sequence_length
    X_sequences = create_sequences(X.values, sequence_length)
    x_anomalies = create_sequences(x_anomalies.values, sequence_length)

    X_train, X_test = train_test_split(X_sequences, test_size=0.2, random_state=42)

    if normalization == 1:
        print("Normalizing using StandardScaler")
        scaler = StandardScaler()
    elif normalization == 2:
        print("Normalizing using MinMaxScaler")
        scaler = MinMaxScaler()
    else:
        print("No normalization applied")
        scaler = None

    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        x_anomalies_scaled = scaler.transform(x_anomalies.reshape(-1, x_anomalies.shape[-1])).reshape(x_anomalies.shape)
    else:
        X_train_scaled, X_test_scaled, x_anomalies_scaled = X_train, X_test, x_anomalies

    input_shape = (sequence_length, X_train.shape[-1])
    
    return X_train_scaled, X_test_scaled, x_anomalies_scaled, input_shape

def extract_time_interval_df(start_time, end_time, sequence_length=sequence_length, normalisation=normalisation):
    bool_start = (dataframe_2['Time_in_years'] >= start_time)
    bool_end = (dataframe_2['Time_in_years'] <= end_time)
    bool_mask = bool_start & bool_end
    
    # Get the indices where the mask is True
    indices = dataframe_2.index[bool_mask].tolist()

    # Get the first and last indices
    start_index = min(indices)

    end_index = max(indices)

    # Extract the data between the start and end indices
    interval_data = dataframe_2.loc[start_index:end_index].copy()
    interval_labels_data = interval_data["Class"].copy()
    columns_to_drop = ['Class', 'Time_in_years']

    for column in columns_to_drop:
        if column in interval_data.columns:
            interval_data.drop(column, axis=1, inplace=True)
        else:
            print(f"Column '{column}' does not exist.")

    # Check if there are enough data points for sequences
    if len(interval_data) < sequence_length:
        print("Not enough data points for sequences. Skipping normalization.")
        return None, None
    
    if normalisation == 1:
       print('Normalisation 1') 
       scaler = StandardScaler()
       interval_data = create_sequences(interval_data.values, sequence_length)
       interval_data = scaler.fit_transform(interval_data.reshape(-1, interval_data.shape[-1])).reshape(interval_data.shape)
       #interval_data = scaler.transform(interval_data.reshape(-1, interval_data.shape[-1])).reshape(interval_data.shape)
       interval_labels_data = create_sequences(interval_labels_data.values, sequence_length)

    if normalisation == 2:
        print('Normalisation 1')
        scaler = MinMaxScaler()
        interval_data = create_sequences(interval_data.values, sequence_length)
        interval_data = scaler.fit_transform(interval_data.reshape(-1, interval_data.shape[-1])).reshape(interval_data.shape)
        interval_labels_data = create_sequences(interval_labels_data.values, sequence_length)
    return interval_data, interval_labels_data
