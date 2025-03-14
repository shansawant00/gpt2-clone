def load_data(file_path):
    import numpy as np
    import pandas as pd

    # Load data from a CSV file
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Example preprocessing: remove null values and normalize
    data = data.dropna()
    normalized_data = (data - data.mean()) / data.std()
    return normalized_data

def split_data(data, train_size=0.8):
    # Split data into training and testing sets
    train_size = int(len(data) * train_size)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data