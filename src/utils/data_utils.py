def load_data(file_path):
    """
    Load data from a specified file path.
    
    Args:
        file_path (str): The path to the data file.
        
    Returns:
        data: The loaded data.
    """
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def preprocess_data(raw_data):
    """
    Preprocess the raw data for the GPT-2 model.
    
    Args:
        raw_data (str): The raw data to preprocess.
        
    Returns:
        processed_data: The preprocessed data.
    """
    # Example preprocessing steps (to be customized)
    processed_data = raw_data.lower()  # Convert to lowercase
    # Add more preprocessing steps as needed
    return processed_data

def save_data(data, file_path):
    """
    Save processed data to a specified file path.
    
    Args:
        data: The data to save.
        file_path (str): The path to save the data file.
    """
    with open(file_path, 'w') as file:
        file.write(data)