import pandas as pd

def read_csv(file_path):
    """
    Read data from a CSV file using pandas.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the data.

    Raises:
    - FileNotFoundError: If the specified file_path does not exist.


    """
    try:
        # Use pandas to read the CSV file into a DataFrame
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError as e:
        # Raise an error if the file is not found
        raise FileNotFoundError(f"Error: File not found at {file_path}") from e


