import pandas as pd
import numpy as np

def log_transform_column(data, column_name):
    """
    Perform a log transformation for a specific column in a pandas DataFrame.

    Parameters:
    - data (pd.DataFrame): The input pandas DataFrame.
    - column_name (str): The name of the column to be log-transformed.

    Returns:
    - pd.DataFrame: The DataFrame with the specified column log-transformed.

    Example:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 7, 8, 9, 10]})
    >>> transformed_data = log_transform_column(data, 'B')
    >>> print(transformed_data)
       A         B
    0  1  1.609438
    1  2  1.945910
    2  3  2.079442
    3  4  2.197225
    4  5  2.302585
    """
    # Check if the specified column exists in the DataFrame
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Perform log transformation for the specified column
    data[column_name] = np.log(data[column_name])

    return data
