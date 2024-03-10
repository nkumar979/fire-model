import pandas as pd
import numpy as np

class DataPreprocessing:
    def __init__(self):
        self.data = None

    def read_csv(self, file_path):
        """
        Read a dataset from a CSV file.

        Parameters:
        - file_path (str): Path to the CSV file.

        Returns:
        - pd.DataFrame: The loaded dataset.
        """
        try:
            self.data = pd.read_csv(file_path)
            return self.data
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None

    def remove_outliers(self, columns=None, z_threshold=3):
        """
        Remove outliers from the dataset using Z-score.

        Parameters:
        - columns (list, optional): List of columns to consider. If None, all numeric columns will be considered.
        - z_threshold (float, optional): Z-score threshold for identifying outliers. Defaults to 3.

        Returns:
        - pd.DataFrame: Dataset with outliers removed.
        """
        if self.data is None:
            print("Error: No dataset loaded. Use read_csv() first.")
            return None

        if columns is None:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        else:
            numeric_columns = columns

        z_scores = np.abs((self.data[numeric_columns] - self.data[numeric_columns].mean()) / self.data[numeric_columns].std())
        mask = (z_scores < z_threshold).all(axis=1)

        self.data = self.data[mask]
        return self.data

# Example Usage:
# Initialize the DataPreprocessing class
data_processor = DataPreprocessing()

# Read a dataset from a CSV file
dataset = data_processor.read_csv("your_dataset.csv")

# Remove outliers from the dataset
dataset_no_outliers = data_processor.remove_outliers()

# Print the cleaned dataset
print(dataset_no_outliers)