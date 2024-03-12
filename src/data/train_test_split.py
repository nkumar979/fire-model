from sklearn.model_selection import train_test_split

def split_train_test(data, target_column, test_size=0.2, random_state=None):
    """
    Split a dataset into training and test sets using scikit-learn.

    Parameters:
    - data (pd.DataFrame): The input pandas DataFrame.
    - target_column (str): The name of the target column to be predicted.
    - test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    - random_state (int or None): Seed for random number generation to ensure reproducibility (default is None).

    Returns:
    - tuple: A tuple containing the training and test sets for features and target.

    Example:
    >>> data = pd.DataFrame({'Feature1': [1, 2, 3, 4, 5], 'Feature2': [5, 7, 8, 9, 10], 'Target': [0, 1, 0, 1, 0]})
    >>> X_train, X_test, y_train, y_test = split_train_test(data, 'Target')
    """
    # Extract features (X) and target variable (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
