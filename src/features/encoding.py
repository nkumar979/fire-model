import pandas as pd

def encode_date_columns(data, month_column, day_of_week_column):
    """
    Numerically encode the month and day of the week columns in a pandas DataFrame.

    Parameters:
    - data (pd.DataFrame): The input pandas DataFrame.
    - month_column (str): The name of the column containing months (e.g., 'Month').
    - day_of_week_column (str): The name of the column containing days of the week (e.g., 'DayOfWeek').

    Returns:
    - pd.DataFrame: The DataFrame with numerically encoded month and day of the week columns.

    Example:
    >>> data = pd.DataFrame({'Month': ['January', 'February', 'March', 'April'],
    ...                      'DayOfWeek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday']})
    >>> encoded_data = encode_date_columns(data, 'Month', 'DayOfWeek')
    >>> print(encoded_data)
       Month  DayOfWeek
    0      1          1
    1      2          2
    2      3          3
    3      4          4
    """
    # Define mapping dictionaries for month and day of the week
    month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

    day_of_week_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5,
                           'Saturday': 6, 'Sunday': 7}

    # Apply mapping to the specified columns
    data[month_column] = data[month_column].map(month_mapping)
    data[day_of_week_column] = data[day_of_week_column].map(day_of_week_mapping)

    return data