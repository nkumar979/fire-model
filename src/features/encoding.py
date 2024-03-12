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
    >>> data = pd.DataFrame({'Month': ['jan', 'feb', 'mar', 'apr'],
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
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

    day_of_week_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5,
                           'sat': 6, 'sun': 7}

    # Apply mapping to the specified columns
    data[month_column] = data[month_column].map(month_mapping)
    data[day_of_week_column] = data[day_of_week_column].map(day_of_week_mapping)

    return data