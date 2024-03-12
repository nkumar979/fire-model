import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_bagged_random_forest(features, target, perform_hyperparameter_tuning=False, save_tuned_hyperparameters=False):
    """
    Train a bagged Random Forest Regressor model with or without hyperparameter tuning.

    Parameters:
    - features (pd.DataFrame): The DataFrame containing the features.
    - target (pd.Series): The target variable.
    - perform_hyperparameter_tuning (bool): Whether to perform hyperparameter tuning (default is False).
    - save_tuned_hyperparameters (bool): Whether to save the tuned hyperparameters (default is False).

    Returns:
    - BaggingRegressor: The trained bagged Random Forest Regressor model.
    - dict or None: The tuned hyperparameters if hyperparameter tuning is performed, otherwise None.

    Example:
    >>> features = pd.DataFrame({'Feature1': [1, 2, 3, 4, 5], 'Feature2': [5, 7, 8, 9, 10]})
    >>> target = pd.Series([10, 20, 30, 40, 50])
    >>> model, tuned_hyperparameters = train_bagged_random_forest(features, target, perform_hyperparameter_tuning=True, save_tuned_hyperparameters=True)
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Define the base Random Forest Regressor
    base_regressor = RandomForestRegressor(random_state=42)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    if perform_hyperparameter_tuning:
        # Perform randomized grid search
        randomized_search = RandomizedSearchCV(base_regressor, param_distributions=param_grid, n_iter=10, cv=5, random_state=42, scoring='neg_mean_squared_error')
        randomized_search.fit(X_train, y_train)

        # Print the best hyperparameters
        best_hyperparameters = randomized_search.best_params_
        print("Best Hyperparameters:", best_hyperparameters)

        # Save tuned hyperparameters if specified
        if save_tuned_hyperparameters:
            with open("tuned_hyperparameters.txt", "w") as file:
                file.write(str(best_hyperparameters))
    else:
        best_hyperparameters = None

    # Create a bagged Random Forest Regressor with the best hyperparameters if available
    bagged_model = BaggingRegressor(base_regressor.set_params(**best_hyperparameters) if best_hyperparameters else base_regressor,
                                    n_estimators=10, random_state=42)

    # Train the model on the training set
    bagged_model.fit(X_train, y_train)

    return bagged_model, best_hyperparameters



