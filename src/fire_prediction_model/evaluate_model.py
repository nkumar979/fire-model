from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score

def eval_predictions(trained_model, X_test, y_test):
    """
    Evaluate a trained Bagged Random Forest Regressor model on a test set.

    Parameters:
    - trained_model (BaggingRegressor): Trained Bagged Random Forest Regressor model.
    - X_test (pd.DataFrame): DataFrame containing features of the test set.
    - y_test (pd.Series): Series containing target values of the test set.

    Returns:
    - None

    Example:
    >>> evaluate_model(trained_model, X_test, y_test)
    """

    # Make predictions on the test set
    predictions = trained_model.predict(X_test)

    # Calculate RMSE and MAPE
    rmse = mean_squared_error(y_test, predictions, squared=False)
    # mape = mean_absolute_percentage_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse:.2f}")
    # print(f"MAPE: {mape:.2f}")
    print(f"Model Score (R²): {r2:.2f}")

    return predictions