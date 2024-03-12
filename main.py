
from src.data import load_data, clean_data, train_test_split
from src.features import log_transform, encoding
from src.fire_prediction_model import predict_fire, evaluate_model

def main():
    raw_data = load_data.read_csv("/Users/nirusenthilkumar/Documents/fire-model/fire-model/data/forestfires.csv")

    cleaned_data = clean_data.handle_missing_data(raw_data)

    features = encoding.encode_date_columns(cleaned_data, 'month', 'day')

    X_train, X_test, y_train, y_test = train_test_split.split_train_test(features, 'area', random_state=1)

    model, _ = predict_fire.train_bagged_random_forest(X_train, y_train)

    predictions = evaluate_model.eval_predictions(model, X_test, y_test)

if __name__ == "__main__":
    main()

