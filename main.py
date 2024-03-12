
from src.data import load_data, clean_data
from src.features import log_transform
from src.fire_prediction_model import predict_fire

def main():
    raw_data = load_data.read_csv("/Users/nirusenthilkumar/Documents/fire-model/fire-model/data/forestfires.csv")

    cleaned_data = clean_data.handle_missing_data(raw_data)

    features = log_transform.log_transform_column(cleaned_data, 'area')

    predict_fire.xgb_model()

if __name__ == "__main__":
    main()

