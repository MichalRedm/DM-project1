"""
Main script responsible for processing the data - normalization/standarization,
feature selection and extraction.
"""

import sys
import pandas as pd
from data_preprocessing import preprocess_data
from feature_extraction import feature_extraction

def main() -> None:
    arg1 = sys.argv[1]
    if arg1 in ("--help", "-h"):
        print("Script designed for the purpose of preprocessing the Cars Dataset.")
        print("Usage: python process.py [OUTPUT FILE NAME]")
        return
    data = pd.read_csv("./CarsData.csv")
    target = "price"
    model_freq = data["model"].value_counts(normalize=True)
    models_over_1percent = model_freq[model_freq > 0.01].index.tolist()
    transmission_common_types = ["Manual", "Semi-Auto", "Automatic"]
    fueulType_common_types = ["Petrol", "Diesel", "Hybrid", "Electric"]
    cat_lists = [models_over_1percent, transmission_common_types, fueulType_common_types]
    data = preprocess_data(data, cat_lists, target)
    # TODO: data preprocessing seems questionable, as some columns (Manufaturer, fuelType, model)
    # seem to be reduced to a single value before one-hot encoding. Fix that issue.
    # TODO: feature selection
    data = feature_extraction(data, target, 5, "PCA", [])
    data.to_csv(arg1, index=False)

if __name__ == "__main__":
    main()
