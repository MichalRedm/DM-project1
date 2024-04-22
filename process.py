"""
Main script responsible for processing the data - normalization/standarization,
feature selection and extraction.
"""

import sys
import pandas as pd
from data_preprocessing import preprocess_data
from feature_extraction import feature_extraction

def main() -> None:
    data = pd.read_csv("./CarsData.csv")
    preprocess_data(data, [])
    # TODO: feature selection
    # TODO: feature extraction
    output_filename = sys.argv[1]
    data.to_csv(output_filename)

if __name__ == "__main__":
    main()
