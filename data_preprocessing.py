import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def reduce_category_nr(data, col, cat_list):
    """
    Filters the column, leaving only selected values - that are in cat_list - 
    and replacing the rest with "Other".
    """

    data[col] = data[col].apply(lambda x: x if x in cat_list else "Other")


def encode_labels(data, col):
    """
    Performs One-Hot encoding for a selected categorical column.
    """

    data_target = data[col].values
    one_hot = preprocessing.OneHotEncoder(categories='auto').fit(data_target.reshape(-1,1))
    one_hots = one_hot.transform(data_target.reshape(-1,1)).todense()
    data[col] = one_hots


def normalize(data, num_cols):
    """
    Normalizes all numerical columns.
    """

    X = data[num_cols]
    norm = MinMaxScaler(feature_range=(0,1)).fit(X)
    transformed_data = norm.transform(X)
    data[num_cols] = transformed_data


def standarize(data, num_cols):
    """
    Standarizes all numerical columns.
    """

    X = data[num_cols]
    scale = StandardScaler().fit(X)
    transformed_data = scale.transform(X)
    data[num_cols] = transformed_data


def preprocess_data(data: pd.DataFrame, cat_lists: list[list[str]]):
    """
    Performes whole data preprocessing: reduces category number,
    One_Hot encodes categorical columns, normalizes and standarizes numerical ones.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset in the format of a pandas dataframe.

    cat_lists : List[List[str]]
        List of lists of categorical for values that will be distinguished from "Other"s
        in order: model, transmission, fuelType, Manufacturer
    ----------
    """

    global num_cols
    num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    cat_cols = ['model', 'transmission', 'fuelType', 'Manufacturer']

    for i in range(len(cat_lists)):
        reduce_category_nr(data, cat_cols[i], cat_lists[i])

    for col in cat_cols:
        encode_labels(data, col)
    
    normalize(data, num_cols)
    standarize(data, num_cols)


# Usage example.


def main() -> None:
    data = pd.read_csv("CarsData.csv")
    model_freq = data['model'].value_counts(normalize=True)
    models_over_1percent = model_freq[model_freq > 0.01].index.tolist()
    transmission_common_types = ["Manual", "Semi-Auto", "Automatic"]
    fueulType_common_types = ["Petrol", "Diesel", "Hybrid", "Electric"]
    cat_lists = [models_over_1percent, transmission_common_types, fueulType_common_types]

    preprocess_data(data, cat_lists)
    print(data.head())

if __name__ == "__main__":
    main()