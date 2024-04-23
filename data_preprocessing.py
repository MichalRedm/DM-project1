import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Any


def reduce_category_nr(data: pd.DataFrame, col: str, cat_list: List[Any]) -> None:
    """
    Filters the column, leaving only selected values - that are in cat_list - 
    and replacing the rest with "Other".

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe containing data for which the filtering shall
        be performed
    
    col : str
        Name of the column that should be filtered.
    
    cat_list : List[Any]
        List of values that shall be preserved after the reduction process.
    """

    data[col] = data[col].apply(lambda x: x if x in cat_list else "Other")


# def encode_labels(data: pd.DataFrame, col: str) -> None:
#     """
#     Performs One-Hot encoding for a selected categorical column.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         Pandas dataframe containing data for which the operation shall
#         be performed.
    
#     col : str
#         Name of the column that should be one-hot encoded.
#     """

#     data_target = data[col].values
#     one_hot = preprocessing.OneHotEncoder(categories='auto').fit(data_target.reshape(-1,1))
#     one_hots = one_hot.transform(data_target.reshape(-1,1)).todense()
#     data[col] = one_hots


def normalize(data: pd.DataFrame, num_cols: List[str]) -> None:
    """
    Normalizes all numerical columns.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe containing data for which the normalization shall
        be performed.
    
    num_cols : List[str]
        List of names of numerical columns for that should be normalized.
    """

    X = data[num_cols]
    norm = MinMaxScaler(feature_range=(0,1)).fit(X)
    transformed_data = norm.transform(X)
    data[num_cols] = transformed_data


def standarize(data: pd.DataFrame, num_cols: List[str]) -> None:
    """
    Standarizes all numerical columns.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe containing data for which the standarization shall
        be performed.
    
    num_cols : List[str]
        List of names of numerical columns for that should be standarized.
    """

    X = data[num_cols]
    scale = StandardScaler().fit(X)
    transformed_data = scale.transform(X)
    data[num_cols] = transformed_data


def preprocess_data(data: pd.DataFrame, cat_lists: List[List[str]], target: str) -> pd.DataFrame:
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
    
    target : str
        Name of the target column for which the value shall be predicted by
        the machine learning algorithm later on.
    """

    data = data.copy()

    num_cols = list(data.select_dtypes([np.number]).columns)
    cat_cols = np.setdiff1d(data.columns, num_cols)
    num_cols.remove(target) # Ignore the target variable.

    for i in range(len(cat_lists)):
        reduce_category_nr(data, cat_cols[i], cat_lists[i])

    data = pd.get_dummies(data, columns=cat_cols, dtype=np.float32) # One-hot encoding.

    # for col in cat_cols:
    #     encode_labels(data, col)
    
    normalize(data, num_cols)
    standarize(data, num_cols)

    return data


# Usage example.


def main() -> None:
    data = pd.read_csv("CarsData.csv")
    model_freq = data["model"].value_counts(normalize=True)
    models_over_1percent = model_freq[model_freq > 0.01].index.tolist()
    transmission_common_types = ["Manual", "Semi-Auto", "Automatic"]
    fueulType_common_types = ["Petrol", "Diesel", "Hybrid", "Electric"]
    cat_lists = [models_over_1percent, transmission_common_types, fueulType_common_types]

    data = preprocess_data(data, cat_lists)
    print(data.head())

if __name__ == "__main__":
    main()
