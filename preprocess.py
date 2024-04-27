"""
Main file responsible for preprocessing the data. Contains function
'preprocess' that transforms the dataset in the form of pandas
DataFrame. When the script is ran from the console, it reads the
dataset from a CSV file and writes the preprocessed dataset
to another file.
"""


import pandas as pd
from typing import Literal
from pipeline import get_full_pipeline


INPUT_DATASET = "./CarsData.csv"
OUTPUT_DATASET = "./CarsDataProcessed.csv"
    

def preprocess(
        df: pd.DataFrame,
        target: str,
        variance_treshold: float = 0.01,
        feature_selection_treshold: float = 200.0,
        feature_extraction_method: Literal["PCA", "LDA"] = "PCA",
        n_components: int | float = 30,
        verbose: bool = False
    ) -> pd.DataFrame:
    """
    Performs preprocessing on the data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset on which the preprocessing shall be performed, in the form of
        a pandas dataframe.
    
    target : str
        Name of the target column (for which the machine learning algorithm
        should predict the value).
    
    variance_treshold : float
        Treshold such that features with variance below it will be removed.
    
    feature_selection_treshold : float
        Treshold used for feature selection.

    feature_extraction_method : Literal["PCA", "LDA"]
        Feature extraction algorithm that should be employed. The following ones
        are supported:
        - PCA (Principal Component Analysis),
        - LDA (Linear Discriminant Analysis).

    n_components : float | int
        Number of components to keep, if it is a positive integer, or variance to be
        kept if it is a float between 0 and 1.

    verbose : bool
        If set to true, the function will print information about the progress
        of data preprocessing.

    Returns
    -------
    Pandas dataframe containing the dataset after preprocessing and the
    target column as the last column.
    """

    # Validate the function parameters.
    assert isinstance(df, pd.DataFrame), "Parameter 'df' must be a pandas DataFrame."
    assert target in df.columns, "Parameter 'target' must be name of a column in DataFrame 'df'."
    assert isinstance(variance_treshold, float) and variance_treshold > 0, "Parameter 'variance_treshold' must be a positive float."
    assert isinstance(feature_selection_treshold, float) and feature_selection_treshold > 0, "Parameter 'feature_selection_treshold' must be a positive float."
    assert feature_extraction_method in ("PCA", "LDA"), "Parameter 'feature_extraction_method' must be 'PCA' or 'LDA'."
    assert isinstance(verbose, bool), "Parameter 'verbose' must be a boolean."
    assert set(num_cols + cat_cols + [target]) == set(df.columns), "Invalid dataset provided."

    full_pipeline = get_full_pipeline(df, target, variance_treshold, feature_selection_treshold,
                                      feature_extraction_method, n_components, verbose)

    # Extract features and target.
    X, y = df.drop(target, axis=1), df[target].to_numpy()

    # Apply the pipeline to preprocess the dataset.
    X_new_np = full_pipeline.fit_transform(X, y)

    # Prepare and return the final DataFrame.
    df = pd.DataFrame(X_new_np, columns=[f"componenet{i}" for i in range(X_new_np.shape[1])])
    df.insert(len(df.columns), target, y)
    return df


def main() -> None:
    df = pd.read_csv(INPUT_DATASET)
    df = preprocess(df, "price", verbose=True)
    df.to_csv(OUTPUT_DATASET, index=False)


if __name__ == "__main__":
    main()
