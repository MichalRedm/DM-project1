"""
Main file responsible for preprocessing the data. Contains function
'preprocess' that transforms the dataset in the form of pandas
DataFrame. When the script is ran from the console, it reads the
dataset from a CSV file and writes the preprocessed dataset
to another file.
"""


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import Perceptron, LassoCV, LogisticRegression, LinearRegression
from typing import Literal


INPUT_DATASET = "./CarsData.csv"
OUTPUT_DATASET = "./CarsDataProcessed.csv"
    

def preprocess(
        df: pd.DataFrame,
        target: str,
        variance_treshold: float = 0.01,
        feature_selection_treshold: float = 200.0,
        feature_extraction_method: Literal["PCA", "LDA"] = "PCA",
        verbose: bool = True
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

    # Extract features and target.
    X, y = df.drop(target, axis=1), df[target].to_numpy()

    # Determine which columns are numerical and which are categorical.
    num_cols = list(X.select_dtypes([np.number]).columns)
    cat_cols = list(np.setdiff1d(X.columns, num_cols))

    # Pipeline to be applied to numerical columns.
    num_pipeline = Pipeline([
        ('Variance_threshold', VarianceThreshold(threshold=variance_treshold)),
        ('std_scaler', StandardScaler())
    ])

    # Pipeline to be applied to categorical columns.
    cat_pipeline = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore')),
        ('Variance_threshold', VarianceThreshold(threshold=variance_treshold)),
    ])

    col_transform = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    full_pipeline = Pipeline([
        ('transform', col_transform),
        ('dense', FunctionTransformer(lambda x: np.array(x.todense()), accept_sparse=True)),
        ('select_from_model', SelectFromModel(LassoCV(), threshold=feature_selection_treshold)), # Feature selection.
        (('pca', PCA()) if feature_extraction_method == "PCA" else ("lda", LDA())) # Feature extraction.
    ], verbose=verbose)

    # Apply the pipeline to preprocess the dataset.
    X_new_np = full_pipeline.fit_transform(X, y)

    # Prepare and return the final DataFrame.
    df = pd.DataFrame(X_new_np, columns=[f"componenet{i}" for i in range(X_new_np.shape[1])])
    df.insert(len(df.columns), target, y)
    return df


def main() -> None:
    df = pd.read_csv(INPUT_DATASET)
    df = preprocess(df, "price")
    df.to_csv(OUTPUT_DATASET, index=False)


if __name__ == "__main__":
    main()
