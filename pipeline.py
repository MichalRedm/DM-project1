"""
File defining pipeline for our data preprocessing
algorithm.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from dataset_info import get_dataset_info
from typing import Literal


class FeatureExtraction(BaseEstimator, TransformerMixin):
    """
    Class used for feature extraction with various algorithms.

    Parameters
    ----------
    n_components : float | int
        Number of components to keep, if it is a positive integer, or variance to be
        kept if it is a float between 0 and 1.
    
    method : Literal["PCA", "LDA"]
        Feature extraction algorithm that should be employed. The following ones
        are supported:
        - PCA (Principal Component Analysis),
        - LDA (Linear Discriminant Analysis).
    """
    def __init__(self, n_components: float | int, method: Literal["PCA", "LDA"]) -> None:
        self.set_params(n_components, method)

    def set_params(self, n_components: float | int, method: Literal["PCA", "LDA"]) -> None:
        """
        Sets the parameters for the feature extraction algorithm.

        Parameters
        ----------
        n_components : float | int
            Number of components to keep, if it is a positive integer, or variance to be
            kept if it is a float between 0 and 1.
        
        method : Literal["PCA", "LDA"]
            Feature extraction algorithm that should be employed. The following ones
            are supported:
            - PCA (Principal Component Analysis),
            - LDA (Linear Discriminant Analysis).
        """
        assert (isinstance(n_components, float) and 0 < n_components < 1) or \
               (isinstance(n_components, int) and n_components > 0), \
               "Parameter 'n_components' must be a float between 0 and 1 or a positive integer."
        self.method = method
        self.n_components = n_components
        self.y = False # if y is used by feature extraction algorithm
        if method == "PCA":
            self.extractor = PCA(n_components=n_components)
        elif method == "LDA":
            self.extractor = LDA(n_components=n_components)
            self.y = True
        else:
            raise ValueError(f"Unsupported feature extraction method: '{method}'.")

    def fit(self, X, y=None) -> "FeatureExtraction":
        """
        Fits the feature extraction algorithm to the data.

        Parameters
        ----------
        X 
            The input features for training.
        y
            The target values corresponding to the training samples in X.

        Returns
        -------
        `self` (an instance of `FeatureExtraction` class fit to the data).
        """
        if self.y:
            self.extractor.fit(X, y) 
        else:
            self.extractor.fit(X)
        return self

    def transform(self, X, y=None) -> np.ndarray:
        """
        Transforms the data by performing feature extraction.

        Parameters
        ----------
        X 
            The input features.
        y
            The target values corresponding to the training samples in X.

        Returns
        -------
        np.ndarray
            An array of transformed features.
        """
        if self.y:
            return self.extractor.transform(X) 
        return self.extractor.transform(X)


def get_full_pipeline(
        df: pd.DataFrame,
        target: str,
        variance_treshold: float = 0.01,
        feature_selection_treshold: float = 200.0,
        feature_extraction_method: Literal["PCA", "LDA"] = "PCA",
        n_components: int | float = 30,
        verbose: bool = False
    ) -> Pipeline:
    """
    Creates a pipeline with selected parameters.

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
        If set to true, the pipeline will print information about the progress
        of data preprocessing when used.

    Returns
    -------
    Pipeline with provided parameters.
    """

    num_cols, cat_cols = get_dataset_info(df, target)

    # Pipeline to be applied to numeric columns.
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
            ('dense', FunctionTransformer(lambda x: np.array(x.todense()), 
                                        accept_sparse=True)),
            ('select_from_model', SelectFromModel(LassoCV(), threshold=feature_selection_treshold)),
            ('feature_extraction', FeatureExtraction(n_components, feature_extraction_method)),
            
        ],
        verbose=verbose,
    )

    return full_pipeline


def get_full_pipeline_with_model(
        df: pd.DataFrame,
        target: str,
        variance_treshold: float = 0.01,
        feature_selection_treshold: float = 200.0,
        feature_extraction_method: Literal["PCA", "LDA"] = "PCA",
        n_components: int | float = 30,
        verbose: bool = False
    ) -> Pipeline:
    """
    Creates a pipeline with selected parameters and a machine learning algorithm attached to it.

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
        If set to true, the pipeline will print information about the progress
        of data preprocessing when used.

    Returns
    -------
    Pipeline with provided parameters and machine learning model.
    """

    full_pipeline = get_full_pipeline(df, target, variance_treshold, feature_selection_treshold,
                                      feature_extraction_method, n_components, verbose)

    full_pipeline_with_model = Pipeline([
        ('full_pipeline', full_pipeline),
        ('model', MLPRegressor(hidden_layer_sizes=(50), batch_size = 8,
                               learning_rate_init = 0.1, verbose = True, max_iter=30))
    ])

    return full_pipeline_with_model
