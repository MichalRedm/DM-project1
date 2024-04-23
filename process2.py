from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
import pandas as pd


INPUT_DATASET = "./CarsData.csv"
OUTPUT_DATASET = "./CarsDataProcessed.csv"


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return X.todense()
    

def preprocess(df: pd.DataFrame, target: str) -> pd.DataFrame:
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

    Returns
    -------
    Pandas dataframe containing the dataset after preprocessing and the
    target column as the last column.
    """

    X, y = df.drop(target, axis=1), df[target].to_numpy()

    num_cols = list(X.select_dtypes([np.number]).columns)
    cat_cols = list(np.setdiff1d(X.columns, num_cols))

    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('ohe', OneHotEncoder())
    ])

    col_transform = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    full_pipeline = Pipeline([
        ('transform', col_transform),
        ('dense', FunctionTransformer(lambda x: np.array(x.todense()), accept_sparse=True)),
        ('pca', PCA(n_components = 0.95))
    ])

    X_new_np = full_pipeline.fit_transform(X, y)

    df = pd.DataFrame(X_new_np, columns=[f"componenet{i}" for i in range(X_new_np.shape[1])])
    df.insert(len(df.columns), target, y)
    return df


def main() -> None:
    df = pd.read_csv(INPUT_DATASET)
    df = preprocess(df, "price")
    df.to_csv(OUTPUT_DATASET, index=False)


if __name__ == "__main__":
    main()
