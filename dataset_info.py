import pandas as pd
from typing import Tuple, List

def get_dataset_info(df: pd.DataFrame, target: str) -> Tuple[List, List]:
    """
    Provides lists of numerical and categorical colums in
    the dataset provided.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe containing the dataset.
    
    target : str
        Name of the target column that should be excluded from returned lists.

    Returns
    -------
    Tuple[List, List]
        List of numerical columns and list of categorical columns.
    """

    num_cols = list(df.select_dtypes(include="number").columns)
    num_cols.remove(target)
    cat_cols = list(set(df.columns).difference(num_cols))
    cat_cols.remove(target)

    return num_cols, cat_cols
