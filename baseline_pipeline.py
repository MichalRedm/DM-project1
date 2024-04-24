"""
File defining baseline to which we will compare our results.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neural_network import MLPRegressor
from dataset_info import *

class Dummy(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X

baseline_num_pipeline = Pipeline([
    ('dummy', Dummy())
])

baseline_cat_pipeline = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore')),
])

baseline_col_transform = ColumnTransformer([
    ('num', baseline_num_pipeline, num_cols),
    ('cat', baseline_cat_pipeline, cat_cols)
])

baseline_full_pipeline = Pipeline([
        ('transform', baseline_col_transform),
        ('model', MLPRegressor(hidden_layer_sizes=(50), batch_size = 8,
                               learning_rate_init = 0.1, verbose = True, max_iter=30))
    ],
    verbose=True
)