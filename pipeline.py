from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import Perceptron, LassoCV


num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
cat_cols = ['model', 'transmission', 'fuelType',  'Manufacturer']

num_pipeline = Pipeline([
    ('Variance_threshold', VarianceThreshold(threshold=0.01)),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('ohe', OneHotEncoder()),
    ('Variance_threshold', VarianceThreshold(threshold=0.01)),
])

col_transform = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

full_pipeline = Pipeline([
    ('transform', col_transform),
    ('dense', FunctionTransformer(lambda x: np.array(x.todense()), accept_sparse=True)),
    ('k_best', SelectFromModel(LassoCV(), threshold=200.0)),
    ('pca', PCA(n_components = 0.95)),
],
    verbose = True,
) 
