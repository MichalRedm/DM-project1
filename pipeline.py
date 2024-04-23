from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import Perceptron, LassoCV, LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from umap.umap_ import UMAP

num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
cat_cols = ['model', 'transmission', 'fuelType',  'Manufacturer']

class FeatureExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: float, method: str):
        self.method = method
        self.n_components = n_components
        self.y = False # if y is used by feature extraction algorithm
        if method == "PCA":
            self.extractor = PCA(n_components=n_components)
        elif method == "LDA":
            self.extractor = LinearDiscriminantAnalysis(n_components=n_components)
            self.y = True
        else:
            raise ValueError(f"Unsupported feature extraction method: '{method}'.")

    def set_params(self, n_components: float, method: str):
        self.method = method
        self.n_components = n_components
        self.y = False # if y is used by feature extraction algorithm
        if method == "PCA":
            self.extractor = PCA(n_components=n_components)
        elif method == "LDA":
            self.extractor = LinearDiscriminantAnalysis(n_components=n_components)
            self.y = True
        else:
            raise ValueError(f"Unsupported feature extraction method: '{method}'.")

    def fit(self, X, y):
        if self.y:
            self.extractor.fit(X, y) 
        else:
            self.extractor.fit(X)
        return self

    def transform(self, X, y=None):
        if self.y:
            return self.extractor.transform(X) 
        return self.extractor.transform(X)


num_pipeline = Pipeline([
    ('Variance_threshold', VarianceThreshold(threshold=0.01)),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ('Variance_threshold', VarianceThreshold(threshold=0.01)),
])

col_transform = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Pipeline parameters set to best chose with grid search
full_pipeline = Pipeline([
        ('transform', col_transform),
        ('dense', FunctionTransformer(lambda x: np.array(x.todense()), 
                                      accept_sparse=True)),
        ('select_from_model', SelectFromModel(LassoCV(), threshold=200)),
        ('feature_extraction', FeatureExtraction(30, 'LDA')),
        ('model', MLPRegressor(hidden_layer_sizes=(50), batch_size = 8, 
                               learning_rate_init = 0.1, verbose = True, max_iter=30))
    ],
    verbose = True,
) 