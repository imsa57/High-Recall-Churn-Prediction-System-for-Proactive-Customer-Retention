from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class TenureBinningTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, bins=[0, 6, 12, 24, 60, np.inf], labels=['0-6m', '6-12m', '1-2y', '2-5y', '5+y']):
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(["Tenure_Category"])

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_binned = pd.cut(X.iloc[:, 0], bins=self.bins, labels=self.labels, include_lowest=True)
        return X_binned.to_numpy().reshape(-1, 1)    
