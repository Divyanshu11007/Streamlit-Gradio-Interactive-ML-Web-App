"""
Model/pipeline definitions shared by train_model.py and app.py.

This module ships with the app because joblib pickles the transformer by
module path (pipeline_defs.IrisFeatureAdder) — the app imports this file
so that joblib.load() can reconstruct the full engineered pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


class IrisFeatureAdder(BaseEstimator, TransformerMixin):
    """Adds engineered geometry features inside the sklearn pipeline so
    everything (feature engineering + scaling + model) travels in one
    joblib artifact — zero train/serve skew:

    - `petal_area`  petal_length × petal_width (the single most
      discriminative quantity in the dataset)
    - `petal_ratio` petal_length / petal_width (shape, scale-free)
    - `sepal_area`  sepal_length × sepal_width
    """

    def fit(self, X, y=None):
        self.output_names_ = BASE_FEATURES + [
            "petal_area",
            "petal_ratio",
            "sepal_area",
        ]
        return self

    @staticmethod
    def _frame(X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=BASE_FEATURES
        )

    def transform(self, X):
        df = self._frame(X)
        return pd.concat(
            [
                df[BASE_FEATURES],
                pd.DataFrame(
                    {
                        "petal_area": df["petal_length"] * df["petal_width"],
                        "petal_ratio": df["petal_length"] / df["petal_width"],
                        "sepal_area": df["sepal_length"] * df["sepal_width"],
                    },
                    index=df.index,
                ),
            ],
            axis=1,
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_names_, dtype=object)
