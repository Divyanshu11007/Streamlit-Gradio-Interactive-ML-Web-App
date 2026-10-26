"""
Model/pipeline definitions shared by train_model.py and app.py.

This module must ship with the app because joblib pickles the transformer
by module path (pipeline_defs.HouseFeatureAdder) — the app imports this
file so that joblib.load() can reconstruct the full engineered pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

BASE_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


class HouseFeatureAdder(BaseEstimator, TransformerMixin):
    """Engineers extra features the raw table lacks, entirely inside the
    sklearn pipeline (so everything travels inside the single joblib file):

    - `bedrooms_ratio`  AveBedrms / AveRooms (much more informative alone)
    - `rooms_per_person` AveRooms / AveOccup
    - `log_population`  log1p(Population) fixes the heavy right tail
    - `region_i`        one-hot of a KMeans cluster over (Latitude,
                        Longitude) → discrete geography the tree can split on
    """

    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _frame(X) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(
            X, columns=BASE_FEATURES
        )

    # -- sklearn API --------------------------------------------------------
    def fit(self, X, y=None):
        df = self._frame(X)
        self.kmeans_: KMeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        )
        self.kmeans_.fit(df[["Latitude", "Longitude"]])
        self.n_clusters_: int = len(np.unique(self.kmeans_.labels_))
        self.output_names_ = (
            BASE_FEATURES
            + ["bedrooms_ratio", "rooms_per_person", "log_population"]
            + [f"region_{i}" for i in range(self.n_clusters_)]
        )
        return self

    def transform(self, X):
        df = self._frame(X)
        zones = self.kmeans_.predict(df[["Latitude", "Longitude"]])

        zone_cols = pd.DataFrame(
            (zones[:, None] == np.arange(self.n_clusters_)).astype(float),
            columns=[f"region_{i}" for i in range(self.n_clusters_)],
            index=df.index,
        )

        engineered = pd.DataFrame(
            {
                "bedrooms_ratio": df["AveBedrms"] / df["AveRooms"],
                "rooms_per_person": df["AveRooms"] / df["AveOccup"].clip(lower=0.5),
                "log_population": np.log1p(df["Population"]),
            },
            index=df.index,
        )

        return pd.concat([df[BASE_FEATURES], engineered, zone_cols], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_names_, dtype=object)
