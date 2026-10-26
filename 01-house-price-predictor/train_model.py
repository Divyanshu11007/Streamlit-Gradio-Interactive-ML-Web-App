"""
Train the California house price prediction model — advanced edition.

What's beyond a basic script:
1. Engineered features: bedrooms ratio, rooms/person, log-population, and
   KMeans geographic clusters (see pipeline_defs.py).
2. Model selection: three candidate regressors compared with 5-fold CV
   on the training set; the winner is refit and evaluated on the hold-out.
3. Permutation importance on the test set → exported for the app's
   "Model insights" tab.
4. A subsample of (lat, lon, price) exported for the app's location map.

Run:  python train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

from pipeline_defs import BASE_FEATURES, HouseFeatureAdder

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"
MAP_PATH = MODEL_DIR / "map_sample.json"

CANDIDATES = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        random_state=42,
    ),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    ),
}

# Heavy-tailed columns get their slider capped at the 99.5% quantile so the
# UI stays usable; Geo/age get the raw min/max.
CAPPED = {"MedInc", "AveRooms", "AveBedrms", "Population", "AveOccup"}


def main() -> None:
    data = fetch_california_housing(as_frame=True)
    X: pd.DataFrame = data.data[BASE_FEATURES]
    y: pd.Series = data.target  # median house value in $100k units

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Model selection via cross-validation -------------------------------
    print("5-fold CV on the training split (R²):")
    cv_results = []
    for name, estimator in CANDIDATES.items():
        pipe = make_pipeline(HouseFeatureAdder(), StandardScaler(), estimator)
        scores = cross_val_score(pipe, X_train, y_train, scoring="r2", cv=5, n_jobs=-1)
        cv_results.append(
            {"model": name, "r2_mean": round(scores.mean(), 4), "r2_std": round(scores.std(), 4)}
        )
        print(f"  {name:>22}: {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(cv_results, key=lambda r: r["r2_mean"])["model"]
    print(f"\nBest model: {best_name}")

    # 2. Final fit + hold-out evaluation ------------------------------------
    model = make_pipeline(
        HouseFeatureAdder(), StandardScaler(), CANDIDATES[best_name]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "r2": round(float(r2_score(y_test, preds)), 4),
        "mae_usd": round(float(mean_absolute_error(y_test, preds)) * 100_000),
        "rmse_usd": round(float(np.sqrt(mean_squared_error(y_test, preds))) * 100_000),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    # 3. Permutation importance ---------------------------------------------
    perm = permutation_importance(
        model, X_test, y_test, scoring="r2", n_repeats=5, random_state=42, n_jobs=-1
    )
    importances = [
        {"feature": feat, "d_r2": round(float(m), 4)}
        for feat, m in sorted(
            zip(BASE_FEATURES, perm.importances_mean),
            key=lambda t: t[1],
        )
        if m > 0
    ]

    # 4. Save artifacts ------------------------------------------------------
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    labels = {
        "MedInc": "Median income ($10k)",
        "HouseAge": "Median house age (years)",
        "AveRooms": "Avg rooms per household",
        "AveBedrms": "Avg bedrooms per household",
        "Population": "Block population",
        "AveOccup": "Avg occupants per household",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
    }
    ranges = {}
    for name in BASE_FEATURES:
        col = X[name]
        top = col.quantile(0.995) if name in CAPPED else col.max()
        ranges[name] = {
            "min": float(col.min()),
            "max": float(top),
            "default": float(col.median()),
        }

    sample = (
        X[["Latitude", "Longitude", "MedInc"]]
        .sample(500, random_state=0)
        .assign(price=lambda d: y.loc[d.index].to_numpy().round(3) * 100_000)
    )
    sample.to_json(
        MAP_PATH,
        orient="records",
        compression=None,
        double_precision=4,
    )

    meta = {
        "model_path": MODEL_PATH.name,
        "model_name": best_name,
        "cv_results": cv_results,
        "importances": importances,
        "map_sample_path": MAP_PATH.name,
        "features": BASE_FEATURES,
        "feature_labels": labels,
        "feature_ranges": ranges,
        **metrics,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved meta   -> {META_PATH}")
    print(f"Saved map    -> {MAP_PATH}")


if __name__ == "__main__":
    main()
