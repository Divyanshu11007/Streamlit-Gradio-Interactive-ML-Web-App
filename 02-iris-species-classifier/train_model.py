"""
Train the iris species classification model — advanced edition.

1. Feature engineering: petal area/ratio, sepal area (pipeline_defs.py).
2. Model selection: three classifiers compared with 5-fold CV on the
   training set; the winner is refit and evaluated on the hold-out.
3. Permutation importance + confusion matrix on the test set → exported
   for the app's "Model insights" tab.

Run:  python train_model.py
"""

import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_defs import BASE_FEATURES, IrisFeatureAdder

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "iris_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"

CANDIDATES = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, random_state=42
    ),
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000),
}


def main() -> None:
    data = load_iris(as_frame=True)
    X = data.data.rename(columns=dict(zip(data.feature_names, BASE_FEATURES)))
    y = data.target
    classes = list(data.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1. Model selection via repeated CV (repeats damp the selection noise
    # that a single 5-fold run has on a 150-row dataset) ---------------------
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    print("5x5-fold CV on the training split (accuracy):")
    cv_results = []
    for name, estimator in CANDIDATES.items():
        pipe = make_pipeline(IrisFeatureAdder(), StandardScaler(), estimator)
        scores = cross_val_score(pipe, X_train, y_train, scoring="accuracy", cv=cv)
        cv_results.append(
            {
                "model": name,
                "acc_mean": round(scores.mean(), 4),
                "acc_std": round(scores.std(), 4),
            }
        )
        print(f"  {name:>22}: {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(cv_results, key=lambda r: r["acc_mean"])["model"]
    print(f"\nBest model: {best_name}")

    # 2. Final fit + hold-out evaluation ------------------------------------
    model = make_pipeline(
        IrisFeatureAdder(), StandardScaler(), CANDIDATES[best_name]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, target_names=classes, digits=3)
    cm = confusion_matrix(y_test, preds).tolist()
    print(f"Test accuracy: {accuracy:.4f}")
    print(report)

    # 3. Permutation importance ---------------------------------------------
    perm = permutation_importance(
        model, X_test, y_test, scoring="accuracy", n_repeats=15, random_state=42
    )
    importances = [
        {"feature": feat, "d_acc": round(float(m), 4)}
        for feat, m in sorted(
            zip(BASE_FEATURES, perm.importances_mean), key=lambda t: t[1]
        )
        if m > 0
    ]

    # 4. Save artifacts -------------------------------------------------------
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    labels = {
        "sepal_length": "Sepal length (cm)",
        "sepal_width": "Sepal width (cm)",
        "petal_length": "Petal length (cm)",
        "petal_width": "Petal width (cm)",
    }
    ranges = {
        name: {
            "min": float(X[name].min()),
            "max": float(X[name].max()),
            "default": float(X[name].median()),
        }
        for name in BASE_FEATURES
    }

    meta = {
        "model_path": MODEL_PATH.name,
        "model_name": best_name,
        "classes": classes,
        "cv_results": cv_results,
        "importances": importances,
        "confusion_matrix": cm,
        "features": BASE_FEATURES,
        "feature_labels": labels,
        "feature_ranges": ranges,
        "accuracy": round(accuracy, 4),
        "report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved meta   -> {META_PATH}")


if __name__ == "__main__":
    main()
