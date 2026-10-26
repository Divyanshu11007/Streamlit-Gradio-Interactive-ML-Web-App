# 🏠 California House Price Predictor — Pro

End-to-end ML deployment project with an **advanced** twist: a custom
feature-engineering pipeline (KMeans geo-clusters + ratios + log
transforms), **cross-validated model selection** between three regressors,
and a three-tab Gradio UI with **real-time** price, map and sensitivity
analysis.

## Highlights

- **Feature engineering inside the pipeline** (`pipeline_defs.py`): a custom
  sklearn transformer adds `bedrooms_ratio`, `rooms_per_person`,
  `log_population`, and 10 one-hot KMeans clusters over (lat, lon) — all
  serialized inside the single `joblib` file.
- **Auto model selection** (`train_model.py`): RandomForest,
  GradientBoosting and HistGradientBoosting compared with 5-fold CV; the
  winner is refit and exported.
- **Three-tab web app** (`app.py`):
  - 🏡 **Predict** — 8 sliders + SF / LA / rural presets; price updates
    live with an ✱ marker on a map of 500 training blocks.
  - 🔍 **Sensitivity** — tornado chart: how a ±10% nudge of each feature
    alone moves the estimate for the *current sliders*.
  - 🧠 **Model insights** — CV table, test metrics (R²/MAE/RMSE) and
    permutation importance.
- **Genuine real-time UI** — every slider change re-runs inference; no
  submit button.

## Structure

```
01-house-price-predictor/
├── pipeline_defs.py       # Shared custom transformer (must ship with app)
├── train_model.py         # Feature engineering, CV selection, export
├── app.py                 # 3-tab Gradio interface
├── requirements.txt
└── model/
    ├── house_price_model.joblib   # Trained sklearn pipeline
    ├── model_metadata.json        # Metrics, CV table, importance, ranges
    └── map_sample.json            # 500 blocks for the location map
```

## Quickstart

```bash
pip install -r requirements.txt
python train_model.py     # trains + exports (~1 min)
python app.py             # → http://127.0.0.1:7860
```

## Metrics (typical)

| Metric | ≈ Value |
|---|---|
| CV R² (train, best model) | ~0.83 ± 0.01 |
| Test R² | ~0.83 |
| Test MAE | ~$30,000 |
| Test RMSE | ~$48,000 |

## Interview talking points

1. Why engineer features inside a sklearn `Pipeline`? → zero
   train/serve skew: everything from KMeans clustering to scaling travels
   in one pickled artifact.
2. Why CV model selection instead of a fixed model? → honest hyper-model
   comparison on train only; test set stays untouched.
3. Why permutation importance for the insights tab? → gauge-free
   attribution computed on held-out data.
