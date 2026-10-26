# 📦 Portfolio: Interactive ML Web Apps

Two end-to-end **scikit-learn + Gradio** deployment projects sharing one
clean architecture — *train → joblib → real-time Gradio UI*. Both apps
re-run inference on every slider move (Gradio `.input` events), with no
submit button.

## Project structure

```
Gradio ML web App/
├── README.md                        # this file
│
├── 01-house-price-predictor/        # Regression (main submission)
│   ├── pipeline_defs.py             # Custom feature engineering transformer
│   ├── train_model.py               # 5-fold CV shootout → best regressor
│   ├── app.py                       # 3-tab UI: predict / sensitivity / insights
│   ├── requirements.txt
│   ├── model/
│   │   ├── house_price_model.joblib # Trained sklearn pipeline (~1 MB)
│   │   ├── model_metadata.json      # Metrics, CV table, importances, ranges
│   │   └── map_sample.json          # 500 blocks for the location map
│   └── ...
│
└── 02-iris-species-classifier/      # Classification (bonus demo)
    ├── pipeline_defs.py             # Custom feature engineering transformer
    ├── train_model.py               # 5×5-fold CV shootout → best classifier
    ├── app.py                       # Sliders → live probabilities + boundary heatmap
    ├── requirements.txt
    └── model/
        ├── iris_model.joblib        # Trained sklearn pipeline
        └── model_metadata.json      # Accuracy, CV table, CM, importance, ranges
```

## What makes each app "Pro"

| | 🏠 House price (main) | 🌸 Iris (bonus) |
|---|---|---|
| Task | Regression ($ per block group) | Classification (3 species) |
| Engineered features (inside the pipeline) | `bedrooms_ratio`, `rooms_per_person`, `log_population`, 10 KMeans lat/long clusters | `petal_area`, `petal_ratio`, `sepal_area` |
| Model selection | 5-fold CV over RF / GB / **HistGB** → 0.8409 CV R² | 5×5-fold repeated CV over RF / GB / HistGB / **LogReg** → 0.9643 CV acc |
| Hold-out test | **R² 0.8445 · MAE $29,907 · RMSE $45,145** | **Accuracy 89.5%** (setosa 0.96 / versicolor 0.86 / virginica 0.88 F1) |
| Inputs | 8 sliders + SF / LA / rural presets | 4 sliders + 3 species presets |
| Signature visuals | Live price + ★ location map + ±10% tornado chart | Live probability bars + class-probability heatmap over the petal plane (★ your flower) |
| Insights tab | CV table, test metrics, permutation importance | CV table, confusion-matrix heatmap, permutation importance |

## Quickstart (VS Code / terminal)

```bash
# main app (from the repo root)
cd 01-house-price-predictor
python -m pip install -r requirements.txt
python train_model.py     # optional: only to refresh the .joblib
python app.py             # → opens http://127.0.0.1:7860

# bonus app
cd ../02-iris-species-classifier
python app.py
```

Works in any local environment (VS Code, plain terminal, PyCharm…).
Windows PowerShell, bash and zsh are all fine — just keep the slashes
straight on Windows (`cd "..."` paths quoted).

## Running on Google Colab

No code changes needed — with three small workflow notes:

1. **Upload** the project folder as a zip, then unzip it in Colab:
   ```python
   from google.colab import files
   files.upload()                      # choose the .zip of a project folder
   !unzip -q 01-house-price-predictor.zip
   %cd 01-house-price-predictor
   ```
2. **Install + retrain in the Colab session** so the joblib file matches
   the runtime's scikit-learn version:
   ```python
   !pip install -q -r requirements.txt
   !python train_model.py
   ```
3. **Launch with `share=True`** for a public `.gradio.live` link (Colab
   has no visible localhost):
   ```python
   import app
   app.demo.launch(share=True)
   ```

Same two lines run identically in VS Code (comment upload/unzip, use
`python app.py` as usual).

## Architecture notes (worth knowing before a review)

- **Zero train/serve skew** — feature engineering (KMeans clustering,
  ratios, log transforms) lives in the pickled pipeline itself, so serving
  can't drift from training logic.
- **Honest model selection** — candidates are compared by CV on the
  *training* split only; the hold-out set is touched exactly once. On the
  iris app a single 5-fold run was too noisy at 150 rows, so selection
  uses RepeatedStratifiedKFold (5×5).
- **Small-data caveat** — with 38 iris test rows, one flower moves
  accuracy by 2.6 points; all four candidates are statistically tied
  (~96% CV). Documented rather than hidden.
- **UI events** — `.input` (user-driven only) instead of `.change`, which
  wrongly fires during page-load hydration in Gradio 6.

## Why this is a strong pair for submission

- One pattern, two tasks → shows you can deploy **regression** (continuous
  pricing) and **classification** (probability outputs) with the same
  serving architecture.
- Real-time inference: predictions, maps and sensitivity analyses update
  on every drag.
- Custom sklearn transformers + model selection + permutation importance →
  depth beyond a "fit-and-predict" tutorial.
- Everything travels in a single `joblib` artifact; the app is
  dataset-agnostic and reproducible from source.
