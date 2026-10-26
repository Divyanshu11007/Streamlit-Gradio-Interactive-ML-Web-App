# 🌸 Iris Species Classifier — Pro

End-to-end ML deployment project (advanced edition): a **CV-selected**
classifier served through an interactive **Gradio** web UI with sliders,
live **probability bars**, and a real-time **decision-boundary heatmap**.

## Highlights

- **Engineered features inside the pipeline** (`pipeline_defs.py`): a custom
  sklearn transformer adds `petal_area`, `petal_ratio`, `sepal_area` — all
  serialized in the single `joblib` artifact (zero train/serve skew).
- **Model selection** (`train_model.py`): RandomForest, GradientBoosting,
  HistGradientBoosting and LogisticRegression compared with **5×5-fold
  RepeatedStratifiedKFold** CV; the winner is refit on the full training
  set and exported.
- **Three-part web app** (`app.py`):
  - 🌸 **Predict** — 4 sliders + setosa/versicolor/virginica presets;
    live probability bars (`gr.Label`).
  - **Boundary heatmap** — for the current slider values, a 41×41 grid of
    class probabilities across the petal length × width plane (sepals
    held fixed), three side-by-side heatmaps + a ★ at your flower.
    Red regions = confident, blue = uncertain.
  - 🧠 **Model insights** — CV table, test accuracy, confusion-matrix
    heatmap and permutation importance.
- **Genuine real-time UI** — `.input` events only (no page-load storms).

## Structure

```
02-iris-species-classifier/
├── pipeline_defs.py       # Shared custom transformer (ships with app)
├── train_model.py         # Feature engineering, 5×5 CV selection, export
├── app.py                 # 3-part Gradio interface
├── requirements.txt
└── model/
    ├── iris_model.joblib      # Trained sklearn pipeline
    └── model_metadata.json    # Accuracy, CV table, CM, importance, ranges
```

## Quickstart

```bash
pip install -r requirements.txt
python train_model.py    # trains + exports (seconds)
python app.py            # → http://127.0.0.1:7860
```

## Results (as trained)

| Model | 5×5 CV accuracy (train) |
|---|---|
| RandomForest | 0.9625 ± 0.0271 |
| GradientBoosting | 0.9591 ± 0.0327 |
| HistGradientBoosting | 0.9538 ± 0.0362 |
| **LogisticRegression (winner)** | **0.9643 ± 0.0375** |

Test accuracy: **89.5%** on the 38-row stratified hold-out (setosa F1
0.957, versicolor 0.857, virginica 0.880).

> Note on small data — with 38 test rows, **one misclassified flower
> moves accuracy by 2.6 points**. All four candidates are statistically
> tied on CV (~0.96); the CV winner wins on train, the test number is
> noise. Point of the story: don't tune against a tiny test set.

## Interview talking points

1. Why engineer features inside a `Pipeline`? → everything from feature
   transforms to scaling travels in one pickled artifact.
2. Why **repeated** CV for selection? → on 150 rows a single 5-fold mean
   has ±3–4% swings; repeats average that out.
3. Why show a probability heatmap? → makes model *uncertainty* visible:
   drag near the versicolor/virginica boundary and watch the probabilities
   split ~50/50.
