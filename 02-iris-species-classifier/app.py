"""
Interactive iris species classifier web app — advanced edition.

Tabs:
- Predict:       4 measurement sliders + 3 species presets; class
                 probabilities update in real time as sliders move.
- Boundary map:  heatmap of each class probability across the petal
                 length × petal width plane (sepals held at the current
                 slider values), with a red star at your current point.
- Model insights: CV model-selection table, test accuracy, confusion
                 matrix heatmap and permutation importance.

Run:  python app.py  ->  open http://127.0.0.1:7860
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline_defs import IrisFeatureAdder  # needed to unpickle the pipeline

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "iris_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"

# --------------------------------------------------------------------------
# Load pretrained artifacts
# --------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

FEATURES: list[str] = meta["features"]
LABELS: dict = meta["feature_labels"]
RANGES: dict = meta["feature_ranges"]
CLASSES: list[str] = meta["classes"]
MODEL_NAME: str = meta["model_name"]

GRID = 41  # resolution of the boundary heatmap


# --------------------------------------------------------------------------
# Prediction + figures
# --------------------------------------------------------------------------
def predict(*vals: float) -> dict[str, float]:
    X = pd.DataFrame([vals], columns=FEATURES)
    proba = model.predict_proba(X)[0]
    return {cls: float(p) for cls, p in zip(CLASSES, proba)}


def predict_proba_1row(*vals: float) -> np.ndarray:
    X = pd.DataFrame([vals], columns=FEATURES)
    return model.predict_proba(X)[0]


def boundary_figure(*vals: float):
    """Class-probability heatmaps over the petal plane, sepals fixed."""
    petal_l = np.linspace(
        RANGES["petal_length"]["min"], RANGES["petal_length"]["max"], GRID
    )
    petal_w = np.linspace(
        RANGES["petal_width"]["min"], RANGES["petal_width"]["max"], GRID
    )
    grid = pd.DataFrame(
        [[vals[0], vals[1], pl, pw] for pl in petal_l for pw in petal_w],
        columns=FEATURES,
    )
    proba = model.predict_proba(grid)
    proba = proba.reshape(GRID, GRID, len(CLASSES))
    current = predict_proba_1row(*vals)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"{cls} — P = {p:.2f}" for cls, p in zip(CLASSES, current)
        ],
        horizontal_spacing=0.05,
    )
    for j, cls in enumerate(CLASSES):
        fig.add_trace(
            go.Heatmap(
                z=proba[:, :, j],
                x=petal_w,
                y=petal_l,
                coloraxis="coloraxis",
                hovertemplate=f"{cls}: %{{z:.2f}}<br>"
                + "petal len %{y:.1f} cm, width %{x:.1f} cm<extra></extra>",
            ),
            row=1,
            col=j + 1,
        )
    for col in range(1, 4):
        fig.add_trace(
            go.Scatter(
                x=[vals[3]],
                y=[vals[2]],
                mode="markers",
                marker={"size": 15, "color": "black", "symbol": "star"},
                name="Your flower",
                showlegend=False,
                hoverinfo="x+y",
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        coloraxis={
            "colorscale": "RdBu_r",
            "cmin": 0,
            "cmax": 1,
            "colorbar": {"title": "Class<br>probability"},
        },
        template="plotly_white",
        height=400,
        margin={"l": 55, "r": 10, "t": 60, "b": 45},
        showlegend=False,
    )
    fig.update_xaxes(title_text="Petal width (cm)", row=1, col=2)
    fig.update_yaxes(title_text="Petal length (cm)", row=1, col=1)
    return fig


def confusion_figure():
    cm = np.array(meta["confusion_matrix"])
    fig = px.imshow(
        cm,
        x=CLASSES,
        y=CLASSES,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion matrix on the {meta['n_test']}-row test set",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin={"l": 90, "r": 10, "t": 60, "b": 40},
    )
    return fig


def importance_figure():
    df = pd.DataFrame(meta["importances"]).rename(columns={"d_acc": "Δ accuracy"})
    fig = px.bar(
        df.sort_values("Δ accuracy"),
        x="Δ accuracy",
        y="feature",
        orientation="h",
        title="Permutation importance (drop in accuracy when a feature is shuffled)",
        labels={"feature": ""},
        color_discrete_sequence=["#4d9fdb"],
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin={"l": 150, "r": 10, "t": 60, "b": 10},
    )
    return fig


def update_report(*vals: float):
    return predict(*vals), boundary_figure(*vals)


# --------------------------------------------------------------------------
# Slider components + species presets
# --------------------------------------------------------------------------
sliders = [
    gr.Slider(
        minimum=RANGES[name]["min"],
        maximum=RANGES[name]["max"],
        value=RANGES[name]["default"],
        step=0.1,
        label=LABELS[name],
        info=name,
    )
    for name in FEATURES
]

PRESETS = {
    "🌷 setosa-like": [5.1, 3.5, 1.4, 0.2],
    "🌸 versicolor-like": [5.9, 2.8, 4.3, 1.3],
    "🌺 virginica-like": [6.5, 3.0, 5.5, 2.0],
}

cv_rows = "\n".join(
    f"| {r['model']} | {r['acc_mean']:.4f} ± {r['acc_std']:.4f} |"
    for r in meta["cv_results"]
)
insights_markdown = (
    "### 🧠 Model insights\n\n"
    f"Winning model (5×5-fold CV on train): **{meta['model_name']}**\n\n"
    "| Model | CV accuracy (train) |\n|---|---|\n"
    + cv_rows
    + "\n\n"
    f"- Test accuracy: **{meta['accuracy'] * 100:.1f}%** "
    f"({meta['n_train']} train / {meta['n_test']} test rows)\n"
    "- With only 38 test rows, one misclassified flower moves accuracy "
    "by 2.6 points — differences between the candidates are CV-tied.\n\n"
    "Features engineered inside the pipeline (`pipeline_defs.py`):\n"
    "`petal_area`, `petal_ratio`, `sepal_area`"
)

# --------------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------------
defaults = [RANGES[name]["default"] for name in FEATURES]
initial_probas = predict(*defaults)
initial_boundary = boundary_figure(*defaults)

with gr.Blocks(title="🌸 Iris Species Classifier — Pro") as demo:
    gr.Markdown(
        "# 🌸 Iris Species Classifier — Pro\n"
        "Move the sliders — probabilities and the boundary heatmap update "
        "**in real time**. Pipeline: feature engineering + StandardScaler + "
        f"{MODEL_NAME}, selected by 5×5-fold CV."
    )

    with gr.Tabs():
        with gr.Tab("🌸 Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Group(sliders)
                    label = gr.Label(
                        value=initial_probas,
                        label="Species probabilities",
                        num_top_classes=3,
                    )
                with gr.Column(scale=2):
                    with gr.Row():
                        preset_buttons = [
                            gr.Button(name, size="sm") for name in PRESETS
                        ]
                    boundary_plot = gr.Plot(initial_boundary, show_label=False)
                    gr.Markdown(
                        "The heatmap holds **sepal** values fixed and sweeps "
                        "the petal plane — red = confident class, blue = "
                        "uncertain. The ★ is your flower's petal point."
                    )
        with gr.Tab("🧠 Model insights"):
            gr.Markdown(insights_markdown)
            with gr.Row():
                gr.Plot(confusion_figure(), show_label=False)
                gr.Plot(importance_figure(), show_label=False)

    # Real-time: user slider moves → probabilities + boundary heatmap.
    # .input = user-driven only (change fires at page load with None values).
    for i, slider in enumerate(sliders):
        api_kwargs = {"api_name": "predict"} if i == 0 else {}
        slider.input(
            fn=update_report,
            inputs=sliders,
            outputs=[label, boundary_plot],
            **api_kwargs,
        )

    # Presets write the sliders AND refresh every report in a single call
    for button, preset in zip(preset_buttons, PRESETS.values()):
        button.click(
            fn=lambda p=list(preset): list(p)
            + [predict(*p), boundary_figure(*p)],
            outputs=sliders + [label, boundary_plot],
        )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        css="footer {visibility: hidden}",
    )
