"""
Interactive California house price prediction web app — advanced edition.

Tabs:
- Predict:       8 sliders + regional presets; price, comparison and a
                 location map update in real time as sliders move.
- Sensitivity:   tornado chart showing how a ±10% nudge of each feature
                 alone changes the estimate for the current inputs.
- Model insights: CV model-selection table, hold-out metrics and
                 permutation feature importance from training.

Run:  python app.py  ->  open http://127.0.0.1:7860
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
import plotly.express as px

from pipeline_defs import HouseFeatureAdder  # needed to unpickle the pipeline

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"
MAP_PATH = MODEL_DIR / "map_sample.json"

# --------------------------------------------------------------------------
# Load pretrained artifacts
# --------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)
with open(MAP_PATH, encoding="utf-8") as f:
    sample_rows = json.load(f)

FEATURES: list[str] = meta["features"]
LABELS: dict = meta["feature_labels"]
RANGES: dict = meta["feature_ranges"]
MODEL_NAME: str = meta["model_name"]

BASE_PRICE_USD = 195_000  # dataset mean, used as a reference point


# --------------------------------------------------------------------------
# Prediction + figures
# --------------------------------------------------------------------------
def predict_price(*vals: float) -> float:
    X = pd.DataFrame([vals], columns=FEATURES)
    return max(model.predict(X)[0], 0.01) * 100_000


def price_markdown(*vals: float) -> str:
    price = predict_price(*vals)
    pct = (price - BASE_PRICE_USD) / BASE_PRICE_USD * 100
    arrow = "📈" if pct >= 0 else "📉"
    return (
        "### 🏠 Estimated median house value in this block group\n"
        f"# ${price:,.0f}\n"
        f"**{arrow} {abs(pct):.1f}% "
        f"{'above' if pct >= 0 else 'below'} average** "
        f"(CA mean ≈ ${BASE_PRICE_USD:,.0f}) · typical error "
        f"± ${meta['rmse_usd']:,} (test RMSE)"
    )


def map_figure(*vals: float):
    df = pd.DataFrame(sample_rows).rename(
        columns={
            "Latitude": "lat",
            "Longitude": "lon",
            "MedInc": "income",
            "price": "price",
        }
    )
    fig = px.scatter(
        df,
        x="lon",
        y="lat",
        color="price",
        color_continuous_scale="Viridis",
        opacity=0.35,
        labels={"lat": "Latitude", "lon": "Longitude", "price": "Train price ($)"},
        title="★ red star = your current sliders' location · dots = 500 training blocks",
    )
    fig.add_scatter(
        x=[vals[7]],
        y=[vals[6]],
        mode="markers",
        marker={"size": 18, "color": "red", "symbol": "star"},
        name="Your location",
    )
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        showlegend=False,
    )
    return fig


def tornado_figure(*vals: float):
    base = predict_price(*vals)
    rows = []
    for i, name in enumerate(FEATURES):
        span = 0.10 * vals[i]
        lo = list(vals)
        lo[i] = max(vals[i] - span, RANGES[name]["min"])
        hi = list(vals)
        hi[i] = min(vals[i] + span, RANGES[name]["max"])
        rows.append(
            {"feature": LABELS[name], "direction": "−10%", "delta": predict_price(*lo) - base}
        )
        rows.append(
            {"feature": LABELS[name], "direction": "+10%", "delta": predict_price(*hi) - base}
        )
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="delta",
        y="feature",
        color="direction",
        orientation="h",
        barmode="group",
        title="Δ predicted price when one feature moves ±10% (others fixed)",
        labels={"delta": "Δ price (USD)", "feature": "", "direction": ""},
        color_discrete_sequence=["#e45858", "#3aa0e4"],
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        template="plotly_white",
        height=470,
        margin={"l": 160, "r": 10, "t": 50, "b": 10},
    )
    return fig


def importance_figure():
    df = pd.DataFrame(meta["importances"]).rename(columns={"d_r2": "Δ R²"})
    fig = px.bar(
        df.sort_values("Δ R²"),
        x="Δ R²",
        y="feature",
        orientation="h",
        title="Permutation importance (drop in R² when a feature is shuffled)",
        labels={"feature": ""},
        color_discrete_sequence=["#4d9fdb"],
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 160, "r": 10, "t": 50, "b": 10},
    )
    return fig


def update_report(*vals: float):
    return price_markdown(*vals), map_figure(*vals), tornado_figure(*vals)


# --------------------------------------------------------------------------
# Slider components + regional presets
# --------------------------------------------------------------------------
sliders = [
    gr.Slider(
        minimum=RANGES[name]["min"],
        maximum=RANGES[name]["max"],
        value=RANGES[name]["default"],
        step=(RANGES[name]["max"] - RANGES[name]["min"]) / 200,
        label=LABELS[name],
        info=name,
    )
    for name in FEATURES
]

PRESETS = {
    "🏙️ SF Bay urban": [9.6, 20, 5.9, 1.03, 3300, 2.8, 37.77, -122.42],
    "🌴 LA coastal suburb": [6.2, 32, 5.3, 1.02, 1900, 2.9, 33.95, -118.45],
    "🏜️ Inland rural": [2.3, 48, 4.7, 1.06, 800, 3.1, 36.70, -119.80],
}

cv_rows = "\n".join(
    f"| {r['model']} | {r['r2_mean']:.4f} ± {r['r2_std']:.4f} |"
    for r in meta["cv_results"]
)
insights_markdown = (
    "### 🧠 Model insights\n\n"
    f"Winning model (5-fold CV on train): **{meta['model_name']}**\n\n"
    "| Model | CV R² (train) |\n|---|---|\n"
    + cv_rows
    + "\n\n**Hold-out test metrics**\n\n"
    f"- R²: **{meta['r2']:.4f}** · MAE: **${meta['mae_usd']:,}** · "
    f"RMSE: **${meta['rmse_usd']:,}**\n"
    f"- Rows: {meta['n_train']:,} train / {meta['n_test']:,} test\n\n"
    "Features engineered inside the pipeline (`pipeline_defs.py`):\n"
    "`bedrooms_ratio`, `rooms_per_person`, `log_population`, "
    "10 KMeans geographic clusters (`region_0…9`)"
)

# --------------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------------
defaults = [RANGES[name]["default"] for name in FEATURES]
initial_md = price_markdown(*defaults)
initial_map = map_figure(*defaults)
initial_tornado = tornado_figure(*defaults)

with gr.Blocks(title="🏠 CA House Price Predictor — Pro") as demo:
    gr.Markdown(
        "# 🏠 California House Price Predictor\n"
        "Move the sliders — price, map and sensitivity update "
        "**in real time**. Pipeline: feature engineering + StandardScaler + "
        f"{MODEL_NAME}, selected by 5-fold CV."
    )

    with gr.Tabs():
        with gr.Tab("🏡 Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_group = gr.Group(sliders)
                    with gr.Row():
                        preset_buttons = [
                            gr.Button(label, size="sm") for label in PRESETS
                        ]
                with gr.Column(scale=2):
                    price_md = gr.Markdown(initial_md)
                    map_plot = gr.Plot(initial_map, show_label=False)
        with gr.Tab("🔍 Sensitivity (±10%)"):
            tornado_plot = gr.Plot(initial_tornado, show_label=False)
            gr.Markdown(
                "Each bar shows how the estimate changes when one feature "
                "moves ±10% while everything else stays fixed at the "
                "**current slider values**. Longest bars = biggest levers."
            )
        with gr.Tab("🧠 Model insights"):
            gr.Markdown(insights_markdown)
            gr.Plot(importance_figure(), show_label=False)

    # Real-time: sliders → price + map + tornado.
    # .input = user-driven only, so nothing fires during page-load
    # hydration (that was the "None" preprocess crash).
    for i, slider in enumerate(sliders):
        api_kwargs = {"api_name": "predict"} if i == 0 else {}
        slider.input(
            fn=update_report,
            inputs=sliders,
            outputs=[price_md, map_plot, tornado_plot],
            **api_kwargs,
        )

    # Presets write the sliders AND refresh every report in a single call
    for button, preset in zip(preset_buttons, PRESETS.values()):
        button.click(
            fn=lambda p=list(preset): list(p)
            + [price_markdown(*p), map_figure(*p), tornado_figure(*p)],
            outputs=sliders + [price_md, map_plot, tornado_plot],
        )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        css="footer {visibility: hidden}",
    )
