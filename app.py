"""
Dashboard for the banana hand inspection model.

    uv run python main.py app        (or: uv run python app.py)

Loads the committed weights and the reports written by the training pipeline.
It never trains anything. Set APP_PORT to use a port other than Gradio's 7860.
"""

import json
import os

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from costing import BASE_RATE, COST_FN, COST_FP, REPORTS
from data import ROOT, pick_device
from review import COST_PER_HOUR, HANDS_PER_DAY, SECONDS_PER_CHECK, outcome

WEIGHTS = ROOT / "weights" / "best.pt"
FRESH_DISPLAY_CONF = 0.25  # Fresh boxes are only drawn for context

# Colours from the fruit itself: leaf green passes, ripe yellow goes to a person, bruise brown is rejected.
GREEN, YELLOW, BROWN = "#3F7D20", "#E3B505", "#6B3E26"
INK, PAPER = "#1E2B24", "#F2F5F0"

# ---------- load once at startup: weights and pipeline outputs, no training ----------
MODEL = YOLO(str(WEIGHTS))
NAMES = MODEL.names
ROTTEN_ID = next(i for i, n in NAMES.items() if n.strip() == "Rotten")
SCORES = pd.read_csv(REPORTS / "image_scores.csv").query("model == 'finetuned'")
METRICS = json.loads((REPORTS / "eval_metrics.json").read_text())
SWEEP = pd.read_csv(REPORTS / "cost_sweep.csv")
COST_SUMMARY = json.loads((REPORTS / "cost_summary.json").read_text())
REVIEW_SUMMARY = json.loads((REPORTS / "review_summary.json").read_text())
STEP6_T = COST_SUMMARY["chosen_threshold"]
LOW0 = REVIEW_SUMMARY["chosen_band"]["t_low"]
HIGH0 = REVIEW_SUMMARY["chosen_band"]["t_high"]


def ordered(low: float, high: float) -> tuple[float, float]:
    return (low, high) if low <= high else (high, low)


def decide(score: float, low: float, high: float) -> str:
    if score >= high:
        return "Reject"
    return "Send to a person" if score >= low else "Pass"


# ---------- the gauge: the one bold element ----------
def gauge(low: float, high: float, score: float | None = None) -> str:
    low, high = ordered(low, high)
    pct = lambda x: f"{100 * min(max(x, 0), 1):.2f}%"
    marker = ""
    if score is not None:
        marker = (f'<div style="position:absolute;left:{pct(score)};top:-6px;bottom:-6px;width:3px;'
                  f'background:{INK};border-radius:2px" title="This hand: {score:.3f}"></div>'
                  f'<div style="position:absolute;left:{pct(score)};top:52px;transform:translateX(-50%);'
                  f'font-size:13px;white-space:nowrap">this hand {score:.2f}</div>')
    return f"""
<div role="img" aria-label="Score line: pass below {low:.2f}, person checks {low:.2f} to {high:.2f}, reject from {high:.2f}"
     style="font-family:Barlow,system-ui,sans-serif;color:{INK};padding:4px 0 28px">
  <div style="position:relative;height:40px;border-radius:6px;overflow:visible;display:flex">
    <div style="width:{pct(low)};background:{GREEN};border-radius:6px 0 0 6px"></div>
    <div style="width:{pct(high - low)};background:{YELLOW}"></div>
    <div style="flex:1;background:{BROWN};border-radius:0 6px 6px 0"></div>
    {marker}
  </div>
  <div style="display:flex;justify-content:space-between;font-size:14px;margin-top:6px">
    <span style="color:{GREEN};font-weight:600">Pass below {low:.2f}</span>
    <span style="color:#8a6d00;font-weight:600">Person checks {low:.2f}–{high:.2f}</span>
    <span style="color:{BROWN};font-weight:600">Reject from {high:.2f}</span>
  </div>
</div>"""


# ---------- check a hand ----------
def predict(image: np.ndarray | None):
    """Run the model once per upload; keep every box so the sliders can redraw without re-running it."""
    if image is None:
        return None
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Gradio gives RGB, Ultralytics expects BGR arrays
    result = MODEL.predict(bgr, conf=0.001, imgsz=640, device=pick_device(), verbose=False)[0]
    b = result.boxes
    return {"image": image, "cls": b.cls.cpu().numpy().astype(int), "conf": b.conf.cpu().numpy(),
            "xyxy": b.xyxy.cpu().numpy()}


def draw(state: dict | None, low: float, high: float):
    low, high = ordered(low, high)
    if state is None:
        return None, "Upload a photo of a hand of bananas to see the model's decision.", gauge(low, high)
    rotten = state["cls"] == ROTTEN_ID
    score = float(state["conf"][rotten].max()) if rotten.any() else 0.0
    canvas = state["image"].copy()
    for c, conf, (x1, y1, x2, y2) in zip(state["cls"], state["conf"], state["xyxy"].astype(int)):
        is_rotten = c == ROTTEN_ID
        if conf < (low if is_rotten else FRESH_DISPLAY_CONF):
            continue
        colour = (107, 62, 38) if is_rotten else (63, 125, 32)  # RGB of BROWN and GREEN
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(canvas, f"{NAMES[c].strip()} {conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    verdict = decide(score, low, high)
    text = (f"### {verdict}\n"
            f"Highest Rotten score: **{score:.3f}**. Brown boxes are rotten areas at or above the pass line "
            f"({low:.2f}); green boxes are fresh bananas found with at least {FRESH_DISPLAY_CONF:.0%} confidence.")
    return canvas, text, gauge(low, high, score)


# ---------- cost and review load ----------
def three_way(low: float, high: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Truth x decision on the 400 test images, and the same rates applied to 1,000 production hands."""
    s, rotten = SCORES.score.to_numpy(), SCORES.defective.to_numpy(bool)
    decisions = np.where(s >= high, "Reject", np.where(s >= low, "Send to a person", "Pass"))
    cols = ["Pass", "Send to a person", "Reject"]
    counts = pd.DataFrame({c: [int((decisions[rotten] == c).sum()), int((decisions[~rotten] == c).sum())]
                           for c in cols}, index=["Truly rotten", "Truly good"])
    per_1000 = counts.div(counts.sum(axis=1), axis=0).mul([1000 * BASE_RATE, 1000 * (1 - BASE_RATE)], axis=0)
    return counts.reset_index(names="Test images (400)"), per_1000.round(1).reset_index(names=f"Per 1,000 hands ({BASE_RATE:.0%} rotten)")


def load_view(low: float, high: float):
    low, high = ordered(low, high)
    o = outcome(SCORES, low, high)
    counts, per_1000 = three_way(low, high)
    kpis = f"""
| | At these lines |
|---|---|
| Expected cost per 1,000 hands | **€{o['cost_per_1000']:,.2f}** (errors €{o['error_cost_per_1000']:,.2f} + checking €{o['labour_cost_per_1000']:,.2f}) |
| Expected cost per day ({HANDS_PER_DAY:,} hands) | **€{o['cost_per_day']:,.0f}** |
| Share sent to a person | **{o['review_share']:.1%}**, about {o['reviews_per_day']:,.0f} hands a day |
| Checking time and people | {o['review_hours_per_day']:.1f} hours a day, {o['people_needed']:.2f} people on 8-hour shifts |
| Missed rotten hands per 1,000 | {o['missed_per_1000']:.1f} (€{COST_FN:.2f} each) |
| Good hands thrown away per 1,000 | {o['false_alarms_per_1000']:.1f} (€{COST_FP:.2f} each) |

Assumptions: {BASE_RATE:.0%} of hands rotten, {SECONDS_PER_CHECK} s per check at €{COST_PER_HOUR:.2f}/hour, the person decides correctly.
Doing nothing costs €{1000 * BASE_RATE * COST_FN:,.0f} per 1,000 hands; a person checking every hand needs about 8.7 people.
"""
    return kpis, counts, per_1000, cost_curve(low, high)


# ---------- cost curve ----------
def cost_curve(low: float, high: float):
    curve = SWEEP[SWEEP.threshold <= 1.0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve.threshold, curve.cost, color=INK, lw=2, label="Model alone: cost at each single line")
    pass_all = 1000 * BASE_RATE * COST_FN
    ax.axhline(pass_all, color="grey", ls=":", label=f"Pass everything: €{pass_all:.0f}")
    ax.scatter([STEP6_T], [curve.loc[np.isclose(curve.threshold, STEP6_T), "cost"].iloc[0]],
               color="red", zorder=5, s=50, label=f"Chosen line (step 6): {STEP6_T:.2f}")
    ax.axvspan(low, high, color=YELLOW, alpha=0.25, label=f"Current review band {low:.2f}–{high:.2f}")
    ax.set_ylim(0, pass_all * 1.5)
    ax.set_xlabel("Line on the Rotten score (reject at or above it)")
    ax.set_ylabel("Expected cost per 1,000 hands (€)")
    ax.set_title(f"Cost against threshold  |  miss €{COST_FN:.2f}, false alarm €{COST_FP:.2f}, {BASE_RATE:.0%} rotten")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------- per-class results (static, from evaluate.py) ----------
def per_class_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model, by_subset in METRICS["box_level"].items():
        for subset, m in by_subset.items():
            for cls, v in m["per_class"].items():
                rows.append({"Model": model, "Test set": subset, "Class": cls, "Boxes": v["instances"],
                             "Precision": v["precision"], "Recall": v["recall"],
                             "AP50": v["AP50"], "AP50-95": v["AP50-95"]})
    box = pd.DataFrame(rows)
    img = pd.DataFrame([{"Model": model, "Test set": subset, **{k: v[k] for k in
                         ("images", "defective", "TP", "FP", "FN", "TN", "precision", "recall", "roc_auc")}}
                        for model, by_subset in METRICS["image_level"].items() for subset, v in by_subset.items()])
    return box, img


BOX_TABLE, IMAGE_TABLE = per_class_tables()
HEADLINE = METRICS["ultralytics_val_finetuned"]["full"]
CONFUSION_PNG = REPORTS / "confusion_matrix_full.png"

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Barlow"), "system-ui", "sans-serif"],
    primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.stone,
)
CSS = f"""
.gradio-container {{max-width: 1100px !important}}
h1 {{font-weight: 700; letter-spacing: -0.01em}}
"""

with gr.Blocks(title="Banana hand inspection", **({} if int(gr.__version__.split(".")[0]) >= 6 else {"theme": THEME, "css": CSS})) as demo:
    gr.Markdown(
        "# Banana hand inspection\n"
        "A YOLO11n model scores each hand of bananas for rot. Two lines on that score decide what happens "
        "to the hand: it passes, a person checks it, or it is rejected. Move the lines to see what each "
        "choice costs and how much work it sends to people.")
    with gr.Row():
        low = gr.Slider(0, 1, value=LOW0, step=0.01, label="Pass below this score")
        high = gr.Slider(0, 1, value=HIGH0, step=0.01, label="Reject at or above this score")
    gauge_html = gr.HTML(gauge(LOW0, HIGH0))
    hand_state = gr.State(None)

    with gr.Tab("Check a hand"):
        with gr.Row():
            upload = gr.Image(type="numpy", label="Photo of a hand of bananas", sources=["upload", "clipboard"])
            shown = gr.Image(type="numpy", label="What the model found", interactive=False)
        verdict = gr.Markdown("Upload a photo of a hand of bananas to see the model's decision.")

    with gr.Tab("Cost and review load"):
        kpis = gr.Markdown()
        with gr.Row():
            matrix_test = gr.Dataframe(label="Decisions on the test images")
            matrix_prod = gr.Dataframe(label="The same rates on 1,000 production hands")
        gr.Markdown("Policies compared at the chosen lines (from the pipeline):")
        gr.Dataframe(pd.read_csv(REPORTS / "review_policies.csv", index_col=0)[
            ["missed_per_1000", "false_alarms_per_1000", "review_share", "people_needed", "cost_per_day"]
        ].round(3).reset_index(names="Policy"))

    with gr.Tab("Cost curve"):
        curve_plot = gr.Plot()
        gr.Markdown(
            f"The curve sweeps a single line (model alone). Its cheapest point, {STEP6_T:.2f}, is the step 6 "
            f"choice: €{COST_SUMMARY['at_chosen_threshold']['cost']:.2f} per 1,000 hands against "
            f"€{1000 * BASE_RATE * COST_FN:.0f} for passing everything. The yellow band is your current review band.")

    with gr.Tab("Per-class results"):
        gr.Markdown(
            f"Positive class: **Rotten**. Headline on the full test set (Ultralytics validator): "
            f"mAP50 **{HEADLINE['mAP50']}**, mAP50-95 **{HEADLINE['mAP50-95']}**. That average hides the gap "
            f"between classes: Fresh boxes are whole bananas and are found almost perfectly; Rotten boxes are "
            f"mostly small spots, about half of which are found. Per hand, which is the decision that matters, "
            f"every rotten test hand was caught. 'unseen' is the 38 test images from photos never used in training.")
        gr.Dataframe(BOX_TABLE, label="Box level, per class")
        gr.Dataframe(IMAGE_TABLE, label="Image level (one decision per hand) at an untuned line of 0.25")
        if CONFUSION_PNG.is_file():
            gr.Image(str(CONFUSION_PNG), label="Box-level confusion matrix, full test set", interactive=False)

    def on_lines(state, lo, hi):
        image, text, g = draw(state, lo, hi)
        return (image, text, g, *load_view(lo, hi))

    line_outputs = [shown, verdict, gauge_html, kpis, matrix_test, matrix_prod, curve_plot]
    upload.change(predict, upload, hand_state).then(on_lines, [hand_state, low, high], line_outputs)
    low.change(on_lines, [hand_state, low, high], line_outputs)
    high.change(on_lines, [hand_state, low, high], line_outputs)
    demo.load(on_lines, [hand_state, low, high], line_outputs)


if __name__ == "__main__":
    launch_kwargs = {"server_port": int(os.environ.get("APP_PORT", 7860))}
    if int(gr.__version__.split(".")[0]) >= 6:
        launch_kwargs.update(theme=THEME, css=CSS)
    demo.launch(**launch_kwargs)
