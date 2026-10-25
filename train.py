"""
Stage: train.

    uv run python main.py train          # the real pipeline, final settings
    uv run python train.py --epochs 1    # quick smoke test to time one epoch

Runs the whole pipeline: downloads the dataset (if needed), describes it,
finetunes a COCO-pretrained YOLO11 model, copies the best checkpoint to
weights/best.pt, records the settings in reports/training_config.json, and
evaluates the model against the baselines (evaluate.py), and sweeps the
threshold against the costs (costing.py), and works out the human review
band and staffing (review.py).
"""

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

import check_leakage
import costing
import describe_data
import evaluate
import review
from data import PROJECT, ROOT, VERSION, WORKSPACE, download_dataset, pick_device

WEIGHTS_DIR = ROOT / "weights"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"

# Final settings. main.py runs this file with no arguments, so these defaults
# are what the submitted model is trained with.
DEFAULTS = {
    "model": "yolo11n.pt",  # COCO-pretrained starting point (Decision #6, revised from yolo11s for time)
    "epochs": 30,
    "patience": 10,  # stop early if val mAP has not improved for this many epochs
    "imgsz": 640,  # native size of the dataset; smaller would shrink the rotten spots
    "batch": 16,
    "seed": 0,
}


def epoch_summary(results_csv: Path) -> dict:
    """Epochs run and the epoch with the highest validation mAP50-95, from ultralytics' log."""
    log = pd.read_csv(results_csv)
    log.columns = log.columns.str.strip()
    best_row = log.loc[log["metrics/mAP50-95(B)"].idxmax()]
    return {
        "epochs_completed": int(log["epoch"].max()),
        "best_epoch": int(best_row["epoch"]),
        "best_val_mAP50-95": round(float(best_row["metrics/mAP50-95(B)"]), 4),
        "best_val_mAP50": round(float(best_row["metrics/mAP50(B)"]), 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, value in DEFAULTS.items():
        parser.add_argument(f"--{name}", type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = download_dataset()
    describe_data.main()  # step 3 stats, plus the image-hash near-duplicate check
    check_leakage.main()  # exact leakage by original photo name; defines the unseen test subset
    device = pick_device()
    print(f"Training {args.model} on {device} for up to {args.epochs} epochs")

    model = YOLO(args.model)
    started = time.time()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        device=device,
        project=str(RUNS),
        name="finetune",
        exist_ok=True,  # overwrite the previous run instead of finetune2, finetune3, ...
        plots=True,
    )
    minutes = (time.time() - started) / 60

    best = Path(model.trainer.best)
    WEIGHTS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS_DIR / "best.pt")

    REPORTS.mkdir(exist_ok=True)
    config = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        **vars(args),
        "device": device,
        "machine": f"{platform.system()} {platform.machine()}",
        "torch": torch.__version__,
        "training_minutes": round(minutes, 1),
        **epoch_summary(best.parent.parent / "results.csv"),
    }
    (REPORTS / "training_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nDone in {minutes:.1f} min. Best weights -> {WEIGHTS_DIR / 'best.pt'}")

    evaluate.main()  # steps 4-5: finetuned vs baselines, on the full and clean test sets
    costing.main()  # step 6: cost sweep and chosen threshold
    review.main()  # step 7: pass / human review / reject, staffing and policy comparison


if __name__ == "__main__":
    main()
