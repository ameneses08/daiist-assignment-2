"""
Steps 4 and 5: evaluate the finetuned model against two baselines on the same test split.

    uv run python evaluate.py        # also runs automatically at the end of train.py

Models compared:
  finetuned   weights/best.pt
  coco        the COCO-pretrained yolo11s used zero-shot; COCO has "banana" but no
              notion of rotten, so banana detections count as Fresh
  trivial     predicts nothing; at image level that means "pass every hand",
              which is also the majority class of the test set

Each is scored on the full test split (400 images, used for decisions) and on the
unseen subset: test images from original photos that never appear in train
(Decision #5, revised; 38 images, reported as a leakage check), at two levels:
  box level    per-class precision, recall, AP50, AP50-95 (the README's metrics)
  image level  one case per image (Decision #4): truly defective if any Rotten
               label, score = highest Rotten confidence predicted

Outputs:
  reports/eval_metrics.json    every number above
  reports/image_scores.csv     one row per test image and model: truth, score, unseen flag
  reports/confusion_matrix_*.png   Ultralytics' box-level confusion matrices (finetuned)
"""

import json
import shutil

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

from data import DATA_DIR, ROOT, SPLIT_DIRS, class_names, download_dataset, pick_device

WEIGHTS = ROOT / "weights" / "best.pt"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
COCO_MODEL = "yolo11s.pt"  # same size as the finetuned model, so the comparison is fair
COCO_TO_OURS = {"banana": "Fresh"}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # AP50 uses the first, AP50-95 averages all ten
PRED_CONF = 0.001  # keep almost every box, as Ultralytics' own validation does
SNAPSHOT_CONF = 0.25  # Ultralytics' default threshold; an untuned snapshot only, step 6 picks the real one
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH = 16  # images per forward pass; keeps GPU memory small


# ---------- data ----------

def test_images() -> list:
    folder = DATA_DIR / SPLIT_DIRS["test"] / "images"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def unseen_names() -> set:
    """Test images from original photos that never appear in train (Decision #5, revised).

    The file names keep the original photo's name (see check_leakage.py), so this
    catches rotated, flipped and noisy copies that an image-hash check cannot.
    """
    leakage_csv = REPORTS / "name_leakage.csv"
    if not leakage_csv.is_file():
        import check_leakage

        check_leakage.main()
    names = pd.read_csv(leakage_csv)
    test = names[names.split == "test"]
    return set(test.loc[~test.original_in_train, "image"])


def load_truth(image_path) -> np.ndarray:
    """Ground-truth boxes as rows of (class, x1, y1, x2, y2), normalised to 0-1."""
    label = DATA_DIR / SPLIT_DIRS["test"] / "labels" / f"{image_path.stem}.txt"
    rows = []
    if label.is_file():
        for line in label.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls, v = int(parts[0]), list(map(float, parts[1:]))
            if len(v) == 4:
                cx, cy, w, h = v
                rows.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            else:  # polygon label: use its bounding box
                xs, ys = v[0::2], v[1::2]
                rows.append((cls, min(xs), min(ys), max(xs), max(ys)))
    return np.array(rows, dtype=float).reshape(-1, 5)


def predict(model, paths, class_map) -> dict:
    """Predictions per image as rows of (class, conf, x1, y1, x2, y2), in our class ids.

    class_map maps the model's class id to ours; ids missing from it are dropped.
    Images go in chunks of BATCH: given a whole list at once, Ultralytics runs it as
    one batch, which needs tens of GB of GPU memory for 400 images.
    """
    out = {}
    for start in range(0, len(paths), BATCH):
        chunk = paths[start:start + BATCH]
        results = model.predict([str(p) for p in chunk], conf=PRED_CONF, iou=0.7, imgsz=640,
                                batch=BATCH, device=pick_device(), verbose=False)
        for path, result in zip(chunk, results):
            b = result.boxes
            cls, conf, xyxy = b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy(), b.xyxyn.cpu().numpy()
            keep = np.array([c in class_map for c in cls], dtype=bool)
            mapped = np.array([class_map[c] for c in cls[keep]], dtype=float)
            out[path.name] = np.column_stack([mapped, conf[keep], xyxy[keep]]).reshape(-1, 6)
    free_gpu_memory()
    return out


def free_gpu_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- box level ----------

def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (n,4) and b (m,4). Images are all 640x640, so
    normalised coordinates give the same IoU as pixels."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """For each prediction and IoU threshold: is it a true positive?

    Highest-confidence predictions claim ground-truth boxes first; each ground-truth
    box can be claimed once, and only by a prediction of the same class.
    """
    tp = np.zeros((len(pred), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(pred) or not len(truth):
        return tp
    iou = box_iou(pred[:, 2:], truth[:, 1:])
    iou[pred[:, 0][:, None] != truth[:, 0][None, :]] = 0
    order = np.argsort(-pred[:, 1])
    for t, threshold in enumerate(IOU_THRESHOLDS):
        claimed = np.zeros(len(truth), dtype=bool)
        for i in order:
            candidates = np.where(~claimed & (iou[i] >= threshold), iou[i], -1)
            j = candidates.argmax()
            if candidates[j] >= 0:
                tp[i, t], claimed[j] = True, True
    return tp


def box_metrics(preds: dict, truths: dict, names: list, subset: set) -> dict:
    tps, confs, pred_cls, target_cls = [], [], [], []
    for name in subset:
        pred, truth = preds.get(name, np.zeros((0, 6))), truths[name]
        tps.append(match(pred, truth))
        confs.append(pred[:, 1])
        pred_cls.append(pred[:, 0])
        target_cls.append(truth[:, 0])
    tp, conf, pcls, tcls = (np.concatenate(x) for x in (tps, confs, pred_cls, target_cls))
    per_class = {n: {"instances": int((tcls == i).sum()), "precision": 0.0, "recall": 0.0,
                     "AP50": 0.0, "AP50-95": 0.0} for i, n in enumerate(names)}
    if len(conf):
        _, _, p, r, _, ap, classes, *_ = ap_per_class(tp, conf, pcls, tcls)
        for k, c in enumerate(classes):
            per_class[names[c]].update(precision=round(float(p[k]), 4), recall=round(float(r[k]), 4),
                                       AP50=round(float(ap[k, 0]), 4),
                                       **{"AP50-95": round(float(ap[k].mean()), 4)})
    return {
        "mAP50": round(float(np.mean([v["AP50"] for v in per_class.values()])), 4),
        "mAP50-95": round(float(np.mean([v["AP50-95"] for v in per_class.values()])), 4),
        "per_class": per_class,
    }


# ---------- image level ----------

def image_scores(preds: dict, truths: dict, rotten_id: int, unseen: set, model_name: str) -> pd.DataFrame:
    rows = []
    for name, truth in truths.items():
        pred = preds.get(name, np.zeros((0, 6)))
        rotten = pred[pred[:, 0] == rotten_id]
        rows.append({"model": model_name, "image": name, "unseen": name in unseen,
                     "defective": bool((truth[:, 0] == rotten_id).any()),
                     "score": float(rotten[:, 1].max()) if len(rotten) else 0.0})
    return pd.DataFrame(rows)


def roc_auc(defective: pd.Series, score: pd.Series) -> float:
    """Probability that a random defective image scores above a random good one."""
    ranks = score.rank()
    n_pos, n_neg = int(defective.sum()), int((~defective).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[defective].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def image_summary(df: pd.DataFrame, threshold: float) -> dict:
    flagged = df.score >= threshold
    tp = int((flagged & df.defective).sum())
    fp = int((flagged & ~df.defective).sum())
    fn = int((~flagged & df.defective).sum())
    tn = int((~flagged & ~df.defective).sum())
    return {
        "images": len(df), "defective": int(df.defective.sum()), "threshold": threshold,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "accuracy": round((tp + tn) / len(df), 4),
        "roc_auc": round(roc_auc(df.defective, df.score), 4),
    }


# ---------- Ultralytics' own validation, as a cross-check and for its plots ----------

def ultralytics_val(model, names: list, subset_name: str, paths: list) -> dict:
    listing = DATA_DIR / f"test_{subset_name}.txt"
    listing.write_text("\n".join(str(p) for p in paths))
    cfg = yaml.safe_load((DATA_DIR / "data_local.yaml").read_text())
    cfg["test"] = str(listing)
    cfg_path = DATA_DIR / f"data_test_{subset_name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    m = model.val(data=str(cfg_path), split="test", imgsz=640, batch=BATCH, device=pick_device(), plots=True,
                  project=str(RUNS), name=f"test_{subset_name}", exist_ok=True, verbose=False)
    for plot in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = m.save_dir / plot
        if src.is_file():
            shutil.copy(src, REPORTS / plot.replace("confusion_matrix", f"confusion_matrix_{subset_name}"))
    per_class = {names[c]: {"precision": round(float(m.box.p[k]), 4), "recall": round(float(m.box.r[k]), 4),
                            "AP50": round(float(m.box.ap50[k]), 4), "AP50-95": round(float(m.box.ap[k]), 4)}
                 for k, c in enumerate(m.box.ap_class_index)}
    return {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
            "per_class": per_class,
            "confusion_matrix": {"labels": names + ["background"],
                                 "rows_are": "predicted", "columns_are": "true",
                                 "matrix": m.confusion_matrix.matrix.astype(int).tolist()}}


# ---------- main ----------

def main() -> None:
    download_dataset()
    names = class_names(DATA_DIR / "data_local.yaml")
    rotten_id = names.index("Rotten")
    REPORTS.mkdir(exist_ok=True)

    paths = test_images()
    unseen = unseen_names()
    truths = {p.name: load_truth(p) for p in paths}
    subsets = {"full": set(truths), "unseen": unseen}
    print(f"Test images: {len(paths)} full, {len(unseen)} from photos never seen in training")

    finetuned = YOLO(str(WEIGHTS))
    coco = YOLO(COCO_MODEL)
    coco_map = {cid: names.index(ours) for cid, coco_name in coco.names.items()
                for key, ours in COCO_TO_OURS.items() if coco_name == key}
    predictions = {
        "finetuned": predict(finetuned, paths, {i: i for i in range(len(names))}),
        "coco": predict(coco, paths, coco_map),
        "trivial": {},
    }

    results = {"classes": names, "positive_class": "Rotten",
               "subsets": {k: len(v) for k, v in subsets.items()}, "box_level": {}, "image_level": {}}
    score_frames = []
    for model_name, preds in predictions.items():
        scores = image_scores(preds, truths, rotten_id, unseen, model_name)
        score_frames.append(scores)
        for subset_name, subset in subsets.items():
            results["box_level"].setdefault(model_name, {})[subset_name] = box_metrics(preds, truths, names, subset)
            results["image_level"].setdefault(model_name, {})[subset_name] = image_summary(
                scores[scores.image.isin(subset)], SNAPSHOT_CONF)

    results["ultralytics_val_finetuned"] = {
        name: ultralytics_val(finetuned, names, name, [p for p in paths if p.name in subset])
        for name, subset in subsets.items()}

    pd.concat(score_frames).to_csv(REPORTS / "image_scores.csv", index=False)
    (REPORTS / "eval_metrics.json").write_text(json.dumps(results, indent=2))
    print_summary(results)


def print_summary(r: dict) -> None:
    print("\n== Box level (README metrics) ==")
    print("All three models scored by the same evaluator in this file, so they compare fairly.")
    print("The Ultralytics line is its official validator on the finetuned model: quote it as the")
    print("headline number, and use the gap to the 'finetuned' row as a consistency check.")
    for subset in ("full", "unseen"):
        print(f"\n[{subset} test]")
        rows = []
        for model, by_subset in r["box_level"].items():
            m = by_subset[subset]
            row = {"model": model, "mAP50": m["mAP50"], "mAP50-95": m["mAP50-95"]}
            for cls, v in m["per_class"].items():
                row.update({f"{cls} P": v["precision"], f"{cls} R": v["recall"], f"{cls} AP50": v["AP50"]})
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
        u = r["ultralytics_val_finetuned"][subset]
        print(f"Ultralytics cross-check (finetuned): mAP50 {u['mAP50']}, mAP50-95 {u['mAP50-95']}")

    print(f"\n== Image level (positive = Rotten), untuned snapshot at conf {SNAPSHOT_CONF} ==")
    for subset in ("full", "unseen"):
        print(f"\n[{subset} test]")
        rows = [{"model": m, **{k: v for k, v in s[subset].items() if k != "threshold"}}
                for m, s in r["image_level"].items()]
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nSaved reports/eval_metrics.json, reports/image_scores.csv, reports/confusion_matrix_*.png")


if __name__ == "__main__":
    main()
