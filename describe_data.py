"""
Step 3: describe the dataset before training on it.

    uv run python describe_data.py

Computes, from the label files themselves:
  - images per split, and how many contain each class or no labels at all
  - boxes (instances) per class per split, and boxes per image
  - box size relative to the image, with COCO small/medium/large buckets
  - near-duplicate images across splits (a leakage check)

Writes reports/dataset_stats.json, reports/box_sizes.png and
reports/duplicate_pairs.csv so REPORT.md and the dashboard can use them.
"""

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import DATA_DIR, ROOT, SPLIT_DIRS, class_names, download_dataset

REPORTS = ROOT / "reports"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMG_SIZE = 640  # Roboflow stretched every image to 640x640
# COCO size buckets, in pixels of area at 640x640
SMALL_MAX = 32**2
MEDIUM_MAX = 96**2
DUPLICATE_MAX_BITS = 4  # dHash Hamming distance at or below this counts as a near-duplicate


def read_boxes(label_file: Path) -> list[tuple[int, float, float]]:
    """Return (class_id, width, height) per box, both as fractions of the image."""
    if not label_file.is_file():
        return []
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, values = int(parts[0]), list(map(float, parts[1:]))
        if len(values) == 4:  # cx cy w h
            w, h = values[2], values[3]
        else:  # polygon: x1 y1 x2 y2 ... -> its bounding box
            xs, ys = values[0::2], values[1::2]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        boxes.append((cls, w, h))
    return boxes


def dhash(image_path: Path) -> int:
    """64-bit difference hash: stays the same under resizing and mild compression."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, bit in enumerate(bits) if bit))


def collect(names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_rows, box_rows = [], []
    for split, folder in SPLIT_DIRS.items():
        image_dir = DATA_DIR / folder / "images"
        label_dir = DATA_DIR / folder / "labels"
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            boxes = read_boxes(label_dir / f"{image_path.stem}.txt")
            row = {"split": split, "image": image_path.name, "path": str(image_path),
                   "n_boxes": len(boxes), "hash": dhash(image_path)}
            for cid, name in enumerate(names):
                row[f"has_{name}"] = any(b[0] == cid for b in boxes)
            image_rows.append(row)
            for cid, w, h in boxes:
                box_rows.append({"split": split, "image": image_path.name,
                                 "class": names[cid], "area_frac": w * h})
    return pd.DataFrame(image_rows), pd.DataFrame(box_rows)


def size_bucket(area_frac: float) -> str:
    pixels = area_frac * IMG_SIZE**2
    if pixels < SMALL_MAX:
        return "small"
    return "medium" if pixels < MEDIUM_MAX else "large"


def near_duplicates(images: pd.DataFrame) -> pd.DataFrame:
    """Pairs of images in different splits whose hashes are nearly identical."""
    pairs = []
    train = images[images.split == "train"]
    train_hashes = np.array(train.hash.tolist(), dtype=np.uint64)
    for split in ("val", "test"):
        for _, row in images[images.split == split].iterrows():
            distances = np.bitwise_count(train_hashes ^ np.uint64(row.hash))
            for j in np.flatnonzero(distances <= DUPLICATE_MAX_BITS):
                pairs.append({"split": split, "image": row.image,
                              "train_image": train.iloc[j].image, "bits_apart": int(distances[j])})
    return pd.DataFrame(pairs, columns=["split", "image", "train_image", "bits_apart"])


def main() -> None:
    names = class_names(download_dataset())
    REPORTS.mkdir(exist_ok=True)
    images, boxes = collect(names)
    boxes["bucket"] = boxes.area_frac.map(size_bucket)
    pd.set_option("display.width", 120)

    print(f"\nClasses: {names}\n")

    print("== Images per split ==")
    per_split = images.groupby("split", sort=False).agg(
        images=("image", "count"),
        no_labels=("n_boxes", lambda s: int((s == 0).sum())),
        **{f"with_{n}": (f"has_{n}", "sum") for n in names},
    )
    print(per_split, "\n")

    print("== Boxes (instances) per class per split ==")
    instances = boxes.pivot_table(index="class", columns="split", values="image",
                                  aggfunc="count", fill_value=0)[list(SPLIT_DIRS)]
    print(instances, "\n")

    print("== Boxes per image ==")
    print(images.groupby("split", sort=False).n_boxes.describe()[["mean", "50%", "max"]], "\n")

    print("== Box area as % of the image, per class ==")
    pct = boxes.groupby("class").area_frac.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print((pct[["5%", "25%", "50%", "75%", "95%"]] * 100).round(2), "\n")

    print("== COCO size buckets (small < 32x32 px, large >= 96x96 px at 640x640) ==")
    buckets = pd.crosstab(boxes["class"], boxes.bucket, normalize="index").reindex(
        columns=["small", "medium", "large"], fill_value=0)
    print((buckets * 100).round(1), "\n")

    dups = near_duplicates(images)
    print(f"== Near-duplicates of train images (dHash <= {DUPLICATE_MAX_BITS} bits) ==")
    for split in ("val", "test"):
        n_images = int((images.split == split).sum())
        n_dup = dups[dups.split == split].image.nunique()
        print(f"{split}: {n_dup} of {n_images} images ({100 * n_dup / max(n_images, 1):.1f}%) "
              f"have a near-identical image in train")
    print("Pairs written to reports/duplicate_pairs.csv; open a few to check by eye.\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in names:
        ax.hist(boxes.loc[boxes["class"] == name, "area_frac"] * 100, bins=50, alpha=0.6, label=name)
    ax.set_xlabel("Box area (% of image)")
    ax.set_ylabel("Number of boxes")
    ax.set_title("Box size by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "box_sizes.png", dpi=150)

    dups.to_csv(REPORTS / "duplicate_pairs.csv", index=False)
    stats = {
        "dataset": {"workspace": "fruit-pnemw", "project": "fruit-jmtdq", "version": 1},
        "classes": names,
        "images_per_split": per_split.astype(int).to_dict(orient="index"),
        "instances_per_class": instances.astype(int).to_dict(orient="index"),
        "box_area_pct_percentiles": (pct[["5%", "25%", "50%", "75%", "95%"]] * 100)
        .round(3).to_dict(orient="index"),
        "size_buckets_pct": (buckets * 100).round(2).to_dict(orient="index"),
        "near_duplicates": {s: int(dups[dups.split == s].image.nunique()) for s in ("val", "test")},
    }
    (REPORTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print("Saved reports/dataset_stats.json and reports/box_sizes.png")


if __name__ == "__main__":
    main()
