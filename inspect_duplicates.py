"""
Show flagged near-duplicate pairs side by side, to check by eye whether they
are really the same photo.

    uv run python inspect_duplicates.py

Writes reports/duplicate_pairs.png: each row is a test image (left) and the
train image it was matched to (right).
"""

import cv2
import numpy as np
import pandas as pd

from data import DATA_DIR, ROOT, SPLIT_DIRS

N_PAIRS = 12
THUMB = 220

pairs = pd.read_csv(ROOT / "reports" / "duplicate_pairs.csv")
pairs = pairs[pairs.split == "test"].drop_duplicates("image").head(N_PAIRS)


def thumb(folder: str, name: str) -> np.ndarray:
    image = cv2.imread(str(DATA_DIR / folder / "images" / name))
    return cv2.resize(image, (THUMB, THUMB))


rows = []
for _, pair in pairs.iterrows():
    left = thumb(SPLIT_DIRS["test"], pair.image)
    right = thumb(SPLIT_DIRS["train"], pair.train_image)
    gap = np.full((THUMB, 10, 3), 255, np.uint8)
    row = np.hstack([left, gap, right])
    cv2.putText(row, f"{pair.bits_apart} bits", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    rows.append(row)

if rows:
    out = ROOT / "reports" / "duplicate_pairs.png"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"Wrote {out} with {len(rows)} pairs (test left, train right).")
else:
    print("No test pairs flagged.")
