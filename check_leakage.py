"""
Leakage check by file name.

    uv run python check_leakage.py

This dataset was uploaded already augmented, and the file names show it:
"rotated_by_60_Screen-Shot-2018-...", "saltandpepper_Screen-Shot-2018-...".
The part from "Screen-Shot" on names the original photo; Roboflow adds
".rf.<hash>" at the end. Copies of the same photo therefore share that part,
which lets us measure leakage exactly, including rotated copies that the
image-hash check in describe_data.py cannot see.

Writes reports/name_leakage.csv: one row per image with its original photo and
whether that photo also appears in train.
"""

import pandas as pd

from data import DATA_DIR, ROOT, SPLIT_DIRS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANCHOR = "Screen-Shot"


def split_name(stem: str) -> tuple[str, str]:
    """(augmentation prefix, original photo) from a Roboflow file stem."""
    base = stem.split(".rf.")[0]
    i = base.find(ANCHOR)
    if i == -1:
        return "(no anchor)", base
    return (base[:i] or "(none)"), base[i:]


def main() -> None:
    rows = []
    for split, folder in SPLIT_DIRS.items():
        for p in sorted((DATA_DIR / folder / "images").iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                prefix, original = split_name(p.stem)
                rows.append({"split": split, "image": p.name, "prefix": prefix, "original": original})
    df = pd.DataFrame(rows)

    print("\n== Augmentation prefixes in file names (digits shown as N) ==")
    print(df.prefix.str.replace(r"\d+", "N", regex=True).value_counts().to_string())

    print("\n== Distinct original photos per split ==")
    print(df.groupby("split", sort=False).agg(images=("image", "count"), originals=("original", "nunique")).to_string())
    copies = df.groupby("original").size()
    print(f"\nCopies per original photo: median {copies.median():.0f}, max {copies.max()}")

    train_originals = set(df.loc[df.split == "train", "original"])
    df["original_in_train"] = df.original.isin(train_originals) & (df.split != "train")
    print("\n== Leakage: images whose original photo also appears in train ==")
    for split in ("val", "test"):
        s = df[df.split == split]
        n = int(s.original_in_train.sum())
        print(f"{split}: {n} of {len(s)} images ({n / len(s):.1%}) | "
              f"{len(s) - n} images from photos never seen in training")

    out = ROOT / "reports" / "name_leakage.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
