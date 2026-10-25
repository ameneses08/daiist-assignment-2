"""
Dataset download and preparation, shared by train.py and describe_data.py.

Downloads the public Roboflow Universe dataset as-is (no fork), so anyone
with a Roboflow key gets exactly the data this pipeline was built on.
"""

from pathlib import Path

import yaml

from api_key import load_api_key

ROOT = Path(__file__).resolve().parent

# The exact dataset this project uses. These three values go in REPORT.md.
WORKSPACE = "fruit-pnemw"
PROJECT = "fruit-jmtdq"
VERSION = 1

EXPORT_FORMAT = "yolov11"  # YOLO txt labels + data.yaml, what ultralytics expects
DATA_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"

# Roboflow names the validation folder "valid"; ultralytics calls the split "val".
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


def download_dataset(force: bool = False) -> Path:
    """Download the dataset once, then reuse it. Returns the path to a clean data.yaml."""
    original_yaml = DATA_DIR / "data.yaml"
    if original_yaml.is_file() and not force:
        print(f"Dataset already at {DATA_DIR}, skipping download.")
    else:
        from roboflow import Roboflow

        print(f"Downloading {WORKSPACE}/{PROJECT} version {VERSION} ...")
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
        version.download(EXPORT_FORMAT, location=str(DATA_DIR), overwrite=True)
    return write_local_yaml(original_yaml)


def write_local_yaml(original_yaml: Path) -> Path:
    """Write data_local.yaml next to Roboflow's data.yaml, leaving the original untouched.

    Two fixes:
    1. Roboflow's paths ("../train/images") are relative in a way ultralytics
       does not always resolve. We give an absolute root plus split folders.
    2. Class names are stripped of stray whitespace ("Rotten " -> "Rotten"),
       because names are compared as exact strings later on.
    """
    cfg = yaml.safe_load(original_yaml.read_text(encoding="utf-8"))

    names = cfg["names"]
    if isinstance(names, dict):  # {0: "Fresh", 1: "Rotten"} style
        names = [names[i] for i in sorted(names)]
    raw_names = [str(n) for n in names]
    clean_names = [n.strip() for n in raw_names]
    if raw_names != clean_names:
        print(f"Note: stripped whitespace from class names {raw_names} -> {clean_names}")

    for split, folder in SPLIT_DIRS.items():
        if not (DATA_DIR / folder / "images").is_dir():
            raise FileNotFoundError(f"Missing {split} split at {DATA_DIR / folder / 'images'}")

    local_cfg = {
        "path": str(DATA_DIR),
        "train": f"{SPLIT_DIRS['train']}/images",
        "val": f"{SPLIT_DIRS['val']}/images",
        "test": f"{SPLIT_DIRS['test']}/images",
        "nc": len(clean_names),
        "names": clean_names,
    }
    local_yaml = DATA_DIR / "data_local.yaml"
    local_yaml.write_text(yaml.safe_dump(local_cfg, sort_keys=False), encoding="utf-8")
    return local_yaml


def pick_device() -> str:
    """CUDA if present, else Apple GPU, else CPU, so the pipeline runs on any machine."""
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def class_names(local_yaml: Path) -> list[str]:
    return yaml.safe_load(local_yaml.read_text(encoding="utf-8"))["names"]


if __name__ == "__main__":
    path = download_dataset()
    print(f"Ready: {path}")
    print(f"Classes: {class_names(path)}")
