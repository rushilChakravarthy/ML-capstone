"""
Download the project datasets from Kaggle into data/raw/.

  used-cars   (regression)      https://www.kaggle.com/datasets/thedevastator/uncovering-factors-that-affect-used-car-prices
                                -> data/raw/                    (autos.csv)
  vehicle-reg (clustering)      https://www.kaggle.com/datasets/aatifahmad123/indian-vehicle-registration-data-202025
                                -> data/raw/vehicle_registration/

(The classification dataset, train.csv, is committed directly.)

Setup (one-time):
    1. pip install kaggle
    2. Create a Kaggle API token: https://www.kaggle.com/settings -> "Create New Token" (downloads kaggle.json)
    3. Place it at ~/.kaggle/kaggle.json (Linux/Mac) or C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows);
       on Linux/Mac also:  chmod 600 ~/.kaggle/kaggle.json

Usage:
    python data/download_dataset.py                 # all downloadable datasets
    python data/download_dataset.py vehicle-reg     # just the clustering dataset
    python data/download_dataset.py used-cars
"""

import os
import sys
import zipfile
from pathlib import Path

RAW = Path(__file__).parent / "raw"
DATASETS = {
    "used-cars": ("thedevastator/uncovering-factors-that-affect-used-car-prices", RAW),
    "vehicle-reg": ("aatifahmad123/indian-vehicle-registration-data-202025", RAW / "vehicle_registration"),
}


def download(api, slug, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {slug} -> {out_dir} ...")
    api.dataset_download_files(slug, path=str(out_dir), unzip=False)
    for zip_path in out_dir.glob("*.zip"):
        print(f"  extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        os.remove(zip_path)
    for f in sorted(out_dir.iterdir()):
        print("  -", f.name)


def main():
    wanted = sys.argv[1:] or list(DATASETS)
    unknown = [w for w in wanted if w not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s) {unknown}. Choose from: {list(DATASETS)}")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit("The 'kaggle' package is not installed. Run: pip install kaggle") from exc
    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json
    for key in wanted:
        download(api, *DATASETS[key])


if __name__ == "__main__":
    main()
