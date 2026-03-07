"""
Combine per-letter Myo armband CSV files from multiple users into a single
training-ready dataset CSV.

Usage (from project root):
    python training/combine_dataset.py
"""

import os
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configurable paths — adjust these if you move the raw data or want a
# different output location.
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("Datasets/dyfav")
OUTPUT_PATH = Path("Datasets/combined_dataset.csv")

# The 17 sensor columns present in every raw CSV (no header row).
SENSOR_COLUMNS = [
    "emg_1", "emg_2", "emg_3", "emg_4",
    "emg_5", "emg_6", "emg_7", "emg_8",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "orient_roll", "orient_pitch", "orient_yaw",
]

# Regex to extract the letter from filenames like "291982754_alphabet_a_right.csv"
FILENAME_PATTERN = re.compile(r"_alphabet_([a-zA-Z])_")

# Regex to extract the user number from folder names like "User1"
USER_FOLDER_PATTERN = re.compile(r"(\d+)")


def parse_label(filename: str) -> str | None:
    """Return the lowercase letter label from a filename, or None on failure."""
    m = FILENAME_PATTERN.search(filename)
    if m:
        return m.group(1).lower()
    return None


def parse_user_id(folder_name: str) -> int | None:
    """Return the integer user ID from a folder name, or None on failure."""
    m = USER_FOLDER_PATTERN.search(folder_name)
    if m:
        return int(m.group(1))
    return None


def combine_dataset() -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {DATASET_ROOT.resolve()}"
        )

    all_frames: list[pd.DataFrame] = []
    file_id = 0

    # Sort user folders for deterministic output order
    user_dirs = sorted(
        [d for d in DATASET_ROOT.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    for user_dir in user_dirs:
        user_id = parse_user_id(user_dir.name)
        if user_id is None:
            print(f"  [WARN] Skipping folder with unparseable name: {user_dir.name}")
            continue

        print(f"Processing {user_dir.name}...")

        csv_files = sorted(user_dir.glob("*.csv"))
        if not csv_files:
            print(f"  [WARN] No CSV files found in {user_dir.name}")
            continue

        for csv_path in csv_files:
            label = parse_label(csv_path.name)
            if label is None:
                print(f"  [WARN] Filename doesn't match expected pattern, skipping: {csv_path.name}")
                continue

            try:
                df = pd.read_csv(csv_path, header=None, names=SENSOR_COLUMNS)
            except Exception as exc:
                print(f"  [WARN] Could not parse {csv_path}, skipping: {exc}")
                continue

            if df.empty:
                print(f"  [WARN] Empty file, skipping: {csv_path.name}")
                continue

            df["label"] = label
            df["user_id"] = user_id
            df["file_id"] = file_id

            print(
                f"  {user_dir.name}/{csv_path.name} → "
                f"label={label}, user_id={user_id}, file_id={file_id} "
                f"({len(df)} rows)"
            )

            all_frames.append(df)
            file_id += 1

    if not all_frames:
        print("No data was loaded — nothing to save.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # ---- Validation --------------------------------------------------------
    sensor_cols_present = [c for c in SENSOR_COLUMNS if c in combined.columns]
    assert len(sensor_cols_present) == 17, (
        f"Expected 17 sensor columns, found {len(sensor_cols_present)}"
    )

    unique_labels = sorted(combined["label"].unique())
    assert all(
        re.fullmatch(r"[a-z]", lbl) for lbl in unique_labels
    ), f"Unexpected label values: {unique_labels}"

    assert combined["user_id"].dropna().apply(
        lambda v: isinstance(v, (int, float)) and float(v).is_integer()
    ).all(), "user_id column contains non-integer values"

    fully_nan_rows = combined.isna().all(axis=1).sum()
    assert fully_nan_rows == 0, f"Found {fully_nan_rows} fully-NaN rows"

    # ---- Summary -----------------------------------------------------------
    print("\n=== Dataset Summary ===")
    print(f"Total rows:        {len(combined)}")
    print(f"Unique users:      {combined['user_id'].nunique()}")
    print(f"Unique labels:     {combined['label'].nunique()}")
    print(f"Unique file_ids:   {combined['file_id'].nunique()}")

    print("\nRows per user:")
    for uid in sorted(combined["user_id"].unique()):
        count = (combined["user_id"] == uid).sum()
        print(f"  user {uid}: {count}")

    print("\nRows per label:")
    for lbl in sorted(combined["label"].unique()):
        count = (combined["label"] == lbl).sum()
        print(f"  {lbl}: {count}")

    # ---- Save --------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(
        f"\n✓ Saved combined dataset to {OUTPUT_PATH} "
        f"(shape: {combined.shape[0]} × {combined.shape[1]})"
    )


if __name__ == "__main__":
    combine_dataset()
