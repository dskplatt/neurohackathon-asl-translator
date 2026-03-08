"""
Validate calibration data before training a personal model.

Checks data/calibration_data.csv for completeness, correctness, and
signal quality. Prints a clear pass/fail for each check.

Usage:
    python scripts/validate_calibration_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

DATA_PATH = "data/calibration_data.csv"

REQUIRED_COLUMNS = [
    "emg_1", "emg_2", "emg_3", "emg_4",
    "emg_5", "emg_6", "emg_7", "emg_8",
    "acc_x", "acc_y", "acc_z",
    "label", "user_id", "file_id",
]

EMG_COLUMNS = [
    "emg_1", "emg_2", "emg_3", "emg_4",
    "emg_5", "emg_6", "emg_7", "emg_8",
]

ALL_LETTERS = set("abcdefghijklmnopqrstuvwxyz")


def main():
    passes = 0
    warnings = 0
    fails = 0
    blocking = False

    def PASS(msg: str):
        nonlocal passes
        passes += 1
        print(f"  [PASS] {msg}")

    def WARN(msg: str):
        nonlocal warnings
        warnings += 1
        print(f"  [WARN] {msg}")

    def FAIL(msg: str):
        nonlocal fails, blocking
        fails += 1
        blocking = True
        print(f"  [FAIL] {msg}")

    print(f"\nValidating {DATA_PATH}...\n")

    # ── Check 1: File exists ─────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        FAIL(f"File not found: {DATA_PATH}")
        print(f"\n{fails} check(s) failed. Cannot proceed.")
        sys.exit(1)
    PASS("File exists")

    df = pd.read_csv(DATA_PATH)

    # ── Check 2: Required columns ────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        FAIL(f"Missing columns: {missing_cols}")
    else:
        PASS(f"All {len(REQUIRED_COLUMNS)} required columns present")

    # ── Check 3: Letters present ─────────────────────────────────────
    present_letters = set(df["label"].unique())
    missing_letters = ALL_LETTERS - present_letters
    n_present = len(present_letters & ALL_LETTERS)
    if n_present == 26:
        PASS("All 26 letters present")
    elif n_present >= 20:
        WARN(f"{n_present}/26 letters present — missing: "
             f"{sorted(missing_letters)}")
    else:
        FAIL(f"Only {n_present}/26 letters present — missing: "
             f"{sorted(missing_letters)}")

    # ── Check 4: Total rows ──────────────────────────────────────────
    total_rows = len(df)
    if total_rows > 5000:
        PASS(f"Total rows: {total_rows:,}")
    else:
        WARN(f"Total rows: {total_rows:,} (expected > 5,000)")

    # ── Check 5: Samples per letter ──────────────────────────────────
    file_ids_per_letter: dict[str, int] = {}
    for letter in sorted(present_letters & ALL_LETTERS):
        letter_data = df[df["label"] == letter]
        n_fids = letter_data["file_id"].nunique()
        file_ids_per_letter[letter] = n_fids

    min_fids = min(file_ids_per_letter.values()) if file_ids_per_letter else 0
    min_letter = min(file_ids_per_letter, key=file_ids_per_letter.get) if file_ids_per_letter else "?"

    if min_fids >= 5:
        PASS(f"Minimum samples per letter: {min_fids} (letter '{min_letter}')")
    elif min_fids >= 2:
        WARN(f"Minimum samples per letter: {min_fids} (letter '{min_letter}') "
             f"— ideally 5")
    else:
        FAIL(f"Letter '{min_letter}' has only {min_fids} sample(s) — need >= 2")

    low_letters = [l for l, n in file_ids_per_letter.items() if 2 <= n < 5]
    for letter in low_letters:
        WARN(f"Letter '{letter}' has only {file_ids_per_letter[letter]} "
             f"file_ids — ideally 5")

    # ── Check 6: NaN values in EMG ───────────────────────────────────
    emg_cols_present = [c for c in EMG_COLUMNS if c in df.columns]
    if emg_cols_present:
        nan_count = df[emg_cols_present].isna().sum().sum()
        nan_pct = nan_count / (total_rows * len(emg_cols_present))
        if nan_count == 0:
            PASS("No NaN values in EMG columns")
        elif nan_pct <= 0.10:
            WARN(f"{nan_count} NaN values in EMG columns ({nan_pct:.1%})")
        else:
            FAIL(f"{nan_count} NaN values in EMG columns ({nan_pct:.1%}) "
                 f"— more than 10%")

    # ── Check 7: EMG RMS ─────────────────────────────────────────────
    if emg_cols_present:
        emg_values = df[emg_cols_present].values.astype(np.float64)
        emg_rms = np.sqrt(np.nanmean(emg_values ** 2))
        if emg_rms > 2.0:
            PASS(f"EMG RMS mean: {emg_rms:.1f} (expected > 2.0)")
        else:
            FAIL(f"EMG RMS mean: {emg_rms:.1f} — below 2.0 "
                 f"(signal is essentially noise)")

    # ── Check 8: EMG value range ─────────────────────────────────────
    if emg_cols_present:
        emg_min = float(np.nanmin(emg_values))
        emg_max = float(np.nanmax(emg_values))
        if emg_min < -1 or emg_max > 1:
            PASS(f"EMG values in range: min={emg_min:.0f}  max={emg_max:.0f}")
        else:
            WARN(f"EMG values look very small: min={emg_min:.3f}  "
                 f"max={emg_max:.3f} — may already be scaled")

    # ── Check 9: file_id uniqueness per label ────────────────────────
    if "file_id" in df.columns and "label" in df.columns:
        fid_label_map = df.groupby("file_id")["label"].nunique()
        cross_label_fids = fid_label_map[fid_label_map > 1]
        if len(cross_label_fids) == 0:
            PASS("file_id is unique per label (no cross-label file_ids)")
        else:
            FAIL(f"{len(cross_label_fids)} file_id(s) map to multiple labels")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{passes} check(s) passed, {warnings} warning(s), {fails} failure(s).\n")

    if blocking:
        print("Fix the failures above before training.")
        sys.exit(1)

    # ── Detailed breakdown ───────────────────────────────────────────
    print("Rows per letter:")
    for letter in sorted(present_letters & ALL_LETTERS):
        n = len(df[df["label"] == letter])
        print(f"  {letter}: {n:>5}", end="")
        if list(sorted(present_letters & ALL_LETTERS)).index(letter) % 6 == 5:
            print()
    print()

    print("File IDs per letter:")
    for letter in sorted(file_ids_per_letter):
        print(f"  {letter}: {file_ids_per_letter[letter]}", end="")
        if list(sorted(file_ids_per_letter)).index(letter) % 6 == 5:
            print()
    print()

    print("Ready to train. Run:")
    print("  python scripts/train_personal_model.py")


if __name__ == "__main__":
    main()
