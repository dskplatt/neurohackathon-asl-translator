"""
Collect personal calibration data from the Myo armband.

Guides the user through signing each ASL letter multiple times while
the system auto-detects signing boundaries via SegmentationStateMachine.
Saves raw EMG + accelerometer frames to data/calibration_data.csv.

Supports resuming: if the CSV already exists, completed letters are skipped
and new data is appended.

Usage:
    python scripts/collect_calibration_data.py
    python scripts/collect_calibration_data.py --samples-per-letter 1   # quick test
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.myo_reader import MyoReader
from src.segmentation import SegmentationStateMachine

# ── Constants ────────────────────────────────────────────────────────────────
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
OUTPUT_PATH = "data/calibration_data.csv"
USER_ID = 10
SAMPLE_TIMEOUT_S = 15

CSV_COLUMNS = [
    "emg_1", "emg_2", "emg_3", "emg_4",
    "emg_5", "emg_6", "emg_7", "emg_8",
    "acc_x", "acc_y", "acc_z",
    "label", "user_id", "file_id",
]

# ── Shared state ─────────────────────────────────────────────────────────────
collected_rows: list[dict] = []
current_label: str | None = None
current_file_id = 0
last_accel = np.zeros(3, dtype=np.float32)
accel_lock = threading.Lock()
sample_event = threading.Event()
samples_this_letter = 0


# ── Callbacks ────────────────────────────────────────────────────────────────

def on_letter_ready(window: np.ndarray) -> None:
    """Called by segmentation when a signing window is detected."""
    global current_file_id, samples_this_letter

    if current_label is None:
        return

    with accel_lock:
        accel = last_accel.copy()

    for i in range(window.shape[0]):
        collected_rows.append({
            "emg_1": float(window[i, 0]),
            "emg_2": float(window[i, 1]),
            "emg_3": float(window[i, 2]),
            "emg_4": float(window[i, 3]),
            "emg_5": float(window[i, 4]),
            "emg_6": float(window[i, 5]),
            "emg_7": float(window[i, 6]),
            "emg_8": float(window[i, 7]),
            "acc_x": float(accel[0]),
            "acc_y": float(accel[1]),
            "acc_z": float(accel[2]),
            "label": current_label,
            "user_id": USER_ID,
            "file_id": current_file_id,
        })

    n_frames = window.shape[0]
    current_file_id += 1
    samples_this_letter += 1
    print(f"    Captured! ({n_frames} frames, file_id={current_file_id - 1})")
    sample_event.set()


def on_accel_frame(accel: np.ndarray) -> None:
    global last_accel
    with accel_lock:
        last_accel = np.asarray(accel, dtype=np.float32).ravel()


def _save_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(path, index=False)


def _load_existing(path: str, samples_per_letter: int) -> dict[str, int]:
    """Load existing CSV and return {letter: n_file_ids} for completed letters."""
    global collected_rows, current_file_id

    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    if df.empty:
        return {}

    collected_rows = df.to_dict("records")
    current_file_id = int(df["file_id"].max()) + 1

    counts: dict[str, int] = {}
    for letter in LETTERS:
        letter_data = df[df["label"] == letter]
        counts[letter] = letter_data["file_id"].nunique()

    done = sum(1 for v in counts.values() if v >= samples_per_letter)
    print(f"Resuming from existing data: {len(df)} rows, "
          f"{done}/26 letters complete")
    for letter in LETTERS:
        n = counts.get(letter, 0)
        if 0 < n < samples_per_letter:
            print(f"  '{letter.upper()}': {n}/{samples_per_letter} samples")

    return counts


def _print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    letters_present = df["label"].nunique()
    total_rows = len(df)

    print(f"\n{'=' * 40}")
    print("Calibration data collection complete!")
    print(f"{'=' * 40}")
    print(f"Total rows:      {total_rows}")
    print(f"Letters covered: {letters_present}/26")

    print("Rows per letter:")
    for letter in LETTERS:
        n = len(df[df["label"] == letter])
        print(f"  {letter}: {n:>5}", end="")
        if LETTERS.index(letter) % 6 == 5:
            print()
    print()

    print("File IDs per letter:")
    for letter in LETTERS:
        n = df[df["label"] == letter]["file_id"].nunique()
        print(f"  {letter}: {n}", end="")
        if LETTERS.index(letter) % 6 == 5:
            print()
    print()

    print(f"Saved to: {OUTPUT_PATH}")
    print(f"\nRun training with:")
    print(f"  python scripts/train_personal_model.py")


def main():
    global current_label, samples_this_letter

    parser = argparse.ArgumentParser(
        description="Collect personal calibration data from the Myo armband.")
    parser.add_argument(
        "--samples-per-letter", type=int, default=5,
        help="Number of signing samples per letter (default: 5)")
    args = parser.parse_args()
    samples_per_letter = args.samples_per_letter

    print(f"\n{'=' * 40}")
    print("ASL Calibration Data Collection")
    print(f"{'=' * 40}")
    print(f"Samples per letter: {samples_per_letter}")
    print(f"Output: {OUTPUT_PATH}\n")

    existing_counts = _load_existing(OUTPUT_PATH, samples_per_letter)

    segmentation = SegmentationStateMachine(
        on_letter_ready=on_letter_ready,
        on_signing_start=lambda: print("    [signing detected]"),
        on_signing_end=lambda: None,
    )

    myo = MyoReader(
        on_emg_frame=segmentation.push_frame,
        on_accel_frame=on_accel_frame,
        on_wave_right=lambda: None,
        on_wave_left=lambda: None,
    )

    try:
        myo.start()
    except Exception as e:
        print(f"\nERROR: Could not connect to Myo armband: {e}")
        print("Make sure:")
        print("  1. The Myo armband is powered on and paired via Bluetooth")
        print("  2. The Myo Connect software is running")
        print("  3. pyomyo is installed: pip install pyomyo")
        sys.exit(1)

    print("Myo connected. Starting data collection...\n")
    time.sleep(1)

    try:
        for letter in LETTERS:
            already_done = existing_counts.get(letter, 0)
            if already_done >= samples_per_letter:
                print(f"[{letter.upper()}] Already has {already_done} samples — skipping")
                continue

            samples_this_letter = already_done
            current_label = letter

            print(f"\n{'=' * 40}")
            print(f"Letter: {letter.upper()}")
            print(f"{'=' * 40}")
            print(f"Sign '{letter.upper()}' clearly {samples_per_letter - already_done} more time(s).")
            print(f"Hold each letter for ~1 second, then fully relax.")
            print(f"The system detects each attempt automatically.\n")

            while samples_this_letter < samples_per_letter:
                sample_event.clear()
                print(f"  [{letter.upper()}] Waiting for sample "
                      f"({samples_this_letter + 1}/{samples_per_letter})...")

                timed_out = not sample_event.wait(timeout=SAMPLE_TIMEOUT_S)

                if timed_out and samples_this_letter < samples_per_letter:
                    print(f"  WARNING: No sample detected in {SAMPLE_TIMEOUT_S}s.")
                    print(f"  Tips: sign more firmly, check band is snug,")
                    print(f"  or adjust thresholds in segmentation.py")

            current_label = None
            print(f"\n  '{letter.upper()}' done — {samples_this_letter} samples captured.")

            _save_csv(collected_rows, OUTPUT_PATH)
            print(f"  Progress saved to {OUTPUT_PATH}")

        _print_summary(collected_rows)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving progress...")
        current_label = None
        if collected_rows:
            _save_csv(collected_rows, OUTPUT_PATH)
            print(f"Saved {len(collected_rows)} rows to {OUTPUT_PATH}")
            print("Re-run this script to resume where you left off.")
    finally:
        myo.stop()


if __name__ == "__main__":
    main()
