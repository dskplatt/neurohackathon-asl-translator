"""
Collect additional calibration samples for specific letters.
Appends new data to the existing calibration_data.csv.
All existing data for all letters is preserved.

Example — add 10 more samples to every letter:
    python scripts/collect_partial_calibration.py \
        --letters a b c d e f g h i j k l m n o p q r s t u v w x y z \
        --samples 10

Example — add 15 samples to just A, S, and E:
    python scripts/collect_partial_calibration.py --letters a s e --samples 15
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
collected_rows: list[dict] = []   # in-memory only — never written until done
current_label: str | None = None
current_file_id = 0               # starts above existing max, set at load time
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


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _load_existing(path: str) -> tuple[pd.DataFrame, int]:
    """
    Load existing CSV. Returns (dataframe, next_file_id).
    next_file_id starts above the current maximum so new rows
    never collide with existing file_ids.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=CSV_COLUMNS), 0

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"WARNING: Could not read existing CSV: {e}")
        return pd.DataFrame(columns=CSV_COLUMNS), 0

    if df.empty:
        return df, 0

    next_id = int(df["file_id"].max()) + 1
    return df, next_id


def _append_and_save(existing: pd.DataFrame,
                     new_rows: list[dict],
                     path: str) -> None:
    """
    Append new_rows to existing dataframe and save.
    All existing rows are preserved exactly.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    new_df = pd.DataFrame(new_rows, columns=CSV_COLUMNS)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(path, index=False)

    print(f"\nAppend complete:")
    print(f"  Existing rows: {len(existing)}")
    print(f"  New rows:      {len(new_df)}")
    print(f"  Total rows:    {len(combined)}")
    print(f"  Saved to:      {path}")


def _sample_counts(df: pd.DataFrame, letters: list[str]) -> dict[str, int]:
    """Return number of unique file_ids per letter."""
    if df.empty:
        return {l: 0 for l in letters}
    return {
        l: int(df[df["label"] == l]["file_id"].nunique())
        for l in letters
    }


def _print_summary(existing: pd.DataFrame,
                   new_rows: list[dict],
                   letters: list[str],
                   samples_per_letter: int) -> None:
    new_df = pd.DataFrame(new_rows, columns=CSV_COLUMNS)
    combined = pd.concat([existing, new_df], ignore_index=True)

    before = _sample_counts(existing, letters)
    after  = _sample_counts(combined, letters)

    print(f"\n{'=' * 42}")
    print("Partial recalibration complete!")
    print(f"{'=' * 42}")
    print(f"New samples per letter: {samples_per_letter}")
    print(f"New rows collected:     {len(new_df)}")
    print()
    print("Updated sample counts:")
    for letter in letters:
        b = before[letter]
        a = after[letter]
        print(f"  {letter.upper()}: {a} total samples  (was {b})")

    print(f"\nAll other letters preserved from existing data.")
    print(f"\nNext steps:")
    print(f"  Fine-tune on these letters only:")
    print(f"    python scripts/finetune_confused.py "
          f"--letters {' '.join(letters)}")
    print(f"  Or retrain the full personal model:")
    print(f"    python scripts/train_personal_model.py")


def main():
    global current_label, samples_this_letter, current_file_id

    parser = argparse.ArgumentParser(
        description="Append calibration samples for specific letters.")
    parser.add_argument(
        "--letters", nargs="+", required=True,
        help="Lowercase letters to collect. e.g. --letters a s e")
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Samples per letter (default: 5, max: 20)")
    args = parser.parse_args()

    # ── Validate arguments ────────────────────────────────────────────────────
    letters_to_collect = [l.lower() for l in args.letters]
    invalid = [l for l in letters_to_collect
               if len(l) != 1 or l not in "abcdefghijklmnopqrstuvwxyz"]
    if invalid:
        print(f"ERROR: Invalid letters: {invalid}")
        print("Letters must be single lowercase a-z characters.")
        sys.exit(1)

    samples_per_letter = args.samples
    if not 1 <= samples_per_letter <= 20:
        print(f"ERROR: --samples must be between 1 and 20.")
        sys.exit(1)

    # ── Load existing data ────────────────────────────────────────────────────
    existing_df, next_file_id = _load_existing(OUTPUT_PATH)
    current_file_id = next_file_id

    if existing_df.empty:
        print(f"WARNING: No existing calibration_data.csv found.")
        print(f"A new file will be created. Consider running full")
        print(f"calibration first (scripts/collect_calibration_data.py).\n")

    # ── Print startup summary and confirm ────────────────────────────────────
    before_counts = _sample_counts(existing_df, letters_to_collect)
    all_letters = list("abcdefghijklmnopqrstuvwxyz")
    preserved = [l for l in all_letters if l not in letters_to_collect]

    print(f"\n{'=' * 42}")
    print("Partial Recalibration")
    print(f"{'=' * 42}")
    print(f"Letters:            {', '.join(l.upper() for l in letters_to_collect)}")
    print(f"Samples per letter: {samples_per_letter}")
    print(f"Total attempts:     {len(letters_to_collect) * samples_per_letter}")
    print()
    print("New samples will be ADDED to existing data.")
    print("All existing data is preserved.")
    print()
    print("Current → new sample counts:")
    for letter in letters_to_collect:
        before = before_counts[letter]
        after  = before + samples_per_letter
        print(f"  {letter.upper()}: {before} samples → will become {after}")

    if preserved:
        print(f"\nPreserved (unchanged): "
              f"{' '.join(l.upper() for l in preserved)}")

    print()
    try:
        input("Press Enter to begin, Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\nCancelled. calibration_data.csv has NOT been modified.")
        sys.exit(0)

    # ── Connect to Myo ────────────────────────────────────────────────────────
    segmentation = SegmentationStateMachine(
        on_letter_ready=on_letter_ready,
        on_signing_start=lambda: current_label and print("    [signing detected]"),
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

    print("\nMyo connected. Starting data collection...\n")
    time.sleep(1)

    # ── Collection loop ───────────────────────────────────────────────────────
    try:
        for letter in letters_to_collect:
            samples_this_letter = 0
            current_label = None

            print(f"\n{'=' * 40}")
            print(f"Letter: {letter.upper()}")
            print(f"{'=' * 40}")
            print(f"Collecting {samples_per_letter} samples of '{letter.upper()}'.\n")

            while samples_this_letter < samples_per_letter:
                current_label = None

                # Rest between samples
                print(f"\n  --- Rest ---")
                time.sleep(2)

                # Prompt user to sign
                sample_event.clear()
                segmentation.reset()
                print(f"  [{letter.upper()}] Sample "
                      f"{samples_this_letter + 1}/{samples_per_letter}")
                print(f"  >>> Sign '{letter.upper()}' now! <<<")

                current_label = letter

                timed_out = not sample_event.wait(timeout=SAMPLE_TIMEOUT_S)

                if timed_out:
                    print(f"  WARNING: No sample detected in {SAMPLE_TIMEOUT_S}s.")
                    print(f"  Tips: sign more firmly, check band is snug,")
                    print(f"  or adjust thresholds in segmentation.py")
                    continue

                # 2-second cooldown after capture
                current_label = None
                print(f"  Cooldown: 2 seconds...")
                time.sleep(2)

            current_label = None
            print(f"\n  '{letter.upper()}' done — "
                  f"{samples_this_letter} samples captured.")

        # ── All letters done — append to CSV ──────────────────────────────────
        _append_and_save(existing_df, collected_rows, OUTPUT_PATH)
        _print_summary(existing_df, collected_rows,
                       letters_to_collect, samples_per_letter)

    except KeyboardInterrupt:
        current_label = None
        print("\n\nCancelled.")
        print("calibration_data.csv has NOT been modified.")
        print(f"({len(collected_rows)} rows collected in memory were discarded)")
    finally:
        myo.stop()


if __name__ == "__main__":
    main()