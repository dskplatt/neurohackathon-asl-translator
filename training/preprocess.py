"""
Preprocess the combined Myo armband dataset into windowed training samples.

Saves RAW (un-normalised) windows to windows.npz so that train.py can fit
the scaler per LOUO fold on training data only (no data leakage).

Aggressive augmentation targets 500+ windows per class from ~93 base.

Usage (from project root):
    python training/preprocess.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = Path("data/combined_dataset.csv")
OUTPUT_PATH = Path("data/windows.npz")
LABEL_MAP_PATH = Path("data/label_map.json")

WINDOW_SIZE = 40   # ~200 ms at 200 Hz
STRIDE = 10        # ~50 ms  — 75 % overlap

CHANNELS = [
    "emg_1", "emg_2", "emg_3", "emg_4",
    "emg_5", "emg_6", "emg_7", "emg_8",
    "acc_x", "acc_y", "acc_z",
]

N_EMG = 8
EMG_INDICES = list(range(N_EMG))
ACC_INDICES = [8, 9, 10]

TIME_SHIFT_MAX = 5
SEED = 42

# ---------------------------------------------------------------------------
# Training mode — controls augmentation intensity
# ---------------------------------------------------------------------------
# True  = full-dataset training (all 9 users, gentler augmentation ~8x)
# False = LOUO cross-validation (aggressive augmentation ~15x)
FULL_TRAINING_MODE = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_label_map() -> dict[str, int]:
    """a -> 0, b -> 1, ..., z -> 25."""
    return {chr(ord("a") + i): i for i in range(26)}


def create_windows(
    df: pd.DataFrame,
    label_map: dict[str, int],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    list[int], dict[int, np.ndarray], dict[int, np.ndarray],
]:
    """Slide a fixed-size window over each file_id group.

    Returns
    -------
    X        : float32 (N, C, T)
    y        : int64   (N,)
    user_ids : int64   (N,)
    file_ids : int64   (N,)
    starts   : list[int]  — start index of each window in its recording
    recordings      : dict  file_id -> (T_full, C) channel array
    recording_labels: dict  file_id -> (T_full,) encoded-label array
    """
    windows: list[np.ndarray] = []
    labels: list[int] = []
    user_ids: list[int] = []
    file_ids: list[int] = []
    starts: list[int] = []

    recordings: dict[int, np.ndarray] = {}
    recording_labels: dict[int, np.ndarray] = {}

    for fid, group in df.groupby("file_id", sort=False):
        data = group[CHANNELS].values.astype(np.float32)       # (T, C)
        lbl_enc = group["label"].map(label_map).values          # (T,)
        uid = int(group["user_id"].iloc[0])

        recordings[fid] = data
        recording_labels[fid] = lbl_enc

        for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            wl = lbl_enc[start:end]
            if wl.min() != wl.max():
                continue
            windows.append(data[start:end].T)                   # (C, T)
            labels.append(int(wl[0]))
            user_ids.append(uid)
            file_ids.append(int(fid))
            starts.append(start)

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(user_ids, dtype=np.int64),
        np.array(file_ids, dtype=np.int64),
        starts,
        recordings,
        recording_labels,
    )


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def aug_gaussian_noise(
    X: np.ndarray, rng: np.random.Generator, std: float = 2.0,
) -> np.ndarray:
    """Add N(0, std) noise to all channels."""
    return X + rng.normal(0, std, size=X.shape).astype(np.float32)


def aug_amplitude_scale(
    X: np.ndarray, rng: np.random.Generator,
    lo: float = 0.6, hi: float = 1.4,
) -> np.ndarray:
    """Per-sample global amplitude scaling."""
    scales = rng.uniform(lo, hi, size=(len(X), 1, 1)).astype(np.float32)
    return X * scales


def aug_per_channel_scale(
    X: np.ndarray, rng: np.random.Generator,
    lo: float = 0.7, hi: float = 1.3,
) -> np.ndarray:
    """Independent scale factor per channel per sample."""
    scales = rng.uniform(lo, hi, size=(len(X), X.shape[1], 1)).astype(np.float32)
    return X * scales


def aug_channel_permutation(
    X: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Randomly permute the 8 EMG channels; keep acc channels in place."""
    aug = X.copy()
    for i in range(len(aug)):
        perm = rng.permutation(N_EMG)
        aug[i, EMG_INDICES, :] = aug[i, perm, :]
    return aug


def aug_time_roll(
    X: np.ndarray, rng: np.random.Generator, max_shift: int = 5,
) -> np.ndarray:
    """Circular time shift (roll) by a random offset."""
    aug = X.copy()
    shifts = rng.integers(-max_shift, max_shift + 1, size=len(aug))
    for i, s in enumerate(shifts):
        aug[i] = np.roll(aug[i], int(s), axis=1)
    return aug


def aug_channel_dropout(
    X: np.ndarray, rng: np.random.Generator, prob: float = 0.15,
) -> np.ndarray:
    """Zero out one random EMG channel with given probability."""
    aug = X.copy()
    mask = rng.random(len(aug)) < prob
    drop_ch = rng.integers(0, N_EMG, size=len(aug))
    for i in np.where(mask)[0]:
        aug[i, drop_ch[i], :] = 0.0
    return aug


def aug_time_shift(
    X: np.ndarray, y: np.ndarray, user_ids: np.ndarray, file_ids: np.ndarray,
    starts: list[int],
    recordings: dict[int, np.ndarray],
    recording_labels: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shift each window start within the original recording."""
    aug_w: list[np.ndarray] = []
    aug_y: list[int] = []
    aug_u: list[int] = []
    aug_f: list[int] = []

    for i in range(len(X)):
        fid = int(file_ids[i])
        rec = recordings[fid]
        rec_lbl = recording_labels[fid]

        offset = int(rng.integers(-TIME_SHIFT_MAX, TIME_SHIFT_MAX + 1))
        new_start = max(0, min(starts[i] + offset, len(rec) - WINDOW_SIZE))
        new_end = new_start + WINDOW_SIZE

        wl = rec_lbl[new_start:new_end]
        if wl.min() != wl.max():
            aug_w.append(X[i])
        else:
            aug_w.append(rec[new_start:new_end].T)

        aug_y.append(int(y[i]))
        aug_u.append(int(user_ids[i]))
        aug_f.append(int(file_ids[i]))

    return (
        np.array(aug_w, dtype=np.float32),
        np.array(aug_y, dtype=np.int64),
        np.array(aug_u, dtype=np.int64),
        np.array(aug_f, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    n_before: int,
    n_after: int,
    y: np.ndarray,
    user_ids: np.ndarray,
    X: np.ndarray,
    label_map: dict[str, int],
) -> None:
    inv = {v: k for k, v in label_map.items()}

    print("\n=== Preprocessing Summary ===")
    print(f"Total windows before augmentation: {n_before}")
    print(f"Total windows after augmentation:  {n_after}")
    n_classes = len(label_map)
    counts = Counter(y.tolist())
    min_c = min(counts.values())
    max_c = max(counts.values())
    print(f"Windows per class range: {min_c} - {max_c}")

    print("\nWindows per user:")
    for uid in sorted(Counter(user_ids.tolist())):
        print(f"  user {uid}: {Counter(user_ids.tolist())[uid]}")

    print(f"\nX shape: {X.shape}  dtype: {X.dtype}")
    print(f"y shape: {y.shape}  unique values: {sorted(np.unique(y).tolist())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)

    # ---- Load ---------------------------------------------------------------
    print(f"Loading {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    print(
        f"  {len(df)} rows, {df['file_id'].nunique()} files, "
        f"{df['user_id'].nunique()} users, {df['label'].nunique()} labels"
    )

    # ---- Label map ----------------------------------------------------------
    label_map = build_label_map()
    LABEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map -> {LABEL_MAP_PATH}")

    # ---- Windowing ----------------------------------------------------------
    print("Creating windows ...")
    X_base, y_base, uids_base, fids_base, starts, recs, rec_lbls = create_windows(
        df, label_map,
    )
    n_before = len(X_base)
    print(f"  {n_before} base windows  (per-window shape: {X_base.shape[1:]})")

    # ---- Augmentation --------------------------------------------------------
    def meta():
        return y_base.copy(), uids_base.copy(), fids_base.copy()

    batches_X = [X_base]
    batches_y = [y_base]
    batches_u = [uids_base]
    batches_f = [fids_base]

    def add(Xa, ya=None, ua=None, fa=None):
        if ya is None:
            ya, ua, fa = meta()
        batches_X.append(Xa)
        batches_y.append(ya)
        batches_u.append(ua)
        batches_f.append(fa)

    if FULL_TRAINING_MODE:
        # Gentler augmentation for full-dataset training (7 variants = 8x)
        # All 9 users provide natural diversity; less augmentation needed
        print("Augmenting (7 variants, full-training mode) ...")
        add(aug_gaussian_noise(X_base, rng, std=1.5))
        add(aug_amplitude_scale(X_base, rng, 0.75, 1.25))
        add(aug_per_channel_scale(X_base, rng, 0.8, 1.2))
        add(aug_channel_permutation(X_base, rng))
        add(aug_time_roll(X_base, rng, max_shift=4))
        Xs, ys, us, fs = aug_time_shift(
            X_base, y_base, uids_base, fids_base,
            starts, recs, rec_lbls, rng,
        )
        add(Xs, ys, us, fs)
        add(aug_channel_dropout(X_base, rng, prob=0.1))
    else:
        # Aggressive augmentation for LOUO (14 variants = 15x)
        # Must compensate for missing user in cross-validation
        print("Augmenting (14 variants, LOUO mode) ...")
        add(aug_gaussian_noise(X_base, rng, std=1.0))
        add(aug_gaussian_noise(X_base, rng, std=3.0))
        add(aug_amplitude_scale(X_base, rng, 0.6, 1.4))
        add(aug_per_channel_scale(X_base, rng, 0.7, 1.3))
        add(aug_channel_permutation(X_base, rng))
        add(aug_time_roll(X_base, rng, max_shift=5))
        Xs, ys, us, fs = aug_time_shift(
            X_base, y_base, uids_base, fids_base,
            starts, recs, rec_lbls, rng,
        )
        add(Xs, ys, us, fs)
        add(aug_channel_dropout(X_base, rng, prob=0.15))
        add(aug_amplitude_scale(
            aug_gaussian_noise(X_base, rng, std=2.0), rng, 0.7, 1.3,
        ))
        add(aug_gaussian_noise(
            aug_channel_permutation(X_base, rng), rng, std=1.5,
        ))
        add(aug_per_channel_scale(
            aug_time_roll(X_base, rng, max_shift=5), rng, 0.7, 1.3,
        ))
        add(aug_channel_dropout(
            aug_amplitude_scale(X_base, rng, 0.6, 1.4), rng, prob=0.2,
        ))
        Xs2, ys2, us2, fs2 = aug_time_shift(
            X_base, y_base, uids_base, fids_base,
            starts, recs, rec_lbls, rng,
        )
        add(aug_amplitude_scale(
            aug_gaussian_noise(Xs2, rng, std=2.0), rng, 0.7, 1.3,
        ), ys2, us2, fs2)
        add(aug_channel_dropout(
            aug_per_channel_scale(
                aug_channel_permutation(X_base, rng), rng, 0.7, 1.3,
            ), rng, prob=0.15,
        ))

    X_all = np.concatenate(batches_X)
    y_all = np.concatenate(batches_y)
    uids_all = np.concatenate(batches_u)
    fids_all = np.concatenate(batches_f)

    n_after = len(X_all)

    is_augmented = np.concatenate([
        np.zeros(n_before, dtype=np.bool_),
        np.ones(n_after - n_before, dtype=np.bool_),
    ])

    print(f"  {n_after} windows after augmentation  (x{n_after / n_before:.0f})")

    # ---- Save RAW (no normalisation) ----------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        X=X_all,
        y=y_all,
        user_ids=uids_all,
        file_ids=fids_all,
        is_augmented=is_augmented,
    )
    print(f"Saved raw windows -> {OUTPUT_PATH}")

    # ---- Summary ------------------------------------------------------------
    print_summary(n_before, n_after, y_all, uids_all, X_all, label_map)


if __name__ == "__main__":
    main()
