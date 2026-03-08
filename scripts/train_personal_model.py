"""
Train a personal ASL classifier on calibration data.

Loads data/calibration_data.csv, windows it identically to training/preprocess.py
(WINDOW_SIZE=40, but STRIDE=5 for more windows from limited data), applies gentle
augmentation, fits a personal scaler, fine-tunes ALL layers of the pretrained model,
and saves the personal model + scaler.

Usage:
    python scripts/train_personal_model.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ASLClassifier

# ── Constants ────────────────────────────────────────────────────────────────
CHANNELS = [
    "emg_1", "emg_2", "emg_3", "emg_4",
    "emg_5", "emg_6", "emg_7", "emg_8",
    "acc_x", "acc_y", "acc_z",
]
WINDOW_SIZE = 40
STRIDE = 5

EPOCHS = 100
LR = 5e-4
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-3
EARLY_STOP_PATIENCE = 20

DATA_PATH = "data/calibration_data.csv"
LABEL_MAP_PATH = "models/label_map.json"
PRETRAINED_MODEL_PATH = "models/classifier_final.pt"
OUTPUT_MODEL_PATH = "models/classifier_personal.pt"
OUTPUT_SCALER_PATH = "models/scaler_personal.joblib"

SEED = 42


# ── Windowing ────────────────────────────────────────────────────────────────

def create_windows(
    df: pd.DataFrame, label_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Slide fixed-size windows over each file_id group (never straddle attempts)."""
    X: list[np.ndarray] = []
    y: list[int] = []

    per_letter: dict[str, int] = {l: 0 for l in label_map}

    for fid, group in df.groupby("file_id", sort=False):
        data = group[CHANNELS].values.astype(np.float32)
        label_str = str(group["label"].iloc[0])
        if label_str not in label_map:
            continue
        label_int = label_map[label_str]

        for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
            window = data[start : start + WINDOW_SIZE]  # (40, 11)
            X.append(window.T)                           # (11, 40) channels first
            y.append(label_int)
            per_letter[label_str] = per_letter.get(label_str, 0) + 1

    inv = {v: k for k, v in label_map.items()}
    print("Windows per letter:")
    for i in range(26):
        letter = inv[i]
        print(f"  {letter}: {per_letter.get(letter, 0)}", end="")
        if i % 6 == 5:
            print()
    print()

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply 3 augmentations to create 4x data (original + 3 variants).

    No channel permutation — the user's pod-to-muscle mapping is the signal.
    """
    X_noise = X + rng.normal(0, 0.05, size=X.shape).astype(np.float32)

    scales = rng.uniform(0.85, 1.15, size=(len(X), 1, 1)).astype(np.float32)
    X_scaled = X * scales

    X_shifted = X.copy()
    for i in range(len(X)):
        shift = int(rng.integers(-3, 4))
        X_shifted[i] = np.roll(X[i], shift, axis=1)

    X_aug = np.concatenate([X, X_noise, X_scaled, X_shifted], axis=0)
    y_aug = np.concatenate([y, y, y, y], axis=0)

    return X_aug, y_aug


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    model: ASLClassifier,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[float, int]:
    """Fine-tune all layers. Returns (best_accuracy, epochs_trained)."""

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=False)

    optimiser = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = ReduceLROnPlateau(optimiser, patience=8, factor=0.5,
                                  min_lr=1e-5)

    best_loss = float("inf")
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    epochs_trained = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimiser.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * batch_X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_X.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        scheduler.step(avg_loss)
        epochs_trained = epoch

        if epoch <= 10 or epoch % 5 == 0 or avg_loss < best_loss:
            print(f"Epoch {epoch:>3d} | loss: {avg_loss:.3f} | acc: {acc:.3f}"
                  + (" <- best" if avg_loss < best_loss else ""))

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 30 and acc < 0.70:
            print(f"\nWARNING: accuracy below 70% at epoch 30 ({acc:.1%}).")
            print("Consider re-collecting calibration data with clearer signing.\n")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, OUTPUT_MODEL_PATH)

    return best_acc, epochs_trained


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load data ────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        print("Run data collection first:")
        print("  python scripts/collect_calibration_data.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print(f"  Letters: {sorted(df['label'].unique())}")
    print(f"  File IDs: {df['file_id'].nunique()}\n")

    # ── Load label map ───────────────────────────────────────────────
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    # ── Windowing ────────────────────────────────────────────────────
    print(f"Windowing (size={WINDOW_SIZE}, stride={STRIDE})...")
    X_base, y_base = create_windows(df, label_map)
    print(f"Base windows: {len(X_base)}  shape: {X_base.shape}\n")

    if len(X_base) == 0:
        print("ERROR: No windows created. Check that calibration data "
              "has enough frames per file_id.")
        sys.exit(1)

    # ── Augmentation ─────────────────────────────────────────────────
    print("Augmenting (3 variants → 4x total)...")
    X_aug, y_aug = augment(X_base, y_base, rng)
    print(f"After augmentation: {len(X_aug)} windows\n")

    # ── Fit personal scaler ──────────────────────────────────────────
    print("Fitting personal scaler...")
    X_flat = X_aug.transpose(0, 2, 1).reshape(-1, 11)  # (N*40, 11)
    personal_scaler = StandardScaler()
    personal_scaler.fit(X_flat)
    joblib.dump(personal_scaler, OUTPUT_SCALER_PATH)
    print(f"Saved scaler → {OUTPUT_SCALER_PATH}")
    print(f"  Mean: {personal_scaler.mean_[:3].round(3)}... "
          f"Scale: {personal_scaler.scale_[:3].round(3)}...\n")

    # ── Apply scaler ─────────────────────────────────────────────────
    N = X_aug.shape[0]
    X_scaled = personal_scaler.transform(X_flat).reshape(N, WINDOW_SIZE, 11)
    X_scaled = X_scaled.transpose(0, 2, 1).astype(np.float32)  # (N, 11, 40)

    # ── Load pretrained model ────────────────────────────────────────
    print("Loading pretrained model...")
    model = ASLClassifier()
    model.load_state_dict(
        torch.load(PRETRAINED_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.unfreeze_all()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    # ── Train ────────────────────────────────────────────────────────
    print(f"Training (lr={LR}, epochs={EPOCHS}, "
          f"batch={BATCH_SIZE}, patience={EARLY_STOP_PATIENCE})...\n")
    t0 = time.time()
    best_acc, epochs_trained = train(model, X_scaled, y_aug, device)
    elapsed = time.time() - t0

    # ── Per-letter window counts ─────────────────────────────────────
    inv = {v: k for k, v in label_map.items()}
    from collections import Counter
    counts = Counter(y_aug.tolist())
    per_letter = [counts.get(i, 0) for i in range(26)]

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 44}")
    print("Personal model training complete!")
    print(f"{'=' * 44}")
    print(f"Best training accuracy: {best_acc:.1%}")
    print(f"Epochs trained:         {epochs_trained}")
    print(f"Training time:          {elapsed:.0f}s")
    print(f"Windows trained on:     {len(X_aug)}  (across 26 letters)")
    print(f"Windows per letter:     min={min(per_letter)}  "
          f"max={max(per_letter)}  avg={np.mean(per_letter):.0f}")
    print()
    print(f"Saved:")
    print(f"  {OUTPUT_MODEL_PATH}")
    print(f"  {OUTPUT_SCALER_PATH}")
    print()
    print("Restart the server to use your personal model.")
    print("The server will automatically detect and load it.")
    print(f"{'=' * 44}")


if __name__ == "__main__":
    main()
