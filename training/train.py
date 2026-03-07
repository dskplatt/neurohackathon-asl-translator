"""ASL classifier training script.

Supports two modes:
  python training/train.py --mode louo   # Leave-One-User-Out CV (default)
  python training/train.py --mode full   # Train on all data, save final model

Usage (from project root):
    python training/train.py
    python training/train.py --mode full
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ASLClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOWS_PATH = Path("data/windows.npz")
MODELS_DIR = Path("models")
LABEL_MAP_PATH = Path("data/label_map.json")
SCALER_FULL_PATH = Path("models/scaler_full.joblib")
TRAINING_STATS_PATH = Path("models/training_stats.json")
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 1.0
MIXUP_ALPHA = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def fit_scaler(X: np.ndarray) -> StandardScaler:
    """Fit a per-channel StandardScaler. X shape: (N, C, T)."""
    N, C, T = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, C)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted scaler. Returns float32 array with same shape as input."""
    N, C, T = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, C)
    scaled = scaler.transform(flat)
    return scaled.reshape(N, T, C).transpose(0, 2, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ASLDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup: interpolate between shuffled pairs within the batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return x * lam + x[idx] * (1 - lam), y, y[idx], lam


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        X_mix, ya, yb, lam = mixup_batch(X_batch, y_batch, MIXUP_ALPHA)
        optimizer.zero_grad()
        logits = model(X_mix)
        loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        running_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == ya).sum().item()
        total += len(y_batch)
    return running_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        running_loss += criterion(logits, y_batch).item() * len(y_batch)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = (preds == labels).mean() * 100
    return running_loss / len(loader.dataset), acc, preds, labels


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    label: str = "",
) -> tuple[dict, float]:
    """Train with early stopping. Returns (best_state_dict, best_metric)."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4,
    )
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_metric = float("inf")
    best_state = None
    best_preds, best_labels = None, None
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
        )

        if val_loader is not None:
            val_loss, val_acc, preds, labels = evaluate(
                model, val_loader, criterion,
            )
            scheduler.step(val_loss)
            metric = -val_acc
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [{label}] ep {epoch:3d}/{MAX_EPOCHS} | "
                f"t_loss {train_loss:.4f} t_acc {train_acc:.1f}% | "
                f"v_loss {val_loss:.4f} v_acc {val_acc:5.1f}% | lr {lr:.1e}"
            )
        else:
            scheduler.step(train_loss)
            metric = train_loss
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [{label}] ep {epoch:3d}/{MAX_EPOCHS} | "
                f"loss {train_loss:.4f} acc {train_acc:.1f}% | lr {lr:.1e}"
            )

        if metric < best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if val_loader is not None:
                best_preds, best_labels = preds, labels
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print(f"  [{label}] Early stopping at epoch {epoch}")
                break

    return best_state, best_metric, best_preds, best_labels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_fold_report(
    user_id: int, best_acc: float,
    preds: np.ndarray, labels: np.ndarray,
    inv_label_map: dict[int, str],
) -> None:
    n_classes = len(inv_label_map)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))

    print(f"\n{'=' * 60}")
    print(f"Held-out user: {user_id}")
    print(f"Best val accuracy: {best_acc:.1f}%")

    print(f"\n  {'Letter':>6}  {'Correct':>7}  {'Total':>5}  {'Acc%':>6}")
    print(f"  {'-' * 30}")
    for cls in range(n_classes):
        total = cm[cls].sum()
        correct = cm[cls, cls]
        acc = correct / total * 100 if total > 0 else 0.0
        print(
            f"  {inv_label_map.get(cls, '?'):>6}  {correct:>7}  "
            f"{total:>5}  {acc:>5.1f}%"
        )

    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat = cm_off.ravel()
    top5 = np.argsort(flat)[::-1][:5]

    print(f"\n  Top 5 confused pairs:")
    print(f"  {'True':>6} -> {'Pred':>6}  {'Count':>5}")
    print(f"  {'-' * 25}")
    for idx in top5:
        r, c = divmod(idx, n_classes)
        if flat[idx] == 0:
            break
        print(
            f"  {inv_label_map.get(r, '?'):>6} -> "
            f"{inv_label_map.get(c, '?'):>6}  {flat[idx]:>5}"
        )
    print()


# ---------------------------------------------------------------------------
# Training stats (for calibration)
# ---------------------------------------------------------------------------
def save_training_stats(
    X_raw: np.ndarray, scaler: StandardScaler,
) -> None:
    """Compute and save EMG statistics needed by calibration."""
    emg_raw = X_raw[:, :8, :]
    rms_per_window = np.sqrt(np.mean(emg_raw ** 2, axis=(1, 2)))
    mean_rms = float(rms_per_window.mean())

    rest_threshold = float(np.percentile(rms_per_window, 10))

    stats = {
        "mean_active_rms": mean_rms,
        "mean_rest_rms": rest_threshold,
        "emg_channel_means": scaler.mean_[:8].tolist(),
        "emg_channel_stds": scaler.scale_[:8].tolist(),
        "acc_channel_means": scaler.mean_[8:11].tolist(),
        "acc_channel_stds": scaler.scale_[8:11].tolist(),
    }

    TRAINING_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved training stats -> {TRAINING_STATS_PATH}")


# ---------------------------------------------------------------------------
# Mode: LOUO cross-validation
# ---------------------------------------------------------------------------
def run_louo(
    X_raw: np.ndarray, y: np.ndarray, user_ids: np.ndarray,
    is_augmented: np.ndarray, label_map: dict[str, int],
) -> None:
    inv_label_map = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)
    n_channels = X_raw.shape[1]

    unique_users = sorted(np.unique(user_ids))
    fold_accuracies: list[float] = []

    for held_out in unique_users:
        print(f"\n{'#' * 60}")
        print(f"# Fold: held-out user = {held_out}")
        print(f"{'#' * 60}")

        train_mask = user_ids != held_out
        test_mask = (user_ids == held_out) & (~is_augmented)

        X_train_raw, y_train = X_raw[train_mask], y[train_mask]
        X_test_raw, y_test = X_raw[test_mask], y[test_mask]

        fold_scaler = fit_scaler(X_train_raw)
        X_train = apply_scaler(X_train_raw, fold_scaler)
        X_test = apply_scaler(X_test_raw, fold_scaler)

        print(f"  Train: {len(X_train)}  Test: {len(X_test)} (originals only)")

        train_loader = DataLoader(
            ASLDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
        )
        test_loader = DataLoader(
            ASLDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False,
        )

        model = ASLClassifier(
            n_channels=n_channels, n_classes=n_classes,
        ).to(DEVICE)
        best_state, best_metric, preds, labels = train_loop(
            model, train_loader, test_loader, label=f"user {held_out}",
        )
        best_acc = -best_metric

        ckpt_path = MODELS_DIR / f"fold_{held_out}_best.pt"
        torch.save(best_state, ckpt_path)
        print(f"  Saved -> {ckpt_path}")

        fold_accuracies.append(best_acc)
        print_fold_report(held_out, best_acc, preds, labels, inv_label_map)

    accs = np.array(fold_accuracies)
    print(f"\n{'=' * 60}")
    print("LOUO Cross-Validation Summary")
    print(f"  Mean accuracy: {accs.mean():.1f}% +/- {accs.std():.1f}%")
    for uid, a in zip(unique_users, fold_accuracies):
        print(f"    user {uid}: {a:.1f}%")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Mode: Full training (all users)
# ---------------------------------------------------------------------------
def run_full(
    X_raw: np.ndarray, y: np.ndarray, label_map: dict[str, int],
) -> None:
    n_classes = len(label_map)
    n_channels = X_raw.shape[1]

    # Fit scaler on all data
    print("Fitting scaler on ALL data ...")
    full_scaler = fit_scaler(X_raw)
    SCALER_FULL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_scaler, SCALER_FULL_PATH)
    print(f"Saved scaler -> {SCALER_FULL_PATH}")

    X_all = apply_scaler(X_raw, full_scaler)

    # Save training stats for calibration
    save_training_stats(X_raw, full_scaler)

    # Train final model
    print(f"\nTraining final model on ALL data ({len(X_all)} samples) ...")
    all_loader = DataLoader(
        ASLDataset(X_all, y), batch_size=BATCH_SIZE, shuffle=True,
    )

    model = ASLClassifier(
        n_channels=n_channels, n_classes=n_classes,
    ).to(DEVICE)

    best_state, _, _, _ = train_loop(model, all_loader, None, label="final")

    final_path = MODELS_DIR / "classifier_final.pt"
    torch.save(best_state, final_path)
    print(f"\nSaved final model -> {final_path}")

    final_label_path = MODELS_DIR / "label_map.json"
    with open(final_label_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map  -> {final_label_path}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL classifier")
    parser.add_argument(
        "--mode", choices=["louo", "full"], default="full",
        help="louo = leave-one-user-out CV; full = train on all data (default)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {WINDOWS_PATH} ...")
    data = np.load(WINDOWS_PATH)
    X_raw = data["X"]
    y = data["y"]
    user_ids = data["user_ids"]
    is_augmented = data["is_augmented"].astype(bool)
    print(f"  X: {X_raw.shape}  y: {y.shape}  users: {np.unique(user_ids)}")
    print(f"  Original: {(~is_augmented).sum()}  Augmented: {is_augmented.sum()}")
    print(f"  Device: {DEVICE}")

    with open(LABEL_MAP_PATH) as f:
        label_map: dict[str, int] = json.load(f)

    if args.mode == "louo":
        run_louo(X_raw, y, user_ids, is_augmented, label_map)
    else:
        run_full(X_raw, y, label_map)


if __name__ == "__main__":
    main()
