"""Integration test: real model predictions → WordResolver → correct word?

Loads the trained ASL classifier and scaler, pulls real samples from the
dataset for each letter of a target word, runs inference to get probability
distributions, and checks whether WordResolver recovers the word.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ASLClassifier
from src.word_resolver import WordResolver

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pipeline():
    """Load trained model, scaler, label map, and dataset."""
    model_path = PROJECT_ROOT / "models" / "classifier_final.pt"
    scaler_path = PROJECT_ROOT / "models" / "scaler_full.joblib"
    label_map_path = PROJECT_ROOT / "data" / "label_map.json"
    data_path = PROJECT_ROOT / "data" / "windows.npz"

    with open(label_map_path) as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)

    data = np.load(data_path)
    X_raw = data["X"]
    y = data["y"]
    is_augmented = data["is_augmented"].astype(bool)

    scaler = joblib.load(scaler_path)

    model = ASLClassifier(n_channels=X_raw.shape[1], n_classes=n_classes).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model, scaler, label_map, inv_label_map, X_raw, y, is_augmented


def apply_scaler(X: np.ndarray, scaler) -> np.ndarray:
    N, C, T = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, C)
    scaled = scaler.transform(flat)
    return scaled.reshape(N, T, C).transpose(0, 2, 1).astype(np.float32)


def pick_samples_for_word(word, label_map, y, is_augmented, rng):
    """Return indices of one original sample per letter in the word."""
    indices = []
    for ch in word.lower():
        label_id = label_map[ch]
        candidates = np.where((y == label_id) & (~is_augmented))[0]
        if len(candidates) == 0:
            candidates = np.where(y == label_id)[0]
        idx = rng.choice(candidates)
        indices.append(idx)
    return indices


@torch.no_grad()
def get_letter_distributions(model, X_samples, inv_label_map):
    """Run model on samples and return list of {letter: prob} dicts."""
    tensor = torch.from_numpy(X_samples).to(DEVICE)
    probs = model.get_probabilities(tensor)  # (N, 26)
    probs_np = probs.cpu().numpy()

    distributions = []
    for row in probs_np:
        dist = {inv_label_map[i]: float(row[i]) for i in range(len(row))}
        distributions.append(dist)
    return distributions


def run_word_test(word, model, scaler, label_map, inv_label_map,
                  X_raw, y, is_augmented, resolver, rng, top_n=5):
    """Test a single word end-to-end. Returns (resolved_word, top_results, distributions)."""
    indices = pick_samples_for_word(word, label_map, y, is_augmented, rng)
    X_samples = apply_scaler(X_raw[indices], scaler)

    distributions = get_letter_distributions(model, X_samples, inv_label_map)

    print(f"\n{'─' * 60}")
    print(f"  Word: \"{word.upper()}\"")
    print(f"  Per-letter top-1 predictions from model:")
    for i, (ch, dist) in enumerate(zip(word, distributions)):
        top_letter = max(dist, key=dist.get)
        top_prob = dist[top_letter]
        true_prob = dist[ch.lower()]
        marker = "✓" if top_letter == ch.lower() else "✗"
        print(f"    slot {i}: true='{ch}' (p={true_prob:.3f})  "
              f"pred='{top_letter}' (p={top_prob:.3f})  {marker}")

    results = resolver.resolve(distributions, top_n=top_n)

    print(f"  WordResolver top-{top_n}:")
    for rank, (w, score) in enumerate(results, 1):
        flag = " ◀" if w == word.lower() else ""
        print(f"    {rank}. {w:15s}  score={score:.2e}{flag}")

    resolved = results[0][0]
    status = "PASS" if resolved == word.lower() else "FAIL"
    print(f"  Result: {status} (resolved '{resolved}', expected '{word.lower()}')")
    return resolved, results, distributions


def main():
    print("=" * 60)
    print("  Integration Test: Model + WordResolver")
    print("=" * 60)

    model, scaler, label_map, inv_label_map, X_raw, y, is_aug = load_pipeline()
    resolver = WordResolver()
    rng = np.random.default_rng(42)

    test_words = ["hello", "world", "cat", "fish", "jump", "sign", "help", "read"]

    results_summary = []
    for word in test_words:
        valid = all(ch in label_map for ch in word.lower())
        if not valid:
            print(f"\n  Skipping '{word}' — letters not all in label map")
            continue
        resolved, _, _ = run_word_test(
            word, model, scaler, label_map, inv_label_map,
            X_raw, y, is_aug, resolver, rng,
        )
        results_summary.append((word, resolved))

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    passed = 0
    for word, resolved in results_summary:
        status = "PASS" if resolved == word.lower() else "FAIL"
        if resolved == word.lower():
            passed += 1
        print(f"  {status}  {word:10s} → {resolved}")

    total = len(results_summary)
    print(f"\n  {passed}/{total} words resolved correctly")

    # Also test: run same word 5 times with different random samples
    # to see how robust the pipeline is
    print(f"\n{'=' * 60}")
    print("  Robustness: 'hello' with 10 different random samples")
    print(f"{'=' * 60}")
    hello_wins = 0
    for trial in range(10):
        trial_rng = np.random.default_rng(trial)
        indices = pick_samples_for_word("hello", label_map, y, is_aug, trial_rng)
        X_samples = apply_scaler(X_raw[indices], scaler)
        distributions = get_letter_distributions(model, X_samples, inv_label_map)
        top_results = resolver.resolve(distributions, top_n=3)
        resolved = top_results[0][0]
        status = "✓" if resolved == "hello" else "✗"
        if resolved == "hello":
            hello_wins += 1
        top3 = ", ".join(f"{w}" for w, _ in top_results)
        print(f"  trial {trial:2d}: {resolved:10s}  top3=[{top3}]  {status}")

    print(f"\n  'hello' resolved {hello_wins}/10 trials")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
