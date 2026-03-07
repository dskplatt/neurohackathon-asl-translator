"""
Inference for ASL letter classification.

Consumes (n_timesteps, 8) EMG windows from segmentation, appends live
accelerometer, resizes to (40, 11), scales, and runs the classifier to
return {letter: probability} for all 26 letters.
"""

import json
import threading
from pathlib import Path

import joblib
import numpy as np
import torch

from src.model import ASLClassifier


class LetterClassifier:
    def __init__(
        self,
        model_path: str = "models/classifier_final.pt",
        scaler_path: str = "models/scaler_full.joblib",
        label_map_path: str = "models/label_map.json",
        device: torch.device | None = None,
    ):
        root = Path(__file__).resolve().parent.parent
        model_path = root / model_path
        scaler_path = root / scaler_path
        label_map_path = root / label_map_path

        for path, desc in [
            (model_path, "Model weights"),
            (scaler_path, "Scaler"),
            (label_map_path, "Label map"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{desc} not found at {path}. "
                    f"Run 'python training/train.py --mode full' first."
                )

        self.model = ASLClassifier(n_channels=11, n_timesteps=40, n_classes=26)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model.eval()

        self.scaler = joblib.load(scaler_path)
        if getattr(self.scaler, "n_features_in_", None) != 11:
            raise ValueError(
                f"Scaler must have n_features_in_=11, got {getattr(self.scaler, 'n_features_in_', None)}"
            )

        with open(label_map_path) as f:
            label_map = json.load(f)
        # Support both {"0": "a", ...} and {"a": 0, ...}
        try:
            self.int_to_letter = {int(k): v for k, v in label_map.items()}
        except (ValueError, TypeError):
            self.int_to_letter = {v: k for k, v in label_map.items()}
        if len(self.int_to_letter) != 26:
            raise ValueError(
                f"label_map must have 26 entries, got {len(self.int_to_letter)}"
            )

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self._last_accel = np.zeros(3, dtype=np.float64)
        self._accel_lock = threading.Lock()

        print(f"LetterClassifier ready on {self.device}")

    def update_accel(self, accel: np.ndarray) -> None:
        """Update latest accelerometer reading (thread-safe). accel: (3,) acc_x, acc_y, acc_z."""
        accel = np.asarray(accel, dtype=np.float64).ravel()
        if accel.size != 3:
            raise ValueError(f"accel must have 3 elements, got {accel.size}")
        with self._accel_lock:
            self._last_accel = accel.copy()

    def preprocess(self, emg_window: np.ndarray) -> torch.Tensor:
        """Convert (n_timesteps, 8) EMG to (1, 11, 40) tensor on self.device."""
        from scipy.interpolate import interp1d

        emg_window = np.asarray(emg_window, dtype=np.float64)
        if emg_window.ndim != 2 or emg_window.shape[1] != 8:
            raise ValueError(
                f"emg_window must be (n_timesteps, 8), got {emg_window.shape}"
            )
        n_timesteps = emg_window.shape[0]

        with self._accel_lock:
            accel = self._last_accel.copy()
        accel_column = np.tile(accel, (n_timesteps, 1))
        window = np.hstack([emg_window, accel_column])

        if n_timesteps != 40:
            x_old = np.linspace(0, 1, n_timesteps)
            x_new = np.linspace(0, 1, 40)
            f = interp1d(x_old, window, axis=0, kind="linear")
            window = f(x_new)

        window = self.scaler.transform(window.astype(np.float32))
        tensor = torch.tensor(window.T, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, emg_window: np.ndarray) -> dict[str, float]:
        """Return {letter: probability} for all 26 letters."""
        with torch.no_grad():
            tensor = self.preprocess(emg_window)
            prob_tensor = self.model.get_probabilities(tensor)
        probs = prob_tensor.squeeze(0).cpu().numpy()
        return {self.int_to_letter[i]: float(probs[i]) for i in range(26)}


if __name__ == "__main__":
    import numpy as np

    classifier = LetterClassifier()

    # Test 1: random window (no real Myo needed)
    fake_window = np.random.randn(50, 8).astype(np.float32) * 20
    result = classifier.predict(fake_window)

    assert isinstance(result, dict), "FAIL: result not a dict"
    assert len(result) == 26, f"FAIL: expected 26 keys, got {len(result)}"
    assert all(k == k.lower() for k in result), "FAIL: keys not lowercase"
    total_prob = sum(result.values())
    assert abs(total_prob - 1.0) < 1e-3, f"FAIL: probs sum to {total_prob}"
    print(f"Test 1 PASS — probs sum to {total_prob:.6f}")

    # Test 2: variable-length windows
    for n_timesteps in [30, 40, 60, 150, 300]:
        window = np.random.randn(n_timesteps, 8).astype(np.float32)
        result = classifier.predict(window)
        assert len(result) == 26
        print(f"Test 2 PASS — n_timesteps={n_timesteps}")

    # Test 3: top prediction is a valid letter
    top_letter = max(result, key=result.get)
    assert top_letter in "abcdefghijklmnopqrstuvwxyz"
    print(f"Test 3 PASS — top letter: {top_letter} ({result[top_letter]:.3f})")

    # Test 4: update_accel works
    classifier.update_accel(np.array([0.1, 0.2, 9.8]))
    result2 = classifier.predict(fake_window)
    assert len(result2) == 26
    print("Test 4 PASS — update_accel + predict")

    print("\nAll inference tests passed.")
