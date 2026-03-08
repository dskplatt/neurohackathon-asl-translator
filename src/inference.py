"""
Inference for ASL letter classification.

Consumes variable-length (n_timesteps, 8) EMG windows from segmentation,
extracts fixed 40-frame sliding sub-windows (matching training), and
classifies using either:
  - Nearest centroid (cosine similarity) if calibration centroids exist
  - Pretrained linear head as fallback
"""

import json
import os
import threading
from pathlib import Path

import joblib
import numpy as np
import torch

from src.model import ASLClassifier

WINDOW_SIZE = 40
STRIDE = 10
REST_THRESHOLD = 5.0


class LetterClassifier:
    def __init__(
        self,
        model_path: str | None = None,
        scaler_path: str | None = None,
        label_map_path: str = "models/label_map.json",
        centroid_path: str = "models/centroids.npz",
        device: torch.device | None = None,
    ):
        root = Path(__file__).resolve().parent.parent

        if model_path is None:
            personal_model = root / "models" / "classifier_personal.pt"
            personal_scaler = root / "models" / "scaler_personal.joblib"
            if personal_model.exists() and personal_scaler.exists():
                model_path = "models/classifier_personal.pt"
                scaler_path = "models/scaler_personal.joblib"
                self._using_personal = True
                print("Using PERSONAL model (calibrated for this user)")
            else:
                model_path = "models/classifier_final.pt"
                scaler_path = "models/scaler_full.joblib"
                self._using_personal = False
                print("Using PRETRAINED model (no personal calibration found)")
        else:
            self._using_personal = "personal" in str(model_path)

        model_path_resolved = root / model_path
        scaler_path_resolved = root / scaler_path
        label_map_path_resolved = root / label_map_path
        self._centroid_path = root / centroid_path

        for path, desc in [
            (model_path_resolved, "Model weights"),
            (scaler_path_resolved, "Scaler"),
            (label_map_path_resolved, "Label map"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{desc} not found at {path}. "
                    f"Run 'python training/train.py --mode full' first."
                )

        self.model = ASLClassifier(n_channels=11, n_timesteps=40, n_classes=26)
        self.model.load_state_dict(
            torch.load(model_path_resolved, map_location="cpu", weights_only=True)
        )
        self.model.eval()

        self.scaler = joblib.load(scaler_path_resolved)
        if getattr(self.scaler, "n_features_in_", None) != 11:
            raise ValueError(
                f"Scaler must have n_features_in_=11, got {getattr(self.scaler, 'n_features_in_', None)}"
            )

        with open(label_map_path_resolved) as f:
            label_map = json.load(f)
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

        self._centroids: np.ndarray | None = None
        if not self._using_personal:
            self.load_centroids()

        print(f"LetterClassifier ready on {self.device}")
        if self._using_personal:
            print("  Personal fine-tuned model ACTIVE")
        elif self._centroids is not None:
            print("  Centroid classifier ACTIVE (26 centroids loaded)")
        else:
            print("  Using pretrained linear head (no centroids found)")

    def reload(self) -> None:
        """Reload model from disk (re-runs auto-detect). Call after training."""
        self.__init__()
        print("Model reloaded.")

    def update_accel(self, accel: np.ndarray) -> None:
        """Update latest accelerometer reading (thread-safe)."""
        accel = np.asarray(accel, dtype=np.float64).ravel()
        if accel.size != 3:
            raise ValueError(f"accel must have 3 elements, got {accel.size}")
        with self._accel_lock:
            self._last_accel = accel.copy()

    def preprocess(self, emg_window: np.ndarray) -> torch.Tensor:
        """Convert variable-length (n, 8) EMG to (1, 11, 40) tensor.

        Uses interpolation when n != 40. Used by calibration for sub-windows.
        """
        from scipy.interpolate import interp1d

        emg_window = np.asarray(emg_window, dtype=np.float64)
        if emg_window.ndim != 2 or emg_window.shape[1] != 8:
            raise ValueError(
                f"emg_window must be (n_timesteps, 8), got {emg_window.shape}"
            )
        n_timesteps = emg_window.shape[0]

        accel_neutral = self.scaler.mean_[8:11].copy()
        accel_column = np.tile(accel_neutral, (n_timesteps, 1))
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

    def load_centroids(self) -> bool:
        """Load centroids from disk. Returns True if loaded."""
        if self._centroid_path.exists():
            data = np.load(self._centroid_path)
            self._centroids = data["centroids"].astype(np.float32)  # (26, 64)
            print("  Reloaded centroids from disk")
            return True
        self._centroids = None
        return False

    def predict(self, emg_window: np.ndarray) -> dict[str, float]:
        """Return {letter: probability} for all 26 letters.

        Extracts 40-frame sliding windows from the segmentation window.
        If centroids are loaded, classifies each chunk by cosine similarity
        to centroids and averages. Otherwise uses the pretrained linear head.
        """
        # #region agent log
        import json as _json, time as _time
        _log_path = "debug-88a71d.log"
        def _dbg(msg, data, hyp):
            try:
                with open(_log_path, "a") as _f:
                    _f.write(_json.dumps({"sessionId":"88a71d","location":"inference.py:predict","message":msg,"data":data,"hypothesisId":hyp,"timestamp":int(_time.time()*1000)}) + "\n")
            except Exception:
                pass
        # #endregion

        emg_window = np.asarray(emg_window, dtype=np.float64)
        if emg_window.ndim != 2 or emg_window.shape[1] != 8:
            raise ValueError(
                f"emg_window must be (n_timesteps, 8), got {emg_window.shape}"
            )
        n = emg_window.shape[0]

        with self._accel_lock:
            accel = self._last_accel.copy()

        # #region agent log
        _dbg("accel_state", {"accel": accel.tolist(), "all_zero": bool(np.all(accel == 0))}, "A")
        # #endregion

        accel_neutral = self.scaler.mean_[8:11].copy()
        accel_column = np.tile(accel_neutral, (n, 1))
        full = np.hstack([emg_window, accel_column])  # (n, 11)
        full_scaled = self.scaler.transform(full.astype(np.float32))

        # #region agent log
        emg_rms_per_frame = np.sqrt(np.mean(emg_window ** 2, axis=1))
        active_mask = emg_rms_per_frame > 12.0
        _dbg("window_and_scaler_stats", {
            "n_frames": int(n),
            "active_pct": round(float(np.mean(active_mask)) * 100, 1),
            "scaled_mean": round(float(np.mean(full_scaled)), 4),
            "scaled_std": round(float(np.std(full_scaled)), 4),
            "scaled_min": round(float(np.min(full_scaled)), 4),
            "scaled_max": round(float(np.max(full_scaled)), 4),
            "accel_raw": [round(float(v), 3) for v in accel],
        }, "B,C,G")
        # #endregion

        W = WINDOW_SIZE
        S = STRIDE

        if n >= W:
            chunks = []
            for start in range(0, n - W + 1, S):
                chunk = full_scaled[start:start + W]  # (40, 11)
                chunks.append(chunk.T)  # (11, 40)
            batch = torch.tensor(np.array(chunks), dtype=torch.float32).to(self.device)
        else:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, n)
            x_new = np.linspace(0, 1, W)
            f = interp1d(x_old, full_scaled, axis=0, kind="linear")
            padded = f(x_new)
            batch = torch.tensor(padded.T, dtype=torch.float32).unsqueeze(0).to(self.device)

        # #region agent log
        _dbg("batch_info", {"n_chunks": int(batch.shape[0]), "using_centroids": self._centroids is not None, "using_personal": self._using_personal}, "B,D")
        # #endregion

        with torch.no_grad():
            features = self.model.get_features(batch)  # (num_chunks, 64)

            # #region agent log
            feat_np_log = features.cpu().numpy()
            feat_norms = np.linalg.norm(feat_np_log, axis=1)
            cosine_sims = []
            if len(feat_np_log) > 1:
                for i in range(min(5, len(feat_np_log))):
                    for j in range(i+1, min(5, len(feat_np_log))):
                        cos = np.dot(feat_np_log[i], feat_np_log[j]) / (np.linalg.norm(feat_np_log[i]) * np.linalg.norm(feat_np_log[j]) + 1e-8)
                        cosine_sims.append(float(cos))
            raw_logits = self.model.classifier_head(features)
            logit_np = raw_logits.cpu().numpy()
            _dbg("feature_analysis", {
                "feat_norm_mean": round(float(np.mean(feat_norms)), 4),
                "feat_norm_std": round(float(np.std(feat_norms)), 4),
                "feat_mean": round(float(np.mean(feat_np_log)), 4),
                "feat_std": round(float(np.std(feat_np_log)), 4),
                "chunk_cosine_sims": [round(c, 4) for c in cosine_sims[:10]],
                "logit_mean": round(float(np.mean(logit_np)), 4),
                "logit_std": round(float(np.std(logit_np)), 4),
                "logit_range": round(float(np.max(logit_np) - np.min(logit_np)), 4),
                "logit_sample_chunk0": [round(float(v), 3) for v in logit_np[0]],
            }, "F,H")
            # #endregion

            if self._centroids is not None:
                feat_np = features.cpu().numpy()
                norms = np.linalg.norm(feat_np, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                feat_np = feat_np / norms

                # Cosine similarity: (num_chunks, 26)
                similarities = feat_np @ self._centroids.T

                # Per-chunk softmax with temperature, then average
                temperature = 0.1
                shifted = (similarities - similarities.max(axis=1, keepdims=True)) / temperature
                exp_sim = np.exp(shifted)
                chunk_probs = exp_sim / exp_sim.sum(axis=1, keepdims=True)
                probs = chunk_probs.mean(axis=0)
            else:
                all_probs = self.model.get_probabilities(batch)  # (n_chunks, 26)

                # #region agent log
                per_chunk_top = torch.argmax(all_probs, dim=1).cpu().numpy().tolist()
                per_chunk_conf = torch.max(all_probs, dim=1).values.cpu().numpy().tolist()
                from collections import Counter
                chunk_votes = Counter(per_chunk_top)
                _dbg("per_chunk_predictions", {
                    "chunk_top_indices": per_chunk_top,
                    "chunk_confidences": [round(c, 3) for c in per_chunk_conf],
                    "vote_counts": {self.int_to_letter.get(k, str(k)): v for k, v in chunk_votes.most_common(5)},
                    "total_chunks": len(per_chunk_top),
                    "dominant_letter": self.int_to_letter.get(chunk_votes.most_common(1)[0][0], "?"),
                    "dominant_pct": round(chunk_votes.most_common(1)[0][1] / len(per_chunk_top) * 100, 1),
                }, "B,D")
                # #endregion

                probs = all_probs.mean(dim=0).cpu().numpy()

        # #region agent log
        top5_idx = np.argsort(probs)[::-1][:5]
        _dbg("final_prediction", {
            "top5": [{"letter": self.int_to_letter[int(i)], "prob": round(float(probs[i]), 4)} for i in top5_idx],
            "entropy": round(float(-np.sum(probs * np.log(probs + 1e-10))), 4),
            "max_prob": round(float(np.max(probs)), 4),
        }, "D,E")
        # #endregion

        return {self.int_to_letter[i]: float(probs[i]) for i in range(26)}


if __name__ == "__main__":
    import numpy as np

    classifier = LetterClassifier()

    fake_window = np.random.randn(50, 8).astype(np.float32) * 20
    result = classifier.predict(fake_window)

    assert isinstance(result, dict), "FAIL: result not a dict"
    assert len(result) == 26, f"FAIL: expected 26 keys, got {len(result)}"
    assert all(k == k.lower() for k in result), "FAIL: keys not lowercase"
    total_prob = sum(result.values())
    assert abs(total_prob - 1.0) < 1e-3, f"FAIL: probs sum to {total_prob}"
    print(f"Test 1 PASS — probs sum to {total_prob:.6f}")

    for n_timesteps in [30, 40, 60, 150, 300]:
        window = np.random.randn(n_timesteps, 8).astype(np.float32)
        result = classifier.predict(window)
        assert len(result) == 26
        print(f"Test 2 PASS — n_timesteps={n_timesteps}")

    top_letter = max(result, key=result.get)
    assert top_letter in "abcdefghijklmnopqrstuvwxyz"
    print(f"Test 3 PASS — top letter: {top_letter} ({result[top_letter]:.3f})")

    classifier.update_accel(np.array([0.1, 0.2, 9.8]))
    result2 = classifier.predict(fake_window)
    assert len(result2) == 26
    print("Test 4 PASS — update_accel + predict")

    print("\nAll inference tests passed.")
