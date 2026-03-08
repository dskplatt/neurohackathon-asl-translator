"""
Calibration system for personalizing the ASL classifier to a new user.

Mode 1 — Signal calibration:
    Collects 6s of EMG (3s rest + 3s active), computes the user's
    RMS levels, and sets segmentation thresholds proportionally.

Mode 2 — Letter calibration (nearest centroid):
    Guides the user through signing all 26 letters multiple times.
    Each captured window is split into 40-frame chunks (matching training).
    Features are extracted from the frozen backbone, L2-normalized, and
    averaged per letter to produce 26 centroids. At inference, cosine
    similarity to centroids replaces the linear head.
    Saves models/centroids.npz.
"""

import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.model import ASLClassifier

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 200


class CalibrationManager:

    def __init__(
        self,
        model: ASLClassifier,
        preprocess_fn: Callable[[np.ndarray], torch.Tensor],
        segmentation,
        broadcast_fn: Callable[[dict], None],
        device: torch.device,
        on_calibration_done: Callable[[], None] | None = None,
    ):
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.segmentation = segmentation
        self.broadcast = broadcast_fn
        self.device = device
        self.on_calibration_done = on_calibration_done or (lambda: None)

        self.state = "idle"
        self.letters = list("abcdefghijklmnopqrstuvwxyz")
        self.letter_to_idx = {l: i for i, l in enumerate(self.letters)}

        # Mode 1
        self._rest_rms: list[float] = []
        self._active_rms: list[float] = []
        self._signal_phase = "rest"
        self._signal_frame_count = 0
        self._PHASE_FRAMES = SAMPLE_RATE * 3

        # Mode 2
        self.current_letter_idx = 0
        self.collected_windows: dict[str, list[np.ndarray]] = {}
        self._cooldown = False
        self._COOLDOWN_SECONDS = 2.0
        self._current_letter_captures: list[np.ndarray] = []
        self.captures_per_letter = 5

    # ── Mode 1: Signal Calibration ──────────────────────────────────

    def start_signal_calibration(self):
        self.state = "signal_cal"
        self._signal_phase = "rest"
        self._signal_frame_count = 0
        self._rest_rms.clear()
        self._active_rms.clear()
        self.broadcast({"type": "calibration_start", "total_letters": 0})
        print("Signal calibration started — relax arm for 3 seconds...")

    def on_signal_frame(self, emg_frame: np.ndarray):
        if self.state != "signal_cal":
            return
        rms = float(np.sqrt(np.mean(np.asarray(emg_frame, dtype=np.float64) ** 2)))
        self._signal_frame_count += 1

        if self._signal_phase == "rest":
            self._rest_rms.append(rms)
            if self._signal_frame_count >= self._PHASE_FRAMES:
                self._signal_frame_count = 0
                self._signal_phase = "active"
                self.broadcast({
                    "type": "calibration_prompt",
                    "letter": "*",
                    "index": 0,
                    "total": 1,
                })
                print("Now make a fist and hold for 3 seconds...")
        elif self._signal_phase == "active":
            self._active_rms.append(rms)
            if self._signal_frame_count >= self._PHASE_FRAMES:
                self._finish_signal_calibration()

    def _finish_signal_calibration(self):
        rest_mean = float(np.mean(self._rest_rms))
        active_mean = float(np.mean(self._active_rms))
        gap = max(active_mean - rest_mean, 1.0)

        active_threshold = rest_mean + gap * 0.45
        rest_threshold = rest_mean + gap * 0.15
        active_threshold = max(active_threshold, rest_threshold + 1.0)

        self.segmentation.update_thresholds(active_threshold, rest_threshold)
        self.state = "idle"
        self.broadcast({"type": "calibration_complete", "accuracy": 0.0})
        print(
            f"Signal calibration done: rest={rest_mean:.1f}, active={active_mean:.1f} "
            f"-> thresholds=({active_threshold:.1f}, {rest_threshold:.1f})"
        )

    # ── Mode 2: Letter Calibration (Nearest Centroid) ──────────────

    def start_letter_calibration(self, captures_per_letter: int = 5):
        self.state = "letter_cal"
        self.current_letter_idx = 0
        self.captures_per_letter = captures_per_letter
        self.collected_windows.clear()
        self._current_letter_captures = []
        self.broadcast({"type": "calibration_start", "total_letters": 26})
        self._prompt_current_letter()

    def _prompt_current_letter(self):
        letter = self.letters[self.current_letter_idx]
        capture_num = len(self._current_letter_captures) + 1
        self.broadcast({
            "type": "calibration_prompt",
            "letter": letter,
            "index": self.current_letter_idx,
            "total": 26,
        })
        print(
            f"Calibration: sign '{letter.upper()}' "
            f"({self.current_letter_idx + 1}/26, "
            f"capture {capture_num}/{self.captures_per_letter})"
        )

    def on_window_captured(self, window: np.ndarray):
        if self.state != "letter_cal" or self._cooldown:
            return
        letter = self.letters[self.current_letter_idx]
        self._current_letter_captures.append(window.copy())
        capture_num = len(self._current_letter_captures)

        if capture_num < self.captures_per_letter:
            self._cooldown = True
            msg = (
                f"Got '{letter.upper()}' ({capture_num}/{self.captures_per_letter})"
                f" — relax, then sign it again"
            )
            print(f"  {msg}")
            self.broadcast({"type": "calibration_rest", "message": msg})

            def _repeat():
                import time
                time.sleep(self._COOLDOWN_SECONDS)
                self._cooldown = False
                self._prompt_current_letter()

            threading.Thread(target=_repeat, daemon=True).start()
            return

        self.collected_windows[letter] = list(self._current_letter_captures)
        self._current_letter_captures = []
        msg = f"Got '{letter.upper()}' done — relax for 2 seconds"
        print(f"  {msg}")
        self.broadcast({
            "type": "calibration_captured",
            "letter": letter,
            "index": self.current_letter_idx,
        })
        self.broadcast({"type": "calibration_rest", "message": msg})
        self.current_letter_idx += 1

        if self.current_letter_idx >= 26:
            self.state = "training"
            print("All letters captured — computing centroids...")
            self.broadcast({"type": "calibration_rest", "message": "Computing centroids..."})
            threading.Thread(target=self._compute_centroids, daemon=True).start()
        else:
            self._cooldown = True

            def _after_cooldown():
                import time
                time.sleep(self._COOLDOWN_SECONDS)
                self._cooldown = False
                self._prompt_current_letter()

            threading.Thread(target=_after_cooldown, daemon=True).start()

    # ── Sub-window splitting (matches training/preprocess.py) ──────

    _WINDOW_SIZE = 40
    _STRIDE = 10

    def _split_into_subwindows(self, window: np.ndarray) -> list[np.ndarray]:
        """Extract fixed 40-frame chunks with stride 10, matching training."""
        n = window.shape[0]
        W = self._WINDOW_SIZE
        S = self._STRIDE
        subs = []
        for start in range(0, n - W + 1, S):
            subs.append(window[start:start + W])
        if not subs and n > 0:
            subs.append(window[:min(n, W)])
        return subs

    # ── Centroid computation ────────────────────────────────────────

    def _compute_centroids(self):
        try:
            self.model.eval()
            centroids = np.zeros((26, 64), dtype=np.float32)
            counts = np.zeros(26, dtype=int)

            for letter, windows in self.collected_windows.items():
                idx = self.letter_to_idx[letter]
                all_features = []

                for window in windows:
                    for sub in self._split_into_subwindows(window):
                        t = self.preprocess_fn(sub).to(self.device)
                        with torch.no_grad():
                            feat = self.model.get_features(t)
                        feat_np = feat.squeeze(0).cpu().numpy()
                        norm = np.linalg.norm(feat_np)
                        if norm > 0:
                            feat_np = feat_np / norm
                        all_features.append(feat_np)

                if all_features:
                    centroid = np.mean(all_features, axis=0).astype(np.float32)
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm
                    centroids[idx] = centroid
                    counts[idx] = len(all_features)

            save_path = ROOT / "models" / "centroids.npz"
            np.savez(save_path, centroids=centroids, letters=self.letters)

            n_letters = int(np.sum(counts > 0))
            total_features = int(np.sum(counts))
            print(f"Saved {n_letters} centroids ({total_features} total feature vectors) to {save_path.name}")

            self.on_calibration_done()

            self.state = "idle"
            self.broadcast({"type": "calibration_complete", "accuracy": 1.0})

        except Exception:
            import traceback
            traceback.print_exc()
            self.state = "idle"
            self.broadcast({"type": "calibration_complete", "accuracy": 0.0})
