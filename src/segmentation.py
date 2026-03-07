"""
EMG-based letter segmentation for real-time ASL translation.

Uses a two-state machine (RESTING / SIGNING) driven by smoothed RMS energy
to detect letter boundaries in a continuous Myo EMG stream.
"""

import threading
from collections import deque

import numpy as np

# ── Tunable constants ────────────────────────────────────────────────────────
SAMPLE_RATE       = 200    # Hz  (Myo EMG rate)
ACTIVE_THRESHOLD  = 15.0   # RMS above this → signing
REST_THRESHOLD    = 8.0    # RMS below this → resting
MIN_WINDOW_MS     = 150    # discard windows shorter than this
MAX_WINDOW_MS     = 1500   # force-close windows longer than this
DEBOUNCE_MS       = 100    # RMS must stay low this long to confirm rest
RMS_SMOOTH_FRAMES = 10     # rolling window size for RMS smoothing (50 ms)


class SegmentationStateMachine:
    """Segments a continuous EMG stream into per-letter windows."""

    def __init__(
        self,
        on_letter_ready,
        on_signing_start=None,
        on_signing_end=None,
        active_threshold: float = ACTIVE_THRESHOLD,
        rest_threshold: float = REST_THRESHOLD,
    ):
        self.on_letter_ready = on_letter_ready
        self.on_signing_start = on_signing_start or (lambda: None)
        self.on_signing_end = on_signing_end or (lambda: None)

        self.active_threshold = active_threshold
        self.rest_threshold = rest_threshold

        self._state = "RESTING"
        self._active_buffer: list[np.ndarray] = []
        self._rms_buffer: deque[float] = deque(maxlen=RMS_SMOOTH_FRAMES)
        self._rest_frame_count = 0
        self._lock = threading.Lock()

    # ── public API ───────────────────────────────────────────────────────

    def push_frame(self, emg_frame: np.ndarray) -> None:
        """Ingest one 8-channel EMG frame.  Thread-safe."""
        emg_frame = np.asarray(emg_frame, dtype=np.float64)

        rms = np.sqrt(np.mean(emg_frame ** 2))

        fire_signing_start = False
        fire_signing_end = False
        fire_letter_ready = False
        fire_force_close = False
        window = None
        discard_msg = None

        with self._lock:
            self._rms_buffer.append(rms)
            smoothed_rms = np.mean(self._rms_buffer)

            if self._state == "RESTING":
                if smoothed_rms > self.active_threshold:
                    self._state = "SIGNING"
                    self._active_buffer = [emg_frame.copy()]
                    self._rest_frame_count = 0
                    fire_signing_start = True

            elif self._state == "SIGNING":
                self._active_buffer.append(emg_frame.copy())

                if smoothed_rms < self.rest_threshold:
                    self._rest_frame_count += 1
                    debounce_frames = int(DEBOUNCE_MS / 1000 * SAMPLE_RATE)

                    if self._rest_frame_count >= debounce_frames:
                        window = np.array(self._active_buffer)
                        duration_ms = len(window) / SAMPLE_RATE * 1000

                        self._state = "RESTING"
                        self._active_buffer = []
                        self._rest_frame_count = 0

                        fire_signing_end = True

                        if MIN_WINDOW_MS <= duration_ms <= MAX_WINDOW_MS:
                            fire_letter_ready = True
                        else:
                            discard_msg = (
                                f"Discarded window: {duration_ms:.0f}ms "
                                f"(out of range)"
                            )
                else:
                    self._rest_frame_count = 0

                max_frames = int(MAX_WINDOW_MS / 1000 * SAMPLE_RATE)
                if self._state == "SIGNING" and len(self._active_buffer) >= max_frames:
                    window = np.array(self._active_buffer)
                    self._state = "RESTING"
                    self._active_buffer = []
                    self._rest_frame_count = 0
                    fire_force_close = True

        # Callbacks outside the lock to avoid deadlock
        if fire_signing_start:
            self.on_signing_start()
        if fire_signing_end:
            self.on_signing_end()
        if fire_letter_ready:
            self.on_letter_ready(window)
        elif discard_msg:
            print(discard_msg)
        if fire_force_close:
            self.on_signing_end()
            self.on_letter_ready(window)
            print("Force-closed window at MAX_WINDOW_MS")

    def update_thresholds(
        self, active_threshold: float, rest_threshold: float
    ) -> None:
        """Hot-swap thresholds (e.g. after calibration)."""
        if active_threshold <= 0 or rest_threshold <= 0:
            raise ValueError("Thresholds must be positive")
        if active_threshold <= rest_threshold:
            raise ValueError("active_threshold must be greater than rest_threshold")

        with self._lock:
            self.active_threshold = float(active_threshold)
            self.rest_threshold = float(rest_threshold)

        print(
            f"Thresholds updated → active={self.active_threshold:.1f}, "
            f"rest={self.rest_threshold:.1f}"
        )

    def reset(self) -> None:
        """Clear buffers and return to RESTING.  Thresholds are kept."""
        with self._lock:
            self._state = "RESTING"
            self._active_buffer = []
            self._rms_buffer.clear()
            self._rest_frame_count = 0


# ── Offline mock test ────────────────────────────────────────────────────────

def run_mock_test() -> None:
    import pandas as pd
    from pathlib import Path

    csv_path = Path(__file__).resolve().parent.parent / "data" / "combined_dataset.csv"
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    df = df[df["user_id"] == 1]

    emg_cols = [f"emg_{i}" for i in range(1, 9)]
    labels_sorted = sorted(df["label"].unique())

    total_windows = 0

    for label in labels_sorted:
        subset = df[df["label"] == label]
        frames_fed = 0
        windows: list[np.ndarray] = []

        sm = SegmentationStateMachine(
            on_letter_ready=lambda w, _w=windows: _w.append(w),
        )

        for _, row in subset.iterrows():
            sm.push_frame(row[emg_cols].values)
            frames_fed += 1

        # Feed a burst of silence to flush any trailing SIGNING state
        silence = np.zeros(8)
        for _ in range(int(DEBOUNCE_MS / 1000 * SAMPLE_RATE) + 5):
            sm.push_frame(silence)

        n_windows = len(windows)
        avg_len_ms = (
            np.mean([len(w) / SAMPLE_RATE * 1000 for w in windows])
            if windows
            else 0.0
        )
        total_windows += n_windows

        print(
            f"  label={label!s:>2}  frames_fed={frames_fed:5d}  "
            f"windows={n_windows:3d}  avg_len={avg_len_ms:6.1f}ms"
        )

    print(f"\nTotal windows detected: {total_windows}")


if __name__ == "__main__":
    run_mock_test()
