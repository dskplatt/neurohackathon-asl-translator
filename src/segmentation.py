"""
EMG-based letter segmentation for real-time ASL translation.

Three-state machine: RESTING → SIGNING → COOLDOWN → RESTING.
A letter is emitted when the user completes a sign, then the system enters
COOLDOWN and waits for a minimum pause (COOLDOWN_MS) *and* the user to rest
before accepting a new sign.
"""

import threading
from collections import deque

import numpy as np

# ── Tunable constants ────────────────────────────────────────────────────────
SAMPLE_RATE       = 200    # Hz  (Myo EMG rate)
ACTIVE_THRESHOLD  = 12.0   # RMS above this → signing
REST_THRESHOLD    = 5.0    # RMS below this → resting
MIN_WINDOW_MS     = 200    # discard windows shorter than this
MAX_WINDOW_MS     = 2000   # force-close windows longer than this
DEBOUNCE_MS       = 300    # RMS must stay low this long to confirm rest
COOLDOWN_MS       = 800    # minimum pause after a letter before accepting the next
RMS_SMOOTH_FRAMES = 15     # rolling window size for RMS smoothing

# Fixed capture mode — matches calibration data collection exactly
FIXED_CAPTURE_MODE   = True    # True = 400-frame fixed windows
                                # False = original RMS rest detection
FIXED_CAPTURE_FRAMES = 400     # must match calibration data frame count


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
        self._cooldown_frame_count = 0
        self._lock = threading.Lock()

    def push_frame(self, emg_frame: np.ndarray) -> None:
        """Ingest one 8-channel EMG frame. Thread-safe."""
        emg_frame = np.asarray(emg_frame, dtype=np.float64)
        rms = np.sqrt(np.mean(emg_frame ** 2))

        fire_signing_start = False
        fire_signing_end = False
        fire_letter_ready = False
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

                if FIXED_CAPTURE_MODE:
                    if len(self._active_buffer) >= FIXED_CAPTURE_FRAMES:
                        window = np.array(self._active_buffer)
                        duration_ms = len(window) / SAMPLE_RATE * 1000

                        window_rms = np.sqrt(np.mean(window ** 2))

                        self._state = "COOLDOWN"
                        self._active_buffer = []
                        self._rest_frame_count = 0
                        self._cooldown_frame_count = 0
                        self._rms_buffer.clear()
                        fire_signing_end = True

                        if window_rms < self.rest_threshold * 1.5:
                            print(f"Discarded false trigger window "
                                  f"(RMS={window_rms:.1f}, "
                                  f"threshold={self.rest_threshold * 1.5:.1f})")
                        else:
                            fire_letter_ready = True
                            print(f"Letter window: {len(window)} frames "
                                  f"({duration_ms:.0f}ms), RMS={window_rms:.1f}")

                else:
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
                                discard_msg = f"Discarded window: {duration_ms:.0f}ms (out of range)"
                    else:
                        self._rest_frame_count = 0

                    max_frames = int(MAX_WINDOW_MS / 1000 * SAMPLE_RATE)
                    if self._state == "SIGNING" and len(self._active_buffer) >= max_frames:
                        window = np.array(self._active_buffer)
                        self._state = "COOLDOWN"
                        self._active_buffer = []
                        self._rest_frame_count = 0
                        self._rms_buffer.clear()
                        fire_signing_end = True
                        fire_letter_ready = True

            elif self._state == "COOLDOWN":
                if smoothed_rms < self.rest_threshold:
                    self._rest_frame_count += 1
                    debounce_frames = int(DEBOUNCE_MS / 1000 * SAMPLE_RATE)
                    if self._rest_frame_count >= debounce_frames:
                        self._state = "RESTING"
                        self._rest_frame_count = 0
                else:
                    self._rest_frame_count = 0

        if fire_signing_start:
            self.on_signing_start()
        if fire_signing_end:
            self.on_signing_end()
        if fire_letter_ready:
            self.on_letter_ready(window)
        elif discard_msg:
            print(discard_msg)

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
        """Clear buffers and return to RESTING. Thresholds are kept."""
        with self._lock:
            self._state = "RESTING"
            self._active_buffer = []
            self._rms_buffer.clear()
            self._rest_frame_count = 0
            self._cooldown_frame_count = 0
