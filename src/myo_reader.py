"""
Myo armband acquisition layer.

Streams EMG (200 Hz, 8 channels) and IMU (50 Hz, accelerometer) data
from a Thalmic Myo via pyomyo, and dispatches pose events for gesture
control (wave-in / wave-out).
"""

import threading

import numpy as np


class MyoReader:
    """Thin wrapper that routes Myo callbacks into the application pipeline."""

    def __init__(
        self,
        on_emg_frame,
        on_accel_frame,
        on_wave_right,
        on_wave_left,
    ):
        self.on_emg_frame = on_emg_frame
        self.on_accel_frame = on_accel_frame
        self.on_wave_right = on_wave_right
        self.on_wave_left = on_wave_left
        self._running = False
        self._myo = None

    # ── Myo callbacks ─────────────────────────────────────────────────

    def _handle_emg(self, emg_data):
        self.on_emg_frame(np.array(emg_data, dtype=np.float32))

    def _handle_imu(self, quat, accel, gyro):
        self.on_accel_frame(np.array(accel, dtype=np.float32))

    def _handle_pose(self, pose):
        from pyomyo.Pose import Pose

        if pose == Pose.WAVE_OUT:
            self.on_wave_right()
        elif pose == Pose.WAVE_IN:
            self.on_wave_left()

    # ── Public API ────────────────────────────────────────────────────

    def start(self):
        from pyomyo import Myo, emg_mode

        self._myo = Myo(mode=emg_mode.FILTERED)
        self._myo.add_emg_handler(self._handle_emg)
        self._myo.add_imu_handler(self._handle_imu)
        self._myo.add_pose_handler(self._handle_pose)

        def _run():
            self._myo.connect()
            self._myo.vibrate(1)
            print("Myo connected — streaming in FILTERED mode")
            while self._running:
                self._myo.run()

        self._running = True
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        if self._myo is not None:
            try:
                self._myo.disconnect()
            except Exception:
                pass
        print("Myo disconnected")
