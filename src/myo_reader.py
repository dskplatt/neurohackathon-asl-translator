"""
Myo armband acquisition layer.

Streams EMG (200 Hz, 8 channels) and IMU (50 Hz, accelerometer) data
from a Thalmic Myo via pyomyo, and dispatches pose events for gesture
control (wave-in / wave-out).
"""

import json
import threading
import time
from pathlib import Path

import numpy as np

# #region agent log
_DBG_LOG_PATH = Path(__file__).resolve().parent.parent / "debug-356814.log"
_dbg_emg_count = 0
_dbg_accel_count = 0
def _dbg_log(location, message, data, hypothesis_id, run_id="initial"):
    try:
        with open(_DBG_LOG_PATH, "a") as f:
            f.write(json.dumps({"sessionId":"356814","id":f"log_{time.time_ns()}","timestamp":int(time.time()*1000),"location":location,"message":message,"data":data,"runId":run_id,"hypothesisId":hypothesis_id}) + "\n")
    except: pass
# #endregion


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
        self._connected = False  # True only after connect() succeeds
        self._myo = None

    # ── Myo callbacks ─────────────────────────────────────────────────

    def _handle_emg(self, emg_data, moving=0):
        # #region agent log
        global _dbg_emg_count
        _dbg_emg_count += 1
        if _dbg_emg_count <= 5 or _dbg_emg_count % 400 == 0:
            arr = np.array(emg_data, dtype=np.float32)
            _dbg_log("myo_reader.py:_handle_emg", "raw EMG frame from Myo", {
                "frame_num": _dbg_emg_count,
                "values": [round(float(v), 2) for v in arr],
                "min": round(float(arr.min()), 2),
                "max": round(float(arr.max()), 2),
                "rms": round(float(np.sqrt(np.mean(arr**2))), 2),
            }, "H3")
        # #endregion
        self.on_emg_frame(np.array(emg_data, dtype=np.float32))

    def _handle_imu(self, quat, accel, gyro):
        # #region agent log
        global _dbg_accel_count
        _dbg_accel_count += 1
        if _dbg_accel_count <= 3 or _dbg_accel_count % 100 == 0:
            raw_arr = np.array(accel, dtype=np.float32)
            scaled_arr = raw_arr / 2048.0
            _dbg_log("myo_reader.py:_handle_imu", "accel from Myo", {
                "frame_num": _dbg_accel_count,
                "raw_values": [round(float(v), 2) for v in raw_arr],
                "scaled_values": [round(float(v), 4) for v in scaled_arr],
            }, "H4")
        # #endregion
        # pyomyo gives raw 16-bit ints; divide by 2048 to convert to g's
        self.on_accel_frame(np.array(accel, dtype=np.float32) / 2048.0)

    def _handle_pose(self, pose):
        from pyomyo.pyomyo import Pose

        try:
            name = pose.name
        except AttributeError:
            name = str(pose)

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
            try:
                self._myo.connect()
                self._myo.vibrate(1)
                self._connected = True
                print("Myo connected — streaming in FILTERED mode")
                while self._running:
                    self._myo.run()
            except Exception as e:
                self._connected = False
                print(f"Myo not available ({e}) — connect Myo armband for transcription")

        self._running = True
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        self._connected = False
        if self._myo is not None:
            try:
                self._myo.disconnect()
            except Exception:
                pass
        print("Myo disconnected")
