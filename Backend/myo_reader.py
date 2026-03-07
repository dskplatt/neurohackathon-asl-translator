"""
myo_reader.py — Myo armband data acquisition module
----------------------------------------------------
Streams 8-channel EMG at 200Hz and handles built-in Myo pose gestures.
Designed to be imported and driven by server.py.

Usage:
    reader = MyoReader(on_emg=my_emg_handler, on_pose=my_pose_handler)
    reader.start()   # non-blocking — spawns background thread
    ...
    reader.stop()
"""

import sys
import threading

try:
    from pyomyo import Myo, emg_mode
except ImportError as e:
    print(f"[ERROR] PyoMyo not available: {e}")
    sys.exit(1)


class MyoReader:
    """
    Wraps the pyomyo connection and runs the acquisition loop
    in a background thread so the FastAPI event loop is never blocked.

    Parameters
    ----------
    on_emg : callable(emg: tuple[int, ...], movement: int)
        Called on every EMG packet (~200Hz).
        emg is a tuple of 8 signed integers in [-128, 127].

    on_pose : callable(pose)
        Called whenever a built-in Myo gesture is detected.
        Relevant values: Pose.WAVE_OUT (submit word), Pose.WAVE_IN (backspace).

    mode : emg_mode
        EMG capture mode. FILTERED is recommended for classification.
    """

    def __init__(self, on_emg, on_pose, mode=emg_mode.FILTERED):
        self._on_emg = on_emg
        self._on_pose = on_pose
        self._mode = mode
        self._myo: Myo | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Connect to the Myo and start the acquisition thread."""
        self._myo = self._connect()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[MyoReader] Acquisition thread started.")

    def stop(self):
        """Signal the acquisition thread to stop and disconnect."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._myo:
            try:
                self._myo.disconnect()
                print("[MyoReader] Myo disconnected.")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect(self) -> Myo:
        print("[MyoReader] Connecting to Myo armband...")
        try:
            m = Myo(mode=self._mode)
            m.add_emg_handler(self._on_emg)
            m.add_pose_handler(self._on_pose)
            m.connect()
            print("[MyoReader] Myo connected successfully.")
            return m
        except Exception as e:
            print(f"[MyoReader] Connection failed: {e}")
            print("[MyoReader] Ensure the Myo is powered on and paired.")
            sys.exit(1)

    def _run_loop(self):
        """Processes incoming Bluetooth packets until stop() is called."""
        while self._running:
            try:
                self._myo.run()
            except Exception as e:
                print(f"[MyoReader] Error in acquisition loop: {e}")
                self._running = False
                break