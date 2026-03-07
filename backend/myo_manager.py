import time
import threading
from collections import deque
from pyomyo import Myo, emg_mode

class MyoManager:
    """
    Manages the Myo armband connection and data streams.
    Runs the pyomyo loop in a background thread to avoid blocking.
    """
    def __init__(self, tty=None):
        self.tty = tty
        self.myo = None
        self.thread = None
        self.running = False
        
        # Buffers for recent data
        # Let's keep the last ~1 second of data for now
        self.emg_buffer = deque(maxlen=200) # 200 Hz
        self.imu_buffer = deque(maxlen=50)  # 50 Hz
        
        # State tracking
        self.is_connected = False
        self.last_emg_timestamp = 0
        self.last_imu_timestamp = 0

    def start(self):
        """Starts the Myo connection and data loop in a background thread."""
        if self.running:
            return

        print("Starting MyoManager...")
        try:
            # Using raw EMG mode for better signal resolution
            self.myo = Myo(mode=emg_mode.SEND_EMG.value, tty=self.tty)
            self.myo.connect()
            self.is_connected = True
            print("Myo connected successfully.")
            
            # Register handlers
            self.myo.add_emg_handler(self._handle_emg)
            self.myo.add_imu_handler(self._handle_imu)
            
            # Start the background thread
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            
        except ValueError as e:
            print(f"Failed to connect to Myo armband. Make sure the USB dongle is plugged in. Error: {e}")
            self.is_connected = False
            self.running = False

    def stop(self):
        """Stops the MyoManager gracefully."""
        print("Stopping MyoManager...")
        self.running = False
        if self.myo and self.is_connected:
            try:
                self.myo.disconnect()
            except Exception as e:
                print(f"Error disconnecting Myo: {e}")
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        self.is_connected = False
        print("MyoManager stopped.")

    def _run_loop(self):
        """The main loop running in the background thread."""
        while self.running and self.is_connected:
            try:
                # This calls the pyomyo event loop to process incoming packets
                # We need to call this frequently
                self.myo.run(10) # process for 10ms
            except Exception as e:
                print(f"Error in Myo run loop: {e}")
                self.is_connected = False
                self.running = False
                break
        print("Myo thread exited.")

    def _handle_emg(self, emg_data):
        """Callback for incoming EMG data. 8 channels."""
        self.last_emg_timestamp = time.time()
        # pyomyo returns a tuple of 8 integers
        self.emg_buffer.append((self.last_emg_timestamp, emg_data))

    def _handle_imu(self, quat, acc, gyro):
        """Callback for incoming IMU data.
        quat: Orientation as a quaternion (W, X, Y, Z)
        acc: Accelerometer data (X, Y, Z) (in g)
        gyro: Gyroscope data (X, Y, Z) (in deg/s)
        """
        self.last_imu_timestamp = time.time()
        # Flatten the IMU data into a single tuple/list
        imu_data = tuple(quat) + acc + gyro
        self.imu_buffer.append((self.last_imu_timestamp, imu_data))

    def get_status(self):
        """Returns the current status of the connection and data streams."""
        now = time.time()
        
        # Consider a stream active if we've received data in the last 1.5 seconds
        emg_active = (now - self.last_emg_timestamp) < 1.5
        imu_active = (now - self.last_imu_timestamp) < 1.5
        
        status = {
            "connected": self.is_connected,
            "running": self.running,
            "emg_active": emg_active,
            "imu_active": imu_active,
            "emg_buffer_size": len(self.emg_buffer),
            "imu_buffer_size": len(self.imu_buffer)
        }
        return status
