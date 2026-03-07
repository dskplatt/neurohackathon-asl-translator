from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from myo_manager import MyoManager

# Global manager instance
myo_manager = MyoManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI application."""
    print("Starting up FastAPI server...")
    # Attempt to connect to Myo and start background stream
    myo_manager.start()
    
    yield # Server is running
    
    # Shutdown
    print("Shutting down FastAPI server...")
    myo_manager.stop()

app = FastAPI(title="ASL Translator Backend", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "ASL Translator Myo Backend API"}

@app.get("/status")
def get_myo_status():
    """Returns the current connection status of the Myo armband."""
    return myo_manager.get_status()

@app.get("/data/emg")
def get_emg_data(count: int = 50):
    """Returns the latest N EMG samples from the buffer."""
    if not myo_manager.is_connected:
        raise HTTPException(status_code=503, detail="Myo armband is not connected.")
    
    # Convert deque to a list and get the last N items
    emg_list = list(myo_manager.emg_buffer)
    # If the user asked for more than we have, return what we have
    return {"emg": emg_list[-count:] if count > 0 else emg_list}

@app.get("/data/imu")
def get_imu_data(count: int = 10):
    """Returns the latest N IMU samples from the buffer."""
    if not myo_manager.is_connected:
        raise HTTPException(status_code=503, detail="Myo armband is not connected.")
    
    imu_list = list(myo_manager.imu_buffer)
    return {"imu": imu_list[-count:] if count > 0 else imu_list}
