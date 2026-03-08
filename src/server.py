"""
ASL translator WebSocket server.

Connects the Myo armband → segmentation → classification → word resolution
pipeline and broadcasts state changes to connected frontends over WebSocket.

    uvicorn src.server:app --reload --port 8000
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.calibration import CalibrationManager
from src.inference import LetterClassifier
from src.myo_reader import MyoReader
from src.segmentation import SegmentationStateMachine
from src.word_resolver import WordResolver

# ── Module-level state ────────────────────────────────────────────────────────

word_buffer: list[dict] = []
connected_clients: set[WebSocket] = set()
loop: asyncio.AbstractEventLoop = None  # type: ignore[assignment]
classifier: LetterClassifier = None  # type: ignore[assignment]
resolver: WordResolver = None  # type: ignore[assignment]
segmentation: SegmentationStateMachine = None  # type: ignore[assignment]
myo_reader: MyoReader = None  # type: ignore[assignment]
calibration_manager: CalibrationManager = None  # type: ignore[assignment]
training_in_progress: bool = False
training_complete: bool = False


# ── Broadcasting ──────────────────────────────────────────────────────────────

async def broadcast(message: dict) -> None:
    payload = json.dumps(message)
    results = await asyncio.gather(
        *(client.send_text(payload) for client in connected_clients),
        return_exceptions=True,
    )
    disconnected = [
        client
        for client, result in zip(list(connected_clients), results)
        if isinstance(result, Exception)
    ]
    for client in disconnected:
        connected_clients.discard(client)


def _broadcast_sync(message: dict) -> None:
    """Thread-safe bridge: schedule broadcast onto the async event loop."""
    asyncio.run_coroutine_threadsafe(broadcast(message), loop)


# ── Pipeline callbacks (called from background threads) ───────────────────────

def _on_emg_frame(emg_frame: np.ndarray) -> None:
    """Route EMG frames to signal calibration when active, else to segmentation."""
    if calibration_manager and calibration_manager.state == "signal_cal":
        calibration_manager.on_signal_frame(emg_frame)
    segmentation.push_frame(emg_frame)


def _on_letter_ready(window: np.ndarray) -> None:
    if calibration_manager and calibration_manager.state in ("letter_cal", "training"):
        if calibration_manager.state == "letter_cal":
            calibration_manager.on_window_captured(window)
        return

    # #region agent log
    import json as _json, time as _time
    _log_path = "debug-88a71d.log"
    window_rms = float(np.sqrt(np.mean(window ** 2)))
    try:
        with open(_log_path, "a") as _f:
            _f.write(_json.dumps({"sessionId":"88a71d","location":"server.py:_on_letter_ready","message":"window_received","data":{"shape": list(window.shape), "window_rms": round(window_rms, 2)}, "hypothesisId":"B","timestamp":int(_time.time()*1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    distribution = classifier.predict(window)
    word_buffer.append(distribution)
    top_letter = max(distribution, key=distribution.get)
    _broadcast_sync({
        "type": "letter_captured",
        "letter_index": len(word_buffer) - 1,
        "top_letter": top_letter,
        "confidence": round(max(distribution.values()), 4),
    })


def _on_wave_right() -> None:
    if not word_buffer:
        return
    candidates = resolver.resolve(word_buffer, top_n=5)
    primary_word, primary_score = candidates[0]
    alternates = [w for w, _ in candidates[1:3]]
    _broadcast_sync({
        "type": "word_resolved",
        "primary": primary_word,
        "score": round(primary_score, 6),
        "alternates": alternates,
    })
    word_buffer.clear()


def _on_wave_left() -> None:
    if not word_buffer:
        return
    word_buffer.pop()
    _broadcast_sync({
        "type": "letter_deleted",
        "remaining_count": len(word_buffer),
    })


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global loop, classifier, resolver, segmentation, myo_reader, calibration_manager

    loop = asyncio.get_event_loop()

    resolver = WordResolver()
    classifier = LetterClassifier()

    segmentation = SegmentationStateMachine(
        on_letter_ready=_on_letter_ready,
        on_signing_start=lambda: (
            _broadcast_sync({"type": "signing_active", "active": True})
            if not calibration_manager or calibration_manager.state == "idle"
            else None
        ),
        on_signing_end=lambda: (
            _broadcast_sync({"type": "signing_active", "active": False})
            if not calibration_manager or calibration_manager.state == "idle"
            else None
        ),
    )

    calibration_manager = CalibrationManager(
        model=classifier.model,
        preprocess_fn=classifier.preprocess,
        segmentation=segmentation,
        broadcast_fn=_broadcast_sync,
        device=classifier.device,
        on_calibration_done=lambda: classifier.load_centroids(),
    )

    def _on_myo_connect_change(connected: bool):
        _broadcast_sync({"type": "myo_status", "myo_connected": connected})

    myo_reader = MyoReader(
        on_emg_frame=_on_emg_frame,
        on_accel_frame=classifier.update_accel,
        on_wave_right=_on_wave_right,
        on_wave_left=_on_wave_left,
        on_connect_change=_on_myo_connect_change,
    )
    try:
        myo_reader.start()
    except Exception as e:
        print(f"Myo not available ({e}) — connect Myo armband for transcription")

    print("Server ready")
    yield

    myo_reader.stop()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "reset",
            "myo_connected": myo_reader._connected if myo_reader else False,
        }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "word_buffer_length": len(word_buffer),
        "connected_clients": len(connected_clients),
        "myo_connected": myo_reader._connected if myo_reader else False,
    }


@app.post("/reset")
async def reset():
    word_buffer.clear()
    await broadcast({"type": "reset"})
    return {"status": "reset"}


# ── Calibration endpoints ─────────────────────────────────────────────────────

@app.post("/calibrate/signal")
async def calibrate_signal():
    if calibration_manager.state != "idle":
        return {"status": "error", "message": f"Already in state: {calibration_manager.state}"}
    calibration_manager.start_signal_calibration()
    return {"status": "started", "mode": "signal"}


@app.post("/calibrate/letters")
async def calibrate_letters():
    if calibration_manager.state != "idle":
        return {"status": "error", "message": f"Already in state: {calibration_manager.state}"}
    calibration_manager.start_letter_calibration()
    return {"status": "started", "mode": "letters"}


@app.post("/calibrate/reload")
async def calibrate_reload():
    """Reload centroids from disk."""
    loaded = classifier.load_centroids()
    if loaded:
        return {"status": "ok", "message": "Centroids reloaded"}
    return {"status": "ok", "message": "No centroids found, using pretrained head"}


@app.get("/calibrate/status")
async def calibrate_status():
    personal_exists = os.path.exists("models/classifier_personal.pt")
    data_exists = os.path.exists("data/calibration_data.csv")

    data_rows = 0
    if data_exists:
        import pandas as pd
        try:
            data_rows = len(pd.read_csv("data/calibration_data.csv"))
        except Exception:
            pass

    return {
        "state": calibration_manager.state,
        "current_letter_idx": calibration_manager.current_letter_idx,
        "centroids_loaded": classifier._centroids is not None,
        "personal_model_exists": personal_exists,
        "calibration_data_exists": data_exists,
        "calibration_data_rows": data_rows,
        "training_in_progress": training_in_progress,
        "training_complete": training_complete,
        "active_model": "personal" if personal_exists else "pretrained",
    }


@app.post("/calibrate/train")
async def start_training():
    global training_in_progress, training_complete

    if training_in_progress:
        return {"status": "already_training"}

    if not os.path.exists("data/calibration_data.csv"):
        return {
            "status": "error",
            "message": "No calibration data found. Run data collection first.",
        }

    training_in_progress = True
    training_complete = False

    def run_training():
        global training_in_progress, training_complete
        try:
            _broadcast_sync({"type": "training_started"})
            result = subprocess.run(
                [sys.executable, "scripts/train_personal_model.py"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                classifier.reload()
                training_complete = True
                _broadcast_sync({
                    "type": "training_complete",
                    "message": "Personal model ready",
                })
                print("Personal model loaded successfully.")
            else:
                _broadcast_sync({
                    "type": "training_failed",
                    "error": result.stderr[-500:],
                })
                print("Training failed:", result.stderr)
        finally:
            training_in_progress = False

    threading.Thread(target=run_training, daemon=True).start()
    return {
        "status": "training_started",
        "message": "Training in background. WebSocket will notify when complete.",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
