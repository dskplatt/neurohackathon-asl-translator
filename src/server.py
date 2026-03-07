"""
ASL translator WebSocket server.

Connects the Myo armband → segmentation → classification → word resolution
pipeline and broadcasts state changes to connected frontends over WebSocket.

    uvicorn src.server:app --reload --port 8000
"""

import asyncio
import json
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

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

def _on_letter_ready(window: np.ndarray) -> None:
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
    global loop, classifier, resolver, segmentation, myo_reader

    loop = asyncio.get_event_loop()

    resolver = WordResolver()
    classifier = LetterClassifier()

    segmentation = SegmentationStateMachine(
        on_letter_ready=_on_letter_ready,
        on_signing_start=lambda: _broadcast_sync({"type": "signing_active", "active": True}),
        on_signing_end=lambda: _broadcast_sync({"type": "signing_active", "active": False}),
    )

    myo_reader = MyoReader(
        on_emg_frame=segmentation.push_frame,
        on_accel_frame=classifier.update_accel,
        on_wave_right=_on_wave_right,
        on_wave_left=_on_wave_left,
    )
    try:
        myo_reader.start()
    except Exception as e:
        print(f"Myo not available ({e}) — mock endpoints still work")

    print("Server ready")
    yield

    myo_reader.stop()


app = FastAPI(lifespan=lifespan)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        await websocket.send_text(json.dumps({"type": "reset"}))
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
        "myo_connected": myo_reader._running if myo_reader else False,
    }


@app.post("/reset")
async def reset():
    word_buffer.clear()
    await broadcast({"type": "reset"})
    return {"status": "reset"}


class MockLetterBody(BaseModel):
    letter: str


@app.post("/mock/letter")
async def mock_letter(body: MockLetterBody):
    letter = body.letter.lower()
    if len(letter) != 1 or letter not in "abcdefghijklmnopqrstuvwxyz":
        return {"status": "error", "message": "letter must be a single a-z character"}

    spread = 0.1 / 25
    distribution = {
        c: (0.9 if c == letter else spread)
        for c in "abcdefghijklmnopqrstuvwxyz"
    }
    word_buffer.append(distribution)
    await broadcast({
        "type": "letter_captured",
        "letter_index": len(word_buffer) - 1,
        "top_letter": letter,
        "confidence": 0.9,
    })
    return {"status": "ok", "letter": letter}


@app.post("/mock/wave_right")
async def mock_wave_right():
    _on_wave_right()
    return {"status": "ok"}


@app.post("/mock/wave_left")
async def mock_wave_left():
    _on_wave_left()
    return {"status": "ok"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
