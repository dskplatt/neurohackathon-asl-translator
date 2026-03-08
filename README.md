# cosign

Real-time ASL fingerspelling transcription using a Myo armband and a personal neural network trained on your own EMG data.

The system reads electromyographic (EMG) signals from a Thalmic Myo armband, classifies individual ASL letters, and resolves them into English words — all streamed live to a browser UI over WebSocket.

## How It Works

```
Myo Armband (BLE)
    ↓  EMG 200 Hz · 8 channels
    ↓  Accelerometer 50 Hz · 3 axes
Segmentation State Machine
    ↓  detects signing vs. resting
Letter Classifier (CNN + BiLSTM)
    ↓  classifies 40-frame EMG windows → A–Z
Word Resolver
    ↓  maps letter probability sequences to English words
Browser UI (WebSocket)
```

1. **Segmentation** monitors the EMG signal's RMS energy. When it crosses a threshold the system enters SIGNING mode and begins capturing 40-frame (200 ms) windows.
2. **Classification** runs each window through a CNN + bidirectional LSTM trained on your personal EMG data. The model outputs a probability distribution over the 26 letters.
3. **Word resolution** takes the accumulated letter distributions and scores candidate English words using a combination of letter probabilities and word frequency (NLTK + wordfreq).
4. A **wave-out gesture** on the Myo triggers word resolution — the top candidate is sent to the frontend and the buffer resets.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Thalmic Myo armband** | Must be paired via Myo Connect |
| **Python 3.11+** | Tested on 3.11 and 3.14 |
| **Node.js 18+** | For the Next.js frontend |

### Python Dependencies

```
numpy
pandas
scipy
scikit-learn
joblib
torch
fastapi
uvicorn
pyomyo
nltk
wordfreq
```

Install with:

```bash
pip install numpy pandas scipy scikit-learn joblib torch fastapi uvicorn pyomyo nltk wordfreq
```

### Node Dependencies

```bash
npm install
```

## Quick Start

Make sure your Myo armband is paired and Myo Connect is running, then:

```bash
npm run dev
```

This starts both the Python backend (FastAPI on port 8000) and the Next.js frontend concurrently. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Personal Calibration

The classifier must be calibrated to **your** muscle signals before it can accurately classify letters. There are two calibration steps:

### 1. Collect Training Data

The calibration UI at [/calibrate](http://localhost:3000/calibrate) walks you through signing each of the 26 letters multiple times while wearing the Myo. The segmentation engine auto-detects when you're signing and captures EMG windows for each letter.

Data is saved to `data/calibration_data.csv`.

You can also collect data from the command line:

```bash
# Full collection — all 26 letters
python scripts/collect_calibration_data.py --samples-per-letter 15

# Add extra samples for specific letters
python scripts/collect_partial_data.py --letters a s e --samples 10
```

To verify the collected data:

```bash
python scripts/validate_calibration_data.py
```

### 2. Train Your Personal Model

Training can be kicked off from the calibration UI or via the API:

```bash
curl -X POST http://localhost:8000/calibrate/train
```

Or directly:

```bash
python scripts/train_personal_model.py
```

This fine-tunes the neural network on your data, fits a personal feature scaler, and saves:

- `models/classifier_personal.pt` — your fine-tuned model weights
- `models/scaler_personal.joblib` — feature scaler fit to your EMG range

The server automatically picks up the personal model on the next request.

### Quick Calibration (Centroids)

For a faster alternative to full retraining, the `/calibrate` page also supports **centroid calibration**: you sign each letter a few times and the system computes per-letter feature centroids. At inference time, letters are classified by cosine similarity to these centroids instead of the linear head. Centroids are saved to `models/centroids.npz`.

## Project Structure

```
├── app/                        Next.js pages
│   ├── page.tsx                Landing page
│   ├── translator/page.tsx     Translator page
│   └── calibrate/              Calibration UI
├── components/
│   ├── ASLTranslator.tsx       Main transcription component
│   └── SigningStatus.tsx       Signing state indicator
├── src/                        Python backend
│   ├── server.py               FastAPI + WebSocket server
│   ├── myo_reader.py           Myo BLE acquisition layer
│   ├── segmentation.py         EMG-based letter segmentation
│   ├── inference.py            Letter classifier (CNN + BiLSTM)
│   ├── model.py                Neural network architecture
│   ├── calibration.py          Signal + centroid calibration
│   └── word_resolver.py        Letter sequences → English words
├── scripts/
│   ├── collect_calibration_data.py
│   ├── collect_partial_data.py
│   ├── train_personal_model.py
│   └── validate_calibration_data.py
├── models/                     Trained weights and scalers
└── data/                       Personal calibration data
```

## API Reference

### WebSocket

**`ws://localhost:8000/ws`**

Streams JSON messages to connected clients:

| Message Type | Fields | Description |
|---|---|---|
| `reset` | `myo_connected` | Sent on connect; clears frontend state |
| `myo_status` | `myo_connected` | Myo connection changed |
| `signing_active` | `active` | User started/stopped signing |
| `letter_captured` | `top_letter`, `confidence`, `letter_index` | A letter window was classified |
| `word_resolved` | `primary`, `score`, `alternates` | Word resolved from letter buffer |

### HTTP

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status, Myo connection, buffer length |
| POST | `/reset` | Clear the word buffer |
| POST | `/calibrate/signal` | Start signal calibration (threshold tuning) |
| POST | `/calibrate/letters` | Start centroid calibration |
| POST | `/calibrate/train` | Train personal model in background |
| POST | `/calibrate/reload` | Reload centroids from disk |
| GET | `/calibrate/status` | Current calibration state |

## Usage Tips

- **Keep the Myo snug** on your forearm — loose fit dramatically increases noise.
- **Calibrate in the same position** you'll use for transcription. Arm angle matters.
- **Wave out** (flick your wrist outward) to resolve the current word.
- If accuracy drops, collect more samples for the problem letters with `collect_partial_data.py` and retrain.
- The Myo will auto-reconnect if the BLE connection drops.

## Built With

- [Next.js](https://nextjs.org/) + [Tailwind CSS](https://tailwindcss.com/) — frontend
- [FastAPI](https://fastapi.tiangolo.com/) — backend server
- [PyTorch](https://pytorch.org/) — neural network
- [pyomyo](https://github.com/PerlinWarp/pyomyo) — Myo armband interface
- [NLTK](https://www.nltk.org/) + [wordfreq](https://github.com/rspeer/wordfreq) — word resolution

---

*Built for the Neurohackathon.*
