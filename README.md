# ASL Fingerspelling Translator

Real-time ASL fingerspelling translation using the Myo armband — no camera, no internet connection required.

## Overview

This project translates American Sign Language (ASL) fingerspelling into English text in real time. Instead of a camera, it reads electromyographic (EMG) signals from a Thalmic Labs Myo armband worn on the forearm. The entire pipeline runs locally over Bluetooth — no cloud services, no internet connection, and no line-of-sight requirement.

Unlike camera-based ASL translators, this approach works regardless of lighting, occlusion, or camera angle. The trade-off is that each user must calibrate a personal model first (~15–30 minutes), because EMG signals vary significantly between individuals based on arm physiology, band placement, and signing style. Calibration involves signing each letter several times while the system records your muscle activity, then training a lightweight neural network on that data.

Once calibrated, the system auto-detects when you're signing, classifies each letter, and resolves sequences of letters into English words using probabilistic word matching.

## How It Works

```
Myo armband → EMG capture → RMS segmentation → CNN+BiLSTM classifier
→ voting across overlapping windows → word resolver → web interface
```

1. **EMG capture** — The Myo armband streams 8-channel EMG at 200 Hz and 3-axis accelerometer data at 50 Hz over Bluetooth.
2. **Segmentation** — A state machine monitors RMS energy to detect when the user is actively signing vs resting.
3. **Classification** — Each signing window is split into overlapping 40-frame sub-windows, classified by a CNN + bidirectional LSTM, and votes are averaged across sub-windows.
4. **Word resolution** — Accumulated letter probability distributions are scored against an English dictionary weighted by word frequency (NLTK + wordfreq).
5. **Wave-out gesture** — A wrist flick triggers word resolution; the top candidate is sent to the browser and the buffer resets.

## Hardware Requirements

- Myo armband (original Thalmic Labs version)
- Bluetooth adapter or built-in Bluetooth
- Any standard laptop or desktop

## Software Requirements

- Python 3.10+
- Node.js 18+
- pyomyo

## Installation

### Backend

```bash
git clone <repo>
cd neurohackathon-asl-translator
pip install -r requirements.txt
```

### Frontend

```bash
npm install
```

## Calibration

Calibration must be completed before the translator can be used. The system trains a personal model specific to your arm and signing style.

```bash
# Step 1 — Collect calibration data (~15 minutes)
# Sign each letter 5+ times when prompted
python scripts/collect_calibration_data.py

# Step 2 — Train your personal model (~15 minutes on CPU)
python scripts/train_personal_model.py
```

### Adding more data to improve accuracy

```bash
# Collect extra samples for specific letters
python scripts/collect_partial_data.py --letters a s e --samples 10

# Or add samples to all letters
python scripts/collect_partial_data.py \
  --letters a b c d e f g h i j k l m n o p q r s t u v w x y z \
  --samples 10

# Retrain after collecting more data
python scripts/train_personal_model.py
```

## Running

```bash
# Start both backend and frontend concurrently
npm run dev
```

Or start them separately:

```bash
# Terminal 1 — start backend
uvicorn src.server:app --port 8000

# Terminal 2 — start frontend
npx next dev
```

Open http://localhost:3000/translator

## Usage

1. Put on the Myo armband on your dominant forearm
2. Open the translator page
3. Sign a letter and hold the position for ~2 seconds
4. The system auto-detects and captures each letter
5. Wave right to resolve the current letters into a word
6. Repeat for the next word

## Project Structure

```
├── app/                        Next.js pages (frontend)
│   ├── page.tsx                Landing page
│   ├── translator/page.tsx     Translator UI
│   └── calibrate/              Calibration UI
├── components/
│   ├── ASLTranslator.tsx       Main transcription component
│   └── SigningStatus.tsx       Signing state indicator
├── src/                        Python backend
│   ├── server.py               FastAPI + WebSocket server
│   ├── myo_reader.py           Myo BLE acquisition layer
│   ├── segmentation.py         EMG-based letter segmentation
│   ├── inference.py            Letter classifier wrapper
│   ├── model.py                CNN + BiLSTM architecture
│   ├── calibration.py          Signal + centroid calibration
│   └── word_resolver.py        Letter sequences → English words
├── scripts/
│   ├── collect_calibration_data.py     Full calibration collection
│   ├── collect_partial_data.py         Add samples for specific letters
│   ├── train_personal_model.py         Fine-tune personal model
│   └── validate_calibration_data.py    Validate data before training
├── training/                   Original dataset training pipeline
├── models/                     Trained weights and scalers (gitignored)
└── data/                       Calibration data (gitignored)
```

## Tech Stack

- **Hardware:** Myo armband (8-channel EMG, 200 Hz, Bluetooth)
- **Backend:** Python, FastAPI, PyTorch, scikit-learn
- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **Model:** CNN + BiLSTM (~85K parameters)
- **Word resolution:** Probabilistic resolver using wordfreq + NLTK

## Limitations

- Requires Myo armband (discontinued hardware — available secondhand)
- Calibration required per user (~15–30 minutes first time)
- Some similar hand-shape letters (A/S/E, M/N) may be occasionally confused
- Best accuracy achieved with consistent band placement
