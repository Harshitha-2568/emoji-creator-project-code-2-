# Emoji Creator Project

This project detects facial emotion from an image or webcam feed and maps the result to an emoji-style output. It includes:

- a CNN training script for 7 emotion classes
- a FastAPI backend for image and optional audio-based prediction
- a Tkinter desktop GUI for live webcam emotion detection
- a React + Vite frontend for browser-based interaction

## Features

- 7-class emotion recognition: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- Training resume support with saved weights and training state
- Face detection using MediaPipe when available, with OpenCV Haar cascade fallback
- Optional audio fusion in the API
- Emotion history logging in SQLite
- Emoji image display for predicted emotions

## Project Structure

```text
emoji-creator-project-code/
├── api.py                  # FastAPI backend
├── gui.py                  # Tkinter desktop app
├── train.py                # CNN training script
├── requirements.txt        # Python dependencies
├── model.weights.h5        # Trained model weights
├── training_state.json     # Last completed training epoch
├── emotion_history.db      # SQLite history database
├── data/
│   ├── train/              # Training images by class
│   └── test/               # Validation/test images by class
├── emojis/                 # Emoji images used by the GUI
└── frontend/               # React + Vite web frontend
```

## Requirements

- Python 3.10 or newer recommended
- `pip`
- Webcam for the desktop GUI
- Node.js 18+ for the frontend

Python packages used by the backend/training app are listed in `requirements.txt`:

```txt
fastapi
uvicorn[standard]
python-multipart
numpy
opencv-python
tensorflow
keras
mediapipe
deepface
```

Note: `gui.py` can also use `Pillow` and optionally `sounddevice`. If those are missing, install them manually:

```bash
pip install pillow sounddevice
```

## Setup

### 1. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
pip install pillow sounddevice
```

### 3. Prepare the dataset

The training script expects this layout:

```text
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Each folder should contain grayscale or color face images for that emotion.

`train.py` expects all 7 classes to exist. If a class is missing from `data/train` but exists in `data/test`, it copies validation images into the training folder for that class.

## Train the Model

Run:

```bash
python train.py
```

By default, training runs for `30` epochs.

To train for a different total number of epochs:

Windows `cmd`:

```bash
set EMOTION_EPOCHS=50
python train.py
```

What training does:

- builds a CNN with grayscale `48x48` input
- loads images from `data/train` and `data/test`
- saves best weights to `model.weights.h5`
- stores progress in `training_state.json`
- resumes from the last saved epoch if weights already exist

## Run the FastAPI Backend

Start the API server with:

```bash
uvicorn api:app --reload
```

The API will usually be available at:

```text
http://127.0.0.1:8000
```

### API Endpoints

#### `GET /health`

Returns service status and the loaded weights file.

#### `POST /predict-emotion`

Form-data inputs:

- `image`: required image file
- `audio`: optional WAV audio file

Returns:

- final predicted emotion
- confidence score
- per-face emotion probabilities
- optional audio emotion probabilities
- detector source
- gender estimate and avatar key

#### `GET /history`

Returns recent prediction history from `emotion_history.db`.

Example:

```text
GET /history?limit=50
```

## Run the Desktop GUI

Start the Tkinter app with:

```bash
python gui.py
```

The GUI:

- opens your webcam
- detects the largest face in frame
- predicts emotion using the trained model
- shows a matching emoji from the `emojis/` folder
- stores prediction history in SQLite

If no trained model exists, the GUI will prompt you to run `train.py` first.

## Run the Frontend

Move into the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Copy the example environment file if needed:

```bash
copy .env.example .env
```

Start the dev server:

```bash
npm run dev
```

The frontend expects the backend API at:

```text
VITE_API_BASE=http://127.0.0.1:8000
```

## Saved Files

- `model.weights.h5`: best trained weights
- `training_state.json`: last completed epoch
- `emotion_history.db`: logged emotion predictions
- `backend.log`: backend runtime log if generated during local runs

## Notes

- `api.py` requires trained weights before startup.
- `DeepFace` is optional at runtime. If it fails to import, emotion prediction still works but gender detection is disabled.
- MediaPipe is used when available; otherwise the app falls back to Haar cascade face detection.
- The API only accepts 16-bit PCM WAV audio for audio analysis.

## Troubleshooting

### No model weights found

Train the model first:

```bash
python train.py
```

### No face detected

- use a clearer face image
- improve lighting
- move closer to the camera

### Frontend cannot connect to backend

- make sure `uvicorn api:app --reload` is running
- confirm `VITE_API_BASE` matches the backend URL

## License

Add your preferred license here.
