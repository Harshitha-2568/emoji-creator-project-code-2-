import io


import os
import sqlite3
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    from deepface import DeepFace
    DEEPFACE_IMPORT_ERROR = None
except Exception as exc:
    DeepFace = None
    DEEPFACE_IMPORT_ERROR = str(exc)


EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
HISTORY_DB_PATH = Path("emotion_history.db")


def build_model():
    model = Sequential()
    model.add(Input(shape=(48, 48, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))
    return model


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.maximum(probs.astype("float32"), 1e-6)
    return probs / float(np.sum(probs))


def avatar_key_from_gender_label(label: str) -> str:
    if label == "female":
        return "girl"
    if label == "male":
        return "boy"
    return "neutral"


def audio_probs_from_signal(x: np.ndarray, sample_rate: int) -> np.ndarray:
    if len(x) < 2:
        return np.array([0.08, 0.06, 0.10, 0.10, 0.36, 0.20, 0.10], dtype="float32")

    rms = float(np.sqrt(np.mean(np.square(x))) + 1e-8)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))

    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    spec_sum = float(np.sum(spectrum)) + 1e-8
    centroid = float(np.sum(freqs * spectrum) / spec_sum)
    centroid_norm = centroid / (sample_rate / 2.0)

    probs = np.array([0.08, 0.06, 0.10, 0.10, 0.36, 0.20, 0.10], dtype="float32")
    if rms < 0.01:
        probs += np.array([0.00, 0.00, 0.02, 0.00, 0.45, 0.25, 0.00], dtype="float32")
    elif rms < 0.03:
        probs += np.array([0.02, 0.00, 0.03, 0.02, 0.20, 0.35, 0.01], dtype="float32")
    elif rms > 0.08 and zcr > 0.12:
        probs += np.array([0.35, 0.02, 0.10, 0.10, 0.00, 0.00, 0.25], dtype="float32")
    elif rms > 0.06 and centroid_norm > 0.35:
        probs += np.array([0.08, 0.00, 0.08, 0.35, 0.00, 0.00, 0.18], dtype="float32")

    if centroid_norm < 0.20 and zcr < 0.08:
        probs += np.array([0.00, 0.00, 0.00, 0.00, 0.10, 0.22, 0.00], dtype="float32")
    if centroid_norm > 0.45 and zcr > 0.11:
        probs += np.array([0.05, 0.00, 0.20, 0.05, 0.00, 0.00, 0.10], dtype="float32")

    return normalize_probs(probs)


def decode_wav_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported.")

    audio = np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def resample_linear(x: np.ndarray, from_sr: int, to_sr: int = 16000) -> np.ndarray:
    if from_sr == to_sr or len(x) == 0:
        return x
    old_idx = np.linspace(0, 1, num=len(x), endpoint=False)
    new_len = max(1, int(len(x) * (to_sr / from_sr)))
    new_idx = np.linspace(0, 1, num=new_len, endpoint=False)
    return np.interp(new_idx, old_idx, x).astype("float32")


class EmotionService:
    def __init__(self):
        self.model = build_model()
        self.weights_path = next(
            (p for p in ["model.weights.h5", "model.h5", "emotion_model.h5"] if os.path.exists(p)),
            None,
        )
        if self.weights_path is None:
            raise RuntimeError("No weights file found. Train first to create model.weights.h5.")
        self.model.load_weights(self.weights_path)

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.mp_detector = None
        self.detector_name = "haar"
        if mp is not None:
            try:
                self.mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                self.detector_name = "mediapipe"
            except Exception as exc:
                self.mp_detector = None
                self.detector_name = "haar"
                print(f"[INFO] MediaPipe face detector unavailable, using Haar: {exc}")
        self.gender_model_available = DeepFace is not None
        if not self.gender_model_available and DEEPFACE_IMPORT_ERROR:
            print(f"[INFO] DeepFace disabled: {DEEPFACE_IMPORT_ERROR}")

    def detect_faces(self, frame_bgr: np.ndarray, gray: np.ndarray):
        if self.mp_detector is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.mp_detector.process(rgb)
            if results.detections:
                h, w = frame_bgr.shape[:2]
                boxes = []
                for detection in results.detections:
                    rel = detection.location_data.relative_bounding_box
                    x = max(0, int(rel.xmin * w))
                    y = max(0, int(rel.ymin * h))
                    bw = min(w - x, int(rel.width * w))
                    bh = min(h - y, int(rel.height * h))
                    if bw > 0 and bh > 0:
                        boxes.append((x, y, bw, bh))
                if boxes:
                    return boxes
        # Robust Haar fallback: normalize contrast and try multiple detector settings.
        h, w = gray.shape[:2]
        min_side = max(36, min(h, w) // 10)
        normalized = cv2.equalizeHist(gray)

        configs = [
            (1.10, 5),
            (1.10, 3),
            (1.05, 3),
        ]
        for scale_factor, min_neighbors in configs:
            faces = self.face_detector.detectMultiScale(
                normalized,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(min_side, min_side),
            )
            if len(faces) > 0:
                return faces

        # Final attempt on downscaled frame for weak/blurred webcam input.
        small = cv2.resize(normalized, (w // 2, h // 2))
        faces_small = self.face_detector.detectMultiScale(
            small,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(max(24, min_side // 2), max(24, min_side // 2)),
        )
        if len(faces_small) > 0:
            upscaled = [(x * 2, y * 2, fw * 2, fh * 2) for (x, y, fw, fh) in faces_small]
            return upscaled

        return []

    def detect_gender(self, face_bgr: np.ndarray) -> dict:
        if not self.gender_model_available:
            return {"label": "unknown", "confidence": 0.0, "source": "unavailable"}
        try:
            analysis = DeepFace.analyze(
                img_path=face_bgr,
                actions=["gender"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )
            result = analysis[0] if isinstance(analysis, list) else analysis
            gender_scores = result.get("gender", {})
            woman_score = float(gender_scores.get("Woman", 0.0))
            man_score = float(gender_scores.get("Man", 0.0))
            if woman_score >= man_score:
                return {"label": "female", "confidence": woman_score / 100.0, "source": "deepface"}
            return {"label": "male", "confidence": man_score / 100.0, "source": "deepface"}
        except Exception:
            return {"label": "unknown", "confidence": 0.0, "source": "failed"}


service: Optional[EmotionService] = None


def ensure_db():
    conn = sqlite3.connect(str(HISTORY_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS emotion_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TEXT NOT NULL,
            emotion_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            detector TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_history(label: str, confidence: float, detector: str):
    conn = sqlite3.connect(str(HISTORY_DB_PATH))
    conn.execute(
        "INSERT INTO emotion_history (detected_at, emotion_label, confidence, detector) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), label, confidence, detector),
    )
    conn.commit()
    conn.close()


app = FastAPI(title="Emotion Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    global service
    ensure_db()
    service = EmotionService()


@app.get("/health")
def health():
    return {"ok": True, "weights": service.weights_path if service else None}


@app.post("/predict-emotion")
async def predict_emotion(
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
):
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    image_bytes = await image.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = service.detect_faces(frame, gray)
    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="No face detected in image")

    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_bgr = frame[y : y + h, x : x + w]
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        raise HTTPException(status_code=422, detail="Invalid face ROI")

    face_input = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0).astype("float32") / 255.0
    face_probs = service.model.predict(face_input, verbose=0)[0]
    face_idx = int(np.argmax(face_probs))
    face_conf = float(np.max(face_probs))

    audio_probs = None
    if audio is not None:
        try:
            raw_audio = await audio.read()
            wav_signal, wav_sr = decode_wav_bytes(raw_audio)
            wav_signal = resample_linear(wav_signal, wav_sr, 16000)
            audio_probs = audio_probs_from_signal(wav_signal, 16000)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {exc}")

    if audio_probs is not None:
        face_weight = 0.80 if face_conf >= 0.60 else 0.65
        fused_probs = (face_weight * face_probs) + ((1.0 - face_weight) * audio_probs)
        detector = f"{service.detector_name}+audio"
    else:
        fused_probs = face_probs
        detector = service.detector_name

    fused_probs = normalize_probs(fused_probs)
    final_idx = int(np.argmax(fused_probs))
    final_conf = float(np.max(fused_probs))
    final_label = EMOTION_LABELS[final_idx]
    gender = service.detect_gender(face_bgr)
    log_history(final_label, final_conf, detector)

    payload = {
        "emotion": final_label,
        "confidence": final_conf,
        "face": {
            "emotion": EMOTION_LABELS[face_idx],
            "confidence": face_conf,
            "probabilities": {k: float(v) for k, v in zip(EMOTION_LABELS, face_probs)},
        },
        "fused_probabilities": {k: float(v) for k, v in zip(EMOTION_LABELS, fused_probs)},
        "detector": detector,
        "gender": gender,
        "avatar_key": avatar_key_from_gender_label(gender["label"]),
    }
    if audio_probs is not None:
        payload["audio"] = {
            "emotion": EMOTION_LABELS[int(np.argmax(audio_probs))],
            "confidence": float(np.max(audio_probs)),
            "probabilities": {k: float(v) for k, v in zip(EMOTION_LABELS, audio_probs)},
        }

    return JSONResponse(payload)


@app.get("/history")
def history(limit: int = 100):
    safe_limit = max(1, min(limit, 1000))
    conn = sqlite3.connect(str(HISTORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, detected_at, emotion_label, confidence, detector
        FROM emotion_history
        ORDER BY id DESC
        LIMIT ?
        """,
        (safe_limit,),
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows], "count": len(rows)}
