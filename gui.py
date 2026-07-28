import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
from collections import deque
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

emotion_model = Sequential()

emotion_model.add(Input(shape=(48, 48, 1)))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

weights_candidates = ['model.weights.h5', 'model.h5', 'emotion_model.h5']
weights_path = next((p for p in weights_candidates if os.path.exists(p)), None)
model_loaded = weights_path is not None
if weights_path is not None:
    emotion_model.load_weights(weights_path)
else:
    print("Error: No model weights found (model.h5/emotion_model.h5). Run train.py first for all 7 emotions.")

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}

BASE_DIR = Path(__file__).resolve().parent
EMOJI_SEARCH_DIRS = [
    BASE_DIR / "emojis",
]
EMOJI_NAME_CANDIDATES = {
    0: ["angry.png"],
    1: ["disgusted.png"],
    2: ["fearful.png"],
    3: ["happy.png"],
    4: ["neutral.png"],
    5: ["sad.png"],
    6: ["surprised.png", "surpriced.png"],
}


def resolve_emoji_path(emotion_index):
    candidates = EMOJI_NAME_CANDIDATES.get(emotion_index, [])
    for emoji_dir in EMOJI_SEARCH_DIRS:
        for name in candidates:
            path = emoji_dir / name
            if path.exists():
                return str(path)
    return None

HISTORY_DB_PATH = Path("emotion_history.db")
history_conn = sqlite3.connect(str(HISTORY_DB_PATH))
history_conn.execute(
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
history_conn.commit()

last_logged_emotion = None
last_logged_ts = datetime.min

if mp is not None:
    try:
        mp_face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        detector_name = "mediapipe"
    except Exception as exc:
        print(f"Info: MediaPipe detector unavailable, using Haar ({exc})")
        mp_face_detector = None
        detector_name = "haar"
else:
    mp_face_detector = None
    detector_name = "haar"


class AudioEmotionModule:
    def __init__(self, sample_rate=16000, seconds=1.0):
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * seconds)
        self.available = sd is not None
        self.stream = None
        self.lock = threading.Lock()
        self.latest_audio = None
        if not self.available:
            print("Info: sounddevice not installed. Audio emotion detection disabled.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            return
        mono = indata[:, 0].copy()
        with self.lock:
            self.latest_audio = mono

    def start(self):
        if not self.available:
            return
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.block_size,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            print(f"Info: audio stream unavailable ({exc}). Audio emotion detection disabled.")
            self.available = False

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def _normalize(self, probs):
        probs = np.maximum(probs, 1e-6)
        return probs / np.sum(probs)

    def get_audio_probs(self):
        # Class order: angry, disgust, fear, happy, neutral, sad, surprise
        if not self.available:
            return None

        with self.lock:
            if self.latest_audio is None:
                return None
            x = self.latest_audio.astype("float32")

        rms = float(np.sqrt(np.mean(np.square(x))) + 1e-8)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))

        spectrum = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.sample_rate)
        spec_sum = float(np.sum(spectrum)) + 1e-8
        centroid = float(np.sum(freqs * spectrum) / spec_sum)
        centroid_norm = centroid / (self.sample_rate / 2.0)

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

        return self._normalize(probs)


global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
global face_detector
cap1 = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
show_text=[-1]
show_confidence=[0.0]
show_face_text=[-1]
show_face_confidence=[0.0]
show_audio_text=[-1]
show_audio_confidence=[0.0]
EMOJI_DISPLAY_SIZE = (260, 260)
VIDEO_FRAME_SIZE = (480, 360)
PREDICT_EVERY_N_FRAMES = 2
UI_REFRESH_MS = 20
EMOJI_REFRESH_MS = 40
SMOOTHING_ALPHA = 0.35
USE_AUDIO_FUSION = False
RECENT_LABEL_WINDOW = 3
FAST_SWITCH_CONFIDENCE = 0.60
LOW_CONF_NEUTRAL_THRESHOLD = 0.40

frame_counter = 0
cached_face_prediction = None
smoothed_fused_prediction = None
recent_fused_labels = deque(maxlen=RECENT_LABEL_WINDOW)

audio_module = AudioEmotionModule()
audio_module.start()


def detect_faces(frame_bgr, gray_frame):
    if mp_face_detector is not None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = mp_face_detector.process(rgb)
        if results.detections:
            h, w = frame_bgr.shape[:2]
            boxes = []
            for detection in results.detections:
                rel_box = detection.location_data.relative_bounding_box
                x = max(0, int(rel_box.xmin * w))
                y = max(0, int(rel_box.ymin * h))
                bw = min(w - x, int(rel_box.width * w))
                bh = min(h - y, int(rel_box.height * h))
                if bw > 0 and bh > 0:
                    boxes.append((x, y, bw, bh))
            if boxes:
                return boxes
    # Haar fallback: equalize for low-light and try strict then relaxed settings.
    gray_eq = cv2.equalizeHist(gray_frame)
    faces = face_detector.detectMultiScale(
        gray_eq, scaleFactor=1.15, minNeighbors=6, minSize=(80, 80)
    )
    if len(faces) == 0:
        faces = face_detector.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
    return faces


def log_emotion_if_needed(emotion_index, confidence):
    global last_logged_emotion, last_logged_ts
    now = datetime.now()
    emotion_label = emotion_dict[emotion_index].strip()
    if emotion_label == last_logged_emotion and (now - last_logged_ts).total_seconds() < 1.5:
        return
    history_conn.execute(
        "INSERT INTO emotion_history (detected_at, emotion_label, confidence, detector) VALUES (?, ?, ?, ?)",
        (
            now.isoformat(timespec="seconds"),
            emotion_label,
            float(confidence),
            detector_name + ("+audio" if audio_module.available else ""),
        ),
    )
    history_conn.commit()
    last_logged_emotion = emotion_label
    last_logged_ts = now

def show_vid():
    global frame_counter, cached_face_prediction, smoothed_fused_prediction
    if not cap1.isOpened():
        lmain.after(500, show_vid)
        return

    flag1, frame1 = cap1.read()
    if not flag1:
        lmain.after(UI_REFRESH_MS, show_vid)
        return

    frame1 = cv2.resize(frame1, VIDEO_FRAME_SIZE)
    frame_counter += 1

    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = detect_faces(frame1, gray_frame)
    show_text[0] = -1
    show_confidence[0] = 0.0
    show_face_text[0] = -1
    show_face_confidence[0] = 0.0
    show_audio_text[0] = -1
    show_audio_confidence[0] = 0.0

    for (x, y, w, h) in num_faces:
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(gray_frame.shape[1], x + w)
        y2 = min(gray_frame.shape[0], y + h)
        cv2.rectangle(frame1, (x1, max(0, y1-15)), (x2, y2+10), (255, 0, 0), 2)
        if model_loaded:
            roi_gray_frame = gray_frame[y1:y2, x1:x2]
            if roi_gray_frame.size == 0:
                continue
            should_predict = (frame_counter % PREDICT_EVERY_N_FRAMES == 0) or (cached_face_prediction is None)
            if should_predict:
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0).astype("float32") / 255.0
                cached_face_prediction = emotion_model.predict(cropped_img, verbose=0)[0]
            face_prediction = cached_face_prediction
            face_idx = int(np.argmax(face_prediction))
            face_conf = float(np.max(face_prediction))

            audio_prediction = audio_module.get_audio_probs() if USE_AUDIO_FUSION else None
            if audio_prediction is not None:
                audio_idx = int(np.argmax(audio_prediction))
                audio_conf = float(np.max(audio_prediction))
                # Reduce audio impact when vision confidence is strong.
                face_weight = 0.95 if face_conf >= 0.60 else 0.80
                fused_prediction_raw = (face_weight * face_prediction) + ((1.0 - face_weight) * audio_prediction)
                show_audio_text[0] = audio_idx
                show_audio_confidence[0] = audio_conf
            else:
                fused_prediction_raw = face_prediction

            if smoothed_fused_prediction is None:
                smoothed_fused_prediction = fused_prediction_raw
            else:
                smoothed_fused_prediction = (
                    (SMOOTHING_ALPHA * fused_prediction_raw)
                    + ((1.0 - SMOOTHING_ALPHA) * smoothed_fused_prediction)
                )

            fused_idx_instant = int(np.argmax(smoothed_fused_prediction))
            fused_conf_instant = float(np.max(smoothed_fused_prediction))
            if fused_conf_instant >= FAST_SWITCH_CONFIDENCE:
                # For clear expressions (e.g., surprise), switch immediately.
                fused_idx = fused_idx_instant
                recent_fused_labels.clear()
                recent_fused_labels.append(fused_idx)
            else:
                recent_fused_labels.append(fused_idx_instant)
                fused_idx = max(set(recent_fused_labels), key=recent_fused_labels.count)

            fused_conf = float(smoothed_fused_prediction[fused_idx])
            if fused_conf < LOW_CONF_NEUTRAL_THRESHOLD:
                fused_idx = 4  # Neutral when prediction is too uncertain.
                fused_conf = float(smoothed_fused_prediction[4])

            cv2.putText(frame1, emotion_dict[fused_idx], (x1+20, max(30, y1-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            show_face_text[0] = face_idx
            show_face_confidence[0] = face_conf
            show_text[0] = fused_idx
            show_confidence[0] = fused_conf
            log_emotion_if_needed(fused_idx, fused_conf)
        else:
            cv2.putText(frame1, "Train model first (run train.py)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        break

    if show_text[0] == -1:
        smoothed_fused_prediction = None
        recent_fused_labels.clear()
        cv2.putText(frame1, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 215, 255), 2, cv2.LINE_AA)

    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(UI_REFRESH_MS, show_vid)


def show_vid2():
    missing_emoji = False
    if show_text[0] in emotion_dict:
        emoji_path = resolve_emoji_path(show_text[0])
        if emoji_path is not None:
            frame2 = cv2.imread(emoji_path)
        else:
            frame2 = None
            missing_emoji = True
        if frame2 is not None and frame2.size > 0:
            pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(pic2)
            img2 = img2.resize(EMOJI_DISPLAY_SIZE, RESAMPLE_LANCZOS)
        else:
            frame2 = np.zeros((EMOJI_DISPLAY_SIZE[1], EMOJI_DISPLAY_SIZE[0], 3), dtype=np.uint8)
            img2 = Image.fromarray(frame2)
    else:
        frame2 = np.zeros((EMOJI_DISPLAY_SIZE[1], EMOJI_DISPLAY_SIZE[0], 3), dtype=np.uint8)
        img2 = Image.fromarray(frame2)

    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    if not model_loaded:
        status_text = "Model missing: run train.py"
    elif show_text[0] == -1:
        status_text = "No face detected"
    else:
        fused = f"Fused: {emotion_dict[show_text[0]].strip()} ({show_confidence[0]*100:.1f}%)"
        face = f"Face: {emotion_dict[show_face_text[0]].strip()} ({show_face_confidence[0]*100:.1f}%)" if show_face_text[0] != -1 else "Face: n/a"
        if show_audio_text[0] != -1:
            audio = f"Audio: {emotion_dict[show_audio_text[0]].strip()} ({show_audio_confidence[0]*100:.1f}%)"
        elif audio_module.available:
            audio = "Audio: listening..."
        else:
            audio = "Audio: disabled"
        if missing_emoji:
            status_text = f"{fused}\n{face}\n{audio}\nEmoji image missing in ./emojis"
        else:
            status_text = f"{fused}\n{face}\n{audio}"
    lmain3.configure(text=status_text,font=('arial',22,'bold'))
    
    lmain2.configure(image=imgtk2)
    lmain2.after(EMOJI_REFRESH_MS, show_vid2)


def on_close():
    audio_module.stop()
    if mp_face_detector is not None:
        mp_face_detector.close()
    history_conn.close()
    if cap1.isOpened():
        cap1.release()
    root.destroy()

if __name__ == '__main__':
    root=tk.Tk()   
    if os.path.exists("logo.png"):
        img = ImageTk.PhotoImage(Image.open("logo.png"))
        heading = Label(root,image=img,bg='black')
    else:
        heading = Label(root, text="Emoji Creator", pady=20, font=('arial',45,'bold'), bg='black', fg='#CDCDCD')
    
    heading.pack() 
    heading2=Label(root,text="Photo to Emoji",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')                                 
    
    heading2.pack()
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)

    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black', justify='left', anchor='w')
    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)
    


    root.title("Photo To Emoji")            
    root.geometry("1400x900+100+10") 
    root['bg']='black'
    root.protocol("WM_DELETE_WINDOW", on_close)
    exitbutton = Button(root, text='Quit',fg="red",command=on_close,font=('arial',25,'bold')).pack(side = BOTTOM)
    show_vid()
    show_vid2()
    root.mainloop()
