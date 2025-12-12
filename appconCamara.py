# app.py (Audio + Video, video estable: /video rápido + YOLO worker)

import os
import time
import wave
import base64
import threading
import queue
from collections import deque
from pathlib import Path
from threading import Lock
from datetime import datetime

import numpy as np
import librosa
import tensorflow as tf

import cv2
from ultralytics import YOLO

from flask import Flask, request, jsonify, render_template, send_from_directory, abort, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# ==============================
# CONFIG
# ==============================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "modelo_cnn_big.h5"
ALERT_AUDIO_DIR = BASE_DIR / "alert_audios"
ALERT_AUDIO_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
WINDOW_SEC = 4.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
N_MELS = 64
CLASS_NAMES = ["ruido", "emergencia"]
ALERT_THRESHOLD = 0.70
ALERT_COOLDOWN_SEC = 8.0

MAX_FRAME_AGE = 10.0
MIN_FRAME_SIZE = 100
MAX_FRAME_SIZE = 500000
STATS_INTERVAL = 5.0
CLEANUP_INTERVAL = 15.0

# ==============================
# FLASK
# ==============================
app = Flask(__name__)
app.config["SECRET_KEY"] = "audio_video_secret_2024"
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=2000000,
)

# ==============================
# STATE
# ==============================
audio_buffer = deque(maxlen=WINDOW_SAMPLES)
alerts = []
last_alert = None
last_alert_time = 0.0

last_gps = {}

last_status = {
    "label": "-",
    "conf": 0.0,
    "noise_prob": 0.0,
    "emerg_prob": 0.0,
    "probs": {"ruido": 0.0, "emergencia": 0.0},
    "is_alert": False,
    "gps": {},
    "ts": None,
    "last_alert": None,
    "video_detections": [],
}

latest_frame = None
frame_timestamp = 0.0
client_count = 0
frame_count = 0
received_frames = 0
rejected_frames = 0
last_frame_time = time.time()
server_start_time = time.time()

state_lock = Lock()

video_queue = queue.Queue(maxsize=1)  # siempre el frame más reciente

# ==============================
# LOAD MODELS
# ==============================
print("🧠 Cargando modelo audio...")
MODEL = None
if MODEL_PATH.exists():
    try:
        MODEL = tf.keras.models.load_model(str(MODEL_PATH))
        print("✅ Audio model OK")
    except Exception as e:
        print("❌ Audio model error:", e)

YOLO_PATH = BASE_DIR / "best_fall_model_stable.pt"
YOLO_MODEL = None
YOLO_CONF_THRESHOLD = 0.50

print("🧠 Cargando YOLO...")
if YOLO_PATH.exists():
    try:
        YOLO_MODEL = YOLO(str(YOLO_PATH))
        print("✅ YOLO OK:", YOLO_MODEL.names)
    except Exception as e:
        print("❌ YOLO error:", e)
else:
    print("⚠️ YOLO no encontrado:", YOLO_PATH)

# ==============================
# AUDIO UTILS
# ==============================


def waveform_to_melspec(y, sr=SAMPLE_RATE):
    target_length = int(sr * WINDOW_SEC)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=1024, hop_length=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-8)

    S_norm = S_norm.astype("float32")
    S_norm = np.expand_dims(S_norm, axis=-1)
    return S_norm


def classify_window(samples_int16: np.ndarray):
    if MODEL is None:
        return "sin_modelo", 0.0, 0.5, 0.5

    y = samples_int16.astype("float32") / 32768.0
    spec = waveform_to_melspec(y, sr=SAMPLE_RATE)
    X = np.expand_dims(spec, axis=0)

    preds = MODEL.predict(X, verbose=0)[0]
    noise_prob = float(preds[0])
    emerg_prob = float(preds[1])

    label_idx = int(np.argmax(preds))
    label = CLASS_NAMES[label_idx]
    conf = float(preds[label_idx])

    return label, conf, noise_prob, emerg_prob


def save_wav(path: Path, samples_int16: np.ndarray, sr=SAMPLE_RATE):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int16.tobytes())

# ==============================
# VIDEO UTILS
# ==============================


def is_valid_jpeg(buf: bytes):
    return bool(buf) and len(buf) >= 10 and buf[0] == 0xFF and buf[1] == 0xD8 and buf[-2] == 0xFF and buf[-1] == 0xD9


def calculate_current_fps():
    global frame_count, last_frame_time
    now = time.time()
    elapsed = now - last_frame_time
    if elapsed <= 0 or frame_count == 0:
        return 0.0
    return round(frame_count / elapsed, 1)

# ==============================
# VIDEO WORKER (YOLO fuera del request)
# ==============================


def video_worker():
    global latest_frame, frame_timestamp
    while True:
        frame_buffer, now_ts = video_queue.get()

        final_image_bytes = frame_buffer
        detections_list = []

        if YOLO_MODEL is not None:
            try:
                np_arr = np.frombuffer(frame_buffer, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    results = YOLO_MODEL(
                        img, conf=YOLO_CONF_THRESHOLD, verbose=False)

                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            detections_list.append(YOLO_MODEL.names[cls_id])

                    annotated = results[0].plot()
                    ok, enc = cv2.imencode(".jpg", annotated)
                    if ok:
                        final_image_bytes = enc.tobytes()
            except Exception as e:
                print("⚠️ YOLO worker error:", e)

        with state_lock:
            latest_frame = final_image_bytes
            frame_timestamp = now_ts
            last_status["video_detections"] = detections_list
            cc = client_count
            fn = received_frames

        if cc > 0:
            try:
                b64 = base64.b64encode(final_image_bytes).decode("utf-8")
                socketio.emit("new-frame", {
                    "image": b64,
                    "timestamp": datetime.now().isoformat(),
                    "size": len(final_image_bytes),
                    "detections": detections_list,
                    "frameNumber": fn
                }, namespace="/")
            except Exception as e:
                print("⚠️ emit error:", e)

# ==============================
# BACKGROUND TASKS
# ==============================


def periodic_stats_task():
    global frame_count, last_frame_time
    while True:
        time.sleep(STATS_INTERVAL)
        with state_lock:
            fps = calculate_current_fps()
            stats_data = {
                "fps": fps,
                "totalFrames": received_frames,
                "connectedClients": client_count,
                "rejectedFrames": rejected_frames,
                "uptime": int(time.time() - server_start_time),
            }
            frame_count = 0
            last_frame_time = time.time()

        if stats_data["connectedClients"] > 0:
            socketio.emit("stats", stats_data, namespace="/")


def periodic_cleanup_task():
    global latest_frame, frame_timestamp
    while True:
        time.sleep(CLEANUP_INTERVAL)
        with state_lock:
            if latest_frame and frame_timestamp > 0:
                age = time.time() - frame_timestamp
                if age > MAX_FRAME_AGE:
                    print(f"🗑️ Frame antiguo descartado ({age:.1f}s)")
                    latest_frame = None
                    frame_timestamp = 0.0


threading.Thread(target=video_worker, daemon=True).start()
threading.Thread(target=periodic_stats_task, daemon=True).start()
threading.Thread(target=periodic_cleanup_task, daemon=True).start()

# ==============================
# ROUTES
# ==============================


@app.route("/")
def index():
    return render_template("panel.html")


@app.route("/panel")
def panel():
    return render_template("panel.html")


@app.route("/video_viewer")
def video_viewer():
    return render_template("video_stream.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "uptime": int(time.time() - server_start_time),
        "model_loaded": MODEL is not None,
        "video": {
            "clients": client_count,
            "frames_received": received_frames,
            "has_frame": latest_frame is not None
        },
        "audio": {
            "alerts": len(alerts),
            "buffer_size": len(audio_buffer)
        }
    }), 200


@app.route("/status")
def status():
    with state_lock:
        data = dict(last_status)
        data["gps"] = dict(last_gps) if last_gps else {}
        data["last_alert"] = last_alert
        data["video"] = {
            "frames_received": received_frames,
            "frames_rejected": rejected_frames,
            "clients_connected": client_count,
            "fps": calculate_current_fps(),
            "has_frame": latest_frame is not None,
            "frame_age": time.time() - frame_timestamp if frame_timestamp > 0 else None
        }
        data["server"] = {"uptime": int(time.time() - server_start_time)}
    return jsonify(data)


@app.route("/latest-frame")
def latest():
    with state_lock:
        cf = latest_frame
        ts = frame_timestamp
        fn = received_frames

    if not cf:
        return jsonify(error="No hay frames"), 404

    age = time.time() - ts
    if age > MAX_FRAME_AGE:
        return jsonify(error="Frame antiguo", age=age), 410

    return Response(cf, mimetype="image/jpeg",
                    headers={"X-Frame-Number": str(fn), "Cache-Control": "no-cache"})


@app.route("/stream")
def stream_mjpeg():
    """
    Stream MJPEG usando el último frame disponible.
    Funciona directo en <img src="/stream"> sin JS.
    """
    def gen():
        last_ts = 0.0
        while True:
            with state_lock:
                frame = latest_frame
                ts = frame_timestamp

            # Solo envía si hay frame nuevo
            if frame and ts != last_ts:
                last_ts = ts
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    + frame
                    + b"\r\n"
                )
            else:
                time.sleep(0.03)  # evita usar CPU al 100%

    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.route("/video", methods=["POST", "OPTIONS"])
def video():
    global received_frames, rejected_frames, frame_count

    if request.method == "OPTIONS":
        return "", 204

    frame_buffer = request.get_data()
    n = len(frame_buffer)
    now_ts = time.time()

    if n < MIN_FRAME_SIZE:
        with state_lock:
            rejected_frames += 1
        return jsonify(error="Frame pequeño"), 400

    if n > MAX_FRAME_SIZE:
        with state_lock:
            rejected_frames += 1
        return jsonify(error="Frame grande"), 413

    if not is_valid_jpeg(frame_buffer):
        with state_lock:
            rejected_frames += 1
        return jsonify(error="JPEG inválido"), 400

    with state_lock:
        received_frames += 1
        frame_count += 1

    # cola max=1: si está llena, botar viejo y meter nuevo
    try:
        video_queue.put_nowait((frame_buffer, now_ts))
    except queue.Full:
        try:
            video_queue.get_nowait()
        except queue.Empty:
            pass
        video_queue.put_nowait((frame_buffer, now_ts))

    return "OK", 200

# ==============================
# SOCKETIO
# ==============================


@socketio.on("connect")
def on_connect():
    global client_count
    with state_lock:
        client_count += 1
        cc = client_count
    print("👤 WS conectado:", request.sid[:8], "total:", cc)
    emit("stats", {
        "fps": calculate_current_fps(),
        "totalFrames": received_frames,
        "connectedClients": cc,
        "rejectedFrames": rejected_frames,
        "uptime": int(time.time() - server_start_time),
    })


@socketio.on("disconnect")
def on_disconnect():
    global client_count
    with state_lock:
        client_count = max(0, client_count - 1)
        cc = client_count
    print("👋 WS desconectado:", request.sid[:8], "total:", cc)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=PORT,
                 debug=False, use_reloader=False)
