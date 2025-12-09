import os
import time
import wave
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np
import librosa
import tensorflow as tf
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
    abort,
)

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "modelo_cnn_big.h5"         # <-- usamos el BIG
ALERT_AUDIO_DIR = BASE_DIR / "alert_audios"
ALERT_AUDIO_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
WINDOW_SEC = 4.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)

N_MELS = 64

CLASS_NAMES = ["ruido", "emergencia"]
ALERT_THRESHOLD = 0.70        # debe coincidir con el del front
ALERT_COOLDOWN_SEC = 8.0      # tiempo mínimo entre alertas nuevas

# ==============================
# ESTADO GLOBAL
# ==============================

app = Flask(__name__)

# Buffer circular de audio (PCM16)
audio_buffer = deque(maxlen=WINDOW_SAMPLES)

# Último GPS recibido
last_gps = {}

# Último estado de clasificación
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
}

# Historial de alertas
alerts = []
last_alert = None
last_alert_time = 0.0

state_lock = Lock()

# ==============================
# CARGA DE MODELO
# ==============================

print("Cargando modelo desde:", MODEL_PATH)
MODEL = tf.keras.models.load_model(str(MODEL_PATH))
print("Modelo cargado OK.")


# ==============================
# UTILIDADES AUDIO
# ==============================

def waveform_to_melspec(y, sr=SAMPLE_RATE):
    """
    Misma transformación que se usó para entrenar:
    - recorta / rellena a 4 s
    - mel-espectrograma
    - dB -> normalización [0,1]
    - salida (64, T, 1)
    """
    target_length = int(sr * WINDOW_SEC)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        pad_width = target_length - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=256,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-8)

    S_norm = S_norm.astype("float32")
    S_norm = np.expand_dims(S_norm, axis=-1)  # (64, T, 1)
    return S_norm


def classify_window(samples_int16: np.ndarray):
    """
    samples_int16: np.int16, tamaño WINDOW_SAMPLES
    Devuelve: label, conf, noise_prob, emerg_prob
    """
    # Escalar a [-1,1] como librosa
    y = samples_int16.astype("float32") / 32768.0

    spec = waveform_to_melspec(y, sr=SAMPLE_RATE)
    X = np.expand_dims(spec, axis=0)  # (1, 64, T, 1)

    preds = MODEL.predict(X, verbose=0)[0]  # [p_ruido, p_emergencia]
    noise_prob = float(preds[0])
    emerg_prob = float(preds[1])

    label_idx = int(np.argmax(preds))
    label = CLASS_NAMES[label_idx]
    conf = float(preds[label_idx])

    return label, conf, noise_prob, emerg_prob


def save_wav(path: Path, samples_int16: np.ndarray, sr=SAMPLE_RATE):
    """
    Guarda un array int16 mono a un .wav
    """
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(sr)
        wf.writeframes(samples_int16.tobytes())


# ==============================
# RUTAS DE HTML
# ==============================

@app.route("/")
def panel():
    """
    Panel principal de detección (segundo HTML que enviaste).
    Guarda el archivo como templates/panel.html
    """
    return render_template("panel.html")


@app.route("/gps", methods=["GET", "POST"])
def gps_view_or_receive():
    """
    - POST: recibe JSON con GPS desde el ESP32
    - GET: muestra el mapa (primer HTML: templates/gps.html)
    """
    global last_gps

    if request.method == "POST":
        data = request.get_json(force=True) or {}
        with state_lock:
            last_gps = data
        return jsonify({"status": "ok"}), 200

    # GET: solo renderiza la plantilla (los parámetros lat/lon/audio se leen en JS)
    return render_template("gps.html")


@app.route("/gps/json")
def gps_json():
    """
    Devuelve el último GPS recibido (para el mapa en vivo).
    """
    with state_lock:
        gps_copy = dict(last_gps) if last_gps else {}
    return jsonify(gps_copy)


# ==============================
# RUTA: AUDIO DESDE ESP32
# ==============================

@app.route("/audio_upload", methods=["POST"])
def audio_upload():
    """
    Recibe fragmentos de audio PCM16 16 kHz desde el ESP32:
    - Content-Type: application/octet-stream
    - X-Audio-Format: pcm16
    - X-Sample-Rate: 16000
    """
    global audio_buffer, last_status, last_alert, last_alert_time, alerts

    raw = request.data
    if not raw:
        return jsonify({"ok": False, "error": "sin datos"}), 400

    # Bytes -> int16
    samples = np.frombuffer(raw, dtype=np.int16)

    # Actualizar buffer circular
    with state_lock:
        audio_buffer.extend(samples.tolist())
        buffer_len = len(audio_buffer)
        gps_copy = dict(last_gps) if last_gps else {}

    if buffer_len < WINDOW_SAMPLES:
        # Aún no tenemos 4s de audio completos
        now_ts = time.time()
        with state_lock:
            last_status.update(
                {
                    "label": "calibrando",
                    "conf": 0.0,
                    "noise_prob": 0.0,
                    "emerg_prob": 0.0,
                    "probs": {"ruido": 0.0, "emergencia": 0.0},
                    "is_alert": False,
                    "gps": gps_copy,
                    "ts": now_ts,
                    "last_alert": last_alert,
                }
            )
        return jsonify({"ok": True, "label": "calibrando"}), 200

    # Tomar la última ventana de 4 s
    with state_lock:
        window_samples = np.array(
            audio_buffer, dtype=np.int16)[-WINDOW_SAMPLES:]
        gps_copy = dict(last_gps) if last_gps else {}
        current_last_alert = last_alert
        current_last_alert_time = last_alert_time

    # Clasificar (fuera del lock, para no bloquear otros hilos)
    label, conf, noise_prob, emerg_prob = classify_window(window_samples)
    is_alert = emerg_prob >= ALERT_THRESHOLD
    now_ts = time.time()

    # Actualizar estado y, si corresponde, crear alerta
    with state_lock:
        last_status = {
            "label": label,
            "conf": conf,
            "noise_prob": noise_prob,
            "emerg_prob": emerg_prob,
            "probs": {
                CLASS_NAMES[0]: noise_prob,
                CLASS_NAMES[1]: emerg_prob,
            },
            "is_alert": is_alert,
            "gps": gps_copy,
            "ts": now_ts,
            "last_alert": last_alert,
        }

        # Crear nueva alerta si supera umbral y respeta cooldown
        if is_alert and (now_ts - last_alert_time > ALERT_COOLDOWN_SEC):
            alert_id = len(alerts) + 1
            filename = f"{alert_id}.wav"
            save_path = ALERT_AUDIO_DIR / filename
            save_wav(save_path, window_samples, sr=SAMPLE_RATE)

            alert_obj = {
                "id": alert_id,
                "ts": now_ts,
                "gps": gps_copy,
                "emerg_prob": emerg_prob,
                "audio_url": f"/alert_audio/{alert_id}.wav",
            }

            alerts.append(alert_obj)
            last_alert = alert_obj
            last_alert_time = now_ts

            # también lo reflejamos en el status
            last_status["last_alert"] = alert_obj

    return jsonify(
        {
            "ok": True,
            "label": label,
            "conf": conf,
            "noise_prob": noise_prob,
            "emerg_prob": emerg_prob,
            "is_alert": is_alert,
        }
    ), 200


# ==============================
# RUTA: STATUS PARA EL PANEL
# ==============================

@app.route("/status")
def status():
    """
    Devuelve el estado actual para el panel:
    {
      label, conf,
      noise_prob, emerg_prob,
      probs: {ruido, emergencia},
      is_alert,
      gps: {...},
      last_alert: {id, ts, gps, emerg_prob, audio_url} | null
    }
    """
    with state_lock:
        data = dict(last_status)  # copia superficial
        # asegurar que tenga gps, aunque no haya clasificación
        if "gps" not in data:
            data["gps"] = dict(last_gps) if last_gps else {}
        # asegurar que last_alert esté sincronizado
        data["last_alert"] = last_alert

    return jsonify(data)


# ==============================
# RUTA: SERVIR AUDIO DE ALERTA
# ==============================

@app.route("/alert_audio/<alert_id>.wav")
def alert_audio(alert_id):
    """
    Sirve el archivo WAV guardado para una alerta.
    """
    filename = f"{alert_id}.wav"
    path = ALERT_AUDIO_DIR / filename
    if not path.exists():
        abort(404)
    return send_from_directory(
        ALERT_AUDIO_DIR,
        filename,
        mimetype="audio/wav",
        as_attachment=False,
    )


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    # Puedes cambiar host/puerto si hace falta
    app.run(host="0.0.0.0", port=5000, debug=True)
