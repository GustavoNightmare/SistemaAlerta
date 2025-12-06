import time
from collections import deque
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, render_template, send_file
import tensorflow as tf

# =====================================================
# CONFIGURACIÓN
# =====================================================

SAMPLE_RATE = 16000      # Hz
DURATION = 4.0           # segundos por ventana
N_MELS = 64

MODEL_PATH = "modelo_sonidos.tflite"
LABELS_PATH = "label_names.npy"

ALERT_THRESHOLD = 0.70   # umbral para emergencia (ajusta a gusto)

AUDIO_BUFFER_SAMPLES = int(SAMPLE_RATE * DURATION)
LAST_AUDIO_PATH = Path("last_audio.wav")

ALERTS_DIR = Path("alerts")         # carpeta para audios de alertas
ALERTS_DIR.mkdir(exist_ok=True)
ALERT_HISTORY_LIMIT = 50            # máximo de alertas guardadas en memoria

# =====================================================
# CARGAR MODELO TFLITE
# =====================================================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_names = np.load(LABELS_PATH, allow_pickle=True)
label_names = [str(x) for x in label_names]

NOISE_LABEL = label_names[0]    # 'ruido'
EMERG_LABEL = label_names[1]    # 'emergencia'

print("Clases cargadas:", label_names)
print("NOISE_LABEL:", NOISE_LABEL)
print("EMERG_LABEL:", EMERG_LABEL)

# =====================================================
# PREPROCESAMIENTO
# =====================================================


def waveform_to_melspec(y, sr=SAMPLE_RATE):
    target_length = int(sr * DURATION)
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
    S_norm = np.expand_dims(S_norm, axis=-1)
    return S_norm


def predict_from_waveform(y):
    spec = waveform_to_melspec(y)
    input_data = np.expand_dims(spec, axis=0)

    expected_shape = tuple(input_details[0]["shape"])

    if input_data.shape != expected_shape:
        _, n_mels_e, time_e, ch_e = expected_shape
        spec_resized = spec
        if spec_resized.shape[1] > time_e:
            spec_resized = spec_resized[:, :time_e, :]
        else:
            pad_width = time_e - spec_resized.shape[1]
            spec_resized = np.pad(
                spec_resized,
                ((0, 0), (0, pad_width), (0, 0)),
                mode="constant",
            )
        input_data = np.expand_dims(spec_resized, axis=0).astype("float32")

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    probs = {label_names[i]: float(p) for i, p in enumerate(output_data)}
    idx = int(np.argmax(output_data))
    label = label_names[idx]
    conf = float(output_data[idx])

    return label, conf, probs

# =====================================================
# ESTADO GLOBAL
# =====================================================


current_result = {
    "label": None,
    "conf": 0.0,
    "probs": {},
}

last_gps_data = {}

audio_buffer = deque(maxlen=AUDIO_BUFFER_SAMPLES)
last_audio_update = 0.0

alerts_history = []      # lista de dicts con info de cada alerta
last_alert_info = None   # última alerta registrada
alert_counter = 0

# =====================================================
# FLASK
# =====================================================

app = Flask(__name__)

# ---------- AUDIO DESDE LA ESP32 ----------


@app.route("/audio_upload", methods=["POST"])
def audio_upload():
    global current_result, last_audio_update
    global alerts_history, last_alert_info, alert_counter

    raw = request.get_data()
    if not raw:
        return jsonify({"error": "empty body"}), 400

    fmt = (request.headers.get("X-Audio-Format") or "pcm16").lower()
    sr = int(request.headers.get("X-Sample-Rate") or SAMPLE_RATE)

    if fmt != "pcm16":
        return jsonify({"error": "unsupported format"}), 400

    if sr != SAMPLE_RATE:
        # Si quieres, aquí podrías re-muestrear.
        pass

    # int16 LE -> float32 [-1,1]
    audio_int16 = np.frombuffer(raw, dtype="<i2")
    y = audio_int16.astype("float32") / 32768.0

    # Añadir al buffer circular
    for s in y:
        audio_buffer.append(s)

    # Cuando tenemos al menos 4 s, hacemos inferencia
    if len(audio_buffer) >= AUDIO_BUFFER_SAMPLES:
        window = np.array(audio_buffer, dtype="float32")

        # Guardar siempre la última ventana por si quieres depurar
        try:
            sf.write(str(LAST_AUDIO_PATH), window, SAMPLE_RATE)
        except Exception as e:
            print("Error guardando last_audio.wav:", e)

        # Inferencia
        label, conf, probs = predict_from_waveform(window)
        emerg_prob = float(probs.get(EMERG_LABEL, 0.0))

        print(f"[AUDIO] Predicción: {label} conf={conf:.3f} probs={probs}")

        current_result = {
            "label": label,
            "conf": conf,
            "probs": probs,
        }
        last_audio_update = time.time()

        # ¿Es alerta?
        is_alert_now = emerg_prob >= ALERT_THRESHOLD
        now_ts = time.time()

        if is_alert_now:
            # Evitar crear 100 alertas seguidas para el mismo evento:
            # solo si ha pasado al menos DURATION s desde la última alerta
            if (last_alert_info is None) or (now_ts - last_alert_info["ts"] > DURATION):
                alert_counter += 1
                alert_id = alert_counter

                alert_path = ALERTS_DIR / f"alert_{alert_id}.wav"
                try:
                    sf.write(str(alert_path), window, SAMPLE_RATE)
                except Exception as e:
                    print("Error guardando audio de alerta:", e)

                gps_snapshot = dict(last_gps_data) if last_gps_data else {}

                alert_info = {
                    "id": alert_id,
                    "ts": now_ts,
                    "emerg_prob": emerg_prob,
                    "label": label,
                    "gps": gps_snapshot,
                    "audio_path": str(alert_path),
                }

                alerts_history.append(alert_info)
                # Limitar tamaño del historial en memoria
                if len(alerts_history) > ALERT_HISTORY_LIMIT:
                    alerts_history = alerts_history[-ALERT_HISTORY_LIMIT:]

                last_alert_info = alert_info
                print(f"[ALERTA] Registrada alerta #{alert_id}")

    return jsonify({"status": "ok"})

# Último audio crudo (para debug opcional)


@app.route("/last_audio.wav")
def last_audio():
    if not LAST_AUDIO_PATH.exists():
        return ("No hay audio todavía", 404)
    return send_file(str(LAST_AUDIO_PATH), mimetype="audio/wav")

# Audio específico de una alerta


@app.route("/alert_audio/<int:alert_id>.wav")
def alert_audio(alert_id):
    for a in alerts_history:
        if a["id"] == alert_id:
            path = a["audio_path"]
            if Path(path).exists():
                return send_file(path, mimetype="audio/wav")
    return ("Audio de alerta no encontrado", 404)

# ---------- GPS ----------


@app.route('/gps', methods=['GET', 'POST'])
def receive_gps():
    global last_gps_data

    if request.method == 'POST':
        last_gps_data = request.get_json(force=True) or {}
        print("Datos GPS recibidos:", last_gps_data)
        return jsonify({"status": "ok"}), 200

    return render_template('gps.html', data=last_gps_data)


@app.route('/gps/json')
def gps_json():
    return jsonify(last_gps_data)

# ---------- ESTADO GLOBAL PARA EL PANEL ----------


@app.route('/status')
def status():
    data = dict(current_result)

    probs = data.get("probs") or {}
    emerg_prob = float(probs.get(EMERG_LABEL, 0.0))

    is_alert = emerg_prob >= ALERT_THRESHOLD

    data["is_alert"] = is_alert
    data["emerg_prob"] = emerg_prob
    data["noise_prob"] = float(probs.get(NOISE_LABEL, 0.0))
    data["gps"] = last_gps_data

    # Info de la última alerta (para el historial en el frontend)
    if last_alert_info is not None:
        alert = {
            "id": last_alert_info["id"],
            "ts": last_alert_info["ts"],
            "emerg_prob": last_alert_info["emerg_prob"],
            "label": last_alert_info["label"],
            "gps": last_alert_info.get("gps", {}),
            "audio_url": f"/alert_audio/{last_alert_info['id']}.wav",
        }
    else:
        alert = None

    data["last_alert"] = alert
    return jsonify(data)

# ---------- VISTA PRINCIPAL ----------


@app.route('/')
def index():
    return render_template('dashboard.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
