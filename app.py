import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import librosa
from flask import Flask, request, jsonify, render_template

import tensorflow as tf

# =========================
# CONFIGURACIÓN
# =========================

SAMPLE_RATE = 16000     # Hz, igual que en el entrenamiento
DURATION = 4.0          # segundos por ventana
N_MELS = 64

MODEL_PATH = "modelo_sonidos4segundos.tflite"
LABELS_PATH = "label_names.npy"

# Etiquetas especiales
ALERT_LABELS = ["disparo", "grito"]  # ajusta a los nombres reales
NOISE_LABEL = "ruido"                # ajusta al nombre real de ruido
ALERT_THRESHOLD = 0.80

# =========================
# CARGAR MODELO TFLITE
# =========================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar nombres de clases
label_names = np.load(LABELS_PATH, allow_pickle=True)

# =========================
# PREPROCESAMIENTO
# =========================


def waveform_to_melspec(y, sr=SAMPLE_RATE):
    """
    y: señal 1D (numpy array) en float32.
    Devuelve mel-espectrograma normalizado (n_mels, time, 1).
    """
    # Duración fija (recortar / rellenar)
    target_length = int(sr * DURATION)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        pad_width = target_length - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")

    # Mel-espectrograma
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=256,
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalizar a [0,1]
    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-8)

    S_norm = S_norm.astype("float32")
    S_norm = np.expand_dims(S_norm, axis=-1)  # (n_mels, time, 1)
    return S_norm


def predict_from_waveform(y):
    """
    Usa el modelo TFLite para predecir a partir de la señal de audio cruda.
    Aplica lógica de tolerancia:
      - Solo considera 'disparo' o 'grito' si conf >= ALERT_THRESHOLD
      - Si no, lo degrada a 'ruido'.
    """
    spec = waveform_to_melspec(y)                # (n_mels, time, 1)
    input_data = np.expand_dims(spec, axis=0)    # (1, n_mels, time, 1)

    # Asegurar que el tamaño coincide con el esperado por el modelo
    expected_shape = input_details[0]['shape']   # p.ej. [1, 64, 122, 1]

    if input_data.shape != tuple(expected_shape):
        _, n_mels_e, time_e, ch_e = expected_shape
        spec_resized = spec

        # Recortar o rellenar en eje de tiempo
        if spec_resized.shape[1] > time_e:
            spec_resized = spec_resized[:, :time_e, :]
        else:
            pad_width = time_e - spec_resized.shape[1]
            spec_resized = np.pad(
                spec_resized,
                ((0, 0), (0, pad_width), (0, 0)),
                mode="constant"
            )

        input_data = np.expand_dims(spec_resized, axis=0).astype("float32")

    # Inferencia TFLite
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Probabilidades por clase
    probs = {str(label_names[i]): float(p) for i, p in enumerate(output_data)}

    # Clase ganadora "cruda"
    idx = int(np.argmax(output_data))
    raw_label = str(label_names[idx])
    raw_conf = float(output_data[idx])

    # Post-procesado
    label = raw_label
    conf = raw_conf

    # Si la clase ganadora es disparo o grito pero con poca confianza,
    # la degradamos a ruido.
    if raw_label in ALERT_LABELS and raw_conf < ALERT_THRESHOLD:
        if NOISE_LABEL in probs:
            label = NOISE_LABEL
            conf = probs[NOISE_LABEL]
        else:
            label = NOISE_LABEL
            conf = 1.0 - raw_conf

    return label, conf, probs


# =========================
# ESTADO GLOBAL (audio + GPS)
# =========================

current_result = {
    "label": None,
    "conf": 0.0,
    "probs": {}
}

last_gps_data = {}


# =========================
# CAPTURA DE AUDIO EN HILO
# =========================


def audio_loop():
    global current_result
    print("🎙 Iniciando captura de audio en tiempo real...")

    while True:
        try:
            # Grabar DURATION segundos del micrófono
            frames = int(DURATION * SAMPLE_RATE)
            audio = sd.rec(
                frames,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()  # esperar a que termine la grabación

            y = audio.flatten()   # pasar a vector 1D

            label, conf, probs = predict_from_waveform(y)

            current_result = {
                "label": label,
                "conf": conf,
                "probs": probs
            }

            # pequeño descanso para no reventar la CPU
            time.sleep(0.1)

        except Exception as e:
            print("❌ Error en audio_loop:", e)
            time.sleep(1.0)


# =========================
# SERVIDOR FLASK
# =========================

app = Flask(__name__)


# ---------- RUTAS GPS ----------

@app.route('/gps', methods=['GET', 'POST'])
def receive_gps():
    global last_gps_data

    if request.method == 'POST':
        last_gps_data = request.get_json(force=True) or {}
        print("Datos GPS recibidos:", last_gps_data)
        return jsonify({"status": "ok"}), 200

    # GET: página con mapa
    return render_template('gps.html', data=last_gps_data)


@app.route('/gps/json')
def gps_json():
    """Devuelve el último dato como JSON (para AJAX)."""
    return jsonify(last_gps_data)


# ---------- ESTADO AUDIO + ALERTA + GPS ----------

@app.route('/status')
def status():
    """
    Devuelve el estado actual del modelo de audio
    + info de alerta + último GPS.
    """
    data = dict(current_result)  # copia

    label = data.get("label")
    conf = float(data.get("conf") or 0.0)

    # ¿Es disparo o grito con confianza suficiente?
    is_alert = label in ALERT_LABELS and conf >= ALERT_THRESHOLD
    data["is_alert"] = is_alert

    # Añadimos último GPS
    data["gps"] = last_gps_data

    return jsonify(data)


# ---------- DASHBOARD PRINCIPAL ----------

@app.route('/')
def index():
    # Panel principal
    return render_template('dashboard.html')


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    # Iniciar hilo de audio
    t = threading.Thread(target=audio_loop, daemon=True)
    t.start()

    # Lanzar Flask
    app.run(host="0.0.0.0", port=5000, debug=True)
