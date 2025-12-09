import io
import wave
from collections import deque

import numpy as np
from flask import Flask, request, jsonify, send_file, render_template

# =========================
# CONFIG
# =========================

SAMPLE_RATE = 16000
RING_SECONDS = 6
PREVIEW_SECONDS = 3

RING_SAMPLES = SAMPLE_RATE * RING_SECONDS

# Buffer circular con las últimas muestras crudas (int16)
audio_ring = deque(maxlen=RING_SAMPLES)

# Parámetros "tipo ESP32"
PARAMS = {
    "noise_gate": 100,              # en unidades de int16 (0..32767)
    "target_level": 0.25,           # nivel deseado [0..1] sobre 32768
    "max_gain": 6.0,
    "min_gain": 0.5,
    "agc_speed": 0.05,
    "compression_threshold": 0.6,   # fracción de full scale
    "compression_ratio": 3.0,
    "apply_processing": True,
}

current_gain = 1.0  # estado del AGC (como currentGain en la ESP)

app = Flask(__name__)


# =========================
# UTILIDADES DE PROCESADO
# =========================

def compress_sample(sample, threshold, ratio):
    """
    sample: float en [-1, 1]
    threshold: fracción (0..1)
    ratio: >1
    """
    abs_s = abs(sample)
    if abs_s > threshold:
        excess = abs_s - threshold
        comp = threshold + excess / ratio
        if sample < 0:
            comp = -comp
    else:
        comp = sample

    # limitar
    if comp > 1.0:
        comp = 1.0
    if comp < -1.0:
        comp = -1.0

    return comp


def process_block_int16(block_int16):
    """
    Aplica noise gate + AGC + compresor como en la ESP32.
    block_int16: np.array int16
    Devuelve: np.array int16 procesado
    """
    global current_gain, PARAMS

    if len(block_int16) == 0:
        return block_int16

    # Copia float [-1, 1]
    samples = block_int16.astype(np.float32) / 32768.0

    noise_gate = float(PARAMS["noise_gate"]) / 32768.0
    target_level = float(PARAMS["target_level"])
    max_gain = float(PARAMS["max_gain"])
    min_gain = float(PARAMS["min_gain"])
    agc_speed = float(PARAMS["agc_speed"])
    thr = float(PARAMS["compression_threshold"])
    ratio = float(PARAMS["compression_ratio"])

    # 1) calcular nivel medio de las muestras que pasan el noise_gate
    mask = np.abs(samples) >= noise_gate
    if np.any(mask):
        avg_level = float(np.mean(np.abs(samples[mask])))
        # AGC estilo ESP32
        if avg_level > target_level * 1.2:
            current_gain -= agc_speed
        elif avg_level < target_level * 0.8:
            current_gain += agc_speed

        if current_gain > max_gain:
            current_gain = max_gain
        if current_gain < min_gain:
            current_gain = min_gain
    else:
        # si está todo por debajo del gate, no tocamos gain
        pass

    # 2) aplicar gate + ganancia + compresor
    out = np.zeros_like(samples, dtype=np.float32)

    for i, s in enumerate(samples):
        # noise gate
        if abs(s) < noise_gate:
            out[i] = 0.0
            continue

        # AGC
        s *= current_gain

        # compresión
        s = compress_sample(s, thr, ratio)

        out[i] = s

    # De vuelta a int16
    out_int16 = np.clip(out * 32767.0, -32768, 32767).astype(np.int16)
    return out_int16


def get_last_block(seconds=PREVIEW_SECONDS):
    """Devuelve las últimas 'seconds' de audio crudo (int16)."""
    n_samples = int(SAMPLE_RATE * seconds)
    if len(audio_ring) == 0:
        return np.array([], dtype=np.int16)

    if len(audio_ring) < n_samples:
        data = np.array(audio_ring, dtype=np.int16)
    else:
        data = np.array(list(audio_ring)[-n_samples:], dtype=np.int16)
    return data


def make_wav_bytes(int16_data):
    """Crea un WAV en memoria y devuelve BytesIO listo para send_file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int16_data.tobytes())
    buf.seek(0)
    return buf


# =========================
# RUTAS FLASK
# =========================

@app.route("/")
def index():
    return render_template("mic_tuner.html")


@app.route("/audio_raw", methods=["POST"])
def audio_raw():
    """
    Recibe audio crudo PCM16 desde la ESP32.
    """
    raw = request.get_data()
    if not raw:
        return jsonify({"error": "empty body"}), 400

    # Interpretar como int16 little-endian
    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size == 0:
        return jsonify({"status": "no_samples"}), 200

    # Añadir al buffer circular
    for s in samples:
        audio_ring.append(int(s))

    return jsonify({"status": "ok", "received": int(samples.size)}), 200


@app.route("/preview.wav")
def preview_wav():
    """
    Devuelve un WAV con los últimos PREVIEW_SECONDS segundos de audio.
    Si apply_processing=True, aplica noise gate + AGC + compresor.
    """
    if len(audio_ring) == 0:
        return ("No hay audio todavía", 404)

    block = get_last_block(PREVIEW_SECONDS)
    if block.size == 0:
        return ("No hay audio válido", 404)

    if PARAMS.get("apply_processing", True):
        processed = process_block_int16(block)
    else:
        processed = block

    wav_buf = make_wav_bytes(processed)
    return send_file(
        wav_buf,
        mimetype="audio/wav",
        as_attachment=False,
        download_name="preview.wav"
    )


@app.route("/params", methods=["GET", "POST"])
def params():
    global PARAMS, current_gain

    if request.method == "GET":
        return jsonify(PARAMS)

    data = request.get_json(force=True, silent=True) or {}
    # Actualizar solo claves válidas
    for key in list(PARAMS.keys()):
        if key in data:
            if key == "apply_processing":
                PARAMS[key] = bool(data[key])
            else:
                try:
                    PARAMS[key] = float(data[key])
                except ValueError:
                    pass

    # Opción: resetear el gain cuando cambias parámetros
    current_gain = 1.0

    return jsonify({"status": "ok", "params": PARAMS})


if __name__ == "__main__":
    # Ejecutar en 0.0.0.0:5001 para que la ESP pueda conectar
    app.run(host="0.0.0.0", port=5001, debug=True)
