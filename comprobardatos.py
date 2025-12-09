import os
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from flask import (
    Flask, render_template, request,
    redirect, url_for, send_from_directory
)

# =========================
# CONFIGURACIÓN BÁSICA
# =========================

BASE_DIR = Path(__file__).resolve().parent

SRC_DIR = BASE_DIR / "emergencia_raw"         # audios crudos
OUT_DIR = BASE_DIR / "emergencia_final"       # recortes de 4 s para entrenar
TRASH_DIR = BASE_DIR / "emergencia_descartada"  # audios descartados

SAMPLE_RATE = 16000
WINDOW_SEC = 4.0

OUT_DIR.mkdir(exist_ok=True)
TRASH_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


def get_next_file():
    """
    Devuelve el siguiente .wav pendiente en emergencia_raw.
    Si no quedan, devuelve None.
    """
    wavs = sorted(SRC_DIR.glob("*.wav"))
    return wavs[0] if wavs else None


# =========================
# RUTAS
# =========================

@app.route("/")
def index():
    audio_path = get_next_file()
    if audio_path is None:
        # No quedan audios por procesar
        return render_template("done.html")

    # Calculamos duración real del audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr if len(y) > 0 else 0.0

    max_start = max(0.0, duration - WINDOW_SEC)

    return render_template(
        "labeler.html",
        audio_file=audio_path.name,
        duration=round(duration, 3),
        window_sec=WINDOW_SEC,
        max_start=round(max_start, 3),
    )


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """
    Sirve el archivo de audio desde emergencia_raw para el reproductor HTML.
    """
    return send_from_directory(SRC_DIR, filename)


@app.route("/save", methods=["POST"])
def save_snippet():
    """
    Guarda un recorte de 4 s a partir de 'start' para el archivo actual.
    Luego mueve el original a emergencia_descartada (o podrías crear
    otra carpeta de 'procesados' si quieres separarlos).
    """
    filename = request.form["filename"]
    start = float(request.form["start"])

    audio_path = SRC_DIR / filename
    if not audio_path.exists():
        return "Archivo no encontrado", 404

    # Cargamos y re-muestreamos a 16 kHz para el dataset
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    start_sample = int(start * SAMPLE_RATE)
    win_samples = int(WINDOW_SEC * SAMPLE_RATE)
    end_sample = start_sample + win_samples

    if start_sample < 0:
        start_sample = 0
    if end_sample > len(y):
        # Rellenar con ceros si se pasa
        pad = end_sample - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    snippet = y[start_sample:end_sample]

    # Nombre de salida: original_sXX.YY.wav
    base = audio_path.stem
    out_name = f"{base}_s{start:.2f}.wav"
    out_path = OUT_DIR / out_name

    sf.write(out_path, snippet, SAMPLE_RATE)
    print(f"[OK] Guardado recorte en {out_path}")

    # Mover original a descartados (ya no lo necesitamos para etiquetar)
    dest = TRASH_DIR / filename
    audio_path.replace(dest)
    print(f"[INFO] Original movido a {dest}")

    return redirect(url_for("index"))


@app.route("/delete", methods=["POST"])
def delete_audio():
    """
    Descarta por completo este audio: se mueve a emergencia_descartada
    sin generar ningún recorte.
    """
    filename = request.form["filename"]
    audio_path = SRC_DIR / filename
    if audio_path.exists():
        dest = TRASH_DIR / filename
        audio_path.replace(dest)
        print(f"[INFO] Audio descartado, movido a {dest}")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Ejecuta en el puerto 5002 para no chocar con otros servidores
    app.run(host="0.0.0.0", port=5002, debug=True)
