import shutil
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    abort,
)

app = Flask(__name__)

# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).resolve().parent

SRC_CLIP_DIR = BASE_DIR / "emergencia_final"   # recortes ya hechos (~4 s)
SRC_RAW_DIR = BASE_DIR / "emergencia_raw"      # audios largos

DEST_GUN_DIR = BASE_DIR / "disparo_final"      # nuevo: disparos
DEST_SCREAM_DIR = BASE_DIR / "grito_final"     # nuevo: gritos
DEST_DISCARD_DIR = BASE_DIR / "emergencia_descartada"

SAMPLE_RATE = 16000
WINDOW_SEC = 4.0


# =========================
# UTILIDADES
# =========================

def get_pending_files():
    """
    Construye la lista de audios pendientes de etiquetar.

    - De emergencia_final: sólo los que NO existen ya en
      disparo_final / grito_final / emergencia_descartada.
    - De emergencia_raw: todos los .wav que sigan en esa carpeta.
      (los procesados se moverán fuera).
    """
    items = []

    # nombres ya procesados (por nombre de archivo)
    processed_names = set()
    for d in [DEST_GUN_DIR, DEST_SCREAM_DIR, DEST_DISCARD_DIR]:
        if d.is_dir():
            for p in d.glob("*.wav"):
                processed_names.add(p.name)

    # 1) clips ya recortados
    if SRC_CLIP_DIR.is_dir():
        for p in sorted(SRC_CLIP_DIR.glob("*.wav")):
            if p.name in processed_names:
                continue
            items.append({
                "mode": "clip",   # ya son ~4s
                "rel_path": str(p.relative_to(BASE_DIR)),
                "name": p.name,
                "folder": "emergencia_final",
            })

    # 2) audios largos
    if SRC_RAW_DIR.is_dir():
        for p in sorted(SRC_RAW_DIR.glob("*.wav")):
            items.append({
                "mode": "raw",
                "rel_path": str(p.relative_to(BASE_DIR)),
                "name": p.name,
                "folder": "emergencia_raw",
            })

    return items


# =========================
# RUTAS FLASK
# =========================

@app.route("/")
def index():
    items = get_pending_files()
    if not items:
        return render_template("done.html")

    try:
        idx = int(request.args.get("idx", 0))
    except (TypeError, ValueError):
        idx = 0

    if idx < 0:
        idx = 0
    if idx >= len(items):
        idx = 0

    item = items[idx]
    audio_url = url_for("serve_audio", rel_path=item["rel_path"])

    return render_template(
        "labeler.html",
        audio_url=audio_url,
        item=item,
        idx=idx,
        total=len(items),
        window_sec=WINDOW_SEC,
    )


@app.route("/audio/<path:rel_path>")
def serve_audio(rel_path):
    """Sirve el archivo de audio desde el disco."""
    full_path = (BASE_DIR / rel_path).resolve()

    # Seguridad básica: que esté dentro del proyecto
    if not str(full_path).startswith(str(BASE_DIR)):
        abort(403)
    if not full_path.is_file():
        abort(404)

    return send_file(str(full_path), mimetype="audio/wav")


@app.route("/save", methods=["POST"])
def save():
    rel_path = request.form.get("rel_path")
    mode = request.form.get("mode")          # "clip" o "raw"
    label = request.form.get("label")        # "disparo", "grito", "descartar"
    idx = int(request.form.get("idx", "0"))

    if not rel_path or not mode or not label:
        return redirect(url_for("index", idx=idx))

    src = (BASE_DIR / rel_path).resolve()
    if not str(src).startswith(str(BASE_DIR)) or not src.exists():
        return redirect(url_for("index", idx=idx))

    # Asegurar carpetas destino
    DEST_GUN_DIR.mkdir(exist_ok=True)
    DEST_SCREAM_DIR.mkdir(exist_ok=True)
    DEST_DISCARD_DIR.mkdir(exist_ok=True)

    # Elegir carpeta destino
    if label == "disparo":
        dest_dir = DEST_GUN_DIR
    elif label == "grito":
        dest_dir = DEST_SCREAM_DIR
    else:
        dest_dir = DEST_DISCARD_DIR

    base_name = src.stem
    dest_path = dest_dir / f"{base_name}.wav"
    counter = 1
    while dest_path.exists():
        dest_path = dest_dir / f"{base_name}_{counter}.wav"
        counter += 1

    # ---------------- clip ya recortado ----------------
    if mode == "clip":
        if label == "descartar":
            # mover el clip a descartados
            shutil.move(str(src), dest_path)
        else:
            # copiar el clip al destino (dejamos el original por si acaso)
            shutil.copy2(str(src), dest_path)

        # siguiente índice (la lista se recalcula sola)
        return redirect(url_for("index", idx=idx))

    # ---------------- audio raw (hay que recortar) -------------
    # modo == "raw"
    start_sec_str = request.form.get("start_sec", "0").replace(",", ".")
    try:
        start_sec = float(start_sec_str)
    except ValueError:
        start_sec = 0.0

    if start_sec < 0:
        start_sec = 0.0

    if label == "descartar":
        # si descartas el raw, simplemente lo movemos
        shutil.move(str(src), dest_path)
        return redirect(url_for("index", idx=idx))

    # Cargar audio, recortar ventana de 4 s y guardar
    y, sr = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
    win_len = int(WINDOW_SEC * sr)

    start_sample = int(start_sec * sr)
    if start_sample < 0:
        start_sample = 0
    if start_sample + win_len <= len(y):
        seg = y[start_sample:start_sample + win_len]
    else:
        seg = y[start_sample:]
        if len(seg) < win_len:
            seg = np.pad(seg, (0, win_len - len(seg)), mode="constant")

    sf.write(str(dest_path), seg, sr, subtype="PCM_16")

    # Mover el raw original a descartados para que no vuelva a salir
    raw_trash = DEST_DISCARD_DIR / src.name
    if not raw_trash.exists():
        shutil.move(str(src), raw_trash)
    else:
        src.unlink(missing_ok=True)

    return redirect(url_for("index", idx=idx))


if __name__ == "__main__":
    print("Iniciando etiquetador en http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=True)
