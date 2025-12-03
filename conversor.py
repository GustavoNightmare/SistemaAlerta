from pathlib import Path
from pydub import AudioSegment

# Parámetros de salida
TARGET_SR = 16000       # frecuencia de muestreo
TARGET_CHANNELS = 1     # mono

# Carpetas de clases que tienes
CLASSES = ["grito", "explosion", "disparos", "ruido"]

# Extensiones de audio que intentaremos convertir
SUPPORTED_EXTS = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".aac"]

BASE_DIR = Path(__file__).resolve().parent


def cargar_y_convertir(path: Path) -> AudioSegment:
    """Carga el archivo y lo normaliza a 16 kHz mono."""
    ext = path.suffix.lower().lstrip(".")  # 'mp3', 'wav', etc.
    if ext == "":
        raise ValueError("Archivo sin extensión reconocible")

    # Le indicamos a pydub el formato explícitamente
    audio = AudioSegment.from_file(path, format=ext)

    # Normalizar a 16kHz mono
    audio = audio.set_frame_rate(TARGET_SR).set_channels(TARGET_CHANNELS)
    return audio


def main():
    for clase in CLASSES:
        # Posibles carpetas de entrada para esa clase
        in_dirs = [
            BASE_DIR / clase,
            BASE_DIR / f"{clase}_wav"
        ]

        # Carpeta de salida final
        out_dir = BASE_DIR / f"{clase}_final"
        out_dir.mkdir(exist_ok=True)

        print(f"\n===== CLASE: {clase} =====")
        print(f"Carpetas de entrada:")
        for d in in_dirs:
            print(f"  - {d} {'(existe)' if d.is_dir() else '(no existe)'}")
        print(f"Carpeta de salida: {out_dir}")

        # Reunir todos los archivos de audio válidos
        files = []
        for d in in_dirs:
            if not d.is_dir():
                continue
            for path in d.iterdir():
                if not path.is_file():
                    continue
                if path.suffix.lower() in SUPPORTED_EXTS:
                    files.append(path)

        if not files:
            print("  ⚠ No encontré archivos de audio para esta clase.")
            continue

        # Orden para que los nombres 1.wav, 2.wav... sean consistentes
        files = sorted(files, key=lambda p: p.name)

        idx = 1
        ok = 0
        fail = 0

        for path in files:
            try:
                audio = cargar_y_convertir(path)
                out_name = f"{idx}.wav"
                out_path = out_dir / out_name
                audio.export(out_path, format="wav")
                print(f"  ✅ {path.name}  →  {out_name}")
                idx += 1
                ok += 1
            except Exception as e:
                print(f"  ❌ ERROR con {path.name}: {e}")
                fail += 1

        print(f"\nResumen para {clase}:")
        print(f"  Convertidos OK : {ok}")
        print(f"  Errores        : {fail}")

    print("\nTerminado. Usa las carpetas *_final para entrenar el modelo. 💾")


if __name__ == "__main__":
    main()
