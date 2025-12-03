from pathlib import Path
import librosa
import shutil

FOLDERS = ["grito_final", "disparos_final", "ruido_final"]
base = Path(__file__).resolve().parent
err_dir = base / "errores_audio"
err_dir.mkdir(exist_ok=True)

for folder_name in FOLDERS:
    folder = base / folder_name
    if not folder.is_dir():
        continue

    print(f"\nRevisando {folder_name}")
    for path in folder.glob("*.wav"):
        try:
            librosa.load(path, sr=None)  # solo para ver si se puede leer
        except Exception as e:
            print(f"  Moviendo {path.name} por error: {e}")
            shutil.move(path, err_dir / path.name)

print("\nListo. Los archivos problemáticos están en 'errores_audio/'.")
