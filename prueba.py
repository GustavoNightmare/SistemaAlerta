import librosa
import numpy as np

SAMPLE_RATE = 16000
DURATION = 2.0  # segundos
N_MELS = 64


def load_audio_fixed(path, sr=SAMPLE_RATE, duration=DURATION):
    # Carga
    y, orig_sr = librosa.load(path, sr=sr)

    # Longitud objetivo
    target_length = int(sr * duration)

    if len(y) > target_length:
        # recortar
        y = y[:target_length]
    else:
        # rellenar con ceros
        pad_width = target_length - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')

    return y


def audio_to_melspec(path):
    y = load_audio_fixed(path)
    # Espectrograma mel
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=256
    )
    # Escala log
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalizar a [0,1]
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm.astype(np.float32)
