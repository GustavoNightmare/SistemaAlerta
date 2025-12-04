import numpy as np
import librosa
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONFIGURACIÓN
# =========================

# Carpetas que se consideran "emergencia" (label 1)
EMERGENCY_FOLDERS = [
    "grito_final",
    "disparos_final",
    "explosion_final",   # opcional, se ignora si no existe
]

# Carpetas que se consideran "ruido normal" (label 0)
NOISE_FOLDERS = [
    "ruido_final",
    # aquí puedes añadir más, ej: "ambiente_calle_final"
]

SAMPLE_RATE = 16000     # Hz
DURATION = 4.0          # segundos
N_MELS = 64

EPOCHS = 60             # con EarlyStopping, no siempre llegará a 60
BATCH_SIZE = 32

# Nombres de clases (orden fijo)
LABEL_NAMES = ["ruido", "emergencia"]  # 0, 1

# =========================
# FUNCIONES DE AUDIO
# =========================


def load_audio_fixed(path, sr=SAMPLE_RATE, duration=DURATION):
    """Carga un audio, lo re-muestrea y lo deja con duración fija."""
    y, orig_sr = librosa.load(path, sr=sr)
    target_length = int(sr * duration)

    if len(y) > target_length:
        y = y[:target_length]
    else:
        pad_width = target_length - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")

    return y


def waveform_to_melspec(y):
    """Convierte señal 1D a mel-espectrograma normalizado [0,1]."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=256,
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-8)

    S_norm = S_norm.astype("float32")
    S_norm = np.expand_dims(S_norm, axis=-1)  # (n_mels, time, 1)
    return S_norm


def file_to_melspec(path):
    y = load_audio_fixed(path)
    return waveform_to_melspec(y)


# =========================
# SPEC-AUGMENT EN EL ESPECTROGRAMA
# =========================

def spec_augment(spec,
                 max_freq_mask=8,
                 max_time_mask=16,
                 num_masks=2):
    """
    Aplica máscaras aleatorias en frecuencia y tiempo (SpecAugment simple).
    spec: (n_mels, time, 1)
    """
    x = spec.copy()
    n_mels, n_frames, _ = x.shape

    for _ in range(num_masks):
        # Máscara en frecuencia
        f = np.random.randint(0, max_freq_mask + 1)
        if f > 0:
            f0 = np.random.randint(0, max(1, n_mels - f + 1))
            x[f0:f0+f, :, :] = 0.0

        # Máscara en tiempo
        t = np.random.randint(0, max_time_mask + 1)
        if t > 0:
            t0 = np.random.randint(0, max(1, n_frames - t + 1))
            x[:, t0:t0+t, :] = 0.0

    return x.astype("float32")


# =========================
# CARGAR DATASET BINARIO
# =========================

def load_dataset_binary():
    X = []
    y = []

    base_dir = Path(__file__).resolve().parent

    def load_from_folders(folder_names, label_idx):
        count = 0
        for folder_name in folder_names:
            folder_path = base_dir / folder_name
            if not folder_path.is_dir():
                print(f"⚠ Carpeta {folder_path} no existe, la salto.")
                continue

            print(
                f"\n📂 Leyendo '{folder_name}' como label {label_idx} ({LABEL_NAMES[label_idx]})")
            file_paths = sorted(folder_path.glob("*.wav"))
            if not file_paths:
                print("  (No encontré .wav en esta carpeta)")
                continue

            for audio_path in file_paths:
                try:
                    spec = file_to_melspec(audio_path)
                    X.append(spec)
                    y.append(label_idx)
                    count += 1
                except Exception as e:
                    print(f"  ❌ Error con {audio_path.name}: {e}")

            print(
                f"  -> {len(file_paths)} archivos procesados en '{folder_name}'")
        return count

    n_noise = load_from_folders(NOISE_FOLDERS, label_idx=0)
    n_emerg = load_from_folders(EMERGENCY_FOLDERS, label_idx=1)

    X = np.array(X)
    y = np.array(y, dtype=np.int64)

    print("\nTamaño total del dataset binario:")
    print("  X:", X.shape)
    print("  y:", y.shape)
    print(f"  ruido      (0): {n_noise}")
    print(f"  emergencia (1): {n_emerg}")

    return X, y


# =========================
# MODELO CNN BINARIO (2 CLASES)
# =========================

def build_model(input_shape, num_classes=2):
    reg = regularizers.l2(1e-4)  # L2 para evitar sobreajuste

    model = models.Sequential([
        layers.Conv2D(16, (3, 3), padding='same',
                      activation='relu',
                      kernel_regularizer=reg,
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(32, (3, 3), padding='same',
                      activation='relu',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), padding='same',
                      activation='relu',
                      kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================
# GENERADOR CON AUGMENTACIÓN
# =========================

def make_train_generator(X, y, batch_size):
    n = len(X)
    indices = np.arange(n)
    while True:
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            batch_x = X[batch_idx]
            batch_y = y[batch_idx]

            # aplicar SpecAugment a cada espectrograma
            batch_x_aug = np.stack([spec_augment(x) for x in batch_x], axis=0)
            yield batch_x_aug, batch_y


# =========================
# MAIN: 70/20/10 SPLIT
# =========================

def main():
    # 1) Cargar dataset binario
    X, y = load_dataset_binary()

    # ------- 70% train, 30% temp -------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # ------- De ese 30%, 2/3 val y 1/3 test -------
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1/3,   # 10% test
        random_state=42,
        stratify=y_temp
    )

    print("\nTamaños finales:")
    print("  Train:", X_train.shape)
    print("  Val  :", X_val.shape)
    print("  Test :", X_test.shape)

    # 2) Crear modelo
    input_shape = X_train.shape[1:]
    model = build_model(input_shape, num_classes=2)
    model.summary()

    # 3) EarlyStopping para evitar sobreentrenar
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    )

    # 4) Generador con SpecAugment
    train_gen = make_train_generator(X_train, y_train, BATCH_SIZE)
    steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    # 5) Evaluar en TEST (10%)
    print("\n===== EVALUACIÓN EN TEST =====")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}")

    # Predicciones para métricas detalladas
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(confusion_matrix(y_test, y_pred))

    # 6) Guardar modelo y nombres de clases
    model.save("modelo_sonidos.h5")
    np.save("label_names.npy", np.array(LABEL_NAMES))

    print("\n✅ Entrenamiento binario terminado.")
    print("Modelo guardado como  modelo_sonidos.h5")
    print("Clases guardadas en  label_names.npy")
    print("Clases:", LABEL_NAMES)


if __name__ == "__main__":
    main()
