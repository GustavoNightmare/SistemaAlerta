import os
from pathlib import Path

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# CONFIGURACIÓN
# =========================

# Carpetas de datos
CLASS_FOLDERS = {
    "grito": "grito_final",
    "disparos": "disparos_final",
    "ruido": "ruido_final",
}

SAMPLE_RATE = 16000     # Hz
DURATION = 2.0          # segundos
N_MELS = 64

EPOCHS = 40
BATCH_SIZE = 32

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


def file_to_melspec(path):
    """Convierte un archivo a mel-espectrograma normalizado [0,1]."""
    y = load_audio_fixed(path)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
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


# =========================
# CARGAR DATASET
# =========================

def load_dataset():
    X = []
    y = []

    label_names = list(CLASS_FOLDERS.keys())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    base_dir = Path(__file__).resolve().parent

    for label_name, folder_name in CLASS_FOLDERS.items():
        folder_path = base_dir / folder_name
        if not folder_path.is_dir():
            print(f"⚠ Carpeta {folder_path} no existe, la salto.")
            continue

        print(f"\n📂 Leyendo clase '{label_name}' de {folder_path}")
        file_paths = sorted(folder_path.glob("*.wav"))

        if not file_paths:
            print("  (No encontré .wav en esta carpeta)")
            continue

        for audio_path in file_paths:
            try:
                spec = file_to_melspec(audio_path)
                X.append(spec)
                y.append(label_to_idx[label_name])
            except Exception as e:
                print(f"  ❌ Error con {audio_path.name}: {e}")

        print(f"  -> {len(file_paths)} archivos procesados para '{label_name}'")

    X = np.array(X)
    y = np.array(y, dtype=np.int64)

    print("\nTamaño total del dataset:")
    print("  X:", X.shape)
    print("  y:", y.shape)

    return X, y, label_names


# =========================
# MODELO CNN
# =========================

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================
# MAIN: 70/20/10 SPLIT
# =========================

def main():
    # 1) Cargar dataset completo
    X, y, label_names = load_dataset()

    # ------- 70% train, 30% temp -------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,            # 30% queda para val+test
        random_state=42,
        stratify=y
    )

    # ------- De ese 30%, 2/3 val y 1/3 test -------
    # 30% * 2/3 = 20%  -> validación
    # 30% * 1/3 = 10%  -> test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1/3,
        random_state=42,
        stratify=y_temp
    )

    print("\nTamaños finales:")
    print("  Train:", X_train.shape)
    print("  Val  :", X_val.shape)
    print("  Test :", X_test.shape)

    # 2) Crear modelo
    input_shape = X_train.shape[1:]
    num_classes = len(label_names)
    model = build_model(input_shape, num_classes)
    model.summary()

    # 3) Entrenar usando validación
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # 4) Evaluar en TEST (10% que el modelo nunca vio)
    print("\n===== EVALUACIÓN EN TEST =====")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}")

    # Predicciones para métricas detalladas
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(confusion_matrix(y_test, y_pred))

    # 5) Guardar modelo y nombres de clases
    model.save("modelo_sonidos.h5")
    np.save("label_names.npy", np.array(label_names))

    print("\n✅ Entrenamiento terminado.")
    print("Modelo guardado como  modelo_sonidos.h5")
    print("Clases guardadas en  label_names.npy")


if __name__ == "__main__":
    main()
