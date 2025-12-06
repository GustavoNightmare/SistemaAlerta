import os
import csv
from pathlib import Path
import random

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# =========================
# CONFIGURACIÓN
# =========================

# Carpetas de datos (ajusta nombres si hace falta)
CLASS_FOLDERS = {
    "ruido": "ruido_final",
    "emergencia": "emergencia_final",
}

SAMPLE_RATE = 16000       # Hz
WINDOW_SEC = 4.0          # tamaño de ventana
STEP_SEC = 4.0            # paso entre ventanas (sin solape)
N_MELS = 64

DO_AUGMENT = True         # activar / desactivar data augmentation
# cuántas versiones aumentadas por ventana (0 = ninguna)
AUG_PER_WINDOW = 1

EPOCHS = 60
BATCH_SIZE = 32

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# =========================
# FUNCIONES DE AUDIO
# =========================

def split_into_windows(y, sr=SAMPLE_RATE):
    """
    Divide un audio en ventanas de WINDOW_SEC con paso STEP_SEC.
    Si la última queda más corta, la rellena con ceros.
    """
    win_len = int(WINDOW_SEC * sr)
    step = int(STEP_SEC * sr)

    if len(y) == 0:
        return []

    windows = []
    for start in range(0, len(y), step):
        end = start + win_len
        chunk = y[start:end]
        if len(chunk) < win_len:
            pad_width = win_len - len(chunk)
            chunk = np.pad(chunk, (0, pad_width), mode="constant")
        windows.append(chunk.astype("float32"))
    return windows


def augment_wave(y, sr=SAMPLE_RATE):
    """
    Data augmentation sencillo:
      - Pequeña variación de ganancia.
      - Ligero desplazamiento temporal.
      - Ruido blanco suave.
    """
    y = y.copy()

    # Ganancia [0.7, 1.3]
    gain = np.random.uniform(0.7, 1.3)
    y *= gain

    # Pequeño shift temporal ±0.25s
    max_shift = int(0.25 * sr)
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        y = np.pad(y, (shift, 0), mode="constant")[: len(y)]
    elif shift < 0:
        y = np.pad(y, (0, -shift), mode="constant")[-len(y):]

    # Ruido blanco suave
    noise_amp = 0.003 * np.random.uniform(0.5, 1.5) * np.max(np.abs(y) + 1e-6)
    noise = noise_amp * np.random.normal(size=y.shape)
    y = y + noise

    # Limitar a [-1, 1] por seguridad
    y = np.clip(y, -1.0, 1.0)
    return y.astype("float32")


def waveform_to_melspec(y, sr=SAMPLE_RATE):
    """
    Convierte una señal 1D a mel-espectrograma normalizado [0,1].
    """
    target_length = int(sr * WINDOW_SEC)
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
    S_norm = np.expand_dims(S_norm, axis=-1)  # (n_mels, time, 1)
    return S_norm


# =========================
# CARGAR DATASET + SEGMENTAR
# =========================

def load_dataset():
    """
    Lee todas las carpetas de CLASS_FOLDERS, segmenta en ventanas de 4 s,
    aplica data augmentation y devuelve:
      X (espectrogramas), y (labels), meta (info por ventana), label_names
    meta[i] = (ruta_archivo, indice_ventana, nombre_clase_original)
    """
    X_list = []
    y_list = []
    meta = []

    label_names = list(CLASS_FOLDERS.keys())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    base_dir = Path(__file__).resolve().parent

    for label_name, folder_name in CLASS_FOLDERS.items():
        folder_path = base_dir / folder_name
        if not folder_path.is_dir():
            print(f"⚠ Carpeta {folder_path} no existe, la salto.")
            continue

        print(f"\n📂 Leyendo clase '{label_name}' de {folder_path}")
        file_paths = sorted(folder_path.rglob("*.wav"))

        if not file_paths:
            print("  (No encontré .wav en esta carpeta)")
            continue

        total_windows = 0

        for audio_path in file_paths:
            try:
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  ❌ Error cargando {audio_path.name}: {e}")
                continue

            windows = split_into_windows(y, sr)
            if not windows:
                continue

            for win_idx, y_win in enumerate(windows):
                # Espectrograma original
                spec = waveform_to_melspec(y_win)
                X_list.append(spec)
                y_list.append(label_to_idx[label_name])
                meta.append((str(audio_path), win_idx, label_name))

                # Aumentación
                if DO_AUGMENT and AUG_PER_WINDOW > 0:
                    for _ in range(AUG_PER_WINDOW):
                        y_aug = augment_wave(y_win, sr)
                        spec_aug = waveform_to_melspec(y_aug)
                        X_list.append(spec_aug)
                        y_list.append(label_to_idx[label_name])
                        meta.append(
                            (str(audio_path), win_idx, label_name + "_aug"))

                total_windows += 1

        print(f"  -> Ventanas generadas (sin contar augment): {total_windows}")

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype=np.int64)

    print("\nTamaño total del dataset (después de segmentar y augment):")
    print("  X:", X.shape)
    print("  y:", y.shape)

    return X, y, meta, label_names


# =========================
# MODELOS
# =========================

def build_cnn_small(input_shape, num_classes):
    """
    Modelo similar al que tenías, pero con dropout y L2.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(32, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_big(input_shape, num_classes):
    """
    Modelo un poco más grande para probar si mejora:
    más filtros y más dropout.
    """
    inputs = layers.Input(shape=input_shape)

    x = inputs
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================
# UTILIDADES: SPLIT + CLASS WEIGHTS
# =========================

def make_splits(X, y, meta):
    indices = np.arange(len(X))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=0.3,
        random_state=SEED,
        stratify=y,
    )

    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp,
        y_temp,
        test_size=1 / 3,
        random_state=SEED,
        stratify=y_temp,
    )

    def subset(arr, idxs):
        return arr[idxs]

    X_train = subset(X, idx_train)
    X_val = subset(X, idx_val)
    X_test = subset(X, idx_test)

    meta_train = [meta[i] for i in idx_train]
    meta_val = [meta[i] for i in idx_val]
    meta_test = [meta[i] for i in idx_test]

    print("\nTamaños finales:")
    print("  Train:", X_train.shape)
    print("  Val  :", X_val.shape)
    print("  Test :", X_test.shape)

    return (
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        X_test, y_test, meta_test,
        idx_train, idx_val, idx_test,
    )


def compute_class_weights(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes)
    total = len(y_train)
    class_weights = {}
    for i in range(num_classes):
        if counts[i] == 0:
            class_weights[i] = 1.0
        else:
            class_weights[i] = float(total) / (num_classes * counts[i])
    print("\nClass weights:", class_weights)
    return class_weights


def save_misclassified_csv(
        filename, idx_split, y_true, y_pred, y_probs, meta, label_names):
    """
    Guarda un CSV con las ventanas mal clasificadas:
      filepath, window_idx, true_label, pred_label, prob_true, prob_pred
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filepath",
            "window_idx",
            "true_label",
            "pred_label",
            "prob_true",
            "prob_pred",
        ])

        for i, global_idx in enumerate(idx_split):
            if y_true[i] == y_pred[i]:
                continue

            filepath, win_idx, _ = meta[global_idx]
            probs = y_probs[i]
            true_label = label_names[y_true[i]]
            pred_label = label_names[y_pred[i]]
            prob_true = float(probs[y_true[i]])
            prob_pred = float(probs[y_pred[i]])

            writer.writerow([
                filepath,
                win_idx,
                true_label,
                pred_label,
                f"{prob_true:.4f}",
                f"{prob_pred:.4f}",
            ])

    print(f"CSV de errores guardado en: {filename}")


# =========================
# MAIN: ENTRENAR Y EVALUAR
# =========================

def run_experiment(name, build_fn,
                   X_train, y_train, meta_train,
                   X_val, y_val,
                   X_test, y_test, meta, label_names,
                   idx_train, idx_val, idx_test):

    num_classes = len(label_names)
    input_shape = X_train.shape[1:]

    model = build_fn(input_shape, num_classes)
    model.summary()

    class_weights = compute_class_weights(y_train, num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"best_{name}.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        verbose=2,
    )

    print(f"\n===== EVALUACIÓN EN TEST ({name}) =====")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}")

    # Predicciones en test
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    print("\nClassification report (TEST):")
    print(classification_report(y_test, y_pred, target_names=label_names))

    print("Matriz de confusión (TEST) (filas = real, columnas = predicho):")
    print(confusion_matrix(y_test, y_pred))

    # CSV de errores en test
    save_misclassified_csv(
        f"errors_{name}_test.csv",
        idx_test, y_test, y_pred, y_probs,
        meta, label_names,
    )

    # (Opcional) también podrías sacar errores de train/val igual que arriba.

    # Guardar modelo final
    model.save(f"modelo_{name}.h5")
    print(f"Modelo guardado como modelo_{name}.h5")

    return test_acc, test_loss


def main():
    X, y, meta, label_names = load_dataset()

    (
        X_train, y_train, meta_train,
        X_val, y_val, meta_val,
        X_test, y_test, meta_test,
        idx_train, idx_val, idx_test,
    ) = make_splits(X, y, meta)

    experiments = [
        ("cnn_small", build_cnn_small),
        ("cnn_big", build_cnn_big),
    ]

    results = []
    for name, fn in experiments:
        print("\n\n########################################")
        print(f"###  Entrenando experimento: {name}")
        print("########################################\n")

        acc, loss = run_experiment(
            name, fn,
            X_train, y_train, meta_train,
            X_val, y_val,
            X_test, y_test, meta,
            label_names,
            idx_train, idx_val, idx_test,
        )
        results.append((name, acc, loss))

    print("\n===== RESUMEN DE EXPERIMENTOS =====")
    for name, acc, loss in results:
        print(f"{name}: acc_test={acc:.4f}, loss_test={loss:.4f}")


if __name__ == "__main__":
    main()
