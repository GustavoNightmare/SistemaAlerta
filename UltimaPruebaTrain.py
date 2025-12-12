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

# --- GPU memory growth ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU disponible. Usando:", gpus)
    except RuntimeError as e:
        print("Error configurando GPU:", e)
else:
    print("⚠ No se encontró GPU. Entrenando en CPU.")

# =========================
# CONFIGURACIÓN
# =========================

CLASS_GROUPS = {
    "ruido": ["ruido_final"],
    "emergencia": ["disparo_final", "grito_final"],
}

SAMPLE_RATE = 16000
WINDOW_SEC = 4.0
STEP_SEC = 4.0  # para ruido
N_MELS = 64

# Mejoras para gritos
EMERG_OFFSETS_SEC = (-1.0, 0.0, 1.0)  # 3 ventanas alrededor del pico
TRIM_TOP_DB = 25                      # recorta silencios (útil en gritos)

# Augmentación (recomendado activar solo para gritos)
DO_AUGMENT_GRITO = True
AUG_PER_GRITO_WINDOW = 1  # 0/1/2... (si lo subes mucho, se puede “pasar”)

EPOCHS = 200
BATCH_SIZE = 32

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# FUNCIONES DE AUDIO
# =========================


def split_into_windows(y, sr=SAMPLE_RATE):
    win_len = int(WINDOW_SEC * sr)
    step = int(STEP_SEC * sr)

    if len(y) == 0:
        return []

    windows = []
    for start in range(0, len(y), step):
        end = start + win_len
        chunk = y[start:end]
        if len(chunk) < win_len:
            chunk = np.pad(chunk, (0, win_len - len(chunk)), mode="constant")
        windows.append(chunk.astype("float32"))
    return windows


def find_best_center_by_rms(y, sr=SAMPLE_RATE):
    frame_len = int(0.25 * sr)  # 250 ms
    hop = frame_len // 2        # 50% solape
    num_frames = max(1, 1 + (len(y) - frame_len) // hop)

    rms_vals = []
    for i in range(num_frames):
        s = i * hop
        e = min(len(y), s + frame_len)
        frame = y[s:e]
        rms_vals.append(float(np.sqrt(np.mean(frame ** 2)))
                        if len(frame) else 0.0)

    best_idx = int(np.argmax(rms_vals))
    best_center = best_idx * hop + frame_len // 2
    return best_center


def extract_emergency_windows(y, sr=SAMPLE_RATE, offsets_sec=EMERG_OFFSETS_SEC):
    """
    Para disparos/gritos:
      - recorta silencios (trim)
      - halla pico de energía (RMS)
      - extrae varias ventanas alrededor del pico (offsets)
    """
    win_len = int(WINDOW_SEC * sr)

    # Trim de silencios (muy útil en gritos grabados con “aire”)
    if len(y) > 0:
        y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)

    if len(y) <= win_len:
        y_pad = np.pad(y, (0, win_len - len(y)), mode="constant")
        return [y_pad.astype("float32")]

    best_center = find_best_center_by_rms(y, sr)

    windows = []
    for off in offsets_sec:
        center = best_center + int(off * sr)
        start = int(center - win_len // 2)
        start = max(0, min(start, len(y) - win_len))
        window = y[start:start + win_len]
        windows.append(window.astype("float32"))
    return windows


def augment_grito(y, sr=SAMPLE_RATE):
    """
    Augmentación ligera para gritos:
      - ganancia
      - pitch shift pequeño
      - time stretch pequeño
    """
    y = y.copy().astype(np.float32)

    # gain
    gain = np.random.uniform(0.75, 1.25)
    y *= gain
    y = np.clip(y, -1.0, 1.0)

    # pitch shift (±2 semitonos)
    if np.random.rand() < 0.7:
        steps = np.random.uniform(-2.0, 2.0)
        try:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        except Exception:
            pass

    # time stretch (0.9–1.1)
    if np.random.rand() < 0.7:
        rate = np.random.uniform(0.9, 1.1)
        try:
            y = librosa.effects.time_stretch(y, rate=rate)
        except Exception:
            pass

    # asegurar longitud exacta 4s
    target_len = int(sr * WINDOW_SEC)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")

    return y.astype("float32")


def waveform_to_melspec(y, sr=SAMPLE_RATE):
    target_length = int(sr * WINDOW_SEC)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")

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
    S_norm = np.expand_dims(S_norm, axis=-1)  # (64, time, 1)
    return S_norm

# =========================
# CARGAR DATASET (con balanceo interno grito/disparo)
# =========================


def load_dataset():
    label_names = list(CLASS_GROUPS.keys())  # ["ruido","emergencia"]
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    # Guardaremos crudo por clase final
    waves_ruido = []
    meta_ruido = []

    # Emergencia separado por subtipo para balance interno
    emerg_by_type = {"disparo_final": [], "grito_final": []}
    meta_emerg_by_type = {"disparo_final": [], "grito_final": []}

    base_dir = Path(__file__).resolve().parent

    # --- ruido ---
    for folder_name in CLASS_GROUPS["ruido"]:
        folder_path = base_dir / folder_name
        if not folder_path.is_dir():
            print(f"⚠ Carpeta {folder_path} no existe, la salto.")
            continue

        print(f"\n📂 Leyendo 'ruido' de '{folder_name}'")
        for audio_path in sorted(folder_path.rglob("*.wav")):
            try:
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  ❌ Error cargando {audio_path.name}: {e}")
                continue

            windows = split_into_windows(y, sr)
            for win_idx, y_win in enumerate(windows):
                waves_ruido.append(y_win.astype("float32"))
                meta_ruido.append((str(audio_path), win_idx, folder_name))

    # --- emergencia (disparo/grito) ---
    print("\n📂 Leyendo 'emergencia' (disparo + grito)")
    for folder_name in CLASS_GROUPS["emergencia"]:
        folder_path = base_dir / folder_name
        if not folder_path.is_dir():
            print(f"⚠ Carpeta {folder_path} no existe, la salto.")
            continue

        for audio_path in sorted(folder_path.rglob("*.wav")):
            try:
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  ❌ Error cargando {audio_path.name}: {e}")
                continue

            windows = extract_emergency_windows(y, sr)
            for win_idx, y_win in enumerate(windows):
                emerg_by_type[folder_name].append(y_win.astype("float32"))
                meta_emerg_by_type[folder_name].append(
                    (str(audio_path), win_idx, folder_name))

    # === Balanceo interno grito vs disparo (oversampling del minoritario) ===
    c_dis = len(emerg_by_type["disparo_final"])
    c_gri = len(emerg_by_type["grito_final"])
    print("\nVentanas emergencia por subtipo (antes de balanceo interno):")
    print(f"  disparo_final: {c_dis}")
    print(f"  grito_final  : {c_gri}")

    target = max(c_dis, c_gri) if max(c_dis, c_gri) > 0 else 0
    print(f"Balanceo interno emergencia -> objetivo por subtipo: {target}")

    emerg_waves = []
    emerg_meta = []

    for t in ["disparo_final", "grito_final"]:
        waves = emerg_by_type[t]
        meta = meta_emerg_by_type[t]
        if len(waves) == 0:
            continue

        if len(waves) < target:
            # oversample con reemplazo
            idxs = np.random.choice(len(waves), size=target, replace=True)
        else:
            idxs = np.random.choice(len(waves), size=target, replace=False)

        for k, idx in enumerate(idxs):
            y_win = waves[idx]
            filepath, win_idx, orig_label = meta[idx]
            emerg_waves.append(y_win)
            emerg_meta.append((filepath, win_idx, orig_label))

            # augment SOLO para gritos (opcional)
            if orig_label == "grito_final" and DO_AUGMENT_GRITO and AUG_PER_GRITO_WINDOW > 0:
                for _ in range(AUG_PER_GRITO_WINDOW):
                    y_aug = augment_grito(y_win, SAMPLE_RATE)
                    emerg_waves.append(y_aug)
                    emerg_meta.append((filepath, win_idx, orig_label + "_aug"))

    # === Balanceo final ruido vs emergencia (downsample mayoritario) ===
    n_ruido = len(waves_ruido)
    n_emerg = len(emerg_waves)
    print("\nVentanas por clase final (antes de balanceo final):")
    print(f"  ruido      : {n_ruido}")
    print(f"  emergencia : {n_emerg}")

    min_count = min(n_ruido, n_emerg) if min(n_ruido, n_emerg) > 0 else 0
    print(
        f"\nBalanceo final ruido vs emergencia: usando {min_count} ventanas por clase.")

    X_list, y_list, meta_all = [], [], []

    # ruido -> sample
    ruido_idxs = np.random.choice(n_ruido, size=min_count, replace=False)
    for idx in ruido_idxs:
        spec = waveform_to_melspec(waves_ruido[idx])
        X_list.append(spec)
        y_list.append(label_to_idx["ruido"])
        meta_all.append(meta_ruido[idx])

    # emergencia -> sample
    emerg_idxs = np.random.choice(n_emerg, size=min_count, replace=False)
    for idx in emerg_idxs:
        spec = waveform_to_melspec(emerg_waves[idx])
        X_list.append(spec)
        y_list.append(label_to_idx["emergencia"])
        meta_all.append(emerg_meta[idx])

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype=np.int64)

    print("\nTamaño total del dataset (después de balanceos):")
    print("  X:", X.shape)
    print("  y:", y.shape)

    return X, y, meta_all, label_names

# =========================
# MODELOS
# =========================


def build_cnn_big(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.6)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# =========================
# SPLITS + WEIGHTS + CSV ERRORES
# =========================


def make_splits(X, y, meta):
    indices = np.arange(len(X))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y, test_size=0.3, random_state=SEED, stratify=y
    )

    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=1/3, random_state=SEED, stratify=y_temp
    )

    def subset(arr, idxs):
        return arr[idxs]

    X_train = subset(X, idx_train)
    X_val = subset(X, idx_val)
    X_test = subset(X, idx_test)

    print("\nTamaños finales:")
    print("  Train:", X_train.shape)
    print("  Val  :", X_val.shape)
    print("  Test :", X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, idx_train, idx_val, idx_test


def compute_class_weights(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes)
    total = len(y_train)
    class_weights = {}
    for i in range(num_classes):
        class_weights[i] = float(
            total) / (num_classes * counts[i]) if counts[i] else 1.0
    print("\nClass weights:", class_weights)
    return class_weights


def save_misclassified_csv(filename, idx_split, y_true, y_pred, y_probs, meta, label_names):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filepath", "window_idx", "orig_label",
            "true_label", "pred_label", "prob_true", "prob_pred"
        ])

        for i, global_idx in enumerate(idx_split):
            if y_true[i] == y_pred[i]:
                continue
            filepath, win_idx, orig_label = meta[global_idx]
            probs = y_probs[i]
            true_label = label_names[y_true[i]]
            pred_label = label_names[y_pred[i]]
            prob_true = float(probs[y_true[i]])
            prob_pred = float(probs[y_pred[i]])

            writer.writerow([
                filepath, win_idx, orig_label,
                true_label, pred_label,
                f"{prob_true:.4f}", f"{prob_pred:.4f}"
            ])

    print(f"CSV de errores guardado en: {filename}")

# =========================
# MAIN
# =========================


def main():
    X, y, meta, label_names = load_dataset()

    X_train, y_train, X_val, y_val, X_test, y_test, idx_train, idx_val, idx_test = make_splits(
        X, y, meta)
    num_classes = len(label_names)
    input_shape = X_train.shape[1:]

    model = build_cnn_big(input_shape, num_classes)
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
            "best_cnn_big_v2.h5",
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
        callbacks=callbacks,
    )

    print("\n===== EVALUACIÓN EN TEST (cnn_big_v2) =====")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}")

    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    print("\nClassification report (TEST):")
    print(classification_report(y_test, y_pred, target_names=label_names))

    print("Matriz de confusión (TEST) (filas = real, columnas = predicho):")
    print(confusion_matrix(y_test, y_pred))

    save_misclassified_csv(
        "errors_cnn_big_v2_test.csv",
        idx_test, y_test, y_pred, y_probs,
        meta, label_names
    )

    model.save("modelo_emerg_vs_ruido_v2.h5")
    print("\n✅ Modelo guardado como modelo_emerg_vs_ruido_v2.h5")


if __name__ == "__main__":
    main()
