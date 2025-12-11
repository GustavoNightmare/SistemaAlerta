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

# --- Comprobar GPU y activar memory growth (DirectML o CUDA) ---
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

# OJO: aquí usamos 3 carpetas, pero solo 2 etiquetas finales:
#   - ruido       -> ruido_final
#   - emergencia  -> disparos_final + gritos_final
CLASS_GROUPS = {
    "ruido": ["ruido_final"],
    "emergencia": ["disparo_final", "grito_final"],
}

SAMPLE_RATE = 16000       # Hz
WINDOW_SEC = 4.0          # tamaño de ventana (segundos)
STEP_SEC = 4.0            # paso entre ventanas para ruido (sin solape)
N_MELS = 64

DO_AUGMENT = False        # ❌ sin augmentación de datos
AUG_PER_WINDOW = 0        # por si acaso

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
    """
    Divide un audio en ventanas de WINDOW_SEC con paso STEP_SEC.
    Si la última queda más corta, la rellena con ceros.
    Usado para la clase 'ruido'.
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


def extract_emergency_window(y, sr=SAMPLE_RATE):
    """
    Para audios de 'emergencia' (disparos / gritos):
      - Si el audio < 4 s: se rellena con ceros.
      - Si es más largo: se busca la zona de mayor energía
        y se recorta UNA sola ventana de 4 s alrededor de ese pico.
    """
    win_len = int(WINDOW_SEC * sr)

    if len(y) <= win_len:
        pad_width = win_len - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
        return y.astype("float32")

    frame_len = int(0.25 * sr)  # 250 ms
    hop = frame_len // 2        # 50% solape

    num_frames = max(1, 1 + (len(y) - frame_len) // hop)
    rms_vals = []

    for i in range(num_frames):
        start = i * hop
        end = start + frame_len
        if end > len(y):
            end = len(y)
        frame = y[start:end]
        if len(frame) == 0:
            rms_vals.append(0.0)
        else:
            rms_vals.append(float(np.sqrt(np.mean(frame ** 2))))

    best_idx = int(np.argmax(rms_vals))
    best_center = best_idx * hop + frame_len // 2

    start_win = int(best_center - win_len // 2)
    if start_win < 0:
        start_win = 0
    if start_win + win_len > len(y):
        start_win = len(y) - win_len

    window = y[start_win:start_win + win_len]
    return window.astype("float32")


def augment_gain_only(y, sr=SAMPLE_RATE):
    """
    Data augmentation SOLO de volumen (por si algún día lo vuelves a activar).
    """
    y = y.copy()
    gain = np.random.uniform(0.75, 1.25)  # subir/bajar volumen
    y *= gain
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
# CARGAR DATASET + BALANCEAR
# =========================


def load_dataset():
    """
    Lee las carpetas de CLASS_GROUPS y devuelve:
      X (espectrogramas), y (labels), meta (info por ventana), label_names

    - 'ruido': se trocea en muchas ventanas de 4 s.
    - 'emergencia': se extrae UNA ventana de 4 s por archivo (máxima energía).
    - disparos_final y gritos_final se agrupan como 'emergencia'.
    - Se balancea por clases finales (ruido / emergencia).
    """
    label_names = list(CLASS_GROUPS.keys())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    # Guardamos primero las ondas crudas por clase final (ruido/emergencia)
    waves_by_class = {i: [] for i in range(len(label_names))}
    meta_by_class = {i: [] for i in range(len(label_names))}

    base_dir = Path(__file__).resolve().parent

    for label_name, folder_list in CLASS_GROUPS.items():
        label_idx = label_to_idx[label_name]
        total_windows_raw = 0

        for folder_name in folder_list:
            folder_path = base_dir / folder_name

            if not folder_path.is_dir():
                print(f"⚠ Carpeta {folder_path} no existe, la salto.")
                continue

            print(
                f"\n📂 Leyendo clase lógica '{label_name}' de carpeta '{folder_name}'")
            file_paths = sorted(folder_path.rglob("*.wav"))

            if not file_paths:
                print("  (No encontré .wav en esta carpeta)")
                continue

            is_emergency = (label_name == "emergencia")

            for audio_path in file_paths:
                try:
                    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                except Exception as e:
                    print(f"  ❌ Error cargando {audio_path.name}: {e}")
                    continue

                if is_emergency:
                    # 1 ventana por archivo (zona de máxima energía)
                    windows = [extract_emergency_window(y, sr)]
                else:
                    # ruido: muchas ventanas
                    windows = split_into_windows(y, sr)

                if not windows:
                    continue

                for win_idx, y_win in enumerate(windows):
                    waves_by_class[label_idx].append(y_win.astype("float32"))
                    # En meta guardamos el tipo específico por si quieres ver
                    # si un error venía de grito o de disparo
                    orig_label = folder_name
                    meta_by_class[label_idx].append(
                        (str(audio_path), win_idx, orig_label)
                    )
                    total_windows_raw += 1

        print(
            f"  -> Ventanas crudas acumuladas para '{label_name}': {total_windows_raw}")

    # --- Balanceo de clases finales (downsampling de la mayoría) ---
    counts = {idx: len(waves_by_class[idx]) for idx in waves_by_class}
    print("\nVentanas por clase final (antes de balancear):")
    for name, idx in label_to_idx.items():
        print(f"  {name}: {counts.get(idx, 0)}")

    min_count = min(counts.values())
    print(
        f"\nBalanceando dataset: usando {min_count} ventanas por clase final.")

    X_list = []
    y_list = []
    meta = []

    for label_name in label_names:
        label_idx = label_to_idx[label_name]
        n = len(waves_by_class[label_idx])
        if n == 0:
            continue

        idxs = list(range(n))
        random.shuffle(idxs)
        keep_idxs = idxs[:min_count]

        print(
            f"  Clase '{label_name}': {n} -> {len(keep_idxs)} (tras balanceo)")

        for idx_wave in keep_idxs:
            y_win = waves_by_class[label_idx][idx_wave]
            filepath, win_idx, orig_label = meta_by_class[label_idx][idx_wave]

            # Original
            spec = waveform_to_melspec(y_win)
            X_list.append(spec)
            y_list.append(label_idx)
            meta.append((filepath, win_idx, orig_label))

            # Augmentation por volumen (DESACTIVADO por defecto)
            if DO_AUGMENT and AUG_PER_WINDOW > 0:
                for _ in range(AUG_PER_WINDOW):
                    y_aug = augment_gain_only(y_win, SAMPLE_RATE)
                    spec_aug = waveform_to_melspec(y_aug)
                    X_list.append(spec_aug)
                    y_list.append(label_idx)
                    meta.append(
                        (filepath, win_idx, orig_label + "_aug")
                    )

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype=np.int64)

    print("\nTamaño total del dataset (después de balancear y augmentar):")
    print("  X:", X.shape)
    print("  y:", y.shape)

    return X, y, meta, label_names

# =========================
# MODELOS
# =========================


def build_cnn_small(input_shape, num_classes):
    """
    CNN pequeña con Dropout + L2 para evitar sobreajuste.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(32, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Conv2D(64, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",  # mejor que MSE para clasificación
        metrics=["accuracy"],
    )
    return model


def build_cnn_big(input_shape, num_classes):
    """
    CNN más grande: más filtros + Dropout fuerte.
    """
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


def build_cnn_medium(input_shape, num_classes):
    """
    Un modelo intermedio con kernels algo más grandes.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(24, (5, 5), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(48, (5, 5), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(96, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(96, activation="relu",
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
    """
    70% train, 20% val, 10% test (aprox) usando dos splits estratificados.
    """
    indices = np.arange(len(X))

    # 70% train, 30% temp
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices,
        y,
        test_size=0.3,
        random_state=SEED,
        stratify=y,
    )

    # De ese 30%: 2/3 val (20%), 1/3 test (10%)
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
        callbacks=callbacks,
    )

    print(f"\n===== EVALUACIÓN EN TEST ({name}) =====")
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
        f"errors_{name}_test.csv",
        idx_test, y_test, y_pred, y_probs,
        meta, label_names,
    )

    # Guardar modelo de este experimento
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
        ("cnn_medium", build_cnn_medium),
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
    best_name = None
    best_acc = -1.0
    for name, acc, loss in results:
        print(f"{name}: acc_test={acc:.4f}, loss_test={loss:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_name = name

    # Copiamos el modelo ganador a un nombre fijo
    if best_name is not None:
        print(f"\n🏆 Mejor modelo: {best_name} (acc_test={best_acc:.4f})")
        # simplemente volvemos a cargarlo y lo guardamos con un nombre estándar
        best_model = tf.keras.models.load_model(f"modelo_{best_name}.h5")
        best_model.save("modelo_emerg_vs_ruido.h5")
        print("Modelo ganador guardado como modelo_emerg_vs_ruido.h5")


if __name__ == "__main__":
    main()
