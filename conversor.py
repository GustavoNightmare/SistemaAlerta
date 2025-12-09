import tensorflow as tf

MODEL_H5 = "modelo_cnn_big.h5"
TFLITE_PATH = "modelo_sonidos_v2.tflite"   # nuevo nombre

# 1. Cargar el modelo Keras
model = tf.keras.models.load_model(MODEL_H5)

# 2. Convertidor TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Opción A: sin cuantizar (float32) – más fácil, mismo comportamiento que ahora
# tflite_model = converter.convert()

# Opción B: cuantización dinámica (más pequeño, algo más rápido)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 3. Guardar
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("TFLite guardado en:", TFLITE_PATH)
