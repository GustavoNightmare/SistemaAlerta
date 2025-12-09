import tensorflow as tf

# Carga el modelo Keras que ya entrenaste
model = tf.keras.models.load_model("modelo_cnn_big.h5")

# Crea el convertidor
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Opcional: optimización para hacerlo más ligero
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Guarda el modelo TFLite
with open("modelo_sonidos_bigcnn.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Modelo convertido y guardado como modelo_sonidos_bigcnn.tflite")
