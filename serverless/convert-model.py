import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('waste_classification_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('waste_classification_
          model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)