import tensorflow as tf
import numpy as np
import ast

image_size = (512, 512)
checkpoint_path = './saved_model/model.keras'

with open('saved_model/class_names.txt', 'r') as file:
  class_names=ast.literal_eval(file.read())

model = tf.keras.models.load_model(checkpoint_path)
# model.summary()

def load_image(img_path):
  return tf.keras.utils.load_img(
    img_path,
    color_mode='rgb',
    target_size=image_size,
    interpolation='nearest',
    keep_aspect_ratio=False,
  )

def predict_image_file(img_path):
  image = load_image(img_path)
  inputs = tf.keras.utils.img_to_array(image)
  inputs = np.array([inputs])
  outputs = model.predict(inputs)
  outputs = tf.nn.softmax(outputs)
  probabilities = np.array(outputs[0]).tolist()
  result = []
  for i, class_name in enumerate(class_names):
    result.append({
      'probability': probabilities[i],
      'class_name': class_name,
  })
  return {'result': result}
