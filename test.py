import tensorflow as tf
import numpy as np
from model import load_image, model, class_names, predict_image_file

img_path = 'test/748dc4e1b7ba92fa29a5dafef579fb2a.png'

## predict with use helper function

result = predict_image_file(img_path)

print(result)

## or predict step by step

image = load_image(img_path)

inputs = tf.keras.utils.img_to_array(image)
inputs = np.array([inputs])

result = model.predict(inputs)

print("class_names:", class_names)
print("result:", result)

label = class_names[result.argmax(axis=1)[0]]

print("label:", label)
