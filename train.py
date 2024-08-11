import tensorflow as tf

dataset_directory = './dataset'
checkpoint_path='./saved_model/model.keras'
image_size = (512, 512)
input_shape = (image_size[0], image_size[1], 3)
num_classes = 2

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
)
base_model.trainable = False

inputs = tf.keras.Input(shape=input_shape)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_directory,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True
)

with open('./saved_model/class_names.txt', 'w') as file:
  file.write(str(dataset.class_names))

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    verbose=1
)

model.fit(
    dataset,
    epochs=5,
    callbacks=[checkpoint_callback],
)

model.save(checkpoint_path)
