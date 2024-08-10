// reference: https://codelabs.developers.google.com/tensorflowjs-transfer-learning-teachable-machine

import * as tf from '@tensorflow/tfjs-node'

let imageSize = 224
let embeddingSize = 1280

async function loadBaseModel() {
  let model = await tf.loadGraphModel(
    'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/large-100-224-feature-vector/1',
    { fromTFHub: true },
  )
  /* warm up the model */
  tf.tidy(() => {
    model.predict(tf.zeros([1, imageSize, imageSize, 3]))
  })
  return model
}

async function loadModelHead(options: { classNames: string[] }) {
  let classSize = options.classNames.length
  let model = tf.sequential()
  model.add(
    tf.layers.dense({
      inputShape: [embeddingSize],
      units: 128,
      activation: 'gelu',
    }),
  )
  model.add(
    tf.layers.dense({
      units: classSize,
      activation: 'softmax',
    }),
  )
  model.summary()
  model.compile({
    optimizer: 'adam',
    loss: classSize === 2 ? 'binaryCrossentropy' : 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })
}

async function loadModel() {
  await loadModelHead({ classNames: ['anime', 'real'] })
}

async function main() {
  await loadModel()
}

main().catch(e => console.error(e))
