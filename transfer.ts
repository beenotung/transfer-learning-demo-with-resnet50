import * as mobilenet from '@tensorflow-models/mobilenet'
import * as tf from '@tensorflow/tfjs-node'
import { readFileSync, readdirSync } from 'fs'
import { readFile } from 'fs/promises'
import { join } from 'path'

let imageSize = 224

type ClassifierOption = {
  nClass: number
}

async function loadBaseModelV2() {
  let model = await mobilenet.load({
    version: 2,
    alpha: 1,
  })
  return model
}

async function loadBaseModelV3() {
  let model = await tf.loadGraphModel(
    'https://www.kaggle.com/models/google/mobilenet-v3/TfJs/large-100-224-feature-vector/1',
    { fromTFHub: true },
  )
  return model
}

async function loadEmbeddingClassifierModel(options: ClassifierOption) {
  // tf.models.modelFromJSON()
  let model = tf.sequential()
  model.add(tf.layers.inputLayer({ inputShape: [1280] }))
  // model.add(tf.layers.globalAveragePooling1d({ inputShape: [1280] }))
  model.add(tf.layers.dropout({ rate: 0.2 }))
  model.add(tf.layers.dense({ units: options.nClass, activation: 'softmax' }))
  return model
}

async function loadSlimModel(options: ClassifierOption) {
  let model = tf.sequential()

  /* image input layer */
  model.add(tf.layers.inputLayer({ inputShape: [imageSize, imageSize, 3] }))

  /* image convolution layers */
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: [3, 3],
      activation: 'gelu',
    }),
  )
  model.add(tf.layers.maxPool2d({ poolSize: [2, 2] }))
  model.add(tf.layers.dropout({ rate: 0.2 }))

  /* embedding feature layer */
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({ units: 256, activation: 'gelu' }))
  model.add(tf.layers.dropout({ rate: 0.2 }))

  /* classification layer */
  model.add(tf.layers.dense({ units: 2, activation: 'softmax' }))

  return model
}

async function loadModel(options: ClassifierOption) {
  let [baseModel, classifierModel] = await Promise.all([
    loadBaseModelV3(),
    loadEmbeddingClassifierModel(options),
  ])
  let optimizer = tf.train.adam(0.01)
  classifierModel.compile({
    optimizer,
    loss:
      // 'categoricalCrossentropy'
      tf.losses.softmaxCrossEntropy,
    metrics: ['accuracy'],
  })
  let classes: string[] = []
  return {
    get classes() {
      return classes
    },
    async predictAsync(inputTensor: tf.Tensor) {
      let embedding = (await baseModel.predictAsync(inputTensor)) as tf.Tensor
      let outputTensor = classifierModel.predict(embedding) as tf.Tensor
      embedding.dispose()
      let predicts = outputTensor.data()
      outputTensor.dispose()
      return predicts
    },
    async train(datasetDir: string) {
      let x: tf.Tensor[] = []
      let y: tf.Tensor[] = []

      let dirnames = readdirSync(datasetDir)
      let classIndex = 0
      for (let dirname of dirnames) {
        classes[classIndex] = dirname
        let dir = join(datasetDir, dirname)
        let filenames = readdirSync(dir)
        for (let filename of filenames) {
          let file = join(dir, filename)
          let input = await loadImage(file)
          x.push(input)
          y.push(tf.tensor([classIndex]))
        }
        classIndex++
      }

      let history = await classifierModel.fit(x, y, {
        shuffle: true,
        batchSize: 32,
        epochs: 5,
        verbose: 1,
        callbacks: [
          {
            onYield(epoch, batch, logs) {},
          },
        ],
      })

      // await history.syncData()
      console.log('history:', history)
    },
  }
}

async function loadImage(file: string) {
  let buffer = await readFile(file)
  let imageTensor = tf.node.decodeImage(buffer)
  const widthToHeight = imageTensor.shape[1] / imageTensor.shape[0]
  let squareCrop
  if (widthToHeight > 1) {
    const heightToWidth = imageTensor.shape[0] / imageTensor.shape[1]
    const cropTop = (1 - heightToWidth) / 2
    const cropBottom = 1 - cropTop
    squareCrop = [[cropTop, 0, cropBottom, 1]]
  } else {
    const cropLeft = (1 - widthToHeight) / 2
    const cropRight = 1 - cropLeft
    squareCrop = [[0, cropLeft, 1, cropRight]]
  }
  // Expand image input dimensions to add a batch dimension of size 1.
  const crop = tf.image.cropAndResize(
    tf.expandDims<tf.Tensor4D>(imageTensor, 1),
    squareCrop,
    [0],
    [imageSize, imageSize],
  )
  return crop.div(255)
}

async function main() {
  // let model = await loadSlimModel({ nClass: 2 })
  let model = await loadModel({ nClass: 2 })
  await model.train('dataset')
  let input = await loadImage('test/748dc4e1b7ba92fa29a5dafef579fb2a.png')
  let predicts = await model.predictAsync(input)
  console.log(predicts)
}

main().catch(e => console.error(e))
