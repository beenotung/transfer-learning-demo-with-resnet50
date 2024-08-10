const tf = require('@tensorflow/tfjs-node')
//const tf = require('@tensorflow/tfjs-node-gpu')

const cocossd = require('@tensorflow-models/coco-ssd')

const fs = require('fs')
const gm = require('gm')

var path = 'dog.jpg'

//https://js.tensorflow.org/api_node/latest/
const buf = fs.readFileSync(path)
const input = tf.node.decodeJpeg(buf)

load(input)

async function load(img) {
  const model = await cocossd.load({ base: 'mobilenet_v2' })
  const predictions = await model.detect(img, 3, 0.2)

  console.log('Predictions: ')
  console.log(predictions)
  drawPredictions(predictions)

  let samplePredictions = [
    {
      bbox: [
        170.0888186097145, 22.4247387945652, 1293.490363061428,
        1114.5569941699505,
      ],
      class: 'cat',
      score: 0.8749409317970276,
    },
  ]
}

async function drawPredictions(predictions) {
  var pic = gm(path)

  for (let i = 0; i < predictions.length; i++) {
    var p = predictions[i].bbox

    pic.fill('transparent')
    pic.stroke('#ff0000', 5)
    pic.drawRectangle(p[0], p[1], p[0] + p[2], p[1] + p[3])

    pic.fill('red')
    pic.stroke('#ff0000', 1)
    pic.fontSize(25)
    pic.drawText(p[0] + 10, p[1] + 25, predictions[i].class)
  }

  pic.write('predicted-' + path, function (err) {
    if (!err) console.log('Done')
  })
}
