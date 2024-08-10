import * as tf from '@tensorflow/tfjs-node'
import * as mobilenet from '@tensorflow-models/mobilenet'
import { readFileSync } from 'fs'

async function main() {
  let loading = ''
  function start(name: string) {
    console.time(name)
    loading = name
  }
  function end() {
    console.timeEnd(loading)
  }
  function next(name: string) {
    end()
    start(name)
  }

  start('load mobilenet model')
  let model = await mobilenet.load({
    version: 2,
    alpha: 1.0,
  })
  next('load image')
  let input = tf.node.decodeJpeg(readFileSync('dog.jpg'))
  next('classify')
  let result = await model.classify(input, 3)
  next('infer')
  let embedding = model.infer(input, true).dataSync()
  end()
  console.log('embedding:', embedding)
  console.log('embedding size:', embedding.length)
}
main().catch(e => console.error(e))
