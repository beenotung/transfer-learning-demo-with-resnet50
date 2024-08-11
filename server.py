from sanic import Sanic
from sanic.response import text, json
import tensorflow as tf
from model import predict_image_file

app = Sanic('ImageAI')

@app.get('/classify')
async def classify(request):
  img_path = request.args.get('img_path')
  result = predict_image_file(img_path)
  return json(result)
