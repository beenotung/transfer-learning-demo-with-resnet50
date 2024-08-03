from sanic import Sanic
from sanic.response import text, json
import tensorflow as tf
import ast
from model import predict_image_file

with open('saved_model/class_names.txt', 'r') as file:
  class_names=ast.literal_eval(file.read())

app = Sanic('ImageAI')

@app.get('/classify')
async def classify(request):
  img_path = request.args.get('img_path')
  result = predict_image_file(img_path)
  return json(result)
