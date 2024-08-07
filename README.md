# Transfer Learning Demo with ResNet50

A generic demo to build custom image classifier of arbitrary classes.

## Key Steps

1. Training: run [train.py](train.py) or [train.ipynb](train.ipynb)
2. Testing: run [test.py](test.py) or [test.ipynb](test.ipynb)
3. Start Server: run `sanic server`
4. Call from HTTP: e.g. `http://127.0.0.1:8000/classify?img_path=test/748dc4e1b7ba92fa29a5dafef579fb2a.png`

## Alternative Base Models

There are multiple alternative base models with smaller file size and less parameters.

Details see [compare.md](./compare.md)
