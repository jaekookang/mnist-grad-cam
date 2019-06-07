'''
Grad-CAM visualization on the web

2019-06-02

## TODO
- [] Add inference code
- [] Make static/main.js
'''
import ipdb as pdb
import os
import cv2
import numpy as np
import ipdb as pdb

from keras.models import load_model
from grad_cam import make_heatmap
from keras.backend import clear_session

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load model

@app.route('/api/mnist', methods=['POST'])
def mnist():
	# input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(28, 28)
	input = np.array(request.json, dtype=np.uint8).reshape(28, 28)
	input = 255 - input
	cv2.imwrite('static/outputs/number.png', input)
	pred_out, gradcam_img = make_heatmap('static/outputs/number.png', 'static/outputs')
	return jsonify(results=pred_out.tolist())

@app.route('/<path:path>')
def static_proxy(path):
    # Serve everything in the 'static' folder
    return app.send_static_file(path)

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
