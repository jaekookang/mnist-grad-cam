'''
Grad-CAM visualization on the web

2019-06-02
'''
import os
import cv2
import numpy as np
import ipdb as pdb

from keras.models import Model
from keras.models import load_model
from keras import backend as K

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

@app.route('/api/mnist', methods=['POST'])
def mnist():
    pass

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
