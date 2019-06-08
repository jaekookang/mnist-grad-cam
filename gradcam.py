'''
Grad-CAM visualizer

2019-06-07

## See:
- https://github.com/eclique/keras-gradcam (mostly used from)
- https://github.com/fchollet
'''
import numpy as np
import cv2
import sys
import os

from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.backend import clear_session
import tensorflow as tf
from tensorflow.python.framework import ops


def load_image(img_file):
    '''Load and preprocess image'''
    nrow, ncol = 28, 28
    img_RGB = cv2.imread(img_file, 1)  # 1=RGB
    img = cv2.imread(img_file, 0)  # 0=grayscale

    img = cv2.resize(img, (nrow, ncol))
    img = img / img.max()  # range: 0-1
    img = img.astype('float32')
    img_resized = img.reshape(1, img.shape[0], img.shape[1], 1)
    return img_resized, img_RGB


def deprocess_image(img):
    '''Convert image from 0-1 scale into 0-255'''
    img = img.copy()
    if np.ndim(img) > 3:
        img = np.squeeze(img)

    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1

    img += 0.5
    img = np.clip(img, 0, 1)

    img *= 255
    if K.image_dim_ordering() == 'th':
        img = img.transpose((1, 2, 0))
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def build_model(model_file='model/mnist_2019-06-02'):
    '''Load trained model'''
    model = load_model(model_file)
    return model


def build_guided_model():
    '''
    Build modified model by applying ReLU activations to gradients
    based on guided backpropagation
    '''
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        # Make GuidedBackProp operation
        @tf.RegisterGradient('GuidedBackProp')
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model


def guided_backprop(input_model, img, layer_name):
    '''Guided backpropagation'''
    input_img = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_img)[0]
    backprop_func = K.function([input_img, K.learning_phase()], [grads])
    grads_val = backprop_func([img, 0])[0]
    return grads_val


def grad_cam(input_model, img, which_class, layer_name):
    '''Visualization using GradCAM'''
    y_class = input_model.output[0, which_class]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_class, conv_output)[0]  # (None,24,24,64)

    gradient_func = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_func([img])
    output, grads_val = output[0, ...], grads_val[0, ...]

    # Compute Global Average Pooling
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)  # sum(weights * conv_output)
    cam = np.maximum(cam, 0)  # ReLU

    # Process CAM
    cam = cv2.resize(cam, img.shape[1:3])
    cam = cam / cam.max()
    return cam


def compute_saliency(img_file, outdir, layer_name, which_class=-1, save=True):
    '''
    Make visualizations
    which_class=-1 means the most probable class
    '''
    clear_session()
    model = build_model()
    guided_model = build_guided_model()

    img, img_RGB = load_image(img_file)

    pred = model.predict(img)
    if which_class == -1:
        which_class = np.argmax(pred)

    gradcam = grad_cam(model, img, which_class, layer_name)
    gb = guided_backprop(model, img, layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]

    guided_gradcam = np.squeeze(guided_gradcam)
    gb = np.squeeze(gb)

    if save:
        # Save GradCAM
        jetcam = cv2.resize(np.uint8(255 * gradcam),
                            (img_RGB.shape[0], img_RGB.shape[1]))
        jetcam = cv2.applyColorMap(jetcam, cv2.COLORMAP_JET)
        # jetcam = (np.float32(jetcam) + img_RGB) / 2
        cv2.imwrite(os.path.join(outdir, 'gradcam.jpg'), jetcam)
        # Save Guided GradCAM
        jetgcam = cv2.resize(guided_gradcam,
                            (img_RGB.shape[0], img_RGB.shape[1]))
        jetgcam = deprocess_image(jetgcam)
        cv2.imwrite(os.path.join(outdir, 'guided_gradcam.jpg'), jetgcam)

    return pred, gradcam, gb, guided_gradcam


if __name__ == '__main__':
    img_file = 'examples/02.jpg'
    outdir = '.'
    # img_file = 'cat_dog.png'

    pred, gradcam, gb, guided_gradcam = compute_saliency(img_file, outdir,
        layer_name='conv2d_2', which_class=-1, save=True)
    print('done')
