'''
This script generates an image with Class Activation Map

Usage:
$ python grad_camp.py examples/01.jpg outputs

2019-06-02 
'''
import ipdb as pdb
import numpy as np
import cv2
import sys
import os

from keras.models import Model
from keras.models import load_model
from train import prepare_mnist
from keras import backend as K
from keras.backend import clear_session


# Make Grad-CAM visualization

def make_heatmap(input_file, out_dir, save_output=True):
    # Load model
    clear_session() # <= This is crucial!
    model = load_model('model/mnist_2019-06-02')
    # Process image
    nrow, ncol = 28, 28
    img_3d = cv2.imread(input_file, 1) # RGB
    img = cv2.imread(input_file, 0) # grayscale
    image_id = os.path.basename(input_file).split('.')[0]
    if (img.shape[0] > 28) | (img.shape[1] > 28):
        img_resized = cv2.resize(img, (nrow, ncol))
        img_resized = img_resized/img_resized.max()
        img_resized = img_resized.astype('float32')
    else:
        img_resized = img/img.max()
        img_resized = img_resized.astype('float32')
    
    # Predict number
    pred_out = model.predict(
        img_resized.reshape(1,img_resized.shape[0],img_resized.shape[1],1))
    target_number = int(np.argmax(pred_out))
    
    ### Get gradient
    # Resize image
    input_image = img_resized.reshape(1,img_resized.shape[0],img_resized.shape[1],1)
    # Make heatmap given model and the target number
    output_vector = model.output[:, target_number]
    # Get the last convolutional layer
    last_conv_layer = model.layers[2]
    # Get the gradient of the given number with regard to the output feature map of the last conv layer
    grads = K.gradients(output_vector, last_conv_layer.output)[0] # (None,24,24,64)
    # Get the mean intensity of the gradient over each feature map (64)
    pooled_grads = K.mean(grads, axis=(0, 1, 2)) # 64
    # Compute gradient given an inputw
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([input_image]) # 64, (24,24,64)

    ### Make heatmap
    # Multiply each channel in the feature map array
    # by 'how important this channel is' with regard to the given number
    for i in range(len(pooled_grads_value)):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
    # Calculate channel-wise mean for the heatmap activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0) # => max(heatmap, 0)
    heatmap /= np.max(heatmap)
    # Resize to the original image
    heatmap_resized = cv2.resize(heatmap, img.shape) # (500, 500)
    heatmap_resized = np.uint8(255*heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    # Apply heatmap on to the original image
    # superimposed_img = heatmap_resized*0.5 + (255-img_3d)
    superimposed_img = 255 - heatmap_resized*0.5 # the smaller, the more active!
    # Save
    if save_output:
        cv2.imwrite(os.path.join(out_dir, image_id+'_gradcam.jpg'), superimposed_img);
        print('Saved')
    clear_session()
    return pred_out, superimposed_img

if __name__ == '__main__':
    # Inputs
    img_file = sys.argv[1]
    out_dir = sys.argv[2]

    # Load model
    # model = load_model('model/mnist_2019-06-02')

    make_heatmap(img_file, out_dir)
