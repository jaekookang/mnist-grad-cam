'''
Train MNIST dataset using CNN

ref:
- https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
'''
from time import strftime, gmtime

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPool2D, Dropout
from keras import backend as K
from keras.optimizers import Adam

class HParam:
    batch_size = 128
    num_classes = 10
    epochs = 1

hp = HParam()


def get_date():
    return strftime('%Y-%m-%d', gmtime())

def prepare_mnist():
    # Prepare mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    nrow, ncol = 28, 28
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, nrow, ncol)
        x_test = x_test.reshape(x_test.shape[0], 1, nrow, ncol)
        input_shape = (1, nrow, ncol)
    else:
        x_train = x_train.reshape(x_train.shape[0], nrow, ncol, 1)
        x_test = x_test.reshape(x_test.shape[0], nrow, ncol, 1)
        input_shape = (nrow, ncol, 1)
    # Change variable type, value range
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # Convert classes into one-hot vector
    y_train = keras.utils.to_categorical(y_train, hp.num_classes)
    y_test = keras.utils.to_categorical(y_test, hp.num_classes)
    return (x_train, y_train), (x_test, y_test), input_shape

def build_cnn(input_shape):
    x = Input(shape=input_shape)
    h = Conv2D(32, (3,3), activation='relu')(x)
    h = Conv2D(64, (3,3), activation='relu')(h)
    h = MaxPool2D(pool_size=(2,2))(h)
    h = Dropout(0.5)(h)
    h = Flatten()(h)
    h = Dense(128, activation='relu')(h)
    h = Dropout(0.5)(h)
    y = Dense(hp.num_classes, activation='softmax')(h)
    model = Model(inputs=[x], outputs=[y])
    return model

def train():
    # Get mnist dataset
    (x_train, y_train), (x_test, y_test), input_shape = prepare_mnist()
    # Build & compile model
    model = build_cnn(input_shape)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    # Run model
    hist = model.fit(x_train, y_train,
                     batch_size=hp.batch_size,
                     epochs=hp.epochs,
                     verbose=1,
                     validation_data=(x_test, y_test))
    # Test model
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')
    # Save model
    model.save(f'model/mnist_{get_date()}')
    print('Model saved!')
    
if __name__ == '__main__':
    train()
