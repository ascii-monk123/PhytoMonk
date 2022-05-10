from ast import Bytes
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


shape = (256, 256)

#read the input image as PIL object
def read_im(enc_image):
    image = Image.open(BytesIO(enc_image))
    return image
#Preprocess image before feeding it to the neural network
def preprocess_im(image:Image.Image):
    image = np.asarray(image.resize(shape))
    image = image / 255.0
    image = np.expand_dims(image, 0)

    return image

#load neural network model
def load_nn():
    model = Sequential()
    #first conv layer
    model.add(Conv2D(filters = 16, kernel_size = (3, 3), padding = 'valid', input_shape = (256, 256, 3)))
    model.add(Activation('relu'))
    #first maxpool layer
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    #second conv layer
    model.add(Conv2D(filters = 32, kernel_size = (3, 3)))
    model.add(Activation('relu'))
    #second maxpool layer
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    #third convolutional layer
    model.add(Conv2D(filters = 64, kernel_size =(3, 3)))
    model.add(Activation('relu'))
    #third maxpool layer
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    #fourth convolutional layer
    model.add(Conv2D(filters = 128, kernel_size = (3, 3)))
    model.add(Activation('relu'))
    #fourth maxpool layer
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    #fifth convolutional
    model.add(Conv2D(filters = 256, kernel_size = (3, 3)))
    #fully connected component

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation = 'softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    model.load_weights('weights/weights.h5')
    return model

nn = load_nn()

#decode prediction type
def decode_prediction(pred):
    im_class = tf.argmax(pred[0], axis=-1)
    return im_class
#predict the disease class
def predict(image:np.ndarray):
    return decode_prediction(nn.predict(image))


