# -*- coding: utf-8 -*
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

class Predictor():

    def __init__(self):
        self.model = load_model('cnn_model.h5')

    def predict(self, img):
        img = img.reshape(1, 784)
        img = img.astype('float32')
        pred = self.model.predict_classes(img)
        return pred
