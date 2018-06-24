# -*- coding: utf-8 -*
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
from PIL import Image
from matplotlib import pylab as plt
from scipy.misc import imresize

class Predictor():

    def __init__(self):
        self.model = load_model('cnn_model.h5')

    def predict(self, img):
        img = img.reshape(1, 784)
        pred = self.model.predict_classes(img)
        return pred

if __name__=='__main__':
    # 画像の読み込み
    img = np.array(Image.open('sample.png').convert('L'))
    img = imresize(img, (28,28))

    # 画像の表示
    plt.imshow(img)
    plt.show()

    img = img.astype('float32')
    prd = Predictor()
    print(prd.predict(img))
