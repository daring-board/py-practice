# -*- coding: utf-8 -*
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ProgbarLogger, ReduceLROnPlateau, LambdaCallback
from keras.optimizers import RMSprop, SGD, Adam
import matplotlib.pyplot as plt

def main():
    batch_size = 128 # バッチサイズ(データサイズ)
    num_classes = 10 # 分類クラス数(今回は0～9の手書き文字なので10)
    epochs = 20 # エポック数(学習の繰り返し回数)

    # mnistデータセット（訓練用データと検証用データ）をネットから取得
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #MNISTデータの表示
    fig = plt.figure(figsize=(9, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
    for i in range(81):
        ax = fig.add_subplot(9, 9, i + 1, xticks=[], yticks=[])
        ax.imshow(x_train[i].reshape((28, 28)), cmap='gray')
    plt.show()

    # 2次元配列から1次元配列へ変換（784次元のベクトル）
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # データ型をfloat32に変換
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 値の正規化(0-255から0.0-1.0に変換）
    x_train /= 255
    x_test /= 255

    # データセットの個数を表示
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # 階調データ（0~255）を2値化（0, 1）
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # モデルの構築
    model = Sequential()

    # Dense：全結合のニューラルネットワークレイヤー
    # 入力層784次元(=28x28)、出力層512次元
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2)) # 過学習防止用：入力の20%を0にする（破棄）
    model.add(Dense(512, activation='relu')) # 活性化関数：relu
    model.add(Dropout(0.2)) # 過学習防止用：入力の20%を0にする（破棄）
    model.add(Dense(num_classes, activation='softmax')) # 活性化関数：softmax
    model.summary()

    # コンパイル（多クラス分類問題）
    model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto'),
    ]

    # 構築したモデルで学習
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=1, validation_data=(x_test, y_test))
    model.save('cnn_model.h5')

    # モデルの検証・性能評価
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
