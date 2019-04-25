# -*- coding= utf8 -*-

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np


def handwriting():
    mnist = fetch_mldata('MNIST original')
    print('-----------')
    print('样本数量：{}，样本特征数：{}'.format(mnist.data.shape[0], mnist.data.shape[1]))
    print('-----------')

    X = mnist.data/255.
    y = mnist.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=1000, random_state=62)

    nlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[100, 100], activation='relu', alpha=1e-5, random_state=62)
    nlp.fit(X_train, y_train)
    print('-----------')
    print('测试数据集得分：{:.2f}%'.format(nlp.score(X_test, y_test)*100))
    print('-----------')

    # 识别图片
    image = Image.open('3.png').convert('F')
    image = image.resize((28, 28))
    arr = []
    for i in range(28):
        for j in range(28):
            pixel = 1.0 - float(image.getpixel((j, i)))/255.
            arr.append(pixel)
    arr1 = np.array(arr).reshape(1, -1)
    print('图中的数字是：{:.0f}'.format(nlp.predict(arr1)[0]))



handwriting()