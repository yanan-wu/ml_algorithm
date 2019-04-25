# -*- coding= utf8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def draw1():
    # 生成一个等差数列
    line = np.linspace(-5, 5, 200)
    # 画出非线性矫正的图形表示
    # relu-非线性矫正;tanh-双曲正切处理
    plt.plot(line, np.tanh(line), label='tanh')
    plt.plot(line, np.maximum(line, 0), label='relu')

    # 设置图注位置
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('relu(x) and tanh(x)')
    plt.show()


def set_params():
    wine = load_wine()
    X = wine.data[:, :2]
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[10, 10], activation='tanh', alpha=1)
    mlp.fit(X_train, y_train)
    print(mlp)

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    # 用不同的色块表示不同的分类
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    z = mlp.predict(np.c_[(xx.ravel(), yy.ravel())]).reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, z)
    # 用散点图画出训练集和测试数据集
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=60)
    # 设定横轴纵轴的范围
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # 设定图题
    plt.title('MLPClassifier:solver=lbfgs')
    plt.savefig('D:/lbfgs.png')
    plt.show()



set_params()