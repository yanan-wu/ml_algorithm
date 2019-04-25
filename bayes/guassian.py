# -*- coding: utf-8 -*-

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np


def guassian1():
    X, y = make_blobs(n_samples=500, centers=5, random_state=8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print('--------------')
    print('模型得分：{:.3f}'.format(gnb.score(X_test, y_test)))
    print('-----------')
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # 用不同的色块表示不同的分类
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    z = gnb.predict(np.c_[(xx.ravel(), yy.ravel())]).reshape(xx.shape)
    plt.pcolormesh(xx, yy, z)
    # 用散点图画出训练集和测试数据集
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='*', edgecolors='k')
    # 设定横轴纵轴的范围
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # 设定图题
    plt.title('gaussianNB')
    plt.savefig('D:\\test.png')
    plt.show()

guassian1()