# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
# 画图工具
import matplotlib.pyplot as plt
import numpy as np


def knn_classification1():
    # 生成样本数为 300，分类为 2 的数据集
    data = make_blobs(n_samples=300, centers=2, random_state=8)
    X, y = data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap(), edgecolors='k')
    plt.show()

    clf = KNeighborsClassifier()
    clf.fit(X, y)

    # 下面的代码用于画图
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.get_cmap(), edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("classification_knn")
    plt.scatter(6.75, 4.81, marker='*', c='red', s=300)
    plt.show()


knn_classification1()