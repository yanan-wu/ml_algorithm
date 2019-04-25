# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def random_forest1():
    wine = datasets.load_wine()
    X = wine.data[:, :2]
    y = wine.target
    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # 设定树的个数 n_estimators; n_jobs=-1 自动使用CPU的全部内核，并行处理
    forest = RandomForestClassifier(n_estimators=6, random_state=3, n_jobs=-1)
    forest.fit(X_train, y_train)
    print(forest)

    # 分别用样本的两个特征值创建图像的横轴和纵轴
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    Z = forest.predict(np.c_[xx.ravel(), yy.ravel()])
    # 给每个分类中的样本分配不同的颜色
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z)

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('random_forest')
    plt.savefig('D:/test.png')
    plt.show()

random_forest1()