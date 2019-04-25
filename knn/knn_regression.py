# -*- coding: utf-8 -*-

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


def knn_regression1():
    X, y = make_regression(n_features=1, n_informative=1, noise=50, random_state=8)
    plt.scatter(X, y, c='orange', edgecolors='k')
    plt.show()

    reg = KNeighborsRegressor(n_neighbors=2)
    # 拟合数据
    reg.fit(X, y)
    # 预测结果可视化
    z = np.linspace(-3, 3, 200).reshape(-1, 1)
    plt.scatter(X, y, c='orange', edgecolors='k')
    plt.plot(z, reg.predict(z), c='k', linewidth=3)
    plt.title('knn_regression')
    plt.show()
    print('-----------')
    print(reg.score(X, y))
    print('-----------')


knn_regression1()