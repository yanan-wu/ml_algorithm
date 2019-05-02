# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder


def show_model_difference():
    rnd = np.random.RandomState(38)
    x = rnd.uniform(-5, 5, size=50)
    # 向数据中添加噪音
    y_no_noise = (np.cos(6*x) + x)
    X = x.reshape(-1, 1)
    y = (y_no_noise + rnd.normal(size=len(x)))/2
    plt.plot(X, y, 'o', c='r')
    plt.show()

    # 生成一个等差数列
    line = np.linspace(-5, 5, 1000, endpoint=False).reshape(-1, 1)
    mlpr = MLPRegressor().fit(X, y)
    knr = KNeighborsRegressor().fit(X, y)

    plt.plot(line, mlpr.predict(line), label='MLP')
    plt.plot(line, knr.predict(line), label='KNN')
    plt.plot(X, y, 'o', c='r')
    plt.legend(loc='best')
    plt.show()


# 装箱可以纠正模型过拟合或者欠拟合的问题
def binning():
    rnd = np.random.RandomState(38)
    x = rnd.uniform(-5, 5, size=50)
    # 向数据中添加噪音
    y_no_noise = (np.cos(6 * x) + x)
    X = x.reshape(-1, 1)
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    # 设置箱体数为 11
    bins = np.linspace(-5, 5, 11)
    # 将数据进行装箱操作
    target_bin = np.digitize(X, bins=bins)
    onehot = OneHotEncoder(sparse=False)
    onehot.fit(target_bin)
    # 使用独热编码转化数据
    X_in_bin = onehot.transform(target_bin)
    print('装箱后的数据形态：{}'.format(X_in_bin.shape))
    print('装箱后的前十个数据点：\n{}'.format(X_in_bin[:10]))

    line = np.linspace(-5, 5, 1000, endpoint=False).reshape(-1, 1)
    new_line = onehot.transform(np.digitize(line, bins=bins))
    mlpr = MLPRegressor().fit(X_in_bin, y)
    knr = KNeighborsRegressor().fit(X_in_bin, y)

    plt.plot(line, mlpr.predict(new_line), label='NEW MLP')
    plt.plot(line, knr.predict(new_line), label='NEW KNN')
    plt.plot(X, y, 'o', c='r')
    plt.legend(loc='best')
    plt.show()


binning()