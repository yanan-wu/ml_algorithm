# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def interaction_features():
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
    print('装箱前的数据形态：{}'.format(X.shape))
    print('装箱后的数据形态：{}'.format(X_in_bin.shape))
    print('装箱后的前十个数据点：\n{}'.format(X_in_bin[:10]))

    line = np.linspace(-5, 5, 1000, endpoint=False).reshape(-1, 1)
    new_line = onehot.transform(np.digitize(line, bins=bins))

    # 将原始数据和装箱后的数据进行堆叠
    X_stack = np.hstack([X, X_in_bin])
    line_stack = np.hstack([line, new_line])

    # 重新训练模型
    mlpr_interact = MLPRegressor().fit(X_stack, y)
    plt.plot(line, mlpr_interact.predict(line_stack), label='MLP for interaction')
    plt.ylim(-4, 4)
    for vline in bins:
        plt.plot([vline, vline], [-5, 5], ':', c='k')
    plt.legend(loc='lower right')
    plt.plot(X, y, 'o', c='r')
    plt.show()


def polynomial_features():
    rnd = np.random.RandomState(38)
    x = rnd.uniform(-5, 5, size=50)
    # 向数据中添加噪音
    y_no_noise = (np.cos(6 * x) + x)
    X = x.reshape(-1, 1)
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    poly = PolynomialFeatures(degree=20, include_bias= False)
    X_poly = poly.fit_transform(X)
    lnr_poly = LinearRegression().fit(X_poly, y)

    line = np.linspace(-5, 5, 1000, endpoint=False).reshape(-1, 1)
    line_poly = poly.transform(line)
    plt.plot(line, lnr_poly.predict(line_poly), label='linear regressor')
    plt.xlim(np.min(X) - 0.5, np.max(X) + 0.5)
    plt.ylim(np.min(y) - 0.5, np.max(y) + 0.5)
    plt.plot(X, y, 'o', c='r')
    plt.legend(loc='lower right')
    plt.show()


polynomial_features()