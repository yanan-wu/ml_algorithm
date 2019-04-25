# -*- coding: utf-8 -*-

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def linear_regression1():
    # n_features: 特征个数 = n_informative（） + n_redundant + n_repeated
    # n_informative：多信息特征的个数
    # n_redundant：冗余信息，informative特征的随机线性组合
    # n_repeated ：重复信息，随机提取n_informative和n_redundant 特征
    # n_classes：分类类别
    # n_clusters_per_class ：某一个类别是由几个cluster构成的
    X, y = make_regression(n_samples=200, n_features=2, n_informative=2, noise=20, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
    lr = LinearRegression().fit(X_train, y_train)
    print('--------------')
    print('lr.coef_:{}'.format(lr.coef_[:]))
    print('lr.intercept_:{}'.format(lr.intercept_))
    print('----------')
    print('测试数据集得分:{:.2f}'.format(lr.score(X_test, y_test)))
    print('-------------')


linear_regression1()