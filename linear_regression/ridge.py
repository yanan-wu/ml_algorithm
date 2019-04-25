# -*- coding: utf-8 -*-

from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def linear_regression1():
    # 糖尿病情数据集
    X, y = load_diabetes().data, load_diabetes().target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
    # alpha 越大，则越降低过拟合程度
    ridge = Ridge(alpha=0.5).fit(X_train, y_train)
    print('--------------')
    print('ridge.coef_:{}'.format(ridge.coef_[:]))
    print('ridge.intercept_:{}'.format(ridge.intercept_))
    print('----------')
    print('训练数据集得分:{:.2f}'.format(ridge.score(X_train, y_train)))
    print('测试数据集得分:{:.2f}'.format(ridge.score(X_test, y_test)))


linear_regression1()