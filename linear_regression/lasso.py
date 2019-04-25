# -*- coding: utf-8 -*-

from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np


def linear_regression1():
    # 糖尿病情数据集
    X, y = load_diabetes().data, load_diabetes().target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
    # alpha 越大，则使用特征数越少
    lasso = Lasso(alpha=0.1, max_iter=100000).fit(X_train, y_train)
    print('--------------')
    print('lasso.coef_:{}'.format(lasso.coef_[:]))
    print('lasso.intercept_:{}'.format(lasso.intercept_))
    print('----------')
    print('训练数据集得分:{:.2f}'.format(lasso.score(X_train, y_train)))
    print('测试数据集得分:{:.2f}'.format(lasso.score(X_test, y_test)))
    print('套索回归使用的特征数：{}'.format(np.sum(lasso.coef_ != 0)))


linear_regression1()