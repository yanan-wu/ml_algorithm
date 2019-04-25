# -*- coding: utf-8 -*-

from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np


def linear_regression1():
    # 糖尿病情数据集
    X, y = load_diabetes().data, load_diabetes().target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
    elastic_net = ElasticNet(alpha=1, l1_ratio=1, max_iter=100000).fit(X_train, y_train)
    print('--------------')
    print('elastic_net.coef_:{}'.format(elastic_net.coef_[:]))
    print('elastic_net.intercept_:{}'.format(elastic_net.intercept_))
    print('----------')
    print('训练数据集得分:{:.2f}'.format(elastic_net.score(X_train, y_train)))
    print('测试数据集得分:{:.2f}'.format(elastic_net.score(X_test, y_test)))
    print('弹性网回归使用的特征数：{}'.format(np.sum(elastic_net.coef_ != 0)))


linear_regression1()