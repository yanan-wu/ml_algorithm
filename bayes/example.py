# -*- coding: utf-8 -*-

# 导入威斯康星乳腺肿瘤数据集
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def example1():
    cancer = load_breast_cancer()
    print('--------------')
    print(cancer.keys())
    print('---------------------')
    print('肿瘤的分类:', cancer['target_names'])
    print('肿瘤的特征:', cancer['feature_names'])
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)
    print('------------------')
    print('训练集数据形态：', X_train.shape)
    print('测试集数据形态：', X_test.shape)
    print('------------------')
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print('----------------')
    print('训练集得分：{:.3f}'.format(gnb.score(X_train, y_train)))
    print('测试集得分：{:.3f}'.format(gnb.score(X_test, y_test)))
    print('----------------')




example1()