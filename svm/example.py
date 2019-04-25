# -*- coding: utf-8 -*-

# 波士顿房价回归分析
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def example():
    boston = load_boston()
    X, y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
    # 对训练集和测试集进行数据预处理
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('--------------')
    print(X_train.shape)
    print(X_test.shape)
    print('-----------------')
    for kernel in ['linear', 'rbf']:
        svr = SVR(kernel=kernel, C=100, gamma=0.1)
        svr.fit(X_train_scaled, y_train)
        print(kernel, '核函数模型训练集得分：{:.3f}'.format(svr.score(X_train_scaled, y_train)))
        print(kernel, '核函数模型测试集得分：{:.3f}'.format(svr.score(X_test_scaled, y_test)))

example()