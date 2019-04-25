# -*- coding: utf-8 -*-

import numpy as np
from sklearn.naive_bayes import BernoulliNB


def bayes1():
    # 分别表示:[刮北风、闷热、多云、天气预报有雨]
    X = np.array([[0, 1, 0, 1],
                  [1, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 1],
                  [0, 1, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 0, 1]])
    # 实际的 7 天中，是否有雨，0-没雨，1-有雨
    y = np.array([0, 1, 1, 0, 1, 0, 0])
    counts = {}
    for label in np.unique(y):
        counts[label] = X[y == label].sum(axis=0)
    print('feature counts:\n{}'.format(counts))
    clf = BernoulliNB()
    clf.fit(X, y)
    Next_day = [[0, 0, 1, 0]]
    pre = clf.predict(Next_day)
    print('------------')
    if pre == [1]:
        print('要下雨了')
    else:
        print('晴天')
    print('--------------')
    # 朴素贝叶斯对于预测具体的数值并不擅长，给出的概率仅供参考
    print('模型预测分类的概率：{}'.format(clf.predict_proba(Next_day)))
    print('--------------')

bayes1()