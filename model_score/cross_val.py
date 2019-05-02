# -*- coding: utf-8 -*-

from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut


def cross_val():
    wine = load_wine()
    svc = SVC(kernel='linear')
    scores = cross_val_score(svc, wine.data, wine.target, cv=6)
    print('交叉得分：{}'.format(scores))
    print('交叉验证平均分：{:.3f}'.format(scores.mean()))


def shuffle_split():
    wine = load_wine()
    svc = SVC(kernel='linear')
    # 随机拆分
    shuffle_split = ShuffleSplit(test_size=.2, train_size=.7, n_splits=10)
    scores = cross_val_score(svc, wine.data, wine.target, cv=shuffle_split)
    print('交叉得分：{}'.format(scores))
    print('交叉验证平均分：{:.3f}'.format(scores.mean()))


# 挨个试试
def leave_one():
    wine = load_wine()
    svc = SVC(kernel='linear')
    cv = LeaveOneOut()
    scores = cross_val_score(svc, wine.data, wine.target, cv=cv)
    print('交叉得分：{}'.format(scores))
    print('交叉验证平均分：{:.3f}'.format(scores.mean()))


shuffle_split()