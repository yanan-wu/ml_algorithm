# -*- coding: utf-8 -*-

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def multinomial1():
    X, y = make_blobs(n_samples=500, centers=5, random_state=8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    mnb = MultinomialNB()
    mnb.fit(X_train_scaled, y_train)
    print('------')
    print('模型得分：{:.3f}'.format(mnb.score(X_test_scaled, y_test)))
    print('------')


multinomial1()

