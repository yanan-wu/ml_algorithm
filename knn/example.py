# -*- coding: utf-8 -*-

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def demo():
    wine_dataset = load_wine()
    # 数据分为训练集和测试集;random_state 为0 每次的拆分情况不同
    X_train, X_test, y_train, y_test = train_test_split(wine_dataset['data'], wine_dataset['target'], random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    # 使用测试集对模型打分
    print('-----------')
    print('测试数据集得分：{:.2f}'.format(knn.score(X_test, y_test)))
    print('-----------')

    #分类预测
    X_new = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]])
    prediction = knn.predict(X_new)
    print("预测的分类结果：{}".format(wine_dataset['target_names'][prediction]))

demo()

