# -*- coding: utf-8 -*-

import graphviz
from sklearn.tree import export_graphviz
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, datasets
from sklearn.model_selection import train_test_split


def decision_tree1():
    wine = datasets.load_wine()
    X = wine.data[:, :2]
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    print(clf)

    # 分别用样本的两个特征值创建图像的横轴和纵轴
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # 给每个分类中的样本分配不同的颜色
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z)

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('decision_tree')
    plt.savefig('D:/test1.png')
    plt.show()

    export_graphviz(clf, out_file='wine.dot', class_names=wine.target_names,
                    feature_names=wine.feature_names[:2], impurity=False, filled=True)
    with open("wine.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)


decision_tree1()