# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import load_wine


def linear1():
    X, y = make_blobs(n_samples=50, centers=2, random_state=6)

    # 创建一个线性内核（linear）的支持向量机模型/高斯内核（rbf）/多项式内核
    clf = svm.SVC(kernel='rbf', C=1000)
    clf.fit(X, y)

    # 将数据点画出
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 生成两个等差数列
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # 把分类边界画出来
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyle=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidths=1, facecolors='none')
    plt.show()


# 画图函数
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


# 绘制等高线的函数
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# 不同核函数的 SVM 对比
def show_differ():
    wine = load_wine()
    X = wine.data[:, :2]
    y = wine.target
    C = 1.0
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(X, y) for clf in models)

    # 设定标题
    titles = ('svc with linear kernel',
              'linearSVC',
              'SVC with RBF kernel',
              'SVC with polynomial kernel')

    # 设定一个子图形的个数和排列方式
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy, alpha=0.8)
        ax.scatter(X0, X1, c=y, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('feature 0')
        ax.set_ylabel('feature 1')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    plt.show()


# 支持向量机 gamma 参数调节
def show_differ1():
    wine = load_wine()
    X = wine.data[:, :2]
    y = wine.target
    C = 1.0
    models = (svm.SVC(kernel='rbf', gamma=0.1, C=C),
              svm.SVC(kernel='rbf', gamma=1, C=C),
              svm.SVC(kernel='rbf', gamma=10, C=C))
    models = (clf.fit(X, y) for clf in models)

    # 设定标题
    titles = ('gamma=0.1',
              'gamma=1',
              'gamma=10')

    # 设定一个子图形的个数和排列方式
    fig, sub = plt.subplots(1, 3, figsize=(10, 3))
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy, alpha=0.8)
        ax.scatter(X0, X1, c=y, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('feature 0')
        ax.set_ylabel('feature 1')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    plt.show()


show_differ()
