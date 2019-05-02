# -*- coding: utf-8 -*-

from sklearn.datasets import load_wine
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV


def simple():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=38)
    best_score = 0
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        for max_iter in [100, 1000, 5000, 10000]:
            lasso = Lasso(alpha=alpha, max_iter=max_iter)
            lasso.fit(X_train, y_train)
            score = lasso.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_parameters = {'alpha': alpha, 'max_iter': max_iter}
    print('模型最高分为：{:.3f}'.format(best_score))
    print('最佳参数配置：{}'.format(best_parameters))


def grid_cross_val():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=38)
    best_score = 0
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        for max_iter in [100, 1000, 5000, 10000]:
            lasso = Lasso(alpha=alpha, max_iter=max_iter)
            scores = cross_val_score(lasso, X_train, y_train, cv=6)
            score = np.mean(scores)
            if score > best_score:
                best_score = score
                best_parameters = {'alpha': alpha, 'max_iter': max_iter}
    print('模型最高分为：{:.3f}'.format(best_score))
    print('最佳参数配置：{}'.format(best_parameters))


def grid_search():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=38)
    params = {'alpha': [0.01, 0.1, 1.0, 10.0], 'max_iter': [100, 1000, 5000, 10000]}
    grid_search = GridSearchCV(Lasso(), params, cv=6)
    grid_search.fit(X_train, y_train)
    print('模型最高分为：{:.3f}'.format(grid_search.score(X_test, y_test)))
    print('最佳参数配置：{}'.format(grid_search.best_params_))


grid_search()