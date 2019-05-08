# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def simple():
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)
    scarer = StandardScaler().fit(X_train)
    X_train_scered = scarer.transform(X_train)
    X_test_scared = scarer.transform(X_test)
    params = {'hidden_layer_sizes': [(50,), (100,), (100, 100)], 'alpha': [0.0001, 0.001, 0.01, 0.1]}
    grid = GridSearchCV(MLPClassifier(max_iter=1600, random_state=38), param_grid=params, cv=3)
    grid.fit(X_train_scered, y_train)
    print('模型最高分为：{:.2f}'.format(grid.best_score_))
    print('最佳参数配置：{}'.format(grid.best_params_))


def pipe_line():
    X, y = make_blobs(n_samples=200, centers=2, cluster_std=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)
    pipeline = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(max_iter=1600, random_state=38))])
    pipeline.fit(X_train, y_train)
    print('使用管道模型的评分：{:.2f}'.format(pipeline.score(X_test, y_test)))

    params = {'mlp__hidden_layer_sizes': [(50,), (100,), (100, 100)], 'mlp__alpha': [0.0001, 0.001, 0.01, 0.1]}
    grid = GridSearchCV(pipeline, param_grid=params, cv=3)
    grid.fit(X_train, y_train)
    print('交叉验证最高分：{:.2f}'.format(grid.best_score_))
    print('模型最佳参数配置：{}'.format(grid.best_params_))
    print('测试集得分：{}'.format(grid.score(X_test, y_test)))


pipe_line()