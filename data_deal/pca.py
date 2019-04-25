# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=62)
    print(X_train.shape, X_test.shape)
    scaler = StandardScaler()
    X = wine.data
    y = wine.target
    X_scaled = scaler.fit_transform(X)
    print(X_scaled.shape)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    print(X_pca.shape)

    X0 = X_pca[wine.target == 0]
    X1 = X_pca[wine.target == 1]
    X2 = X_pca[wine.target == 2]

    plt.scatter(X0[:, 0], X0[:, 1], c='b', s=60, edgecolors='k')
    plt.scatter(X1[:, 0], X1[:, 1], c='g', s=60, edgecolors='k')
    plt.scatter(X2[:, 0], X2[:, 1], c='r', s=60, edgecolors='k')

    plt.legend(wine.target_names, loc='best')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()


pca()