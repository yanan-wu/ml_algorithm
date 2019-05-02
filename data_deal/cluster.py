# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import DBSCAN


def show_data():
    blobs = make_blobs(random_state=1, centers=1)
    X_blobs = blobs[0]
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c='r', edgecolors='k')
    plt.show()


def k_means():
    blobs = make_blobs(random_state=1, centers=1)
    X_blobs = blobs[0]
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_blobs)

    x_min, x_max = X_blobs[:, 0].min() - 0.5, X_blobs[:, 0].max() + 0.5
    y_min, y_max = X_blobs[:, 1].min() - 0.5, X_blobs[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               aspect='auto', origin='lower')
    plt.plot(X_blobs[:, 0], X_blobs[:, 1], 'r.', markersize=5)
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=3, color='b', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def cluster():
    blobs = make_blobs(random_state=1, centers=1)
    X_blobs = blobs[0]
    # 凝聚聚类
    linkage = ward(X_blobs)
    dendrogram(linkage)
    ax = plt.gca()
    plt.xlabel('sample index')
    plt.ylabel('cluster distance')
    plt.show()


def dbscan():
    blobs = make_blobs(random_state=1, centers=1)
    X_blobs = blobs[0]
    db = DBSCAN(eps=1, min_samples=10)
    clusters = db.fit_predict(X_blobs)
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=clusters, s=60, edgecolors='k')
    plt.xlabel('feature 0')
    plt.ylabel('feature 1')
    plt.show()


dbscan()