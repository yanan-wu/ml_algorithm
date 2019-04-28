# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF


def get_face_data():
    # 载入人脸数据集
    faces = fetch_lfw_people(min_faces_per_person=20, resize=0.8)
    image_shape = faces.images[0].shape
    # 打印照片
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), subplot_kw={'xticks': (), 'yticks': ()})
    for target, image, ax in zip(faces.target, faces.images, axes.ravel()):
        ax.imshow(image, cmap=plt.cm.get_cmap('gray'))
        ax.set_title(faces.target_names[target])
    plt.show()

    # 建立模型
    print('-----------')
    X_train, X_test, y_train, y_test = train_test_split(faces.data/255, faces.target, random_state=62)
    mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=62, max_iter=400)
    mlp.fit(X_train, y_train)
    print('模型识别准确率：{:.2f}'.format(mlp.score(X_test, y_test)))


def pca_whiten():
    # 载入人脸数据集
    faces = fetch_lfw_people(min_faces_per_person=20, resize=0.8)
    X_train, X_test, y_train, y_test = train_test_split(faces.data / 255, faces.target, random_state=62)
    # pca 的白化功能
    pca = PCA(whiten=True, n_components=0.9, random_state=62).fit(X_train)
    X_train_whiten = pca.transform(X_train)
    X_test_whiten = pca.transform(X_test)
    print('白化后的数据形态：{}'.format(X_train_whiten.shape))
    mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=62, max_iter=400)
    mlp.fit(X_train_whiten, y_train)
    print('模型识别准确率：{:.2f}'.format(mlp.score(X_test_whiten, y_test)))


def nmf():
    # 载入人脸数据集
    faces = fetch_lfw_people(min_faces_per_person=20, resize=0.8)
    X_train, X_test, y_train, y_test = train_test_split(faces.data / 255, faces.target, random_state=62)
    # 非负矩阵特征提取
    nmf = NMF(n_components=105, random_state=62).fit(X_train)
    X_train_nmf = nmf.transform(X_train)
    X_test_mnf = nmf.transform(X_test)
    print('NMF 处理后的数据形态：{}'.format(X_train_nmf.shape))
    mlp = MLPClassifier(hidden_layer_sizes=[100, 100], random_state=62, max_iter=400)
    mlp.fit(X_train_nmf, y_train)
    print('模型识别准确率：{:.2f}'.format(mlp.score(X_test_mnf, y_test)))


pca_whiten()