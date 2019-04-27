# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt


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


get_face_data()