import torch
import numpy as np
import os
import matplotlib.pyplot as plt

THRESHOLD = 0.001


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    similarity = np.dot(a, b.T) / (a_norm * b_norm)

    return similarity


def angular_distance(a, b):
    cos = cosine_distance(a, b)
    return 1 - (np.arccos(cos) / np.pi)


def get_distribution_folder(parrent_folder, sub_folder):
    folder_names = os.listdir(parrent_folder)

    rand = np.random.randint(len(folder_names))
    while folder_names[rand] == sub_folder:
        rand = np.random.randint(len(folder_names))

    list_npy = os.listdir(os.path.join(parrent_folder, sub_folder))
    list_npy_negative = os.listdir(os.path.join(parrent_folder, folder_names[rand]))

    min_len = len(list_npy_negative) if len(list_npy) > len(list_npy_negative) else len(list_npy)

    positive = []
    negative = []

    for npy in range(1000):
        idx = np.random.randint(min_len)
        x1 = np.load(os.path.join(parrent_folder, sub_folder, list_npy[idx]))
        idx = np.random.randint(min_len)
        x2 = np.load(os.path.join(parrent_folder, sub_folder, list_npy[idx]))
        x3 = np.load(os.path.join(parrent_folder, folder_names[rand], list_npy_negative[idx]))

        positive.append(angular_distance(x1, x2))
        negative.append(angular_distance(x1, x3))

    return np.array(positive), np.array(negative)


def round_plot(value, threshold):
    x = np.arange(0, 1.0001, threshold)
    y = np.zeros(x.shape)
    for i in value:
        idx = round(i, 2) / threshold
        if idx != idx:
            continue
        y[int(idx)] += 1

    return x, y


def draw_plot(parrent_folder, sub_folder):
    positive, negative = get_distribution_folder(parrent_folder, sub_folder)
    x_positive, y_positive = round_plot(positive, THRESHOLD)
    print(x_positive)
    plt.bar(x_positive, y_positive, width=0.008)
    plt.show()


draw_plot("/Users/ntdat/Downloads/face_features/", "/Users/ntdat/Downloads/face_features/3")
