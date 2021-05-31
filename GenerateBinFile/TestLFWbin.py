import os
import pickle
import cv2
import mxnet as mx
import numpy as np
import torch


def get_val_pair(path, name):
    ver_path = os.path.join(path, name + ".bin")
    print(ver_path)
    assert os.path.exists(ver_path)
    data_set, issame = load_bin(ver_path)
    print('ver', name)
    return data_set, issame


def load_bin(path, image_size=[112, 112]):
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_list = []
    for flip in [0, 1]:
        data = torch.zeros((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)

        # test show img
        # print(img.asnumpy().shape)
        # cv2.imshow("ALoAlo", img.asnumpy())
        # key = cv2.waitKey()
        # if key == ord('q'):
        #     return

        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = mx.nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = torch.tensor(img.asnumpy())
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


data = get_val_pair("/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/Code/GenerateBinFile", "data")
print(len(data[1]))
