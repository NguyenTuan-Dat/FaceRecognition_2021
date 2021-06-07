import torch
import torch.nn as nn
import sys
from vit_pytorch import ViT_face
from vit_pytorch import ViTs_face
from util.utils import get_val_data, perform_val
from IPython import embed
import sklearn
import cv2
import numpy as np
from image_iter import FaceDataset
import torch.utils.data as data
import argparse
import os

MULTI_GPU = False
DEVICE = torch.device("cuda:0")

haar_cascade = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    similarity = np.dot(a, b.T) / (a_norm * b_norm)

    return similarity


def l2_distance(a, b):
    diff = np.subtract(a, b)
    dist = np.sum(np.square(diff))
    print("dist: ", dist)
    return dist


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='', help='training set directory')
    parser.add_argument('--test_dir', default='', help='test set directory')
    parser.add_argument('--network', default='VITs',
                        help='training set directory')
    parser.add_argument('--target', default='lfw,talfw,sllfw,calfw,cplfw,cfp_fp,agedb_30',
                        help='')
    parser.add_argument('--batch_size', type=int, help='', default=20)
    parser.add_argument('--database_dir', type=str, default="/content/embeddings/")
    return parser.parse_args(argv)


def main(args):
    if args.network == 'VIT':
        model = ViT_face(
            image_size=112,
            patch_size=8,
            loss_type='CosFace',
            GPU_ID=DEVICE,
            num_class=93431,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.network == 'VITs':
        model = ViTs_face(
            loss_type='CosFace',
            GPU_ID=DEVICE,
            num_class=93431,
            image_size=112,
            patch_size=8,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    model_root = args.model
    model.load_state_dict(torch.load(model_root))

    model.to(DEVICE)
    model.eval()

    database, name_ids = load_database(args.database_dir)

    print("len database: {}, len name_ids: {}".format(len(database), len(name_ids)))

    print("=======================embedding database done!!!")

    img_test = cv2.imread(args.target)
    unknows, faces = embedding(img_test, model)
    print("len unknows: {}, len faces: {}".format(len(unknows), faces.shape))
    find_person(database, name_ids, unknows, faces)


def embedding(img, model):
    embed = None
    img_cropped_faces = None
    with torch.no_grad():
        try:
            cropped_faces = haar_cascade.detectMultiScale(img, 1.3, 5)
            if cropped_faces is not None:
                img_cropped_faces = []
                for (x, y, w, h) in cropped_faces:
                    if w > h:
                        h = w
                    else:
                        w = h
                    cropped_face = img[y: y + h, x: x + w]
                    cropped_face = cv2.resize(cropped_face, (112, 112))
                    img_cropped_faces.append(cropped_face)
                img_cropped_faces = np.array(img_cropped_faces)
                tensor_cropped_faces = torch.from_numpy(np.transpose(img_cropped_faces, (0, 3, 1, 2)))
                tensor_cropped_faces = tensor_cropped_faces.type(torch.float32)
                embed = model(tensor_cropped_faces.to(DEVICE)).cpu()
                print(embed.shape)
                return embed, img_cropped_faces
        except Exception as ex:
            print(ex)
    return embed, img_cropped_faces


def load_database(path_to_folder):
    emb_names = os.listdir(path_to_folder)
    embs = []
    for emb_name in emb_names:
        emb = np.load(os.path.join(path_to_folder, emb_name))
        embs.append(emb)

    return embs, emb_names


def find_person(database, name_ids, unknows, face_unknows):
    for i in range(len(unknows)):
        unknow = unknows[i]
        unknow = unknow.unsqueeze(0)
        face = face_unknows[i]
        if face is not None and unknow is not None:
            cv2.imwrite("/content/output/" + str(i) + ".jpg", face)
            print(str(i) + ".jpg")
            distancies = list()
            for idx in range(len(database)):
                if database[idx] is None:
                    continue
                person = database[idx][0].unsqueeze(0)
                loss = torch.tensor([100])
                # print("person shape: {}, unknow shape: {}".format(person.shape, unknow.shape))
                if person is not None:
                    loss = cosine_distance(unknow, person)
                    # print(loss.shape)
                    # print("{}, {:>30}: {}".format(idx, name_ids[idx], loss))
                # print("loss: {}".format(loss))
                distancies.append(np.absolute(np.min(loss)))
            distancies = np.array(distancies)
            argmin = np.argmin(distancies)
            min = np.min(distancies)
            print("min: {}, argmin: {}, name: {}".format(min, argmin, name_ids[argmin]))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
