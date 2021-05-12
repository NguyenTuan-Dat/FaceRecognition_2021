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
from facenet_pytorch import MTCNN
from google.colab.patches import cv2_imshow

MULTI_GPU = False
DEVICE = torch.device("cuda:0")

mtcnn = MTCNN(image_size=112, margin=0, keep_all=True, post_process=False, device='cuda:0')
l2 = torch.nn.MSELoss()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='', help='training set directory')
    parser.add_argument('--test_dir', default='', help='test set directory')
    parser.add_argument('--network', default='VITs',
                        help='training set directory')
    parser.add_argument('--target', default='lfw,talfw,sllfw,calfw,cplfw,cfp_fp,agedb_30',
                        help='')
    parser.add_argument('--batch_size', type=int, help='', default=20)
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

    database, name_ids = embedding_database(args.test_dir, model)
    unknows, faces = embedding(args.target, model)
    find_person(database, name_ids, unknows, faces)


def embedding(img, model):
    embed = None
    img_cropped_faces = None
    with torch.no_grad():
        try:
            cropped_faces = mtcnn(img)
            img_cropped_faces = np.transpose(cropped_faces, (0, 2, 3, 1))
            if cropped_faces is not None:
                embed = model(cropped_faces.to(DEVICE)).cpu()
                return embed, img_cropped_faces
        except Exception as ex:
            print(ex)
    return embed, img_cropped_faces


def embedding_database(path_to_dirs, model):
    list_dir = os.listdir(path_to_dirs)

    embeds = list()
    name_ids = list()

    for dir in list_dir:
        path_to_dir = path_to_dirs + dir + "/"
        img_names = os.listdir(path_to_dir)
        for img_name in img_names:
            img = cv2.imread(path_to_dir + img_name)
            embed, _ = embedding(img, model)
            embeds.append(embed)
            name_ids.append(dir + "/" + img_name)

            print("{}: {}".format(dir + "/" + img_name, embed))

    return embeds, name_ids


def find_person(database, name_ids, unknows, face_unknows):
    for i in range(len(unknows)):
        unknow = unknows[i]
        face = face_unknows[i]
        cv2_imshow(face)
        distancies = list()
        for person in database:
            loss = l2(unknow, person)
            print(loss)
            distancies.append(loss)
        distancies = np.array(distancies)
        argmin = np.argmin(distancies)
        print(name_ids[argmin])


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
