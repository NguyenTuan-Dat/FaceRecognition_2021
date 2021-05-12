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

MULTI_GPU = False
DEVICE = torch.device("cuda:0")

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
    if args.network == 'VIT' :
        model = ViT_face(
            image_size=112,
            patch_size=8,
            loss_type='CosFace',
            GPU_ID= DEVICE,
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

    with torch.no_grad():
        mtcnn = MTCNN(margin=0, keep_all=True, post_process=False, device='cuda:0')

        img_names = os.listdir(args.test_dir)
        for img_name in img_names:
            img = cv2.imread(args.test_dir + img_name)
            cropped_faces = mtcnn(img)
            embedding = model(cropped_faces)
            print(embedding)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))