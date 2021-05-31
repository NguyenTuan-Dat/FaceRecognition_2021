import pickle
import mxnet as mx
import os

PATH_TO_DATA = ""
PATH_TO_PAIRS_FILE = ""
OUTPUT_PATH = "./data.bin"


def get_pairs_file(path_to_data, path_to_pairs_file):
    """
    pairs file's format (with 0 is other person and 1 is one person):

    name_img1 name_img2 0/1
    name_img3 name_img4 0/1
    name_img5 name_img6 0/1
    name_imgx name_imgy 0/1
    name_imgz name_imgt 0/1
    """

    path_to_imgs = []
    is_sames = []

    pairs = open(path_to_pairs_file, 'r').read().split('\n')

    for pair in pairs:
        if pair == "":
            break
        img1_name, img2_name, is_same = pair.split(" ")
        path_to_img1 = os.path.join(path_to_data, img1_name)
        path_to_img2 = os.path.join(path_to_data, img2_name)
        assert os.path.exists(path_to_img1)
        assert os.path.exists(path_to_img2)

        path_to_imgs.append(path_to_img1)
        path_to_imgs.append(path_to_img2)

        if is_same == "1":
            is_sames.append(True)
        if is_same == "0":
            is_sames.append(False)

    return path_to_imgs, is_sames


def create_bin_files(path_to_data, path_to_pairs_file):
    path_to_imgs, is_sames = get_pairs_file(path_to_data, path_to_pairs_file)

    imgs = []
    i = 0
    for path_to_img in path_to_imgs:
        with open(path_to_img, 'rb') as fin:
            img = fin.read()
            imgs.append(img)
            i += 1
            if i % 1000 == 0:
                print('loading ', i)
    with open(OUTPUT_PATH, "wb") as output_file:
        pickle.dump((imgs, is_sames), output_file, protocol=pickle.HIGHEST_PROTOCOL)


create_bin_files(
    "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/VN-celeb-resize1/Train/",
    "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/Code/GenerateBinFile/train.txt")
