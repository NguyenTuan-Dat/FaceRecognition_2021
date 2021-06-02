import os
import cv2

PATH_TO_DATA = "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/VN-celeb-resize1/Train/"
OUTPUT = "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/VN-celeb-resize1/Train-folder/"

os.mkdir(OUTPUT)
list_imgs = os.listdir(PATH_TO_DATA)

for i in range(1, 801):
    list_img_of_person = [name for name in list_imgs if name.split("_")[0] == str(i)]
    os.mkdir(os.path.join(OUTPUT, str(i)))
    for img_name in list_img_of_person:
        img = cv2.imread(os.path.join(PATH_TO_DATA, img_name))

        cv2.imwrite(os.path.join(OUTPUT, str(i), img_name), img)
