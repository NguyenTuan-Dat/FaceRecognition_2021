import os
import random

PATH_TO_DATA = "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/VN-celeb-resize1/Train/"
OUTPUT = "./train.txt"

pairs = []
list_name_img = os.listdir(PATH_TO_DATA)
list_same_person_imgs = []

for i in range(1, 101):
    same_person_imgs = [name_img for name_img in list_name_img if name_img.split("_")[0] == str(i)]
    list_same_person_imgs.append(same_person_imgs)

print(list_same_person_imgs)

with open(OUTPUT, "w") as f:
    for i in range(2500):
        person_1 = random.randint(0, (len(list_same_person_imgs) - 1))
        person_2 = random.randint(0, (len(list_same_person_imgs) - 1))
        while person_2 == person_1:
            person_2 = random.randint(0, (len(list_same_person_imgs) - 1))

        img1_person_1 = random.randint(0, (len(list_same_person_imgs[person_1]) - 1))
        img2_person_1 = random.randint(0, (len(list_same_person_imgs[person_1]) - 1))
        print("len[person1]: {}, rand_1: {}, rand_2: {}".format(len(list_same_person_imgs[person_1]), img1_person_1,
                                                                img2_person_1))
        while img1_person_1 == img2_person_1:
            img2_person_1 = random.randint(0, len(list_same_person_imgs[person_1]) - 1)

        f.write(list_same_person_imgs[person_1][img1_person_1]
                + " " + list_same_person_imgs[person_1][img2_person_1]
                + " " + "1\n")

        img3_person_1 = random.randint(0, len(list_same_person_imgs[person_1]) - 1)
        img1_person_2 = random.randint(0, len(list_same_person_imgs[person_2]) - 1)
        f.write(list_same_person_imgs[person_1][img3_person_1]
                + " " + list_same_person_imgs[person_2][img1_person_2]
                + " " + "0\n")
