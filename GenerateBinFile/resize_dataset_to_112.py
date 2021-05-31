import cv2
import os

PATH_TO_FOLDER = "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/VN-celeb/"
OUTPUT_FOLDER = "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/VN-celeb-resize1/"
SIZE = (112, 112)

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

folders = os.listdir(PATH_TO_FOLDER)

for folder in folders:
    try:
        folder_output_path = OUTPUT_FOLDER
        if not os.path.exists(folder_output_path):
            os.mkdir(folder_output_path)

        folder_path = os.path.join(PATH_TO_FOLDER, folder)
        img_names = os.listdir(folder_path)
        count = 0
        for img_name in img_names:
            img = cv2.imread(os.path.join(folder_path, img_name))
            img = cv2.resize(img, SIZE)

            name_img = folder + "_"
            for i in range(4 - len(str(count))):
                name_img += "0"
            name_img += str(count) + ".jpg"
            cv2.imwrite(os.path.join(folder_output_path, name_img), img)
            count += 1
            print("resized img: {}".format(img_name))
    except Exception as ex:
        print(ex)
