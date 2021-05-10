import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import sleep
import os

PATH_TO_DIR = "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/Data/"
SIZE_W = 500

list_dir = os.listdir(PATH_TO_DIR)

for video in list_dir:
    if "Test" in video:
        continue

    video_name = video.split(".")[0]

    if os.path.exists(PATH_TO_DIR + video_name) == False:
        os.mkdir(PATH_TO_DIR + video_name)

    count = 0
    count_frame = 0

    cap = cv2.VideoCapture(PATH_TO_DIR + video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, channel = frame.shape

        if count_frame % 20 == 0:
            print("./" + video_name + "/image_" +
                  str(count) + ".jpg")
            cv2.imwrite(PATH_TO_DIR + video_name + "/image_" +
                        str(count) + ".jpg", frame)
            count += 1
        count_frame += 1

        # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # plt.imshow(frame)
        # plt.show()
        # sleep(0.1)
        if 0xFF == ord('q'):
            break

    cap.release()
