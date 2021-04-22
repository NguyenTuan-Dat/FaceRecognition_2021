import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import sleep
import os

SIZE_W = 500
count = 0

test_img = cv2.imread("../input/cityscapes-image-pairs/cityscapes_data/train/1.jpg")
plt.imshow(test_img)
plt.show()

cap = cv2.VideoCapture(
    "../input/data-nckh-facerecognition/102180162.h264")

while(cap.isOpened()):
    ret, frame = cap.read()
    height, width, channel = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (SIZE_W, int(height/width*SIZE_W)))
    print((int(width/height*SIZE_W), SIZE_W))
    cv2.imwrite("./image_" + str(count) + ".jpg", frame)
    count+=1

    # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plt.imshow(frame)
    plt.show()
    sleep(0.1)
    if 0xFF == ord('q'):
        break

cap.release()
