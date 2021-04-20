import numpy as np
import cv2
from imgrender import render

SIZE_W = 1000

cap = cv2.VideoCapture(
    '/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/Data/Test_0_20210413074108.h264')

while(cap.isOpened()):
    ret, frame = cap.read()
    height, width, chanel = frame.shape
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.resize(frame, (SIZE_W, int(height/width*SIZE_W)))
    print((int(width/height*SIZE_W), SIZE_W))

    render(frame, scale=(40, 60))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
