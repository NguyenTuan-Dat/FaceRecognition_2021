import numpy as np
import cv2
import matplotlib.pyplot as plt

SIZE_W = 500

cap = cv2.VideoCapture(
    '../input/data-nckh-facerecognition/102180155.h264')

while(cap.isOpened()):
    ret, frame = cap.read()
    height, width, chanel = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (SIZE_W, int(height/width*SIZE_W)))
    print((int(width/height*SIZE_W), SIZE_W))

    # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plt.imshow(frame)
    plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
