import cv2
import os

PATH_TO_VIDEOS = "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/Data/"
OUPUT_FOLDER = "/Users/ntdat/Tài liệu/Nghiên cứu nhận dạng khuôn mặt/Data Id/"

video_names = os.listdir(PATH_TO_VIDEOS)
for video_name in video_names:
    if "Test" in video_name:
        continue
    path_to_video = PATH_TO_VIDEOS + video_name
    NAME_ID = video_name.split('.')[0]
    OUPUT_FOLDER_ID = OUPUT_FOLDER + NAME_ID + "/"

    if not os.path.exists(OUPUT_FOLDER_ID):
        os.mkdir(OUPUT_FOLDER_ID)

    video = cv2.VideoCapture(path_to_video)
    count = 0

    while(video.isOpened()):
        ret, frame = video.read()
        if frame is None:
            break
        w, h, c = frame.shape
        frame_show = cv2.resize(frame, (500, int(w/h*500)))
        cv2.imshow("AloAlo " + NAME_ID, frame_show)
        if cv2.waitKey() == ord('s'):
            cv2.imwrite(OUPUT_FOLDER_ID + NAME_ID + "_" + str(count) + ".jpg", frame)
            print("Save image: " + OUPUT_FOLDER_ID + NAME_ID + "_" + str(count) + ".jpg")
            count += 1

    video.release()