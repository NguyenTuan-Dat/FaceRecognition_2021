import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='GenerateData')

parser.add_argument("-v", "--video_input", type=str,
                    default='/content/Data/Test_0_20210413073736.h264')
parser.add_argument("-o", "--output_folder", type=str,
                    default='/content/image_raw/')

args = parser.parse_args()

print(*args.output_folder)

if not os.path.exists(*args.output_folder):
    os.mkdir(*args.output_folder)

video_input = cv2.VideoCapture(*args.video_input)
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imwrite(*args.output_folder + "image_raw_" +
                str(count) + ".jpg", frame)
    count += 1

cap.release()
