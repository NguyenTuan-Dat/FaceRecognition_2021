import cv2

img = cv2.imread("/Users/ntdat/Downloads/best-face-oil.png")
h, w, c = img.shape
print(h, w, c)
h_3 = int(h/3)
w_3 = int(w/3)
stride = int(h_3*0.7)
for i in range(5):
    for j in range(5):
        h_end = int(i*stride + h_3)
        w_end = int(j*stride + w_3)
        print("h_3: {}, w_3: {}, j:{}, stride: {}".format(h_3, w_3, j, stride))
        if h_end > h:
            print("aloalo")
            h_end = h
        if w_end > w:
            w_end = w
        print("{}, {}, {}, {}| h: {}, w: {}".format(i*stride, h_end,
                                                    j*stride, w_end, h_end-i*stride, w_end - j*stride))
        crop_img = img[int(i*stride):h_end, int(j*stride):w_end]
        cv2.imwrite("/Users/ntdat/Downloads/{}_{}.png".format(i, j), crop_img)
