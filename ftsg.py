from main import BASELINE_FRAMES
import cv2
import numpy as np

BASELINE_FRAMES = 1700
input_path = "COL780-A1-Data/ptz/input/"
kernel = np.ones((5,5),np.uint8)

def pre_process(frame):
    fr = cv2.GaussianBlur(frame,(3,3),0)
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

    return fr

def post_process(frame, th = 20):
    fr = cv2.GaussianBlur(frame,(3,3),0)
    ret, fr = cv2.threshold(fr, th, 255, cv2.THRESH_BINARY)
    fr = cv2.morphologyEx(fr, cv2.MORPH_CLOSE, kernel)
    fr = cv2.morphologyEx(fr, cv2.MORPH_OPEN, kernel)

    return fr


backSubtractor = cv2.createBackgroundSubtractorKNN()

frame1 = cv2.imread(input_path + "in" + str(1).zfill(6) + ".jpg")
prvs = pre_process(frame1)
backSubtractor.apply(frame1)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

for i in range(1, BASELINE_FRAMES+1):
    frame2 = cv2.imread(input_path + "in" + str(i).zfill(6) + ".jpg")
    next = pre_process(frame2)

    fgmask = post_process(backSubtractor.apply(next))

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 7, 15, 3, 7, 1.5, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgr = np.bitwise_and(post_process(bgr), fgmask)

    cv2.imshow("bg", bgr)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
