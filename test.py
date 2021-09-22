import vibe
import updated_flow
import cv2
import numpy as np

BASELINE_FRAMES = 1700
input_path = "COL780-A1-Data/illumination/input/"
kernel = np.ones((3,3),np.uint8)

def pre_process(frame):
    fr = cv2.medianBlur(frame, 3)
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

    return fr

def post_process(frame, th = 10):
    # fr = cv2.medianBlur(frame, 3)
    ret, fr = cv2.threshold(frame, th, 255, cv2.THRESH_BINARY)
    fr = cv2.dilate(fr, kernel)
    fr = cv2.morphologyEx(fr, cv2.MORPH_OPEN, kernel)
    fr = cv2.morphologyEx(fr, cv2.MORPH_CLOSE, kernel)

    return fr

bs = vibe.Vibe_Subsense()
bs_knn = cv2.createBackgroundSubtractorMOG2()

frame1 = cv2.imread(input_path + "in" + str(1).zfill(6) + ".jpg")
frame1 = cv2.resize(frame1, (320,240))
prvs = pre_process(frame1)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

for i in range(1, BASELINE_FRAMES+1):
    print(i)
    frame = cv2.imread(input_path + "in" + str(i).zfill(6) + ".jpg")
    frame = cv2.resize(frame, (320,240))
    frame = pre_process(frame)

    mask = bs_knn.apply(frame)
    mask = post_process(mask, th=10)

    flow = cv2.calcOpticalFlowFarneback(prvs, frame, None, 0.5, 10, 20, 3, 7, 1.5, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgr = post_process(bgr)

    bs.apply(frame, bgr)

    res = bs.get()
    if not res is None:
        # res = post_process(res, 20)
        cv2.imshow("image", frame)
        cv2.imshow("fg", mask)
        cv2.imshow("flow", bgr)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break