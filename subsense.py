import numpy as np
import cv2

def intensity(frame, x, y):
    return 0.0722 * frame[x, y, 0] + 0.7152 * frame[x, y, 1] + 0.2126 * frame[x, y, 2]

def lbsp_per_pixel(frame, threshold, x, y):
    h, w, c = frame.shape

    value = 0

    p = 0
    cur_intensity = intensity(frame, x, y)

    if x-2>=0 and y-2>=0 and abs(cur_intensity - intensity(frame, x-2, y-2)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x-2>=0 and y>=0 and abs(cur_intensity - intensity(frame, x-2, y)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x-2>=0 and y+2<w and abs(cur_intensity - intensity(frame, x-2, y+2)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1

    if x-1>=0 and y-1>=0 and abs(cur_intensity - intensity(frame, x-1, y-1)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x-1>=0 and y>=0 and abs(cur_intensity - intensity(frame, x-1, y)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x-1>=0 and y+1<w and abs(cur_intensity - intensity(frame, x-1, y+1)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1

    if x>=0 and y-2>=0 and abs(cur_intensity - intensity(frame, x, y-2)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x>=0 and y-1>=0 and abs(cur_intensity - intensity(frame, x, y-1)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x>=0 and y+1<w and abs(cur_intensity - intensity(frame, x, y+1)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x>=0 and y+2<w and abs(cur_intensity - intensity(frame, x, y+2)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1

    if x+1<h and y-1>=0 and abs(cur_intensity - intensity(frame, x+1, y-1)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x+1<h and y>=0 and abs(cur_intensity - intensity(frame, x+1, y)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x+1<h and y+1<w and abs(cur_intensity - intensity(frame, x+1, y+1)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1

    if x+2<h and y-2>=0 and abs(cur_intensity - intensity(frame, x+2, y-2)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x+2<h and y>=0 and abs(cur_intensity - intensity(frame, x+2, y)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1
    if x+2<h and y+2<h and abs(cur_intensity - intensity(frame, x+2, y+2)) <= threshold * cur_intensity:
        value += (1 << p)
    p += 1

    return value

def compute_lbsp(frame, threshold):
    h, w, c = frame.shape

    LBSP = np.zeros((h, w), dtype=int)
    I = np.zeros((h, w), dtype=int)

    for i in range(h):
        for j in range(w):
            LBSP[i, j] = lbsp_per_pixel(frame, threshold, i, j)
            I[i, j] = intensity(frame, i, j)

    return LBSP, I

def hamming_distance(v1, v2):
    return bin(v1 ^ v2).count("1")
