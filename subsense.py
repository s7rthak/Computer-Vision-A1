import numpy as np
import cv2
import sys
import os
from multiprocessing import Pool

def intensity(frame, x, y):
    return 0.0722 * frame[x, y, 0] + 0.7152 * frame[x, y, 1] + 0.2126 * frame[x, y, 2]

def intensity_vec(frame):
    return 0.0722 * frame[:, :, 0] + 0.7152 * frame[:, :, 1] + 0.2126 * frame[:, :, 2]

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

    pool = Pool(os.cpu_count())
    LBSP = np.array(pool.starmap(lbsp_per_pixel, [(frame, threshold, i, j) for i in range(h) for j in range(w)])).reshape((h,w))
    I = intensity_vec(frame)

    return LBSP, I

def f(x):
    return bin(x).count("1")

def hamming_distance(v1, v2):
    return f(v1 ^ v2)

def hamming_distance_vec(X, Y):
    return np.vectorize(f)(X ^ Y)

import random

class Subsense:
    def __init__(self, v_init, r_init, t_init, n_min, learn_rate, n_models=50):
        self.background_models = []
        self.n_models = n_models
        self.r_color_0 = 30
        self.r_lbsp_0 = 3

        self.prev_map = None
        self.d_min = 0.5
        self.v = None
        self.r = None
        self.t = None

        # Store the initial values
        self.v_init = v_init
        self.r_init = r_init
        self.t_init = t_init

        self.n_min = n_min
        self.learn_rate = learn_rate
        
    def apply(self, img):
        h, w, c = img.shape

        # If it is the first image frame, then initialize parameters
        if self.prev_map is None: 
            self.h = h
            self.w = w
            self.c = c

            self.prev_map = np.full((self.h, self.w), 0)
            self.v = np.full((self.h,self.w), self.v_init*1.0)
            self.r = np.full((self.h,self.w), self.r_init*1.0)
            self.t = np.full((self.h,self.w), self.t_init*1.0)

        # Initialize d-values
        d_min = np.full((self.h, self.w), sys.maxsize*1.0)
        d_max = np.full((self.h, self.w), 0.0)

        # Set r-values
        r_color = self.r * self.r_color_0
        r_lbsp = np.power(np.full((self.h, self.w), 2.0), self.r) + self.r_lbsp_0
        print(self.r)

        self.curr_segmentation = np.full((self.h, self.w), 1)
        curr_lbsp, curr_intensity = compute_lbsp(img, 0.3)
        
        # Until we don't have enough background models, append to our models collection.
        if len(self.background_models) < self.n_models:
            self.background_models.append((curr_lbsp, curr_intensity))

        # Find the differences in color-lbsp of current frame and our present background models.
        matches = np.zeros((self.h, self.w), dtype=int)
        for i in range(len(self.background_models)):
            lbsp_diff = hamming_distance_vec(self.background_models[i][0], curr_lbsp)
            intensity_diff = np.absolute(self.background_models[i][1] - curr_intensity)
            d_max = np.maximum(d_max, intensity_diff)
            d_min = np.minimum(d_min, intensity_diff)

            fg_mask = np.logical_or(np.greater_equal(lbsp_diff, r_lbsp), np.greater_equal(intensity_diff, r_color))
            curr_match = np.where(fg_mask, 1, 0)
            matches = matches + curr_match

        # print(matches)

        # For all pixel positions, if number of matches is greater than n_min, then declare as background else it is foreground.
        self.curr_segmentation = np.where(matches < self.n_min, 0, 1)

        # Decide the pixels where we want to update models.
        update = np.random.binomial(1, 1/(self.t+1e-6))
        update_mask = np.logical_and(update == 1, self.curr_segmentation == 0)

        update_model = np.full((self.h, self.w), -1)
        update_model[update_mask] = random.randint(0, len(self.background_models) - 1)

        update_pixel_coordinates = np.argwhere(update_model != -1)
        for i in range(len(update_pixel_coordinates)):
            x, y = update_pixel_coordinates[i][0], update_pixel_coordinates[i][1]
            k = update_model[x, y]
            self.background_models[k][0][x, y], self.background_models[k][1][x, y] = curr_lbsp[x, y], curr_intensity[x, y]
                
        #Now for updates
        self.d_min = (1 - self.learn_rate) * self.d_min + self.learn_rate * d_min/(d_max+1e-6)
        
        self.xor_store = self.curr_segmentation ^ self.prev_map
        self.prev_map = self.curr_segmentation
        
        mask = self.xor_store == 1
        self.v[mask] = self.v[mask] + 1
        self.v[~mask] = self.v[~mask] - 0.1
        self.v = np.maximum(0.0, self.v)

        mask = self.r >= (1 + self.d_min*2) ** 2
        self.r[mask] = self.r[mask] + self.v[mask]
        self.r[~mask] = self.r[~mask] - (1/(self.v[~mask]+1e-6))
        self.r = np.maximum(1.0, self.r)
        
        mask = self.curr_segmentation == 1
        self.t[mask] = self.t[mask] + 1/(self.v[mask]*self.d_min[mask]+1e-6)
        self.t[~mask] = self.t[~mask] - (self.v[~mask]/(self.d_min[~mask]+1e-6))
        self.t = np.maximum(2.0, self.t)
        self.t = np.minimum(256.0, self.t)


        return 255 * self.curr_segmentation
        