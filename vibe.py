import numpy as np
import cv2
import random
import time
from multiprocessing import Pool
import os
import itertools


from numpy.core.fromnumeric import size

def intensity(frame, x, y):
    # return 0.0722 * frame[x, y, 0] + 0.7152 * frame[x, y, 1] + 0.2126 * frame[x, y, 2]
    return float(frame[x, y])

def lbsp_per_pixel(frame, x, y, threshold=0.3):
    h, w = frame.shape

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

def compute_lbsp(frame, threshold=0.3):
    h, w = frame.shape

    LBSP = np.zeros((h, w), dtype=int)

    pool = Pool(os.cpu_count())
    LBSP = np.array(pool.starmap(lbsp_per_pixel, [(frame, i, j, threshold) for i in range(h) for j in range(w)])).reshape((h,w))

    return LBSP

def f(x):
    return bin(x).count("1")

def hamming_distance_vec(X, Y):
    return np.vectorize(f)(X ^ Y)

all_moves = list(itertools.product([-1,0,1], repeat=2))
all_moves.remove((0,0))
two_moves = list(itertools.product([-2,0,2], repeat=2))
two_moves.remove((0,0))
lbsp_moves = all_moves + two_moves

def random_neighbour(x, y, h, w):
    choices =  np.array((x, y)) + np.array(all_moves)
    choices = choices[np.logical_and(choices[:, 0] >= 0, choices[:, 0] < h)]
    choices = choices[np.logical_and(choices[:, 1] >= 0, choices[:, 1] < w)]

    return random.choice(choices)

def update_lbsp(x, y, Background_LBSP_Models, Background_Models, n, h, w):
    cur_bg_lbsp_model = Background_LBSP_Models[:,:,n]
    cur_bg_model = Background_Models[:,:,n]

    choices =  np.array((x, y)) + np.array(lbsp_moves)
    choices = choices[np.logical_and(choices[:, 0] >= 0, choices[:, 0] < h)]
    choices = choices[np.logical_and(choices[:, 1] >= 0, choices[:, 1] < w)]

    for i in range(choices.shape[0]):
        cur_bg_lbsp_model[choices[i][0], choices[i][1]] = lbsp_per_pixel(cur_bg_model, choices[i][0], choices[i][1])

    return cur_bg_lbsp_model


class Vibe_Subsense:
    def __init__(self, R=30, rand_samples=16, bootstrap=35, n_min=2, use_subsense=False):
        self.height = None
        self.width = None
        self.got_first_frame = False
        self.ready_to_go = False
        self.n_min = n_min
        self.rand_samples = rand_samples
        self.frame_number = 0
        self.bootstrap_frames = bootstrap
        self.R = R
        self.subsense = use_subsense
    
    def number_plus_noise(self, number):
        number = number + random.randint(-10, 10)
        if number > 255:
            number = 255
        if number < 0:
            number = 0
        return np.uint8(number)

    def noise_to_image(self, frame):
        noisy = frame.astype(np.int16)
        noisy = frame + np.random.randint(-10, 11, size=(self.height, self.width))
        noisy[noisy > 255] = 255
        noisy[noisy < 0] = 0
        noisy = noisy.astype(np.uint8)

        return noisy

    def random_neighbour(self, x, y):
        choices =  np.array((x, y)) + np.array(all_moves)
        choices = choices[np.logical_and(choices[:, 0] >= 0, choices[:, 0] < self.height)]
        choices = choices[np.logical_and(choices[:, 1] >= 0, choices[:, 1] < self.width)]

        return random.choice(choices)

    def apply(self, frame, flow_bg=None):
        if not self.got_first_frame:
            self.height, self.width = frame.shape[0], frame.shape[1]
            self.got_first_frame = True
            self.N = self.bootstrap_frames
            self.R = np.ones((self.height, self.width, self.N), dtype=np.uint8) * self.R
            self.S = np.ones((self.height, self.width, self.N), dtype=np.uint8) * 5

            self.result_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            self.bg_models = np.zeros((self.height, self.width, self.N), dtype=np.uint8)
            self.bg_models_lbsp = np.zeros((self.height, self.width, self.N), dtype=int)
            self.compare_matrix = np.zeros((self.height, self.width, self.N), dtype=np.uint8) 

        if self.frame_number < self.N:
            self.bg_models[:,:,self.frame_number] = self.noise_to_image(frame)
            if self.subsense:
                self.bg_models_lbsp[:,:,self.frame_number] = compute_lbsp(self.bg_models[:,:,self.frame_number])
        
        if self.subsense:
            self.lbsp = compute_lbsp(frame)
            self.lbsp_3d = np.tile(self.lbsp[...,None],self.N)

        frame_3d = np.tile(frame[...,None],self.N)
        if self.subsense:
            self.compare_matrix = np.logical_and(np.abs(self.bg_models - frame_3d) < self.R, hamming_distance_vec(self.bg_models_lbsp, self.lbsp_3d) < self.S)
        else:
            self.compare_matrix = np.abs(self.bg_models - frame_3d) < self.R

        curr_match = np.sum(np.where(self.compare_matrix, 1, 0), axis=2)
        self.result_mask = np.where(curr_match >= 2, 0, 255)
        self.result_mask = self.result_mask.astype(np.uint8)

        if flow_bg is None: 
            self.update_mask = self.result_mask
        else:
            self.update_mask = flow_bg

        replace_pixel = np.random.randint(0, self.rand_samples, size=(self.height, self.width))
        replace_neighbour_pixel = np.random.randint(0, self.rand_samples, size=(self.height, self.width))

        replace_mask = np.logical_and(replace_pixel == 0, self.update_mask == 0)
        replace_neighbour_mask = np.logical_and(replace_neighbour_pixel == 0, self.update_mask == 0)

        self.bg_models[replace_mask, random.randint(0, self.N-1)] = frame[replace_mask]
        (x_indices, y_indices) = np.where(replace_neighbour_mask)
        for i in range(len(x_indices)):
            x_n, y_n = random_neighbour(x_indices[i],y_indices[i],self.height,self.width)
            self.bg_models[x_n,y_n,random.randint(0, self.N-1)] = frame[x_indices[i], y_indices[i]]

        if self.subsense:
            for x in range(0, self.height):
                for y in range(0, self.width):
                    if self.update_mask[x, y] == 0:
                        rand = random.randint(0, self.rand_samples-1)
                        if rand == 0:
                            rand_model = random.randint(0, self.N-1)
                            self.bg_models_lbsp[:,:,rand_model] = update_lbsp(x,y,self.bg_models_lbsp,self.bg_models,rand_model,self.height,self.width)

                        rand = random.randint(0, self.rand_samples-1)
                        if rand == 0:
                            rand_model = random.randint(0, self.N-1)
                            self.bg_models_lbsp[:,:,rand_model] = update_lbsp(x_n,y_n,self.bg_models_lbsp,self.bg_models,rand_model,self.height,self.width)
        
        #self.result_mask = cv2.medianBlur(self.result_mask, 7)
        self.frame_number += 1
        if self.frame_number > self.bootstrap_frames:
            self.ready_to_go = True

    def get(self,mode):
        if mode == 'create':
            return None
        if self.ready_to_go:
            return self.result_mask
        return None

# Path to video for processing
