import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

BASELINE_FRAMES = 1700
input_path = "illumination/input/"

int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)

class OpticalFlow(object):
    def __init__(self):
        self.min_max_speed = 20
        self.img_color_curr = None
        self.img_gray_curr = None
        self.img_color_prev = None
        self.cnt_img = 0

    def insert_image(self, img_color):
        self.cnt_img += 1
        self.img_color_prev = self.img_color_curr
        self.img_color_curr = img_color
        self.flow_magnitude = None

    def compute_optical_flow(self):
        if self.cnt_img == 1:
            return self.get_black_image(depth = 0)
        else:
            # Compute optical flow: vx and vy
            gray_curr = cv2.cvtColor(self.img_color_curr,cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.img_color_prev,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
            # Compute magnitude
            flow_vx=flow[...,0]
            flow_vy=flow[...,1]
            self.flow_magnitude = np.sqrt(flow_vx**2+flow_vy**2)
            
            return self.flow_magnitude

    def flow_to_image(self, flow_magnitude):
        flow_uint8 = (flow_magnitude*10).astype(np.uint8)
        flow_img = cv2.cvtColor(flow_uint8, cv2.COLOR_GRAY2BGR)
        return flow_img

    def get_black_image(self, depth):
        s = self.img_color_curr.shape
        if depth == 0:
            return np.zeros((s[0], s[1]))
        else:
            return np.zeros((s[0], s[1], depth))

    def get_mask_of_moving(self):
        if self.cnt_img == 1:
            return self.get_black_image(depth = 0)
        else:
            V = self.flow_magnitude
            mask =  V / max(self.min_max_speed, V.max())
            mask[mask>1]=1
            mask = mask**(0.5)
            return mask
            
        
    

class ModelBackground(object):
    def __init__(self, change_rate = 0.03):
        self.change_rate = change_rate
        self.img_color_bg = None
        self.cnt_img = 0
        

    def insert_image(self, img_color):
        self.cnt_img += 1
        if self.cnt_img == 1: # Init background model
            if 0: # init with black
                self.img_color_bg = np.zeros_like(img_color)
            else: # init with first image
                self.img_color_bg = img_color.copy()

        # Make background model similar to the current image
        self.img_color_bg = (1 - self.change_rate) * self.img_color_bg + \
                                self.change_rate * img_color

    def get_background_image(self):
        return self.img_color_bg.astype( np.uint8 )

    def get_mask_of_foreground(self, img_color_curr, min_max_diff = 200):
        I1 = img_color_curr.astype(np.float)
        I2 = self.img_color_bg.astype(np.float)
        d = I1 - I2
        mask = np.sqrt( d[..., 0]**2 + d[..., 1]**2 + d[..., 2]**2 )
        mask = mask / max(min_max_diff, mask.max())
        mask[mask>1]=1
        mask = mask**(0.5)
        return mask
        

def mask2image(mask):
    m = (mask * 255).astype(np.uint8)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)

def mask2gray(mask):
    return (mask * 255).astype(np.uint8)

def setMaskOntoImage(mask, img):
    mask = mask.reshape(img.shape[0], img.shape[1], 1)
    res = img * mask
    res = res.astype(np.uint8)
    return res
