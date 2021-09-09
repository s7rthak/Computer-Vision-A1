""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args

BASELINE_FRAMES = 1700
ILLUMINATION_FRAMES = 301
JITTER_FRAMES = 1150
DYNAMIC_FRAMES = 1189
PTZ_FRAMES = 1130

dataset_dir = "COL780-A1-Data/"
kernel = np.ones((5,5),np.uint8)

def baseline_bgs(args):
    os.makedirs(args.out_path, exist_ok=True)

    file_handle = open(args.eval_frames, 'r')
    lines_list = file_handle.readlines()

    eval_start, eval_end = (int(val) for val in lines_list[0].split())
    file_handle.close()

    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=15, detectShadows=False)
    backSub2 = cv2.createBackgroundSubtractorKNN()

    for i in range(1, BASELINE_FRAMES+1):
        frame_name = "in" + str(i).zfill(6) + ".jpg"

        frame = cv2.imread(args.inp_path + frame_name)

        bilateral = cv2.bilateralFilter(frame, 15, 75, 75)
        fgMask = backSub2.apply(bilateral)
        closing = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        if i >= eval_start and i <= eval_end:
            pred_name = "gt" + str(i).zfill(6) + ".png"
            cv2.imwrite(args.out_path + pred_name, opening)


def illumination_bgs(args):
    #TODO complete this function
    pass


def jitter_bgs(args):
    #TODO complete this function
    pass


def dynamic_bgs(args):
    #TODO complete this function
    pass


def ptz_bgs(args):
    #TODO: (Optional) complete this function
    pass


def main(args):
    if args.category not in "bijdp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)