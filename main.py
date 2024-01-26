DATA_DIR = "ADL-Rundle-6"
import argparse
import itertools
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.iou import BoundingBox, intersection_box, iou
from src.tracker import Tracker

det_file = (
    "ADL-Rundle-6/det/det.txt"
    if os.path.exists("ADL-Rundle-6/det/clean_det.csv")
    else "ADL-Rundle-6/det/det.txt"
)

if not os.path.exists("ADL-Rundle-6/det/clean_det.csv"):
    det_df = pd.read_csv(det_file, sep=",", header=None)
    det_df.columns = [
        "frame",
        "id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "conf",
        "x",
        "y",
        "z",
    ]
    det_df.to_csv("ADL-Rundle-6/det/clean_det.csv", index=False)
else:
    det_df = pd.read_csv("ADL-Rundle-6/det/clean_det.csv", sep=",", header=0)

BOUNDING_BOX_DIR = "ADL-Rundle-6/bounding_boxes"
IMG_DIR = "ADL-Rundle-6/img1"
IMG_FILE_LIST = sorted(os.listdir(IMG_DIR))

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-n", "--n-frame", type=int, default=1, help="Frame number")
arg_parser.add_argument("-g", "--gif", action="store_true", help="Create gif")

args = arg_parser.parse_args()
nb_frame = args.n_frame
save_gif = args.gif
img_file_list = IMG_FILE_LIST[:nb_frame]

threshold = 0.5
cur_id = 0

tracker = Tracker(det_file, img_file_list)
tracker.print_info()
