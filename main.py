import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.hungarian_tracker import HungarianTracker
from src.parsers import get_track_args
from src.tracker import Tracker

BOUNDING_BOX_DIR = "ADL-Rundle-6/bounding_boxes"
IMG_DIR = "ADL-Rundle-6/img1"
IMG_FILE_LIST = sorted(os.listdir(IMG_DIR))
DATA_DIR = "ADL-Rundle-6"
DET_FILE = (
    "ADL-Rundle-6/det/det.txt"
    if os.path.exists("ADL-Rundle-6/det/clean_det.csv")
    else "ADL-Rundle-6/det/det.txt"
)

args = get_track_args()
nb_frame = args.n_frame
save_gif = args.gif
if args.all:
    nb_frame = len(IMG_FILE_LIST)
img_file_list = IMG_FILE_LIST[:nb_frame]
if args.hungarian:
    print("Using Hungarian algorithm")
    tracker = HungarianTracker(DET_FILE, img_file_list)

    # sys.exit()
else:
    print("Using greedy algorithm")
    tracker = Tracker(DET_FILE, img_file_list)
tracker.print_info()
tracker.iou_tracking("produced/h_tracking.csv")
# tracker.generate_gif(gif_file="produced/bounding_boxes.gif", nb_frames=nb_frame)
