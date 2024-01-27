import os
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.hungarian_tracker import HungarianTracker
from src.parsers import get_track_args
from src.tracker import Tracker
from src.video_generator import generate_video

BOUNDING_BOX_DIR = "ADL-Rundle-6/bounding_boxes"
IMG_DIR = "ADL-Rundle-6/img1"
IMG_FILE_LIST = sorted(os.listdir(IMG_DIR))
DATA_DIR = "ADL-Rundle-6"
DET_FILE = (
    "ADL-Rundle-6/det/det.txt"
    if os.path.exists("ADL-Rundle-6/det/clean_det.csv")
    else "ADL-Rundle-6/det/det.txt"
)
OUTPUT_DIR = "produced"
OUTPUT_CSV = "produced/h_tracking.csv"
OUTPUT_VIDEO = "produced/output.avi"

args = get_track_args()
nb_frame = args.n_frame

save_video = args.video

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
tracker.iou_tracking(OUTPUT_CSV)


if save_video:
    print("Generating video")
    generate_video(
        output_file=OUTPUT_VIDEO,
        file_name_pattern="ADL-Rundle-6/img1/%06d.jpg",
        tracker=tracker,
        max_frame=nb_frame,
    )
