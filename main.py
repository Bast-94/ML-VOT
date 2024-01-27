import os
import sys
from glob import glob

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config_manager import ConfigManager
from src.hungarian_tracker import HungarianTracker
from src.parsers import get_track_args
from src.tracker import Tracker
from src.video_generator import generate_video

config = ConfigManager("config/config.yml")

args = get_track_args()
nb_frame = args.n_frame
save_video = args.video

IMG_FILE_LIST = glob(os.path.join(config.IMG_DIR, "*.jpg"))
if args.all:
    nb_frame = len(IMG_FILE_LIST)
img_file_list = glob(os.path.join(config.IMG_DIR, "*.jpg"))


if args.hungarian:
    print("Using Hungarian algorithm")
    tracker = HungarianTracker(config.DET_FILE, img_file_list)

else:
    print("Using greedy algorithm")
    tracker = Tracker(config.DET_FILE, img_file_list)


tracker.print_info()
tracker.iou_tracking(config.OUTPUT_CSV)


if save_video:
    print("Generating video")
    generate_video(
        output_file=config.OUTPUT_VIDEO,
        file_name_pattern="ADL-Rundle-6/img1/%06d.jpg",
        tracker=tracker,
        max_frame=nb_frame,
    )
