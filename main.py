import os
import sys
from glob import glob

import cv2
import numpy as np
import termcolor
from scipy.optimize import linear_sum_assignment

from src.config_manager import ConfigManager
from src.hungarian_tracker import HungarianTracker
from src.kalman_tracker import KalmanTracker
from src.parsers import get_track_args
from src.tracker import Tracker
from src.utils import load_det_file
from src.video_generator import generate_video

config = ConfigManager("config/config.yml")

args = get_track_args()

if args.commands == "test":
    print("Test")
    det_df = load_det_file(config.DET_FILE)
    frame_data = det_df[det_df.frame == 1]
    # convert frame_data to list of dict
    frame_data = frame_data.to_dict("records")
    print(frame_data)
    sys.exit(0)

nb_frame = args.n_frame
save_video = args.video
output_csv = args.output_csv
IMG_FILE_LIST = glob(os.path.join(config.IMG_DIR, "*.jpg"))
if args.all:
    nb_frame = len(IMG_FILE_LIST)
img_file_list = IMG_FILE_LIST[:nb_frame]


if args.hungarian:
    tracker = HungarianTracker(config.DET_FILE, img_file_list)
elif args.kalman:
    tracker = KalmanTracker(config.DET_FILE, img_file_list)
else:
    tracker = Tracker(config.DET_FILE, img_file_list)


tracker.print_info()
tracker(output_csv=output_csv)


if save_video is not None:
    print("Generating video")
    generate_video(
        output_file=args.video,
        file_name_pattern="ADL-Rundle-6/img1/%06d.jpg",
        tracker=tracker,
        max_frame=nb_frame,
    )
