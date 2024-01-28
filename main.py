import os
import sys
from glob import glob

import cv2
import numpy as np
import termcolor
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

from src.box_encoder import BoxEncoder
from src.config_manager import ConfigManager
from src.hungarian_tracker import HungarianTracker
from src.iou import (BoundingBox, bb_to_np, bb_with_dim_and_centroid, centroid,
                     intersection_box, iou)
from src.kalman_tracker import KalmanTracker
from src.nn_tracker import NNTracker
from src.parsers import get_track_args
from src.tracker import Tracker
from src.utils import load_det_file
from src.video_generator import generate_video

config = ConfigManager("config/config.yml")

args = get_track_args()
IMG_FILE_LIST = glob(os.path.join(config.IMG_DIR, "*.jpg"))
if args.commands == "test":
    print("Test")
    import torch
    import torch.nn.functional as F

    nn_tracker = NNTracker(config.DET_FILE, IMG_FILE_LIST)
    nn_tracker.print_info()
    nn_tracker.init_first_frame()
    encoded = nn_tracker.encode_frame(nn_tracker.current_tracks)

    print(nn_tracker.similarity_matrix())

    sys.exit(0)

nb_frame = args.n_frame
save_video = args.video
output_csv = args.output_csv
if nb_frame is not None:
    nb_frame = int(nb_frame)
else:
    nb_frame = len(IMG_FILE_LIST)
img_file_list = IMG_FILE_LIST[:nb_frame]


if args.hungarian:
    tracker = HungarianTracker(config.DET_FILE, img_file_list)
elif args.kalman:
    tracker = KalmanTracker(config.DET_FILE, img_file_list)
elif args.nn:
    tracker = NNTracker(config.DET_FILE, img_file_list)
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
