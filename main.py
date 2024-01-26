import os

from src.track_parser import get_args
from src.tracker import Tracker
from src.hungarian_tracker import HungarianTracker

BOUNDING_BOX_DIR = "ADL-Rundle-6/bounding_boxes"
IMG_DIR = "ADL-Rundle-6/img1"
IMG_FILE_LIST = sorted(os.listdir(IMG_DIR))
DATA_DIR = "ADL-Rundle-6"
DET_FILE = (
    "ADL-Rundle-6/det/det.txt"
    if os.path.exists("ADL-Rundle-6/det/clean_det.csv")
    else "ADL-Rundle-6/det/det.txt"
)

args = get_args()
nb_frame = args.n_frame
save_gif = args.gif
img_file_list = IMG_FILE_LIST[:nb_frame]
if args.hungarian:
    print("Using Hungarian algorithm")
    tracker = HungarianTracker(DET_FILE, img_file_list)
else:
    print("Using greedy algorithm")
    tracker = Tracker(DET_FILE, img_file_list)
tracker.print_info()
tracker.generate_gif(gif_file="produced/bounding_boxes.gif", nb_frames=nb_frame)
