DATA_DIR = "ADL-Rundle-6"
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
import imageio
import itertools
from src.iou import BoundingBox, intersection_box, iou

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
img_file_list = sorted(os.listdir(IMG_DIR))

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-n", "--n-frame", type=int, default=1, help="Frame number")
arg_parser.add_argument("-g", "--gif", action="store_true", help="Create gif")

args = arg_parser.parse_args()
nb_frame = args.n_frame
save_gif = args.gif
img_file_list = img_file_list[:nb_frame]

threshold = 0.5
cur_id = 0


for n_frame, img_file in tqdm(enumerate(img_file_list, start=1)):
    img_index = n_frame - 1
    img_file = img_file_list[img_index]
    img = Image.open(os.path.join(IMG_DIR, img_file))

    frame_data = det_df[det_df["frame"] == n_frame]
    next_frame_data = det_df[det_df["frame"] == n_frame + 1]
    opencv_img = cv2.imread(os.path.join(IMG_DIR, img_file))
    for i, row1 in enumerate(frame_data.index):
        
        for j, row2 in enumerate(next_frame_data.index):
            
            bb1 = BoundingBox(
                frame_data["bb_left"][row1],
                frame_data["bb_top"][row1],
                frame_data["bb_width"][row1],
                frame_data["bb_height"][row1],
            )
            bb2 = BoundingBox(
                next_frame_data["bb_left"][row2],
                next_frame_data["bb_top"][row2],
                next_frame_data["bb_width"][row2],
                next_frame_data["bb_height"][row2],
            )
            iou_score = iou(bb1, bb2)
            if det_df.loc[row1, "id"] == -1:
                det_df.loc[row1, "id"] = cur_id
                cur_id += 1
            if iou_score >= threshold:
                det_df.loc[row2, "id"] = det_df.loc[row1, "id"]
            if save_gif:
                cv2.rectangle(
                    opencv_img,
                    (int(bb1.bb_left), int(bb1.bb_top)),
                    (int(bb1.bb_left + bb1.bb_width ), int(bb1.bb_top + bb1.bb_height)),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    opencv_img,
                    str(det_df.loc[row1, "id"]),
                    (int(bb1.bb_left), int(bb1.bb_top)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(
                    os.path.join(BOUNDING_BOX_DIR, img_file),
                    opencv_img,
                )
    
    det_df.to_csv("ADL-Rundle-6/det/restults.csv", index=False)
    

if save_gif:
    print("Creating gif...")
    images = []
    for filename in sorted(os.listdir(BOUNDING_BOX_DIR))[:nb_frame]:
        images.append(imageio.imread(os.path.join(BOUNDING_BOX_DIR, filename)))
    imageio.mimsave("ADL-Rundle-6/bounding_boxes.gif", images, duration=0.5)
