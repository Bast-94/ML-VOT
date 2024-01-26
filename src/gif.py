import os

import cv2
import imageio
import pandas as pd
from tqdm import tqdm

from src.iou import BoundingBox

BOUNDING_BOX_DIR = "./ADL-Rundle-6/bounding_boxes"


def update_gif(opencv_img, id, bb1, img_file, bounding_box_dir):
    cv2.rectangle(
        opencv_img,
        (int(bb1.bb_left), int(bb1.bb_top)),
        (
            int(bb1.bb_left + bb1.bb_width),
            int(bb1.bb_top + bb1.bb_height),
        ),
        (0, 0, 255),
        2,
    )
    cv2.putText(
        opencv_img,
        str(id),
        (int(bb1.bb_left), int(bb1.bb_top)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(
        os.path.join(bounding_box_dir, img_file),
        opencv_img,
    )


def generate_gif(
    result_csv,
    img_file_list,
    gif_file="ADL-Rundle-6/bounding_boxes.gif",
    img_dir="ADL-Rundle-6/img1",
    nb_frames=10,
):
    df = pd.read_csv(result_csv, sep=",", header=0)
    img_file_list = img_file_list[:nb_frames]
    for n_frame, img_file in tqdm(enumerate(img_file_list, start=1)):
        res_df = df[df["frame"] == n_frame]
        opencv_img = cv2.imread(os.path.join(img_dir, img_file))
        for row1 in res_df.index:
            bb1 = BoundingBox(
                res_df["bb_left"][row1],
                res_df["bb_top"][row1],
                res_df["bb_width"][row1],
                res_df["bb_height"][row1],
            )
            update_gif(opencv_img, row1, bb1, img_file)
    images = []
    print("Generating gif...")
    bounded_box_files = sorted(os.listdir(BOUNDING_BOX_DIR))[:nb_frames]
    for filename in tqdm(bounded_box_files):
        images.append(imageio.imread(os.path.join(BOUNDING_BOX_DIR, filename)))
    imageio.mimsave(gif_file, images, duration=0.5)
    print("Gif saved at {}".format(gif_file))
