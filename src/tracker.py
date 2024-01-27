import os

import cv2
import pandas as pd
from tqdm import tqdm

from src.iou import BoundingBox, intersection_box, iou
from src.utils import load_det_file

BOUNDING_BOX_DIR = "./ADL-Rundle-6/bounding_boxes"
import imageio
import numpy as np


class Tracker:
    def __init__(self, det_file: str, img_file_list: list, threshold: float = 0.5):
        self.det_df = load_det_file(det_file)
        self.cur_id = 0
        self.img_file_list = img_file_list
        self.result_df = None
        self.frame_idx = 1
        self.threshold = 0.5

    def print_info(self):
        print(f"nb frame: {len(self.img_file_list)}")

    def get_frame(self, n_frame: int):
        return self.det_df[self.det_df.frame == n_frame]

    def get_bound_box(self, frame_data: pd.DataFrame, row: int):
        return BoundingBox(
            frame_data["bb_left"][row],
            frame_data["bb_top"][row],
            frame_data["bb_width"][row],
            frame_data["bb_height"][row],
        )

    def iou_perframe(self):
        threshold = self.threshold
        tracks = self.get_frame(self.frame_idx)
        detections = self.get_frame(self.frame_idx + 1)
        for row1 in tracks.index:
            best_iou = 0
            for row2 in detections.index:
                bb1 = self.get_bound_box(tracks, row1)
                bb2 = self.get_bound_box(detections, row2)
                iou_score = iou(bb1, bb2)

                if self.result_df.loc[row1, "id"] == -1:
                    self.result_df.loc[row1, "id"] = self.cur_id
                    self.cur_id += 1
                if iou_score >= threshold and iou_score > best_iou:
                    self.result_df.loc[row2, "id"] = self.result_df.loc[row1, "id"]
                    best_iou = iou_score

    def next_frame(self):
        self.frame_idx += 1

    def iou_tracking(self, output_csv: str, threshold: float = 0.5):
        self.result_df = self.det_df.copy()
        while self.frame_idx < len(self.img_file_list):
            self.iou_perframe()
            self.next_frame()

        self.result_df.to_csv(output_csv, index=False)

    def update_gif(self, opencv_img, id, bb1, img_file, bounding_box_dir):
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
        self,
        gif_file="ADL-Rundle-6/bounding_boxes.gif",
        img_dir="ADL-Rundle-6/img1",
        nb_frames=10,
    ):
        if self.result_df is None:
            self.iou_tracking("ADL-Rundle-6/result.csv")
        df = self.result_df
        img_file_list = self.img_file_list[:nb_frames]

        for n_frame, img_file in tqdm(enumerate(img_file_list, start=1)):
            res_df = df[df["frame"] == n_frame]
            opencv_img = cv2.imread(os.path.join(img_dir, img_file))
            for row1 in res_df.index:
                bb1 = self.get_bound_box(res_df, row1)
                id = res_df.loc[row1, "id"]
                self.update_gif(
                    opencv_img=opencv_img,
                    id=id,
                    bb1=bb1,
                    img_file=img_file,
                    bounding_box_dir=BOUNDING_BOX_DIR,
                )

        print("Generating gif...")
        images = []
        bounded_box_files = sorted(os.listdir(BOUNDING_BOX_DIR))[:nb_frames]
        for filename in tqdm(bounded_box_files):
            images.append(imageio.imread(os.path.join(BOUNDING_BOX_DIR, filename)))
        imageio.mimsave(gif_file, images, duration=0.5)
        print("Gif saved at {}".format(gif_file))
