import os
from typing import Any

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
        print(f"Simple Tracker with threshold {self.threshold}")

    def get_frame(self, n_frame: int):
        return self.result_df[self.result_df.frame == n_frame]

    def get_bounding_box(self, frame_data: pd.DataFrame, row: int):
        return BoundingBox(
            frame_data["bb_left"][row],
            frame_data["bb_top"][row],
            frame_data["bb_width"][row],
            frame_data["bb_height"][row],
        )

    def apply_matching(self):
        threshold = self.threshold
        tracks = self.get_frame(self.frame_idx)
        detections = self.get_frame(self.frame_idx + 1)
        for row1 in tracks.index:
            best_iou = 0
            for row2 in detections.index:
                bb1 = self.get_bounding_box(tracks, row1)
                bb2 = self.get_bounding_box(detections, row2)
                iou_score = iou(bb1, bb2)

                if self.result_df.loc[row1, "id"] == -1:
                    self.result_df.loc[row1, "id"] = self.cur_id
                    self.cur_id += 1
                if iou_score >= threshold and iou_score > best_iou:
                    self.result_df.loc[row2, "id"] = self.result_df.loc[row1, "id"]
                    best_iou = iou_score

    def next_frame(self):
        self.frame_idx += 1

    def init_first_frame(self):
        assert self.frame_idx == 1, print("First frame must be 1")
        first_frame = self.get_frame(self.frame_idx)
        for row in first_frame.index:
            self.result_df.loc[row, "id"] = self.cur_id
            self.cur_id += 1
    

    def track(self, output_csv: str):
        print("Tracking")
        self.result_df = self.det_df.copy()
        self.init_first_frame()
        for _ in tqdm(self.img_file_list):
            self.apply_matching()
            self.next_frame()

        self.result_df.to_csv(output_csv, index=False)
        print(f"Tracking done, result saved in {output_csv}")
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.track(*args, **kwds)
