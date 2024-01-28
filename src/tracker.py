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
        self.current_tracks = []
        self.current_detections = []

    def print_info(self) -> None:
        print(f"Simple Tracker with threshold {self.threshold}")

    # same method as get_frame, but with different return type
    def get_frame(self, n_frame: int) -> list[dict[str, Any]]:
        return self.det_df[self.det_df["frame"] == n_frame].to_dict("records")

    @staticmethod
    def get_bounding_box(line_dict: dict[str, Any]) -> BoundingBox:
        return BoundingBox(
            line_dict["bb_left"],
            line_dict["bb_top"],
            line_dict["bb_width"],
            line_dict["bb_height"],
        )

    def apply_matching(self) -> None:
        for track in self.current_tracks:
            best_iou = 0
            bounding_box_1 = self.get_bounding_box(track)
            for detection in self.current_detections:
                bb2 = self.get_bounding_box(detection)
                iou_score = iou(bounding_box_1, bb2)
                if iou_score >= self.threshold and iou_score > best_iou:
                    detection["id"] = track["id"]
                    best_iou = iou_score

    def next_frame(self) -> None:
        self.frame_idx += 1

    def init_first_frame(self) -> None:
        assert self.frame_idx == 1, print("First frame must be 1")
        self.current_tracks = self.get_frame(self.frame_idx)
        self.current_detections = self.get_frame(self.frame_idx + 1)
        for track in self.current_tracks:
            track["id"] = self.cur_id
            self.cur_id += 1

    def write_track_to_result(self) -> None:
        # check that id is not twice in current_tracks
        assert len(self.current_tracks) == len(
            set([track["id"] for track in self.current_tracks])
        ), print([track["id"] for track in self.current_tracks])
        self.result_df = pd.concat([self.result_df, pd.DataFrame(self.current_tracks)])

    def update_track_and_detection(self) -> None:
        self.current_tracks = self.current_detections
        self.current_detections = self.get_frame(self.frame_idx + 1)

    def track(self, output_csv: str) -> None:
        print("Tracking")
        self.result_df = pd.DataFrame(columns=self.det_df.columns)
        self.init_first_frame()
        for _ in tqdm(self.img_file_list):
            self.apply_matching()
            self.write_track_to_result()
            self.next_frame()
            self.update_track_and_detection()

        self.result_df.to_csv(output_csv, index=False, header=False)
        print(f"Tracking done, result saved in {output_csv}")

    def __call__(self, output_csv: str) -> Any:
        self.track(output_csv)
