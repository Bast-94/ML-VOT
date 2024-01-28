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

    def print_info(self):
        print(f"Simple Tracker with threshold {self.threshold}")

    def get_frame2(self, n_frame: int):
        return self.det_df[self.result_df.frame == n_frame]

    # same method as get_frame, but with different return type
    def get_frame(self, n_frame: int) -> list[dict[str, Any]]:
        return self.det_df[self.det_df["frame"] == n_frame].to_dict("records")

    def get_bounding_box(self, frame_data: pd.DataFrame | dict, row: int):
        return BoundingBox(
            frame_data["bb_left"][row],
            frame_data["bb_top"][row],
            frame_data["bb_width"][row],
            frame_data["bb_height"][row],
        )

    @staticmethod
    def get_bounding_box2(line_dict: dict[str, Any]) -> BoundingBox:
        return BoundingBox(
            line_dict["bb_left"],
            line_dict["bb_top"],
            line_dict["bb_width"],
            line_dict["bb_height"],
        )

    def apply_matching2(self):
        threshold = self.threshold
        tracks = self.current_tracks
        detections = self.current_detections
        for row1 in tracks.index:
            best_iou = 0
            for row2 in detections.index:
                bb1 = self.get_bounding_box(tracks, row1)
                bb2 = self.get_bounding_box(detections, row2)
                iou_score = iou(bb1, bb2)

                if tracks.loc[row1, "id"] == -1:
                    tracks.loc[row1, "id"] = self.cur_id
                    self.cur_id += 1
                if iou_score >= threshold and iou_score > best_iou:
                    detections.loc[row2, "id"] = tracks.loc[row1, "id"]
                    best_iou = iou_score
        self.current_tracks = tracks
        self.current_detections = detections

    def apply_matching(self):
        for track in self.current_tracks:
            best_iou = 0
            for detection in self.current_detections:
                bb1 = self.get_bounding_box2(track)
                bb2 = self.get_bounding_box2(detection)
                iou_score = iou(bb1, bb2)
                if track["id"] == -1:
                    track["id"] = self.cur_id
                    self.cur_id += 1
                if iou_score >= self.threshold and iou_score > best_iou:
                    
                    detection["id"] = track["id"]
                    best_iou = iou_score

    def next_frame(self):
        self.frame_idx += 1

    def init_first_frame(self):
        assert self.frame_idx == 1, print("First frame must be 1")
        self.current_tracks = self.get_frame(self.frame_idx)
        self.current_detections = self.get_frame(self.frame_idx + 1)
        for track in self.current_tracks:
            track["id"] = self.cur_id
            self.cur_id += 1

    def write_track_to_result(self):
        self.result_df = pd.concat([self.result_df, pd.DataFrame(self.current_tracks)])

    def update_track_and_detection(self):
        self.current_tracks = self.current_detections
        self.current_detections = self.get_frame(self.frame_idx + 1)

    def track(self, output_csv: str):
        print("Tracking")
        self.result_df = pd.DataFrame(columns=self.det_df.columns)
        self.init_first_frame()
        for _ in tqdm(self.img_file_list):
            self.apply_matching()
            self.write_track_to_result()
            self.next_frame()
            self.update_track_and_detection()

        self.result_df.to_csv(output_csv, index=False)
        print(f"Tracking done, result saved in {output_csv}")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.track(*args, **kwds)
