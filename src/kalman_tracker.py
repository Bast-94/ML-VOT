import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.hungarian_tracker import HungarianTracker
from src.iou import (BoundingBox, bb_to_np, bb_with_dim_and_centroid, centroid,
                     intersection_box, iou)
from src.kalman_filter import KalmanFilter


class KalmanTracker(HungarianTracker):
    def __init__(self, det_file: str, img_file_list: list):
        super().__init__(det_file, img_file_list)
        self.kalman_filter_map = {}

    def print_info(self):
        print("Kalman Tracker")

    def similarity_matrix(self):
        similarity_matrix = np.zeros(
            (len(self.current_tracks), len(self.current_detections))
        )
        for t, track in enumerate(self.current_tracks):
            track_id = track["id"]
            assert self.kalman_filter_map.get(track_id) is not None, print(
                f"Kalman filter not initialized for track {track_id}"
            )
            prediction = self.kalman_filter_map[track_id].predict()           
            x, y = prediction[0, 0], prediction[1, 0]
            
            wt, ht = track["bb_width"], track["bb_height"]
            
            bbt = bb_with_dim_and_centroid(np.array([x, y]), wt, ht)
            for d, detection in enumerate(self.current_detections):
                bbd = self.get_bounding_box(detection)
                similarity_matrix[t, d] = iou(bbt, bbd)
        return similarity_matrix

    def init_first_frame(self):
        print("Initializing first frame")
        assert self.frame_idx == 1, print("First frame must be 1")
        self.current_tracks = self.get_frame(self.frame_idx)
        self.current_detections = self.get_frame(self.frame_idx + 1)
        for track in self.current_tracks:
            assert track["id"] == -1, print("Track id must be -1")
            track["id"] = self.cur_id
            self.cur_id += 1
            self.kalman_filter_map[track["id"]] = KalmanFilter.tracking_kalman_filter()

    def update_detection(self):
        for detection in self.current_detections:
            if detection["id"] == -1:
                detection["id"] = self.cur_id
                self.cur_id += 1
                self.kalman_filter_map[
                    detection["id"]
                ] = KalmanFilter.tracking_kalman_filter()

    def clean_kalman_filter_map(self):
        for track in self.current_tracks:
            track_id = track["id"]
            if self.kalman_filter_map.get(track_id) is None:
                continue
            if track_id not in [d["id"] for d in self.current_detections]:
                del self.kalman_filter_map[track_id]

    def apply_matching(self):
        #print("Applying matching at frame", self.frame_idx)
        similarity_matrix = self.similarity_matrix()
        row_ind, col_ind = linear_sum_assignment(1 - similarity_matrix)

        for row_idx, col_idx in zip(row_ind, col_ind):
            track_id = self.current_tracks[row_idx]["id"]
            detection = self.current_detections[col_idx]
            
            detection["id"] = track_id
            center = centroid(self.get_bounding_box(detection))
            assert center.shape == (2,), print(
                f"center must be a 2d vector, got {center.shape}"
            )
            
            (self.kalman_filter_map[track_id].update(center))
            
        
        self.update_detection()
        #self.clean_kalman_filter_map()
