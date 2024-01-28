import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.iou import BoundingBox, intersection_box, iou
from src.tracker import Tracker


class HungarianTracker(Tracker):
    def __init__(self, det_file: str, img_file_list: list):
        super().__init__(det_file, img_file_list)

    def print_info(self):
        print("Hungarian Tracker")

    def similarity_matrix(self):
        similarity_matrix = np.zeros(
            (len(self.current_tracks), len(self.current_detections))
        )
        for t, track in enumerate(self.current_tracks):
            for d, detection in enumerate(self.current_detections):
                bbt = self.get_bounding_box2(track)
                bbd = self.get_bounding_box2(detection)
                similarity_matrix[t, d] = iou(bbt, bbd)

        return similarity_matrix

    def update_detection(self):
        for detection in self.current_detections:
            if detection["id"] == -1:
                detection["id"] = self.cur_id
                self.cur_id += 1
        
    def apply_matching(self):
        
        similarity_matrix = self.similarity_matrix()
        row_ind, col_ind = linear_sum_assignment(1 - similarity_matrix)
        for row_idx, col_idx in zip(row_ind, col_ind):
            self.current_detections[col_idx]["id"] = self.current_tracks[row_idx]["id"]
        self.update_detection()
