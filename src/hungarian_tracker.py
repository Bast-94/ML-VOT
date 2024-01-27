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
        
    def similarity_matrix(
        self,
        n_frame: int = None,
        tracks: pd.DataFrame = None,
        detections: pd.DataFrame = None,
    ):
        if tracks is None or detections is None:
            if n_frame is None:
                raise ValueError(
                    "n_frame or frame_data and next_frame_data must be set"
                )
            tracks = self.get_frame(n_frame)
            detections = self.get_frame(n_frame + 1)

        similarity_matrix = np.zeros((len(tracks), len(detections)))
        for i, row1 in enumerate(tracks.index):
            bb1 = self.get_bounding_box(tracks, row1)
            for j, row2 in enumerate(detections.index):
                bb2 = self.get_bounding_box(detections, row2)
                similarity_matrix[i, j] = iou(bb1, bb2)
        return similarity_matrix

    def iou_perframe(self):
        tracks = self.get_frame(self.frame_idx)
        detections = self.get_frame(self.frame_idx + 1)
        similarity_matrix = self.similarity_matrix(tracks=tracks, detections=detections)
        row_ind, col_ind = linear_sum_assignment(1 - similarity_matrix)

        for row_idx, col_idx in zip(row_ind, col_ind):
            self.result_df.loc[detections.index[col_idx], "id"] = self.result_df.loc[
                tracks.index[row_idx], "id"
            ]

        for row in detections.index:
            if self.result_df.loc[row, "id"] == -1:
                self.result_df.loc[row, "id"] = self.cur_id
                self.cur_id += 1
        return detections
