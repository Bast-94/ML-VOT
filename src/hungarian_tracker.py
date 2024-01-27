import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.iou import BoundingBox, intersection_box, iou
from src.tracker import Tracker


class HungarianTracker(Tracker):
    def __init__(self, det_file: str, img_file_list: list):
        super().__init__(det_file, img_file_list)

    def similarity_matrix(
        self,
        n_frame: int = None,
        frame_data: pd.DataFrame = None,
        next_frame_data: pd.DataFrame = None,
    ):
        if frame_data is None or next_frame_data is None:
            if n_frame is None:
                raise ValueError(
                    "n_frame or frame_data and next_frame_data must be set"
                )
            frame_data = self.get_frame(n_frame)
            next_frame_data = self.get_frame(n_frame + 1)

        similarity_matrix = np.zeros((len(frame_data), len(next_frame_data)))
        for i, row1 in enumerate(frame_data.index):
            bb1 = self.get_bound_box(frame_data, row1)
            for j, row2 in enumerate(next_frame_data.index):
                bb2 = self.get_bound_box(next_frame_data, row2)
                similarity_matrix[i, j] = iou(bb1, bb2)
        return similarity_matrix

    def iou_perframe(self):
        tracks = self.get_frame(self.frame_idx)
        detections = self.get_frame(self.frame_idx + 1)
        similarity_matrix = self.similarity_matrix(
            frame_data=tracks, next_frame_data=detections
        )
        row_ind, col_ind = linear_sum_assignment(1 - similarity_matrix)
        for row_idx, col_idx in zip(row_ind, col_ind):
            if similarity_matrix[row_idx, col_idx] > kargs.get("threshold", 0.5):
                self.result_df.loc[tracks.index[row_idx], "id"] = self.result_df.loc[
                    detections.index[col_idx], "id"
                ]
            else:
                self.result_df.loc[tracks.index[row_idx], "id"] = self.cur_id
                self.cur_id += 1
