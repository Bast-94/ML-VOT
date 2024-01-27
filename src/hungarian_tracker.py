import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from src.iou import BoundingBox, intersection_box, iou
from src.tracker import Tracker
import pandas as pd

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
    def iou_perframe(self, **kargs):
        threshold = kargs.get("threshold", 0.5)
        tracks = self.get_frame(self.frame_idx)
        detections = self.get_frame(self.frame_idx + 1)
        similarity_matrix = self.similarity_matrix(
            frame_data=tracks, next_frame_data=detections
        )
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        for i, row in enumerate(tracks.index):
            if self.result_df.loc[row, "id"] == -1:
                self.result_df.loc[row, "id"] = self.cur_id
                self.cur_id += 1
            if similarity_matrix[i, col_ind[i]] >= threshold:
                self.result_df.loc[
                    detections.index[col_ind[i]], "id"
                ] = self.result_df.loc[row, "id"]
