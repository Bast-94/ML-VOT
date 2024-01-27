import numpy as np
import pandas as pd

from src.hungarian_tracker import HungarianTracker
from src.iou import BoundingBox, intersection_box, iou, centroid , np_to_bb, bb_to_np
from src.kalman_filter import KalmanFilter


class KalmanTracker(HungarianTracker):
    def __init__(self, det_file: str, img_file_list: list):
        super().__init__(det_file, img_file_list)
        self.kalman_filter_map = {}

    def print_info(self):
        print("Kalman Tracker")

    

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
        for t, track_index in enumerate(tracks.index):
            
            track_id = tracks.loc[track_index, "id"]
            predict_center = self.kalman_filter_map[track_id].predict()
            bbt = np_to_bb(predict_center)
            for d, detection_index in enumerate(detections.index):
                bb2 = self.get_bounding_box(detections, detection_index)
                similarity_matrix[t, d] = iou(bbt, bb2)

        return similarity_matrix
    def iou_perframe(self):
        if (self.frame_idx == 1):
            detections = self.get_frame(self.frame_idx)
            for row in detections.index:
                detection_id = detections.loc[row, "id"]
                self.kalman_filter_map[self.cur_id] = KalmanFilter()
            
        super().iou_perframe()
        
