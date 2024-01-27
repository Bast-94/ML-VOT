import numpy as np
import pandas as pd

from src.hungarian_tracker import HungarianTracker
from src.iou import BoundingBox, intersection_box, iou, centroid , np_to_bb, bb_to_np
from src.kalman_filter import KalmanFilter


class KalmanTracker(HungarianTracker):
    def __init__(self, det_file: str, img_file_list: list):
        super().__init__(det_file, img_file_list)
        self.kalman_filter_map = {}
        self.current_track = None
        self.current_detection = None

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
            assert self.kalman_filter_map.get(track_id) is not None, print(f"Kalman filter not initialized for track {track_id}")
            predict_center = self.kalman_filter_map[track_id].predict()
            bbt = np_to_bb(predict_center)
            for d, detection_index in enumerate(detections.index):
                bb2 = self.get_bounding_box(detections, detection_index)
                similarity_matrix[t, d] = iou(bbt, bb2)

        return similarity_matrix
    
    def init_first_frame(self):
        print("Initializing first frame")
        assert self.frame_idx == 1, print("First frame must be 1")
        self.current_track = self.result_df[self.result_df.frame == self.frame_idx]
        self.current_detection = self.result_df[self.result_df.frame == self.frame_idx + 1]
        for row in self.current_track.index:
            self.result_df.loc[row, "id"] = self.cur_id
            self.kalman_filter_map[self.cur_id] = KalmanFilter.tracking_kalman_filter()
            self.cur_id += 1
    
    def update_detection(self, detections: pd.DataFrame):
        print(f"Updating detections for frame {self.frame_idx}")
        for row in detections.index:
            if self.result_df.loc[row, "id"] == -1:
                self.result_df.loc[row, "id"] = self.cur_id
                self.kalman_filter_map[self.cur_id] = KalmanFilter.tracking_kalman_filter()
                self.cur_id += 1
        return detections
    
    
        
