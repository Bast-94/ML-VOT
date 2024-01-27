import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracker import Tracker
from tqdm import tqdm

class HungarianTracker(Tracker):
    def __init__(self, det_file: str, img_file_list: list):
        super().__init__(det_file, img_file_list)
    

    

    def iou_tracking(self, output_csv: str):
        print("Using Hungarian algorithm")
        self.result_df = self.det_df.copy()
        for n_frame, img_file in tqdm(enumerate(self.img_file_list, start=1)):
            frame_data = self.get_frame(n_frame)
            next_frame_data = self.get_frame(n_frame + 1)
            similarity_matrix_df = self.similarity_matrix(frame_data=frame_data, next_frame_data=next_frame_data)
            cost = 1 - similarity_matrix_df
            row_indices, col_indices = linear_sum_assignment(cost)
            assert cost.shape[0] == len(frame_data), f"{cost.shape[0]} != {len(frame_data)}"
            assert cost.shape[1] == len(next_frame_data), f"{cost.shape[1]} != {len(next_frame_data)}"
            print(cost)
            
            for detection, track in zip(row_indices, col_indices):
                print(f"{detection} -> {track}")
        self.result_df.to_csv(output_csv, index=False)
