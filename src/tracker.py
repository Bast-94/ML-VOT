import pandas as pd
from tqdm import tqdm

from src.iou import BoundingBox, intersection_box, iou
from src.utils import load_det_file


class Tracker:
    def __init__(self, det_file: str, img_file_list: list):
        self.det_df = load_det_file(det_file)
        self.cur_id = 0
        self.img_file_list = img_file_list
        self.result_df = None

    def print_info(self):
        print(f"nb frame: {len(self.img_file_list)}")

    def get_frame(self, n_frame: int):
        return self.det_df[self.det_df.frame == n_frame]

    def get_bound_box(self, frame_data: pd.DataFrame, row: int):
        return BoundingBox(
            frame_data["bb_left"][row],
            frame_data["bb_top"][row],
            frame_data["bb_width"][row],
            frame_data["bb_height"][row],
        )

    def iou_tracking(self, output_csv: str, threshold: float = 0.5):
        for n_frame, img_file in tqdm(enumerate(self.img_file_list, start=1)):
            frame_data = self.get_frame(n_frame)
            next_frame_data = self.get_frame(n_frame + 1)
            for i, row1 in enumerate(frame_data.index):
                best_iou = 0
                for j, row2 in enumerate(next_frame_data.index):
                    bb1 = self.get_bound_box(frame_data, row1)
                    bb2 = self.get_bound_box(next_frame_data, row2)
                    iou_score = iou(bb1, bb2)

                    if self.det_df.loc[row1, "id"] == -1:
                        self.det_df.loc[row1, "id"] = cur_id
                        cur_id += 1
                    if iou_score >= threshold and iou_score > best_iou:
                        self.det_df.loc[row2, "id"] = self.det_df.loc[row1, "id"]

        self.result_df = self.det_df
        self.det_df.to_csv(output_csv, index=False)
