import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.box_encoder import BoxEncoder
from src.iou import (BoundingBox, bb_to_np, bb_with_dim_and_centroid, centroid,
                     intersection_box, iou)
from src.kalman_tracker import KalmanTracker


class NNTracker(KalmanTracker):
    def __init__(self, det_file: str, img_file_list: list):
        super().__init__(det_file, img_file_list)
        self.box_encoder = BoxEncoder()

    def print_info(self):
        print("NN Tracker")
        print(f"Using model {self.box_encoder.model.__class__.__name__}")
        print(f"Number of frames: {len(self.img_file_list)}")

    def encode_frame(self, lines: list[dict]) -> torch.Tensor:
        frame_idx = lines[0]["frame"] - 1
        img = BoxEncoder.img_to_tensor(self.img_file_list[frame_idx])
        bb_list = [self.get_bounding_box(line) for line in lines]
        return self.box_encoder.encode_bounding_boxes(img, bb_list)

    def similarity_matrix(self):
        track_boxes = self.encode_frame(self.current_tracks)
        detection_boxes = self.encode_frame(self.current_detections)
        assert track_boxes.shape[1] == detection_boxes.shape[1]
        assert track_boxes.shape[0] == len(self.current_tracks)
        assert detection_boxes.shape[0] == len(self.current_detections)

        similarity_matrix = np.zeros(
            (len(self.current_tracks), len(self.current_detections))
        )
        for t, track_box in enumerate(track_boxes):
            prediction = self.kalman_filter_map[self.current_tracks[t]["id"]].predict()
            x, y = prediction[0, 0], prediction[1, 0]
            wt, ht = (
                self.current_tracks[t]["bb_width"],
                self.current_tracks[t]["bb_height"],
            )
            bbt = bb_with_dim_and_centroid(np.array([x, y]), wt, ht)
            for d, detection_box in enumerate(detection_boxes):
                cosine_sim = F.cosine_similarity(track_box, detection_box, dim=0)
                bbd = self.get_bounding_box(self.current_detections[d])
                similarity_matrix[t, d] = cosine_sim * 0.5 + iou(bbt, bbd) * 0.5
        del track_boxes
        del detection_boxes
        return similarity_matrix
