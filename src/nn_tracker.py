import cv2
import numpy as np
import torch

from src.box_encoder import BoxEncoder
from src.hungarian_tracker import HungarianTracker
from src.iou import (BoundingBox, bb_to_np, bb_with_dim_and_centroid, centroid,
                     intersection_box, iou)
import torch.nn.functional as F

class NNTracker(HungarianTracker):
    def __init__(self, det_file: str, img_file_list: list):
        super().__init__(det_file, img_file_list)
        self.box_encoder = BoxEncoder()

    def print_info(self):
        print("NN Tracker")

    def encode_frame(self, lines: list[dict]) -> torch.Tensor:
        frame_idx = lines[0]["frame"] -1
        img = BoxEncoder.img_to_tensor(self.img_file_list[ frame_idx])
        bb_list = [self.get_bounding_box(line) for line in lines]
        return self.box_encoder.encode_bounding_boxes(img, bb_list)
        
        

    def similarity_matrix(self):
        
        track_boxes = self.encode_frame(self.current_tracks)
        detection_boxes = self.encode_frame(self.current_detections)
        assert track_boxes.shape[1] == detection_boxes.shape[1]
        assert track_boxes.shape[0] == len(self.current_tracks)
        assert detection_boxes.shape[0] == len(self.current_detections)
        
        similarity_matrix = np.zeros((len(self.current_tracks), len(self.current_detections)))
        for i, track_box in enumerate(track_boxes):
            for j, detection_box in enumerate(detection_boxes):
                similarity_matrix[i, j] = F.cosine_similarity(track_box, detection_box, dim=0)
        return similarity_matrix
