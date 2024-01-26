
from src.tracker import Tracker

class HungarianTracker(Tracker):
    def __init__(self, det_file: str, img_file_list: list):
        super().__init__(det_file, img_file_list)
        