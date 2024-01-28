from src.hungarian_tracker import HungarianTracker

class NNTracker(HungarianTracker):
    def __init__(self, det_file: str, img_file_list: list):
        super().__init__(det_file, img_file_list)
    
    def print_info(self):
        print("NN Tracker")
    
    