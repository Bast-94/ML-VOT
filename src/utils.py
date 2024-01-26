import os

import pandas as pd


def load_det_file(det_file):
    if not os.path.exists("ADL-Rundle-6/det/clean_det.csv"):
        det_df = pd.read_csv(det_file, sep=",", header=None)
        det_df.columns = [
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z",
        ]
        det_df.to_csv("ADL-Rundle-6/det/clean_det.csv", index=False)
    else:
        det_df = pd.read_csv("ADL-Rundle-6/det/clean_det.csv", sep=",", header=0)
    return det_df
