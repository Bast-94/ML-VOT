import glob

import cv2
from tqdm import tqdm

from src.tracker import Tracker


def update_frame(tracker: Tracker, cur_frame, id, bb1):
    opencv_img =cv2.rectangle(
        cur_frame,
        (int(bb1.bb_left), int(bb1.bb_top)),
        (
            int(bb1.bb_left + bb1.bb_width),
            int(bb1.bb_top + bb1.bb_height),
        ),
        (0, 0, 255),
        2,
    )
    cv2.putText(
        opencv_img,
        str(id),
        (int(bb1.bb_left), int(bb1.bb_top)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def generate_video(
    output_file="output.avi",
    file_name_pattern="ADL-Rundle-6/img1/%06d.jpg",
    tracker: Tracker = None,
    max_frame: int = None,
):
    if tracker is None:
        tracker = Tracker("ADL-Rundle-6/det/det.txt", "ADL-Rundle-6/img1")
    if tracker.result_df is None:
        tracker.iou_tracking("produced/h_tracking.csv")
    cap = cv2.VideoCapture(file_name_pattern)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 25
    keep_reading, cur_frame = cap.read()

    frame_size = (cur_frame.shape[1], cur_frame.shape[0])
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    nb_frame = len(glob.glob(file_name_pattern.replace("%06d", "*")))
    nb_frame = nb_frame if max_frame is None else min(nb_frame, max_frame)
    for i in tqdm(range(nb_frame)):
        frame_data = tracker.result_df[tracker.result_df["frame"] == i + 1]
        for row in frame_data.index:
            id = frame_data.loc[row, "id"]
            bb1 = tracker.get_bounding_box(frame_data, row)
            update_frame(tracker, cur_frame, id, bb1)

        out.write(cur_frame)
        keep_reading, cur_frame = cap.read()
        if not keep_reading:
            break

    out.release()
    cap.release()
