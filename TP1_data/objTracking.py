from src.kalman_filter import KalmanFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
VIDEO_PATH = 'videos/vid1.mp4'
if __name__ == '__main__':
    # Create a Kalman Filter object
    kf = KalmanFilter(dt=0.1, u_x=1, u_y=1, x_std_meas=0.1, y_std_meas=0.1)
    cap = cv2.VideoCapture('randomball.avi')
    fig = plt.figure()
    for i in range(10):
        #frame = cap.read()[1]
        
        kf.predict()
        kf.update([i, i])
        

