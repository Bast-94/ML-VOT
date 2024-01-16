DATA_DIR = 'ADL-Rundle-6'
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
import imageio
import itertools
from src.iou import BoundingBox, intersection_box, iou
det_file = 'ADL-Rundle-6/det/det.txt' if os.path.exists('ADL-Rundle-6/det/clean_det.csv') else 'ADL-Rundle-6/det/det.txt'

if not os.path.exists('ADL-Rundle-6/det/clean_det.csv'):
    det_df = pd.read_csv(det_file, sep=',', header=None)
    det_df.columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    det_df.to_csv('ADL-Rundle-6/det/clean_det.csv', index=False)
else:
    det_df = pd.read_csv('ADL-Rundle-6/det/clean_det.csv', sep=',', header=0)

BOUNDING_BOX_DIR = 'ADL-Rundle-6/bounding_boxes'
IMG_DIR = 'ADL-Rundle-6/img1'
img_file_list = sorted(os.listdir(IMG_DIR))

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-n', '--n-frame', type=int, default=6, help='Frame number')
arg_parser.add_argument('-g', '--gif', action='store_true', help='Create gif')

args = arg_parser.parse_args()
nb_frame = args.n_frame
save_gif = args.gif
img_file_list = img_file_list[:nb_frame]

for n_frame, img_file in tqdm(enumerate(img_file_list, start=1)):
    img_file = img_file_list[n_frame-1]
    img = Image.open(os.path.join(IMG_DIR, img_file))
    frame_data = det_df[det_df['frame']==n_frame]
    opencv_img = cv2.imread(os.path.join(IMG_DIR, img_file))
    row_cominations = list(itertools.combinations(frame_data.index, 2))
    iou_matrix = np.zeros((len(row_cominations), len(row_cominations)))
    for row1, row2 in row_cominations:
        bb1 = BoundingBox(frame_data.loc[row1, 'bb_left'], frame_data.loc[row1, 'bb_top'], frame_data.loc[row1, 'bb_width'], frame_data.loc[row1, 'bb_height'])
        bb2 = BoundingBox(frame_data.loc[row2, 'bb_left'], frame_data.loc[row2, 'bb_top'], frame_data.loc[row2, 'bb_width'], frame_data.loc[row2, 'bb_height'])

        

    for row in (frame_data.iloc):
        bb_left = int(row['bb_left'])
        bb_top = int(row['bb_top'])
        bb_width = int(row['bb_width'])
        bb_height = int(row['bb_height'])
        if save_gif:
            cv2.rectangle(opencv_img, (bb_left, bb_top), (bb_left+bb_width, bb_top+bb_height), (0,0,255), 2)
            cv2.putText(opencv_img, str(row['id']), (bb_left, bb_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.imwrite(os.path.join(BOUNDING_BOX_DIR, img_file), opencv_img)
        


if save_gif:
    
    print('Creating gif...')
    images = []
    for filename in sorted(os.listdir(BOUNDING_BOX_DIR))[:10]:
        images.append(imageio.imread(os.path.join(BOUNDING_BOX_DIR, filename)))
    imageio.mimsave('ADL-Rundle-6/bounding_boxes.gif', images, duration=0.5)
        
        
    

    