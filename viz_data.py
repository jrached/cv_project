#!/usr/bin/env python3 

import numpy as np 
import cv2 
import PIL 
from PIL import Image 
import pandas as pd 
import matplotlib.pyplot as plt 

####################
### Visualizing ####
####################

def load_image(img_file, depth=False): 
    """
    Given an mp4 file, load the image 

    Inputs: 
        - img_file, path to mp4 image file 
    Outputs: 
        - imgage as numpy array 
    """ 
    if not depth: 
        return np.array(Image.open(img_file)).astype(np.uint8)
    else: 
        return np.array(Image.open(img_file)) 

def load_poses(pose_file): 
    return pd.read_csv(pose_file).to_numpy().reshape(-1, 4, 4) 

def viz_split(split_index, fps, num_samples=1000):
    """
    Visualize 1000 datapoint split 
    """

    # Paths to images.
    dir_path = f'/home/jarched/cv_project_code/project/data/kimera'
    pose_path = f'/home/jrached/cv_project_code/project/data/kimera/split{split_index}/poses'
    depth_path = f'/home/jrached/cv_project_code/project/data/kimera/split{split_index}/depth'
    rgb_path = f'/home/jrached/cv_project_code/project/data/kimera/split{split_index}/color'

    start = split_index * num_samples
    for i in range(start, start + num_samples):
        depth_image = load_image(depth_path + f"/depth_img{i}.png", depth=True)
        color_image = load_image(rgb_path + f"/color_img{i}.png")

        # Use opencv visualizer 
        image = color_image
        cv2.imshow('Color', image[:, :, ::-1]) 
        cv2.imshow('Depth', depth_image)
        print(np.max(depth_image))
        key = cv2.waitKey((int(1000 / fps))) & 0xFF 

        if key == ord('q'):
            return False  
    
    return True 

if __name__ == '__main__': 
    split_num = 3
    fps = 30
    _ = viz_split(split_num, fps)