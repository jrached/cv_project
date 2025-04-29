#!/usr/bin/env python3

import numpy as np 
import matplotlib.pyplot as pyplot 
from mpl_toolkits.mplot3d import Axes3D 
import robotdatapy 
import datetime as dt 
from robotdatapy.data import ImgData, PoseData 
from robotdatapy.transform import T_FLURDF 
import pandas as pd 
from scipy.spatial.transform import Rotation as Rot 

import argparse 
import os 
import cv2 
from PIL import Image 

# Get RAFT's flow visualizer 
import sys
sys.path.append('../project')
sys.path.append('../RAFT/core')
from utils import flow_viz

# From project 
from viz_data import load_image, load_poses


#############################
### Data Extraction Code ####
#############################

# Create generators for depth, color, and poses 
def make_generators(path_to_split, split_index, num_samples=1000):
    """
    Make generators for depth, color, and poses 
    from the data extracted from the bags.

    Inputs: 
        - pose_data: vehicle pose data, used for camera extrinsics
        - rgb_data: color image data, used for visualizations
        - depth_data: depth data, used for geometric optical flow

    Outputs: 
        - pose_generator
        - depth_generator
        - rgb_generator
    """ 

    path_to_split += f'/split{split_index}'

    start = split_index * num_samples
    def depth_generator():
        for i in range(start, start + num_samples): 
            yield load_image(path_to_split + f'/depth/depth_img{i}.png', depth=True) / 1000 # Convert to meters

    def rgb_generator(): 
        for i in range(start, start + num_samples): 
            yield load_image(path_to_split + f'/color/color_img{i}.png') 
        
    poses = load_poses(path_to_split + f'/poses/poses.csv')
    def pose_generator(): 
        for pose in poses: 
            yield pose 

    return depth_generator(), pose_generator(), rgb_generator()


###############################
### Actual Scene Flow Code ####
###############################

def scene_flow(depth_generator, pose_generator, rgb_generator, intrinsics, cam_to_rover):
    """
    Computes the scene flow giving a video as a list of frame objects. 
    Each object contains the extrinsics, intrinsics, depth, and image for 
    that frame. 

    inputs: 
        - frames, a video represented as a list of frame objects. 
    outputs: 
        - The scene flow for each frame.
    """

    inv_intrinsics = np.linalg.inv(intrinsics)
    rover_to_cam = np.linalg.inv(cam_to_rover) 

    for depth, pose in zip(depth_generator, pose_generator):
        prev_depth = depth
        prev_pose = pose 
        break

    h, w = prev_depth.shape 
    prev_flow = np.zeros((h, w, 2))

    alpha = 0.1
    beta = 0.01
    # frame_flow = []
    for curr_depth, curr_pose, img in zip(depth_generator, pose_generator, rgb_generator):
        # Create grid, flatten it in raster order, and homogenize it
        # pixel_coords_c1 = np.mgrid[:w, :h].reshape(h*w, 2) #TODO: Double check shape
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        pixel_coords_c1 = np.stack((xs, ys), axis=-1).reshape(-1, 2)
        pixel_coords_c1 = np.hstack((pixel_coords_c1, np.ones((h*w, 1))))

        # print(pixel_coords_c1)

        # Unproject: X_c1 = Z * K**-1 @ x_pix1 
        prev_depth_copy = prev_depth.copy() 
        prev_depth = prev_depth[pixel_coords_c1[:, 1].astype(int), pixel_coords_c1[:, 0].astype(int)]
        points_c1f = prev_depth.reshape(h*w, 1) * (inv_intrinsics @ pixel_coords_c1.T).T
        points_c1f = np.hstack((points_c1f, np.ones((h*w, 1))))         

        # Transform to world frame 
        extrinsics = prev_pose @ cam_to_rover  
        points_wf = (extrinsics @ points_c1f.T).T 

        # Transform to cam2 frame 
        extrinsics = rover_to_cam @ np.linalg.inv(curr_pose)
        points_c2f = (extrinsics @ points_wf.T).T[:, :-1] 

        # print(points_c2f)
        
        # Project 3d points to cam2 image 
        pixel_coords_c2 = (intrinsics @ points_c2f.T).T 
        pixel_coords_c2 = pixel_coords_c2 / pixel_coords_c2[:, 2:3]

        # print(pixel_coords_c2)

        # Compute flow 
        flow = np.clip((pixel_coords_c2[:, :-1] - pixel_coords_c1[:, :-1]).reshape(h, w, 2), -15, 15)
        # flow = (pixel_coords_c2[:, :-1] - pixel_coords_c1[:, :-1]).reshape(h, w, 2)
        flow = alpha * flow + (1 - alpha) * prev_flow
        prev_flow = flow 

        print(flow)
        # print(flow.shape)
        viz_video(img, flow, 30)

        #Update depth and pose 
        prev_depth = beta * curr_depth + (1 - beta) * prev_depth_copy 
        prev_pose = curr_pose 
    
    # return frame_flow



##########################
### Visualization Code ###
##########################
def viz_video(img, flo, fps):
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img[:, :, [2, 1, 0]], flo[:, :, [2, 1, 0]]], axis=0)

    cv2.imshow('image', img_flo/255.0)
    
    # Wait for fps 
    # key = cv2.waitKey(int(1000 / fps)) & 0xFF 
    key = cv2.waitKey(1) & 0xFF 

    # Break if q is pressed or video finished
    if key == ord('q'):
        return False
    
    return True


if __name__=="__main__":
    # Extract data and make generators
    split_number = 3
    path_to_split = f'/home/jrached/cv_project_code/project/data/kimera'
    depth_generator, pose_generator, rgb_generator = make_generators(path_to_split, split_number)
    
    # Depth intrinsics TODO: Write these to a csv file
    depth_intrinsics = np.array([[386.88305664,   0.0,          312.75875854],
                                [  0.0,         386.88305664, 237.32907104],
                                [  0.0,         0.0,          1.0         ]])

    # Compute cam_to_rover transform 
    T_rc1 = np.array([[0, 0, 1, 0], 
                     [0, 1, 0, 0], 
                     [-1, 0, 0, 0], 
                     [0, 0, 0, 1]]) 
    T_rc2 = np.array([[0, 1, 0, 0], 
                      [-1, 0, 0, 0], 
                      [0, 0, 1, 0], 
                      [0, 0, 0, 1]])
    T_rc = T_rc1 @ T_rc2 

    # Run scene flow 
    flow = scene_flow(depth_generator, pose_generator, rgb_generator, depth_intrinsics, T_rc) 
    
    # RGB intrinsics TODO: Write to csv file 
    rgb_intrinsics = np.array([[380.80969238,   0.0,          315.84698486],
                               [0.0,            380.53787231, 238.04495239],
                               [0.0,            0.0,          1.0         ]])
