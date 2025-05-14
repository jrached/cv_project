#!/bin/user/env python3 

import os 
import sys
import subprocess
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
import cv2 
from PIL import Image 
import torch 

# Imports from project 
from scene_flow import load_image, load_poses, main

MAX_DEPTH = 8

def make_generators_extended(path_to_split, split_index, num_samples=500):
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
            yield np.clip(load_image(path_to_split + f'/depth/depth_img{i}.png', depth=True) / 1000, 0.0, MAX_DEPTH) # Convert to meters

    def rgb_generator(): 
        for i in range(start, start + num_samples): 
            yield load_image(path_to_split + f'/color/color_img{i}.png') 
        
    quad_poses = load_poses(path_to_split + f'/poses/quad_poses_rdp.csv') 
    def quad_pose_generator(): 
        for pose in quad_poses: 
            yield pose 

    scout_poses = load_poses(path_to_split + f'/poses/scout_poses_rdp.csv') 
    def scout_pose_generator(): 
        for pose in scout_poses: 
            yield pose 

    dog_poses = load_poses(path_to_split + f'/poses/dog_poses_rdp.csv') 
    def dog_pose_generator(): 
        for pose in dog_poses: 
            yield pose 

    depth_intrinsics = pd.read_csv(path_to_split + "/../intrinsics/depth_intrinsics.csv").to_numpy().reshape(3, 3)

    return depth_generator(), quad_pose_generator(), scout_pose_generator(), dog_pose_generator(), rgb_generator(), depth_intrinsics


def compute_trajectory_error(path_to_flow, depth_generator, quad_pose_generator, scout_pose_generator, dog_pose_generator, rgb_generator, depth_intrinsics, cam_to_quad):
    cap = cv2.VideoCapture(path_to_flow)

    # Check if video opened successfully 
    if not cap.isOpened():
        print(f"Error: Could not open video file.")
    else: 
        print(f"Video file, opened successfully!")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n Total frames: {frame_count}, FPS: {fps}")

    error_arr = []
    for depth, quad_pose, scout_pose in zip(depth_generator, quad_pose_generator, scout_pose_generator):

        # Get video frame 
        ret, flow_frame = cap.read() 
        if type(flow_frame) != np.ndarray: 
            error_np = np.array(error_arr)
            return error_np, np.mean(error_np), np.std(error_np)

        if not ret: 
            print("End of video or error ocurred.")
            break

        # Compress rgb image into grayscale by averaging each channel
        gray_flow_frame = 0.33 * flow_frame[:, :, 0] + 0.33 * flow_frame[:, :, 1] + 0.33 * flow_frame[:, :, 2]
        gray_flow_frame = gray_flow_frame.reshape(-1,) 

        # Generate pixel coordinates
        h, w, _ = flow_frame.shape
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        pixel_coords = np.stack((xs, ys), axis=0).reshape(-1, 2)
        pixel_coords = np.hstack((pixel_coords, np.ones((h*w, 1))))

        # Unproject onto 3D only pixels with depth that were also picked up by residual flow 
        flat_depth = depth.flatten()
        valid_mask = np.isfinite(flat_depth) & (flat_depth > 0) & (gray_flow_frame > 0) 
        valid_depth = flat_depth[valid_mask] 
        valid_coords = pixel_coords[valid_mask]
        flat_shape = valid_coords.shape[0] 
        if flat_shape == 0: continue # If we don't have valid pixel coords 
        points_3d = valid_depth.reshape(-1, 1) * (np.linalg.inv(depth_intrinsics) @ valid_coords.T).T 
        points_3d = np.hstack((points_3d, np.ones((flat_shape, 1))))

        # Transform into world frame 
        extrinsics = quad_pose @ cam_to_quad 
        points_wf = (extrinsics @ points_3d.T).T

        # Compute average pose of moving obstacle 
        avg_pose = np.mean(points_wf, axis=0)[:3]
        error = np.linalg.norm(np.abs(avg_pose - scout_pose[:3, 3]))
        error_arr.append(error)
    
    error_np = np.array(error_arr)

    return error_np, np.mean(error_np), np.std(error_np)

def test_split(split_index):
    cam_to_quad = T_FLURDF
    path_to_split = "/home/jrached/cv_project_code/project/data/highbay1"
    path_to_flow = f"/home/jrached/cv_project_code/project/data/highbay_out/split{split_index}/flow_masked_image.mp4"
    depth_generator, quad_pose_generator, scout_pose_generator, dog_pose_generator, rgb_generator, depth_intrinsics = make_generators_extended(path_to_split, split_index)
    error_arr, mean_error, std_error = compute_trajectory_error(path_to_flow, depth_generator, quad_pose_generator, scout_pose_generator, dog_pose_generator, rgb_generator, depth_intrinsics, cam_to_quad)
    return error_arr, mean_error, std_error 

if __name__ == '__main__':
    ########### To run: ####################################################
    #python3 scene_flow.py --model=models/raft-things.pth --path=data/highbay1 
    ########################################################################
    split_index = 3
    flow_thresh = 5.0 
    alpha = 0.8

    threshs = np.linspace(1.0, 5.0, 20)
    for thresh in threshs:
        main(alpha, thresh, split_index)
        error_arr, mean_error, std_error = test_split(split_index)
        print(f"For thresh: {thresh} achieved mean error: {mean_error}, std: {std_error},")
        # print(f"Array length: {len(error_arr)}")
