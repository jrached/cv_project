#!/usr/bin/env python3

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

# Get RAFT's functions
sys.path.append('../RAFT/core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


# From project 
sys.path.append('../project')
from viz_data import load_image, load_poses

DEVICE = 'cuda'
MAX_DEPTH = 8

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
            yield np.clip(load_image(path_to_split + f'/depth/depth_img{i}.png', depth=True) / 1000, 0.5, MAX_DEPTH) # Convert to meters

    def rgb_generator(): 
        for i in range(start, start + num_samples): 
            yield load_image(path_to_split + f'/color/color_img{i}.png') 
        
    poses = load_poses(path_to_split + f'/poses/poses_rdp.csv')
    def pose_generator(): 
        for pose in poses: 
            yield pose 

    return depth_generator(), pose_generator(), rgb_generator()


###############################
### Actual Scene Flow Code ####
###############################

def smooth_flow(alpha, curr_flow, prev_flow):
    """
    Running average to smooth flow signal
    
    Inputs: 
        - alpha, running average weight
        - Current flow value 
        - Previous flow value 
    Outputs: 
        - Smoothed value 
    """
    return alpha * curr_flow + (1 - alpha) * prev_flow

def scene_flow(curr_depth, prev_depth, curr_pose, prev_pose, intrinsics, inv_intrinsics):
    """
    Computes the scene flow for each frame. 

    inputs: 
        - Current depth image
        - Previous depth image 
        - Current camera pose (extrinsics) 
        - Previous camera pose (extrinsics) 
        - Camera intrinsics 
        - Inverse of camera intrinsics (to avoid computation)  
    outputs: 
        - The scene flow for each frame
        - Current depth, to be prev depth on next iteration 
        - Current pose, to be prev pose on next iteration 
    """ 
    # Generate pixel coordinates for frame 1 
    h, w = prev_depth.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    pixel_coords_c1 = np.stack((xs, ys), axis=-1).reshape(-1, 2)
    pixel_coords_c1 = np.hstack((pixel_coords_c1, np.ones((h*w, 1))))

    # Unproject pixel coords to 3D scene: X_c1 = Z * K**-1 @ x_pix1 
    prev_depth = prev_depth[pixel_coords_c1[:, 1].astype(int), pixel_coords_c1[:, 0].astype(int)]
    points_c1f = prev_depth.reshape(h*w, 1) * (inv_intrinsics @ pixel_coords_c1.T).T
    points_c1f = np.hstack((points_c1f, np.ones((h*w, 1))))         

    # Transform 3D poitns from frame 1 frame to world frame 
    extrinsics = prev_pose
    points_wf = (extrinsics @ points_c1f.T).T 

    # Transform to frame 2 frame 
    extrinsics = np.linalg.inv(curr_pose)
    points_c2f = (extrinsics @ points_wf.T).T[:, :-1] 
    
    # Project 3D points to frame 2 image 
    points_c2f = points_c2f / points_c2f[:, 2:3]
    pixel_coords_c2 = (intrinsics @ points_c2f.T).T 

    # Compute flow as displacement of pixel coordinates 
    geo_flow = (pixel_coords_c2[:, :-1] - pixel_coords_c1[:, :-1]).reshape(h, w, 2)

    return geo_flow, curr_depth, curr_pose 

def prep_image(image): 
    """
    Preps image to be right dimensions for RAFT optical flow model 
    """
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    return image[None].to(DEVICE)  

def raft_optical_flow(model, img, prev_img):
    """
    Computes RAFT optical flow 

    inputs: 
        - RAFT model 
        - current image
        - previous image (already prepped for RAFT)
    outputs: 
        - RAFT optical flow
        - current image, to be next iteration's previous image 
    """
    img = prep_image(img)
    padder = InputPadder(prev_img.shape)
    prev_img, img = padder.pad(prev_img, img)
    _, flow_up = model(prev_img, img, iters=20, test_mode=True)
    raft_flow = flow_up.permute(2, 3, 1, 0).squeeze(dim=-1).cpu().numpy()

    return raft_flow, img 

def post_process_flow(curr_depth, prev_depth, flow, prev_flow, alpha, flow_thresh, scale_by_depth=False):
    """
    Postprocess flow by smoothing and removing noise under a threshold. Optionally, scale the flow by depth in order to detect flow 
    at longer distances (mixed performance). 

    Inputs: 
        - Current depth, float 
        - Previous depth, float
        - Current flow, float  
        - Flow threshold, float 
        - Scale by depth, bool, (Optional) 
    """
    # Smooth flow 
    flow = smooth_flow(alpha, flow, prev_flow)
    prev_flow = flow 

    # Scale flow by depth 
    h, w = curr_depth.shape
    if scale_by_depth:
        avg_depth = (1 / MAX_DEPTH) * (curr_depth.reshape(h, w, 1) + prev_depth.reshape(h, w, 1)) / 2
        scaled_flow = avg_depth * flow 
    else: 
        scaled_flow = flow
        
    # Threshold to remove noisy flow (norm of flow vectors)
    flow_norms = np.linalg.norm(scaled_flow, axis=-1).reshape(h, w, 1)
    mask = flow_norms.copy()
    mask[flow_norms < flow_thresh] = 0
    mask[flow_norms >= flow_thresh] = 1
    mask = np.broadcast_to(mask, (h, w, 2))
    masked_flow = mask * scaled_flow 

    return masked_flow, prev_flow  

def raft_and_scene_flow(args, intrinsics, video_writers, split_index=14, viz=False):
    """
    Computes residuals between RAFT optical flow and geometric optical flow.  

    inputs: 
        - args, arguments for RAFT
        - Camera intrinsics 
        - OpenCV video writers to save flow videos
        - Split_index (Optional)
        - Viz, bool, whether to visualize flow frames as they are generated (Optional)
    outputs: 
        - The residuals for each frame 
    """
    # Initialize RAFT model  
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    # Get data generators 
    path_to_split = args.path 
    depth_generator, pose_generator, rgb_generator = make_generators(path_to_split, split_index)
    inv_intrinsics = np.linalg.inv(intrinsics)

    # Extract openCV video writers 
    vw1, vw2, vw3, vw4, vw5 = video_writers 

    with torch.no_grad():
        # Initialize pose, depth, img.
        for prev_depth, prev_pose, prev_img in zip(depth_generator, pose_generator, rgb_generator):
            prev_img = prep_image(prev_img)
            break

        # Smoothing parameters
        h, w = prev_depth.shape 
        prev_flow = np.zeros((h, w, 2))
        alpha = 0.4
        flow_thresh = 4

        for curr_depth, curr_pose, img in zip(depth_generator, pose_generator, rgb_generator):
            vw1.write(img[:, :, ::-1])

            # Compute geometric flow 
            geo_flow, prev_depth, prev_pose = scene_flow(curr_depth, prev_depth, curr_pose, prev_pose, intrinsics, inv_intrinsics)
            vw2.write(flow_viz.flow_to_image(geo_flow)[:, :, ::-1])

            # Compute RAFT flow  
            raft_flow, prev_img = raft_optical_flow(model, img, prev_img)
            vw3.write(flow_viz.flow_to_image(raft_flow)[:, :, ::-1])

            # Compute residuals  
            flow = raft_flow - geo_flow
            vw4.write(flow_viz.flow_to_image(flow)[:, :, ::-1])

            # Clean up signal 
            processed_flow, prev_flow = post_process_flow(curr_depth, prev_depth, flow, prev_flow, alpha, flow_thresh)
            vw5.write(flow_viz.flow_to_image(processed_flow)[:, :, ::-1])

            # Visualize flow  
            if viz: 
                viz_frame(img, processed_flow)

    for vw in video_writers: 
        vw.release()
    cv2.destroyAllWindows()


##########################
### Visualization Code ###
##########################
def viz_frame(img, flo):
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img[:, :, [2, 1, 0]], flo[:, :, [2, 1, 0]]], axis=0)

    cv2.imshow('image', img_flo/255.0)
    
    # Wait for fps 
    key = cv2.waitKey(1) & 0xFF 

    # Break if q is pressed or video finished
    if key == ord('q'):
        return False
    
    return True

def play_video(dir_path, videos):
    """
    Plays videos of all stages of generating flow residuals (5 videos).

    Inputs: 
        - Path to videos
        - Name of videos

    Outputs :
        - 3x3 video grid 
    """

    num_vids = len(videos) 
    cap = [None] * num_vids
    ret = [None] * num_vids 
    frame = [None] * num_vids 

    for i, v in enumerate(videos):
        cap[i] = cv2.VideoCapture(dir_path + v)

        # Check if video opened successfully 
        if not cap[i].isOpened():
            print(f"Error: Could not open video file {v[1:]}.")
        else: 
            print(f"Video file, {v[1:]}, opened successfully!")
    
        # Get video properties (e.g., frame cound and frame width)
        frame_count = int(cap[i].get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap[i].get(cv2.CAP_PROP_FPS)
        print(f"\nVideo: {v[1:]}, Total frames: {frame_count}, FPS: {fps}")

    # Display each frame as 3x3 grid 
    while True: 
        for i, c in enumerate(cap):
            if c is not None:
                ret[i], frame[i] = c.read() 

        i, name = 0, "" 
        while i < num_vids:
            if not ret[i]: 
                print("End of video or error ocurred.")
                break 

            # Extract video names 
            v_name = videos[i][1:].split(".")[0]
            name += v_name + ", "

            # Stack frames 
            if i % 3 != 0: 
                frame_row = np.hstack((frame_row, frame[i]))

            # Stack rows
            if i % 3 == 0 or i == num_vids - 1:
                if i == 0:
                    # Initialize row 
                    frame_row = frame[0]
                elif i <= 3: 
                    # Initialize grid 
                    stacked_frame = frame_row
                else: 
                    _ , stacked_width, _ = stacked_frame.shape
                    row_height, row_width, row_channels = frame_row.shape

                    # If row doesn't have 3 videos, pad row 
                    if row_width != stacked_width: 
                        pad = np.zeros((row_height, stacked_width - row_width, row_channels), dtype=np.uint8)
                        frame_row = np.hstack((frame_row, pad)) 

                    stacked_frame = np.vstack((stacked_frame, frame_row))

                # Initialize new row                 
                frame_row = frame[i]
            i += 1
                
        # Display the frame 
        cv2.imshow(name, stacked_frame) 

        # Wait for (1 / fps) ms for key press to continue or exit if 'q' is pressed
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break 
    
    for i in range(num_vids):
        cap[i].release()
    cv2.destroyAllWindows()

#############
### Utils ###
#############

def empty_out_dir(out_path):
    subprocess.run(["rm", "-rf", out_path + "/*"])

if __name__=="__main__":
    ######## To run: ####################################################
    # python3 scene_flow_clean.py --model=models/raft-things.pth --path=data/kimera2 
    #####################################################################
    out_path = '/home/jrached/cv_project_code/project/data/out'
    videos = ["/original.mp4", "/geometric_flow.mp4", "/raft_flow.mp4", "/residual_flow.mp4", "/processed_flow.mp4"]
    only_viz = True

    if not only_viz: 
        ### Params 
        split_number = 13
        fps = 15 # 30.0
        frame_size = (640, 480) # OpenCV uses (width, height) order

        ### RAFT Params 
        path_to_raft = '/home/jrached/cv_project_code/RAFT/'
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()
        args.model = path_to_raft + args.model 

        # RGB intrinsics TODO: Write to csv file 
        rgb_intrinsics = np.array([[380.80969238,   0.0,          315.84698486],
                                [0.0,            380.53787231, 238.04495239],
                                [0.0,            0.0,          1.0         ]])

        # Depth intrinsics TODO: Write these to a csv file
        depth_intrinsics = np.array([[386.88305664,   0.0,          312.75875854],
                                    [  0.0,         386.88305664, 237.32907104],
                                    [  0.0,         0.0,          1.0         ]])

        # Get openCV video writers 
        empty_out_dir(out_path)
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        video_writers = [cv2.VideoWriter(out_path + file_name, fourcc, fps, frame_size, isColor=True) for file_name in videos]

        # Run RAFT and scene flow 
        flow = raft_and_scene_flow(args, depth_intrinsics, video_writers, split_index=split_number)

    else: 
        vid_name = "/residual_flow.mp4"
        print(f"\n Video path : {out_path + vid_name}\n")
        play_video(out_path, videos)

    