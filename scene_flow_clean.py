#!/usr/bin/env python3

"""
TODO: 
    1. Make base threshold a param
"""

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
import glob 

from processing_utils import save_frames, make_generators, extract_data_from_bags, save_from_bag, make_generators_from_bags
from viz_utils import write_to_video, viz_frame, play_video
from floros import floros, flow_to_image, prep_image 

# Get RAFT's functions
sys.path.append('../RAFT/core')
from raft import RAFT

DEVICE = 'cuda'
MAX_DEPTH = 12 # 8 
PX_RADIUS = 0.35

###############################
### Actual Scene Flow Code ####
###############################
def generate_dataset(args, alpha, flow_thresh, videos, viz=False):
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
    depth_generator, bd_pose_generator, px_pose_generator, rgb_generator, intrinsics = make_generators_from_bags(args)
    inv_intrinsics = np.linalg.inv(intrinsics)

    with torch.no_grad():
        # Initialize pose, depth, img.
        for prev_depth, prev_bd_pose, prev_px_pose, prev_img in zip(depth_generator, bd_pose_generator, px_pose_generator, rgb_generator):
            prev_img = prep_image(prev_img)
            h, w = prev_depth.shape 
            prev_flow = np.zeros((h, w, 2))
            break

        frame_index = 0 #TODO: Can you wrap enumerate() around zip() and prepend frame_index to curr_depth, ... ? 
        for curr_depth, curr_bd_pose, curr_px_pose, img in zip(depth_generator, bd_pose_generator, px_pose_generator, rgb_generator):
           
            # Run floros
            prev_depth, prev_bd_pose, prev_px_pose, prev_img, prev_flow, outputs = floros(model, alpha, flow_thresh, curr_depth, curr_bd_pose, curr_px_pose, img, intrinsics, inv_intrinsics, prev_depth, prev_bd_pose, prev_px_pose, prev_img, prev_flow)

            # Save outputs 
            outputs = flow_to_image(outputs) 
            save_frames(args, outputs, videos, frame_index)
            frame_index += 1

            # Optionally visualize flow online 
            if viz: 
                viz_frame(img, outputs[6])
    
############
### Main ###
############
def main(alpha, flow_thresh, only_viz=False, viz_live=False):

    # General params
    fps = 15 
    frame_size = (848, 480) # OpenCV uses (width, height) order
    videos = ["original", "geometric_flow", "raft_flow", "residual_flow", "processed_flow", "noisy_mask", "noiseless_mask"]

    # RAFT Params 
    path_to_raft = '/home/jrached/cv_project_code/RAFT/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--data_path', help="input dataset")
    parser.add_argument('--out_path', help="path to store processed data")
    parser.add_argument('--viz_path', help="path to store data for visualization purposes")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    raft_args = parser.parse_args()
    raft_args.model = path_to_raft + raft_args.model 

    if only_viz: 
        write_to_video(raft_args, videos, fps, frame_size)
        play_video(raft_args.viz_path, videos)

    else: 
        generate_dataset(raft_args, alpha, flow_thresh, videos, viz=viz_live)


if __name__=="__main__":
    ######## To run: ####################################################
    # python3 scene_flow_clean.py --model=models/raft-things.pth --data_path=data/kimera2 --out_path=/desired/out/path --viz_path=/desired/viz/path
    # python3 scene_flow_clean.py --model=models/raft-things.pth --data_path=data/highbay1 --out_path=/desired/out/path --viz_path=/desired/viz/path
    # python3 scene_flow_clean.py --model=models/raft-things.pth --data_path=data/filter_net/new_dataset/test1 --out_path=data/filter_net/new_processed_flow/test1 --viz_path=data/filter_net/viz/test1
    # Direct version, give bag directory and processes everything from there: python3 scene_flow_clean.py --model=models/raft-things.pth --data_path=data/filter_net/new_bags/test1/bag_1 --out_path=data/filter_net/new_processed_flow/test1 --viz_path=data/filter_net/viz/test1
    #####################################################################

    # Residuals params
    alpha = 0.8
    flow_thresh = 5.0
    main(alpha, flow_thresh)

