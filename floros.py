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

# Get RAFT's functions
sys.path.append('../RAFT/core')
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'
MAX_DEPTH = 12 # 8 
PX_RADIUS = 0.35

########################
### HELPER FUNCTIONS ###
########################
def prep_image(image): 
    """
    Preps image to be right dimensions for RAFT optical flow model 
    """
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    return image[None].to(DEVICE)  

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

def flow_to_image(outputs):
    return  [outputs[0], 
            flow_viz.flow_to_image(outputs[1]), 
            flow_viz.flow_to_image(outputs[2]), 
            flow_viz.flow_to_image(outputs[3]), 
            flow_viz.flow_to_image(outputs[4]), 
            outputs[5], 
            outputs[6]]

def project_pose(img_shape, bd_pose, px_pose, intrinsics, quad_to_cam):
    """
    Project the pose of the dynamic obstacle onto the image to produce a mask 

    Inputs: 
        - img_shape, depth image height and width
        - Pose
        - Intrinsics
        - quad_to_cam transform  

    Outputs: 
        - masked image 
    """

    # Get height and width  
    h, w = img_shape

    # Project position 
    pos = px_pose[:4, 3].reshape(4, 1)    
    extrinsics = (quad_to_cam @ np.linalg.inv(bd_pose))    
    points_3d = (extrinsics @ pos)[:3, :]
    # depth = points_3d[2:]
    pix_coords = intrinsics @ points_3d
    depth = pix_coords[2:].copy()
    pix_coords /= depth
    pix_coords = pix_coords[:2, :] 

    # Make mask
    obj_radius = int(intrinsics[0, 0] * PX_RADIUS / depth[0, 0])
    img = np.zeros(img_shape, dtype=np.uint8) 
    u, v = int(pix_coords[0, 0]), int(pix_coords[1, 0]) 
    if 0 <= u < w and 0 <= v < h: 
        cv2.circle(img, (u, v), obj_radius, 255, thickness=-1)

    return img 


######################
### CORE FUNCTIONS ###
######################
def scene_flow(curr_depth, prev_depth, curr_pose, prev_pose, intrinsics, inv_intrinsics, cam_to_rover=np.eye(4)):
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

    # Remove coords with no depth measurements 
    prev_depth = prev_depth.flatten()
    valid_mask = np.isfinite(prev_depth) & (prev_depth > 0) & (prev_depth < MAX_DEPTH)
    valid_depth = prev_depth[valid_mask]
    valid_coords_c1 = pixel_coords_c1[valid_mask, :]
    flat_shape = valid_coords_c1.shape[0]
    
    # Unproject pixel coords to 3D scene: X_c1 = Z * K**-1 @ x_pix1 
    points_c1f = valid_depth.reshape(-1, 1) * (inv_intrinsics @ valid_coords_c1.T).T
    points_c1f = np.hstack((points_c1f, np.ones((flat_shape, 1))))         

    # Transform 3D poitns from frame 1 frame to world frame 
    extrinsics =  prev_pose @ cam_to_rover
    points_wf = (extrinsics @ points_c1f.T).T 

    # Transform to frame 2 frame 
    extrinsics = np.linalg.inv(cam_to_rover) @ np.linalg.inv(curr_pose) 
    points_c2f = (extrinsics @ points_wf.T).T[:, :-1] 
    
    # Project 3D points to frame 2 image 
    points_c2f = points_c2f / points_c2f[:, 2:3]
    pixel_coords_c2 = (intrinsics @ points_c2f.T).T 

    # Compute flow as displacement of pixel coordinates 
    geo_flow_flat = (pixel_coords_c2[:, :-1] - valid_coords_c1[:, :-1])

    # Rebuild image to right shape without bad values 
    output = np.full((h * w, 2), fill_value=0.0)
    output[valid_mask] = geo_flow_flat 
    geo_flow = output.reshape(h, w, 2)

    return geo_flow, curr_depth, curr_pose 

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
    base_thresh = 0.0 # TODO make a param 
    flow_norms = np.linalg.norm(scaled_flow, axis=-1).reshape(h, w, 1)
    mask = flow_norms.copy()
    mask[flow_norms < base_thresh + flow_thresh * curr_depth.reshape(h, w, 1) / 8] = 0
    mask[flow_norms >= base_thresh + flow_thresh * curr_depth.reshape(h, w, 1) / 8] = 1
    mask = np.broadcast_to(mask, (h, w, 2))
    masked_flow = mask * scaled_flow 

    return masked_flow, prev_flow  

def generate_dynamic_mask(raw_img, processed_flow):
    h, w, _ = raw_img.shape
    masked_frame = raw_img.copy()
    processed_flow_norms = np.broadcast_to(np.linalg.norm(processed_flow, axis=-1).reshape(h, w, 1), (h, w, 3))
    masked_frame[processed_flow_norms == 0] = 0 
    masked_frame[processed_flow_norms != 0] = 255
    return masked_frame

def generate_noiseless_mask(masked_frame, depth_shape, prev_bd_pose_copy, prev_px_pose, intrinsics):
    noisy_masked_frame = masked_frame.copy() 
    noise_mask = project_pose(depth_shape, prev_bd_pose_copy, prev_px_pose, intrinsics, np.linalg.inv(T_FLURDF))
    noisy_masked_frame[noise_mask == 0] = 0
    return noisy_masked_frame

def floros(model, alpha, flow_thresh, curr_depth, curr_bd_pose, curr_px_pose, img, intrinsics, inv_intrinsics, prev_depth, prev_bd_pose, prev_px_pose, prev_img, prev_flow):
    prev_bd_pose_copy = prev_bd_pose.copy()
    
    #compute geometric flow 
    geo_flow, prev_depth, prev_bd_pose = scene_flow(curr_depth, prev_depth, curr_bd_pose, prev_bd_pose, intrinsics, inv_intrinsics, cam_to_rover=T_FLURDF)

    # Compute RAFT flow  
    raft_flow, prev_img = raft_optical_flow(model, img, prev_img)

    # Compute residuals  
    residual_flow = raft_flow - geo_flow
    
    # Clean up signal 
    processed_flow, prev_flow = post_process_flow(curr_depth, prev_depth, residual_flow, prev_flow, alpha, flow_thresh, scale_by_depth=True)

    # Generate dynamic obstacle masks
    noisy_mask = generate_dynamic_mask(img, processed_flow)
    noiseless_mask = generate_noiseless_mask(noisy_mask, curr_depth.shape, prev_bd_pose_copy, prev_px_pose, intrinsics)
    prev_px_pose = curr_px_pose

    # For convenience, store outputs in a list
    outputs = [img, 
               geo_flow, 
               raft_flow, 
               residual_flow, 
               processed_flow, 
               noisy_mask, 
               noiseless_mask]

    return prev_depth, prev_bd_pose, prev_px_pose, prev_img, prev_flow, outputs