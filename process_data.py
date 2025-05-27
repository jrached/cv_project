#!/bin/usr/env python3 
import sys 
import glob 
from robotdatapy.transform import T_FLURDF 
import numpy as np 
import pandas as pd 
import cv2 
from PIL import Image 
import subprocess 

sys.path.append('.') 
from scene_flow import load_image 

MAX_DEPTH = 8 
PX_RADIUS = 0.35 

def make_generators(data_path):
    """
    Makes generators from data from a given path. 

    Inputs:
        - Path to a video 

    Outputs: 
        - depth_generator
        - pose_generator 
        - rgb_generator 
    """

    # Create paths 
    depth_path = data_path + '/depth'
    color_path = data_path + '/color'
    bd_poses_path = data_path + '/poses/bd_poses.csv'
    px_poses_path = data_path + '/poses/px_poses.csv'
    intrinsics_path = data_path + '/intrinsics/depth_intrinsics.csv'

    # Make depth image generator 
    num_frames = len(glob.glob(depth_path + '/*'))
    def depth_generator():
        for i in range(num_frames):
            yield np.clip(load_image(depth_path + f'/depth_img{i}.png', depth=True) / 1000, -0.5, MAX_DEPTH)
    
    # Make color image generator 
    def color_generator():
        for i in range(num_frames): 
            yield load_image(color_path + f'/color_img{i}.png') 

    # Make pose generator 
    bd_poses = pd.read_csv(bd_poses_path).to_numpy().reshape(-1, 4, 4)
    def bd_pose_generator():
        for pose in bd_poses:
            yield pose 
    
    # Make pose generator 
    px_poses = pd.read_csv(px_poses_path).to_numpy().reshape(-1, 4, 4)
    def px_pose_generator():
        for pose in px_poses:
            yield pose 

    # Get intrinsics 
    intrinsics = pd.read_csv(intrinsics_path).to_numpy().reshape(3, 3)

    return depth_generator(), bd_pose_generator(), px_pose_generator(), color_generator(), intrinsics  


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

def save_image(img, img_path, index):
    img = Image.fromarray(img) # Highbay is on RGB 
    img.save(img_path + f"/masked_img{index}.png")


def main(path, test_num=1):
    # Define paths 
    data_path = path + f'/dataset/test{test_num}'
    dest_path = path + f'/processed/test{test_num}/masked'
    subprocess.run(['rm', '-r', dest_path])
    subprocess.run(['mkdir', '-p', dest_path]) 
    
    # Make generators 
    depth_generator, bd_pose_generator, px_pose_generator, color_generator, intrinsics = make_generators(data_path)

    # Iterate through generators  
    img_index = 0
    quad_to_cam = np.linalg.inv(T_FLURDF) 
    for depth_img, bd_pose, px_pose, color_img in zip(depth_generator, bd_pose_generator, px_pose_generator, color_generator): 
        masked_img = project_pose(depth_img.shape, bd_pose, px_pose, intrinsics, quad_to_cam)

        # Save image 
        save_image(masked_img, dest_path, img_index)

        # Update index 
        img_index += 1


def viz_mask(path, test_num=1, fps=15):
    """
    Visualize original and masked image
    """

    # Paths to images.
    color_path = path + f'/dataset/test{test_num}/color'
    masked_path = path + f'/processed/test{test_num}/masked'

    num_imgs = len(glob.glob(masked_path + '/*'))
    for i in range(num_imgs):
        masked_image = load_image(masked_path + f"/masked_img{i}.png")
        color_image = load_image(color_path + f"/color_img{i}.png")

        # Use opencv visualizer 
        cv2.imshow('Color', color_image[..., ::-1]) 
        cv2.imshow('Masked', masked_image)
        key = cv2.waitKey((int(1000 / fps))) & 0xFF 

        if key == ord('q'):
            return False  
    
    return True 


if __name__=='__main__':
    path = '/home/jrached/cv_project_code/project/data/filter_net'
    # main(path, test_num=1)
    viz_mask(path, test_num=1, fps=15)






