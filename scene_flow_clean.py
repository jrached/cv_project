#!/usr/bin/env python3

"""
TODO: 
    1. Add a viz_path fild to args 
    2. Modify save_frames() so that it writes the datat to the training set and the viz directory
    3. Get num_frames in video
    4. Move floros code to tis own file, 
       reading/writing code to its own file,
       visualizing code to its own file,
       keep this as the main file 
    5. Make a function to store poses and twist in the input training data folder 
    6. Make base threshold a param
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

# Get RAFT's functions
sys.path.append('../RAFT/core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'
MAX_DEPTH = 12 # 8 
PX_RADIUS = 0.35

#############################
### Data Extraction Code ####
#############################
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

# Create generators for depth, color, and poses 
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
            yield load_image(depth_path + f'/depth_img{i}.png', depth=True) / 1000
    
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

def flow_to_image(outputs):
    return  [outputs[0], 
            flow_viz.flow_to_image(outputs[1]), 
            flow_viz.flow_to_image(outputs[2]), 
            flow_viz.flow_to_image(outputs[3]), 
            flow_viz.flow_to_image(outputs[4]), 
            outputs[5], 
            outputs[6]]

def run_floros(args, alpha, flow_thresh, viz=False):
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
    depth_generator, bd_pose_generator, px_pose_generator, rgb_generator, intrinsics = make_generators(args.data_path)
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
            save_frames(outputs, args.out_path, frame_index)
            frame_index += 1

            # Optionally visualize flow online 
            if viz: 
                viz_frame(img, outputs[6])


##########################
### Visualization Code ###
##########################
def write_to_video(frames_path, out_path, videos, num_frames, fps, frame_size):

    # Get openCV video writers 
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video_writers = [cv2.VideoWriter(os.path.join(out_path, f"{vid_name}.mp4"), fourcc, fps, frame_size, isColor=True) for vid_name in videos]
    
    for i in range(num_frames):
        for j, vw in enumerate(video_writers):
            img = load_image(os.path.join(frames_path, f"{videos[j]}/image{i}.png"))
            vw.write(img) 

    for vw in video_writers: 
        vw.release()
    cv2.destroyAllWindows()

def viz_frame(img, frame, is_image):
    # map flow to rgb image
    if not is_image:
        frame = flow_viz.flow_to_image(frame)

    img_flo = np.concatenate([img[:, :, [2, 1, 0]], frame[:, :, [2, 1, 0]]], axis=0)

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
    
        # Get video properties (e.g., frame count and frame width)
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
                        pad = 255 * np.ones((row_height, stacked_width - row_width, row_channels), dtype=np.uint8)
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

def get_intrinsics(intrinsics_path):
    return pd.read_csv(intrinsics_path).to_numpy().reshape(3, 3)

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


def save_frames(outputs, out_path, index):
    """
    Saves masks in specified directory

    TODO: Also needs to save poses and twists
    """
    # Extract data
    img, geo_flow, raft_flow, residual_flow, processed_flow, noisy_mask, noiseless_mask = outputs

    ##########################
    ### SAVE TRAINING DATA ###
    ##########################
    
    # Make inputs and targets directory 
    inputs = out_path + '/inputs' 
    targets = out_path + '/targets'

    # Compute zero based index (there's a cleaner way to do this)
    num_digits = 5
    index_str = str(index) 
    zeros = "".join(["0" for _ in range(num_digits - len(index_str))])
    lex_index = zeros + index_str 

    # Convert images to uint8 
    noisy_mask = noisy_mask.astype(np.uint8)
    noiseless_mask = noiseless_mask.astype(np.uint8)
    
    #Save data 
    noisy_mask = Image.fromarray(noisy_mask, mode='RGB') #
    noisy_mask.save(inputs + f"/frame{lex_index}.png")
    noiseless_mask = Image.fromarray(noiseless_mask, mode='RGB')
    noiseless_mask.save(targets + f"/frame{lex_index}.png") 

############
### Main ###
############

def main(alpha, flow_thresh, test_num, only_viz=False):

    fps = 15 
    frame_size = (848, 480) # OpenCV uses (width, height) order
    data_path = f'/home/jrached/cv_project_code/project/data/filter_net/dataset/test{test_num}'
    out_path = f'/home/jrached/cv_project_code/project/data/filter_net/processed_flow/test{test_num}'
    viz_path = '/home/jrached/cv_project_code/project/data/filter_net/viz'
    video_path = f'{viz_path}/video/test{test_num}'
    frames_path = f'{viz_path}/frames/test{test_num}'
    videos = ["original", "geometric_flow", "raft_flow", "residual_flow", "processed_flow", "noisy_mask", "noiseless_mask"]
    viz_live = False
    only_viz = False

    if only_viz: 
        num_frames = None #TODO: Get the number of frames in each video 
        write_to_video(frames_path, video_path, fps, num_frames, frame_size)
        play_video(viz_path, videos)

    else: 
        ### RAFT Params 
        path_to_raft = '/home/jrached/cv_project_code/RAFT/'
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--data_path', help="input dataset")
        parser.add_argument('--out_path', help="path to store processed data")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        raft_args = parser.parse_args()
        raft_args.model = path_to_raft + raft_args.model 

        # Make inputs and targets directory 
        inputs = out_path + '/inputs' 
        targets = out_path + '/targets'
        empty_out_dir(out_path)
        subprocess.run(["mkdir", "-p", inputs, targets])

        # Copy bd poses and twists to inputs directory
        subprocess.run(["cp", data_path + '/poses/bd_poses.csv', out_path + '/inputs/.'])
        subprocess.run(["cp", data_path + '/poses/bd_twists.csv', out_path + '/inputs/.'])

        # Get extrinsics 
        # depth_intrinsics = get_intrinsics('/home/jrached/cv_project_code/project/data/highbay1/intrinsics/depth_intrinsics.csv')

        # Run RAFT and scene flow 
        run_floros(raft_args, alpha, flow_thresh, viz=viz_live)


if __name__=="__main__":
    ######## To run: ####################################################
    # python3 scene_flow.py --model=models/raft-things.pth --data_path=data/kimera2 --out_path=/desired/out/path
    # python3 scene_flow.py --model=models/raft-things.pth --data_path=data/highbay1 --out_path=/desired/out/path
    # python3 scene_flow.py --model=models/raft-things.pth --data_path=data/filter_net/dataset/test1 --out_path=/desired/out/path
    #####################################################################

    # Residuals params
    alpha = 0.8
    flow_thresh = 5.0

    # General params
    num_tests = 7

    for test_num in range(1, num_tests + 1):
        main(alpha, flow_thresh, test_num)

