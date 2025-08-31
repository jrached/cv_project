import os 
import sys
import subprocess
import numpy as np 
import matplotlib.pyplot as pyplot 
from mpl_toolkits.mplot3d import Axes3D 
import robotdatapy 
import datetime as dt 
from robotdatapy.data import ImgData, PoseData, GeneralData 
from robotdatapy.transform import T_FLURDF 
import pandas as pd 
from scipy.spatial.transform import Rotation as R

import argparse 
import cv2 
from PIL import Image 
import torch 
import glob 

########################
### Helper Functions ###
########################
def interpolate(depth_data, state_data, is_twist):
    """
    Interpolates state_data into depth_data using rdp's built in procedure
    """
    depth_times = depth_data.times
    interp_data = []
    for t in depth_times:
        if is_twist: 
            lin = state_data.data(t).twist.linear
            ang = state_data.data(t).twist.angular
            twist_data = np.array([lin.x, lin.y, lin.z, ang.x, ang.y, ang.z])
            interp_data.append(twist_data)
        else: 
            interp_data.append(state_data.pose(t))

    return np.stack(interp_data)

def twist_from_pose(pose, times):
    """ 
    Takes in an array of poses and returns an array of twists obtained from the 
    first order approximation of those poses 
    
    Inputs: 
        - pose, ndarray (n, 16)
        - times, array (n,)

    Outputs: 
        - twist, ndarray (n, 6) 
    """

    # Get positions from pose 
    n, _ = pose.shape 
    pos = pose[:, [3, 7, 11]]

    # Compute time intervals 
    times = times.reshape(n, 1)
    times_k = times[:-1, :]
    times_kp1 = times[1:, :]
    delta_t = times_kp1 - times_k # Note has shape (n-1, 1)

    # Compute linear velocity 
    pos_k = pos[:-1, :]
    pos_kp1 = pos[1:, :] 
    delta_pos = pos_kp1 - pos_k 
    vel = delta_pos / delta_t 
    vel = np.vstack((vel[0, :], vel)) 

    # Reconstruct rotation matrices 
    rot_mat = np.zeros((n, 3, 3)) 
    rot_mat[:, 0, 0:3] = pose[:, 0:3]
    rot_mat[:, 1, 0:3] = pose[:, 4:7]
    rot_mat[:, 2, 0:3] = pose[:, 8:11]

    # Compute change in orientation 
    rot_mat_k = rot_mat[:-1, :, :] 
    rot_mat_kp1 = rot_mat[1:, :, :]
    rot_mat_rel = np.einsum('nij,njk->nik', rot_mat_k.transpose(0, 2, 1), rot_mat_kp1)
    rotvecs = R.from_matrix(rot_mat_rel).as_rotvec() 
    omega_body = rotvecs / delta_t 
    omega_world = np.einsum('nij,nj->ni', rot_mat_k, omega_body)
    omega_world = np.vstack((omega_world[0, :], omega_world))

    # Concatenate linear and angular velocities 
    twist = np.hstack((vel, omega_world)) 
    return twist 

#############################
### EXTRACTION FUNCTIONS ####
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

def get_intrinsics(intrinsics_path):
    return pd.read_csv(intrinsics_path).to_numpy().reshape(3, 3)

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
            yield load_image(depth_path + f'/depth_img{i}.png', depth=True) / 1000 # Scale from mm to m
    
    # Make color image generator 
    def color_generator():
        for i in range(num_frames): 
            yield load_image(color_path + f'/color_img{i}.png') 

    # Make pose generator 
    bd_poses = load_poses(bd_poses_path)
    def bd_pose_generator():
        for pose in bd_poses:
            yield pose 
    
    # Make pose generator 
    px_poses = load_poses(px_poses_path)
    def px_pose_generator():
        for pose in px_poses:
            yield pose 

    # Get intrinsics 
    intrinsics = get_intrinsics(intrinsics_path)

    return depth_generator(), bd_pose_generator(), px_pose_generator(), color_generator(), intrinsics  

def save_frames(args, outputs, videos, frame_index, save_viz_data=False):
    """
    Saves frames in specified directory
    """
    # Extract data
    img, geo_flow, raft_flow, residual_flow, processed_flow, noisy_mask, noiseless_mask = outputs

    ##########################
    ### SAVE TRAINING DATA ###
    ##########################
    
    # # Make inputs and targets directory 
    out_path = args.out_path
    inputs = os.path.join(out_path, 'inputs') 
    targets = os.path.join(out_path, 'targets')

    # Compute zero based frame_index (there's a cleaner way to do this)
    num_digits = 5
    frame_index_str = str(frame_index) 
    zeros = "".join(["0" for _ in range(num_digits - len(frame_index_str))])
    lex_frame_index = zeros + frame_index_str 

    # Convert images to uint8 
    noisy_mask = noisy_mask.astype(np.uint8)
    noiseless_mask = noiseless_mask.astype(np.uint8)
    
    #Save data 
    noisy_mask = Image.fromarray(noisy_mask, mode='RGB') #
    noisy_mask.save(inputs + f"/frame{lex_frame_index}.png")
    noiseless_mask = Image.fromarray(noiseless_mask, mode='RGB')
    noiseless_mask.save(targets + f"/frame{lex_frame_index}.png") 

    #####################
    ### SAVE VIZ DATA ###
    #####################

    if save_viz_data:
        # Convert images to uint8
        img = img.astype(np.uint8)
        geo_flow = geo_flow.astype(np.uint8)
        raft_flow = raft_flow.astype(np.uint8)
        residual_flow = residual_flow.astype(np.uint8)
        processed_flow = processed_flow.astype(np.uint8)
        noisy_mask = noisy_mask.astype(np.uint8)
        noiseless_mask = noiseless_mask.astype(np.uint8)

        # Convert arrays to image 
        img = Image.fromarray(img, mode='RGB')
        geo_flow = Image.fromarray(geo_flow, mode='RGB')
        raft_flow = Image.fromarray(raft_flow, mode='RGB')
        residual_flow = Image.fromarray(residual_flow, mode='RGB')
        processed_flow = Image.fromarray(processed_flow, mode='RGB')
        noisy_mask = Image.fromarray(noisy_mask, mode='RGB')
        noiseless_mask = Image.fromarray(noiseless_mask, mode='RGB')

        # Save data 
        viz_path = args.viz_path
        frames_path = os.path.join(viz_path, f'frames')
        img.save(os.path.join(frames_path, f'{videos[0]}/frame{frame_index}'))
        geo_flow.save(os.path.join(frames_path, f'{videos[1]}/frame{frame_index}'))
        raft_flow.save(os.path.join(frames_path, f'{videos[2]}/frame{frame_index}'))
        residual_flow.save(os.path.join(frames_path, f'{videos[3]}/frame{frame_index}'))
        processed_flow.save(os.path.join(frames_path, f'{videos[4]}/frame{frame_index}'))
        noisy_mask.save(os.path.join(frames_path, f'{videos[5]}/frame{frame_index}'))
        noiseless_mask.save(os.path.join(frames_path, f'{videos[6]}/frame{frame_index}'))


##############################
### EXTRACT DATA FROM BAGS ###
##############################
def load_from_bags(bag_path): 
    # Data params 
    time_tol = 10 

    # Topics 
    bd_pose_topic = '/BD01/world'
    bd_twist_topic = '/BD01/mocap/twist'
    px_pose_topic = '/PX02/world' 
    px_twist_topic = '/PX02/mocap/twist'
    depth_topic = '/BD01/camera/depth/image_rect_raw'
    rgb_topic = '/BD01/camera/color/image_raw'
    depth_info_topic = '/BD01/camera/color/camera_info'
    rgb_info_topic = '/BD01/camera/color/camera_info'

    # For each depth, image, pose extract data 
    bd_pose_data = PoseData.from_bag(
        path=bag_path,
        topic=bd_pose_topic,
        time_tol=time_tol
    )
    bd_twist_data = GeneralData.from_bag(
        path=bag_path, 
        topic=bd_twist_topic, 
        time_tol=time_tol
    )
    px_pose_data = PoseData.from_bag(
        path=bag_path,
        topic=px_pose_topic,
        time_tol=time_tol
    )
    px_twist_data = GeneralData.from_bag(
        path=bag_path,
        topic=px_twist_topic, 
        time_tol=time_tol
    )
    rgb_data = ImgData.from_bag(
        path=bag_path, 
        topic=rgb_topic, 
        camera_info_topic=rgb_info_topic, 
        time_tol=time_tol,
        compressed=False
    )
    depth_data = ImgData.from_bag(
        path=bag_path, 
        topic=depth_topic,
        camera_info_topic=depth_info_topic, 
        time_tol=time_tol, 
        compressed=False
    )
    


    return depth_data, rgb_data, bd_pose_data, bd_twist_data, px_pose_data, px_twist_data

def save_from_bag(dest_path, depth_data, rgb_data, bd_interp_poses, bd_interp_twist, px_interp_poses, px_interp_twist):
    """ 
    Function to store depth images, color images, and poses in a directory.

    Inputs: 
        - depth_data, rdp depth object
        - rgb_interp_data, list of rgb frames interpolated using rdp 
        - interp_poses, a list of interpolated poses (same number of samples as depth_data)
        - bd_interp_poses, an array of poses interpolated using rdp method 
    Outputs: 
        - None, data is written to files 
    """

    # Change split number dynamically 
    pose_path = dest_path + f'/poses'
    depth_path = dest_path + f'/depth'
    rgb_path = dest_path + f'/color'
    intrinsics_path = dest_path + "/intrinsics"

    # 1. Make directory 
    subprocess.run(["mkdir", "-p", dest_path])
    # 2. Make images, depth, poses subdirectories 
    subprocess.run(["mkdir", "-p", rgb_path, depth_path, pose_path, intrinsics_path])

    # (ii) Flatten poses in raster order and save them as csv
    num_frames = len(depth_data.times)  

    bd_interp_poses = bd_interp_poses[:num_frames, :, :]
    bd_interp_poses_flat = bd_interp_poses.reshape(num_frames, 16)
    pose_df = pd.DataFrame(bd_interp_poses_flat) 
    pose_df.to_csv(pose_path + "/bd_poses.csv", index=False)

    px_interp_poses = px_interp_poses[:num_frames, :, :]
    px_interp_poses_flat = px_interp_poses.reshape(num_frames, 16)
    pose_df = pd.DataFrame(px_interp_poses_flat) 
    pose_df.to_csv(pose_path + "/px_poses.csv", index=False)

    # bd_interp_twist = twist_from_pose(bd_interp_poses_flat, depth_data.times) # TODO: Either make this optional or delete as we have twist already
    bd_interp_twist = bd_interp_twist[:num_frames, :]
    twist_df = pd.DataFrame(bd_interp_twist) 
    twist_df.to_csv(pose_path + "/bd_twists.csv", index=False) 

    # px_interp_twist = twist_from_pose(px_interp_poses_flat, depth_data.times)
    px_interp_twist = px_interp_twist[:num_frames, :]
    twist_df = pd.DataFrame(px_interp_twist) 
    twist_df.to_csv(pose_path + "/px_twists.csv", index=False) 
 
    flat_intrinsics = depth_data.K.reshape(-1, 1) 
    intrinsics_df = pd.DataFrame(flat_intrinsics)
    intrinsics_df.to_csv(intrinsics_path + "/depth_intrinsics.csv", index=False)

    # 4. Write data to respective directories
    for i in range(num_frames): 
        # (i) For rgb and depth use function from PIL probably 
        depth_image = depth_data.img(depth_data.times[i])
        depth_image = Image.fromarray(depth_image) 
        depth_image.save(depth_path + f"/depth_img{i}.png")

        rgb_image = rgb_data.img(depth_data.times[i]) 
        rgb_image = Image.fromarray(rgb_image) # Highbay is on RGB 
        rgb_image.save(rgb_path + f"/color_img{i}.png")

def generators_from_bag(out_path, depth_data, rgb_data, bd_interp_poses, bd_interp_twist, px_interp_poses):
    """
    Makes generators directly from robotdatapy loaded data

    Inputs:
        - Path to to dataset inputs folder

    Outputs: 
        - depth_generator
        - pose_generator 
        - rgb_generator 
    """

    # Make inputs and targets directories
    inputs = os.path.join(out_path, 'inputs') 
    targets = os.path.join(out_path, 'targets')
    subprocess.run(["mkdir", "-p", inputs, targets])

    # Save odometry to dest folder 
    num_frames = len(depth_data.times) 
    bd_interp_poses = bd_interp_poses[:num_frames-1, :, :]
    bd_interp_poses_flat = bd_interp_poses.reshape(num_frames, 16)
    pose_df = pd.DataFrame(bd_interp_poses_flat) 
    pose_df.to_csv(inputs + "/bd_poses.csv", index=False)

    bd_interp_twist = bd_interp_twist[:num_frames-1, :]
    twist_df = pd.DataFrame(bd_interp_twist) 
    twist_df.to_csv(inputs + "/bd_twists.csv", index=False)  

    # Make depth image generator 
    def depth_generator():
        for i in range(num_frames):
            yield depth_data.img(depth_data.times[i]) / 1000 # Scale from mm to m
    
    # Make color image generator 
    def color_generator():
        for i in range(num_frames): 
            yield rgb_data.img(depth_data.times[i])

    # Make pose generator 
    def bd_pose_generator():
        for pose in bd_interp_poses:
            yield pose 
    
    # Make pose generator 
    def px_pose_generator():
        for pose in px_interp_poses:
            yield pose 

    # Get intrinsics 
    intrinsics = depth_data.K

    return depth_generator(), bd_pose_generator(), px_pose_generator(), color_generator(), intrinsics  


def extract_data_from_bags(bag_path, dest_path): 
    #1. Extract data
    depth_data, rgb_data, bd_pose_data, bd_twist_data, px_pose_data, px_twist_data = load_from_bags(bag_path)

    #2. Interpolate 
    bd_interp_poses = interpolate(depth_data, bd_pose_data, False)
    bd_interp_twist = interpolate(depth_data, bd_twist_data, True)
    px_interp_poses = interpolate(depth_data, px_pose_data, False)
    px_interp_twist = interpolate(depth_data, px_twist_data, True)

    #3. Save all data at once 
    save_from_bag(dest_path, depth_data, rgb_data, bd_interp_poses, bd_interp_twist, px_interp_poses, px_interp_twist) 

def make_generators_from_bags(args):
    #1. Extract data
    depth_data, rgb_data, bd_pose_data, bd_twist_data, px_pose_data, _ = load_from_bags(args.data_path)

    #2. Interpolate 
    bd_interp_poses = interpolate(depth_data, bd_pose_data, False)
    bd_interp_twist = interpolate(depth_data, bd_twist_data, True)
    px_interp_poses = interpolate(depth_data, px_pose_data, False)

    #3. Make generators
    return generators_from_bag(args.out_path, depth_data, rgb_data, bd_interp_poses, bd_interp_twist, px_interp_poses)

    
if __name__ == '__main__': 
    # Extract and save data
    bag_path = f'/home/jrached/cv_project_code/project/data/filter_net/new_bags/test1'
    dest_path = f'/home/jrached/cv_project_code/project/data/filter_net/new_dataset/test1'
    extract_data_from_bags(bag_path, dest_path)
    