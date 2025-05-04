import os 
import subprocess 
import numpy as np 
import robotdatapy 
from robotdatapy.data import ImgData, PoseData 
from scipy.spatial.transform import Slerp 
from scipy.spatial.transform import Rotation as R 
import PIL 
from PIL import Image 
import pandas as pd 


################################
### Data Reading and Writing ###
################################

def extract_data(): 
    # Data params 
    time_tol = 10 

    # File paths 
    pose_bag = '/home/jrached/cv_project_code/kimera_dataset/thoth_vio'
    depth_bag = '/home/jrached/cv_project_code/kimera_dataset/thoth' 

    # Topics 
    pose_topic = '/thoth/kimera_vio_ros/odometry'
    depth_topic = '/thoth/forward/depth/image_rect_raw'
    rgb_topic = '/thoth/forward/color/image_raw/compressed'
    depth_info_topic = '/thoth/forward/depth/camera_info'
    rgb_info_topic = '/thoth/forward/color/camera_info'

    # For each depth, image, pose extract data 
    pose_data = PoseData.from_bag(
        path=pose_bag,
        topic=pose_topic,
        time_tol=time_tol
    )
    rgb_data = ImgData.from_bag(
        path=depth_bag, 
        topic=rgb_topic, 
        camera_info_topic=rgb_info_topic, 
        time_tol=time_tol,
        compressed=True
    )
    depth_data = ImgData.from_bag(
        path=depth_bag, 
        topic=depth_topic,
        camera_info_topic=depth_info_topic, 
        time_tol=time_tol, 
        compressed=False
    )

    return depth_data, rgb_data, pose_data 

def save_data(dest_path, split_index, depth_data, rgb_data, interp_poses_rdp, n=1000):
    """ 
    Function to store depth images, color images, and poses in 1000 datapoint splits in a directory 
    named /kimera/split01/images, /kimera/split01/depth, /kimera/split01/poses.
    There should be 16 splits. 

    Inputs: 
        - split_index, numnber of split currently storing
        - depth_data, rdp depth object
        - rgb_interp_data, list of rgb frames interpolated using rdp 
        - interp_poses, a list of interpolated poses (same number of samples as depth_data)
        - interp_poses_rdp, a list of poses interpolated using rdp method 
        - n, number of samples per split (always 1000)
    Outputs: 
        - None, data is written to files 
    """
    # Change split number dynamically 
    dir_path = dest_path + f'/split{split_index}'
    pose_path = dest_path + f'/split{split_index}/poses'
    depth_path = dest_path + f'/split{split_index}/depth'
    rgb_path = dest_path + f'/split{split_index}/color'

    # 1. If directory exists, delete directory.
    subprocess.run(["rm", "-rf", dir_path])
    # 2. Make directory /kimera/split01
    subprocess.run(["mkdir", "-p", dir_path])
    # 3. Make images, depth, poses subdirectories 
    subprocess.run(["mkdir", "-p", rgb_path, depth_path, pose_path])

    # 4. Write data to respective directories
    start = split_index * n
    for i in range(start, start + n): 
        # (i) For rgb and depth use function from PIL probably 
        depth_image = depth_data.img(depth_data.times[i])
        depth_image = Image.fromarray(depth_image) 
        depth_image.save(depth_path + f"/depth_img{i}.png")
    
        # print(depth_data.times)
        # print(depth_data.times[i])
        rgb_image = rgb_data.img(depth_data.times[i]) 
        # rgb_image = rgb_interp_data[i]
        rgb_image = Image.fromarray(rgb_image[:, :, ::-1])  
        rgb_image.save(rgb_path + f"/color_img{i}.png")
        
    # (ii) For poses flatten them in raster order and save them as csv
    # interp_poses = np.stack(interp_poses)[start: start + n, :, :] # Get first n poses as numpy array  
    # num_poses, _, _ = interp_poses.shape  
    # interp_poses_flat = interp_poses.reshape(num_poses, 16)

    # pose_df = pd.DataFrame(interp_poses_flat) 
    # pose_df.to_csv(pose_path + '/poses.csv', index=False)

    # Repeat for rdp interpolated poses 
    interp_poses_rdp = np.stack(interp_poses_rdp)[start: start + n, :, :] # Get first n poses as numpy array  
    num_poses, _, _ = interp_poses_rdp.shape 
    interp_poses_rdp_flat = interp_poses_rdp.reshape(num_poses, 16)

    pose_df = pd.DataFrame(interp_poses_rdp_flat) 
    pose_df.to_csv(pose_path + "/poses_rdp.csv", index=False)
    

#####################
### Interpolation ###
#####################

def interpolate(poses, source_times, target_times):
    """
    Given a list of poses at times source_times, return a list of poses for each time in target_times.
    Target times has more times than source times, but they must be contained within the range of source times
    in order to be interpolated. 
    Inputs: 
        - poses, the poses to interpolate 
        - source_times, the poses key_times 
        - target_times, the times that we want to interpolate poses to (from depth times) 
    Outputs: 
        interpolated_poses, a list of poses of length equals to len(target_times).
    """
    # For whatever reasons you have to zero the times
    source_times += -source_times[0]
    target_times += -target_times[0]
    source_times[-1] = target_times[-1] # target range <= source range

    # Get rotation and translation components from 4x4 poses 
    poses = np.stack(poses)
    Rs = poses[:, 0:3, 0:3]
    ts = poses[:, 0:3, -1]

    # Perform slerp interpolation with scipy's Slerp
    rots = R.from_matrix(Rs)
    slerp = Slerp(source_times, rots)
    interp_rots = slerp(target_times)
    interp_Rs = interp_rots.as_matrix()

    # Perform linear interpolation on translation 
    interp_ts = np.array([np.interp(target_times, source_times, ts[:, i]) for i in range(3)]).T

    # Reconstruct and return poses 
    interp_poses = []
    for R_i, t_i in zip(interp_Rs, interp_ts): 
        pose = np.eye(4)
        pose[:3, :3] = R_i
        pose[:3, 3] = t_i

        interp_poses.append(pose) 

    return interp_poses 

def interpolate_rdp(depth_data, pose_data, rgb_data):
    """
    Interpolates pose_data into depth_data using rdp's built in procedure
    """

    depth_times = depth_data.times

    interp_poses = []
    for t in depth_times:
        interp_poses.append(pose_data.pose(t))

    # interp_rgb = [] 
    # for t in depth_times: 
    #     interp_rgb.append(rgb_data.img(t))


    return interp_poses, 0 

#####################
### Main Function ###
#####################
def main(dest_path): 
    #1. Extract the data
    depth_data, rgb_data, pose_data = extract_data()

    #2. Interpolate both ways 
    poses = [pose_data.pose(pose_data.times[i]) for i in range(len(pose_data.times))]  
    # interp_poses_m1 = interpolate(poses, pose_data.times, depth_data.times) # Manual method
    interp_poses_m2, interp_rgb = interpolate_rdp(depth_data, pose_data, rgb_data) # RDP method 

    #3. Save data in 16 splits 
    for i in range(16):
        save_data(dest_path, i, depth_data, rgb_data, interp_poses_m2)

if __name__ == '__main__': 
    # Extract and save data 
    # dest_path = f'/home/jrached/cv_project_code/project/data/kimera2'
    # main(dest_path)
    pass