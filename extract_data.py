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
    pose_bag = '/home/jrached/cv_project_code/highbay_dataset/test1/odometry'
    depth_bag = '/home/jrached/cv_project_code/highbay_dataset/test1/cv_bag2' 

    # Topics 
    quad_pose_topic = '/BD01/world'
    scout_pose_topic = '/SCOUT2/world'
    dog_pose_topic = '/RAVAGE/world'
    depth_topic = '/camera/camera/depth/image_rect_raw'
    rgb_topic = '/camera/camera/color/image_raw'
    depth_info_topic = '/camera/camera/color/camera_info'
    rgb_info_topic = '/camera/camera/color/camera_info'

    # For each depth, image, pose extract data 
    quad_pose_data = PoseData.from_bag(
        path=pose_bag,
        topic=quad_pose_topic,
        time_tol=time_tol
    )
    scout_pose_data = PoseData.from_bag(
        path=pose_bag,
        topic=scout_pose_topic,
        time_tol=time_tol
    )
    dog_pose_data = PoseData.from_bag(
        path=pose_bag,
        topic=dog_pose_topic,
        time_tol=time_tol
    )
    rgb_data = ImgData.from_bag(
        path=depth_bag, 
        topic=rgb_topic, 
        camera_info_topic=rgb_info_topic, 
        time_tol=time_tol,
        compressed=False
    )
    depth_data = ImgData.from_bag(
        path=depth_bag, 
        topic=depth_topic,
        camera_info_topic=depth_info_topic, 
        time_tol=time_tol, 
        compressed=False
    )

    return depth_data, rgb_data, quad_pose_data, scout_pose_data, dog_pose_data   

def save_data(dest_path, split_index, depth_data, rgb_data, quad_interp_poses, scout_interp_poses, dog_interp_poses, n=500):
    """ 
    Function to store depth images, color images, and poses in 1000 datapoint splits in a directory 
    named /kimera/split01/images, /kimera/split01/depth, /kimera/split01/poses.
    There should be 16 splits. 

    Inputs: 
        - split_index, numnber of split currently storing
        - depth_data, rdp depth object
        - rgb_interp_data, list of rgb frames interpolated using rdp 
        - interp_poses, a list of interpolated poses (same number of samples as depth_data)
        - quad_interp_poses, a list of poses interpolated using rdp method 
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

    # (ii) For poses flatten them in raster order and save them as csv
    start = split_index * n
    quad_interp_poses = np.stack(quad_interp_poses) # Get first n poses as numpy array 
    l_max = min(start + n, quad_interp_poses.shape[0] -1) 
    quad_interp_poses = quad_interp_poses[start: l_max, :, :]
    num_poses, _, _ = quad_interp_poses.shape 
    quad_interp_poses_flat = quad_interp_poses.reshape(num_poses, 16)

    pose_df = pd.DataFrame(quad_interp_poses_flat) 
    pose_df.to_csv(pose_path + "/quad_poses_rdp.csv", index=False)

    scout_interp_poses = np.stack(scout_interp_poses)[start: l_max, :, :] # Get first n poses as numpy array  
    num_poses, _, _ = scout_interp_poses.shape 
    scout_interp_poses_flat = scout_interp_poses.reshape(num_poses, 16)

    pose_df = pd.DataFrame(scout_interp_poses_flat) 
    pose_df.to_csv(pose_path + "/scout_poses_rdp.csv", index=False)

    dog_interp_poses = np.stack(dog_interp_poses)[start: l_max, :, :] # Get first n poses as numpy array  
    num_poses, _, _ = dog_interp_poses.shape 
    dog_interp_poses_flat = dog_interp_poses.reshape(num_poses, 16)

    pose_df = pd.DataFrame(dog_interp_poses_flat) 
    pose_df.to_csv(pose_path + "/dog_poses_rdp.csv", index=False)

    # 4. Write data to respective directories
    for i in range(start, l_max): 
        # (i) For rgb and depth use function from PIL probably 
        depth_image = depth_data.img(depth_data.times[i])
        depth_image = Image.fromarray(depth_image) 
        depth_image.save(depth_path + f"/depth_img{i}.png")

        rgb_image = rgb_data.img(depth_data.times[i]) 
        # rgb_image = Image.fromarray(rgb_image[:, :, ::-1]) #Kimera is on BGR  
        rgb_image = Image.fromarray(rgb_image) # Highbay is on RGB 
        rgb_image.save(rgb_path + f"/color_img{i}.png")
        

    

#####################
### Interpolation ###
#####################
def interpolate_rdp(depth_data, pose_data):
    """
    Interpolates pose_data into depth_data using rdp's built in procedure
    """

    depth_times = depth_data.times

    interp_poses = []
    for t in depth_times:
        interp_poses.append(pose_data.pose(t))


    return interp_poses

#####################
### Main Function ###
#####################
def main(dest_path): 
    #1. Extract the data
    depth_data, rgb_data, quad_pose_data, scout_pose_data, dog_pose_data = extract_data()

    #2. Interpolate both ways 
    quad_interp_poses = interpolate_rdp(depth_data, quad_pose_data) # RDP method 
    scout_interp_poses = interpolate_rdp(depth_data, scout_pose_data) # RDP method 
    dog_interp_poses = interpolate_rdp(depth_data, dog_pose_data) # RDP method 

    #3.5 Save intrinsics once 
    intrinsics_path = dest_path + "/intrinsics"
    flat_intrinsics = depth_data.K.reshape(-1, 1) 
    intrinsics_df = pd.DataFrame(flat_intrinsics)
    intrinsics_df.to_csv(intrinsics_path + "/depth_intrinsics.csv", index=False)
    
    #3. Save data in 16 splits 
    for i in range(16):
        save_data(dest_path, i, depth_data, rgb_data, quad_interp_poses, scout_interp_poses, dog_interp_poses)

     


if __name__ == '__main__': 
    # Extract and save data 
    dest_path = f'/home/jrached/cv_project_code/project/data/highbay1'
    main(dest_path)
    pass