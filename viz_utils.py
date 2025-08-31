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

from processing_utils import load_image

# Get RAFT's functions
sys.path.append('../RAFT/core')
from utils import flow_viz


##########################
### Visualization Code ###
##########################
def write_to_video(args, videos, fps, frame_size):
    # Get openCV video writers 
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video_writers = [cv2.VideoWriter(os.path.join(args.out_path, f"videos/{vid_name}.mp4"), fourcc, fps, frame_size, isColor=True) for vid_name in videos]
    
    # Iterate through frames in each video folder and write to video
    frames_path = os.path.join(args.data_path, f"frames/{videos[0]}")
    num_frames = len([f for f in os.listdir(frames_path) if os.path.isfile(os.path.join(frames_path,f))]) # This assumes all videos will have the same number of frames (should be true)
    for i in range(num_frames):
        for j, vw in enumerate(video_writers):
            img = load_image(os.path.join(args.data_path, f"frames/{videos[j]}/frame{i}.png"))
            vw.write(img) 

    for vw in video_writers: 
        vw.release()
    cv2.destroyAllWindows()

def viz_frame(img, frame, is_image=True):
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

def play_video(viz_path, videos):
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
        cap[i] = cv2.VideoCapture(os.path.join(viz_path, f"videos/{v}.mp4"))

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