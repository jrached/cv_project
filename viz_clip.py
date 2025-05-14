import sys
# sys.path.append('core')
sys.path.append('/home/jrached/cv_project_code/RAFT/core')
sys.path.append('../project')

from scene_flow import make_generators 
from scene_flow import scene_flow 

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


import time 
from torch.cuda.amp import autocast

DEVICE = 'cuda'

def load_video(filename, play=False):
    # Get video 
    cap = cv2.VideoCapture(filename)

    # Get frames per second 
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check that video opened successfuly 
    if not cap.isOpened():
        print("Error: Could not open video")
        return 

    frames = [] 
    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        if play: 
            cv2.imshow('frame',frame)

        # Append frame to list 
        # print(f"This is the frame shape: {frame.shape}")
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        frames.append(frame[None].to(DEVICE))

        # Wait for fps 
        key = cv2.waitKey(int(1000 / fps)) & 0xFF   
        # Break if q is pressed or video finished
        if key == ord('q') or ret==False :
            break
    
    # Release video capture object and close all windows 
    cap.release()
    cv2.destroyAllWindows()

    return frames, fps 


def viz_video(img, flo, fps):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    print(flo)

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # flo = flow_to_rgb(flo)
    img_flo = np.concatenate([img[:, :, [2, 1, 0]], flo[:, :, [2, 1, 0]]], axis=0)

    cv2.imshow('image', img_flo/255.0)
    
    # Wait for fps 
    # key = cv2.waitKey(int(1000 / fps)) & 0xFF 
    key = cv2.waitKey(1) & 0xFF 

    # Break if q is pressed or video finished
    if key == ord('q'):
        return False
    
    return True

def demo_video(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)

    model.eval()

    with torch.no_grad():
        # Load video from path 
        frame_path = glob.glob(os.path.join(args.path, '*.mp4'))[0] 
        # print(f"This is the frame path: {frame_path}")
        frames, fps = load_video(frame_path)
        
        # Iterate through the frames in the video 
        start_time = time.time() 
        prev_frame = frames[0]
        for i, frame in enumerate(frames[1:]):

            # Pad frames to fit model input (presumably)
            padder = InputPadder(prev_frame.shape)
            prev_frame, frame = padder.pad(prev_frame, frame)

            # Get optical flow of each frame with RAFT and play video 
            flow_low, flow_up = model(prev_frame, frame, iters=20, test_mode=True)
            if not viz_video(prev_frame, flow_up, fps):
                break

            # Update previous frame 
            prev_frame = frame
        
        return (i + 1) / (time.time() - start_time)

def demo_kimera(args, split_index=5):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)

    model.eval()
    
    with torch.no_grad():
        # Load video from path 
        path_to_split = args.path
 
        _, _, rgb_generator = make_generators(path_to_split, split_index) 

        # Get first frame 
        for frame in rgb_generator: 
            frame = torch.from_numpy(frame).permute(2, 0, 1).float()
            prev_frame = frame[None].to(DEVICE)
            break 

        # Iterate through the frames in the video
        for frame in rgb_generator:
            
            # Prepare image for model 
            frame = torch.from_numpy(frame).permute(2, 0, 1).float()
            frame = frame[None].to(DEVICE)

            # Pad frames to fit model input 
            padder = InputPadder(prev_frame.shape)
            prev_frame, frame = padder.pad(prev_frame, frame)

            # Get optical flow of each frame with RAFT and play video 

            flow_low, flow_up = model(prev_frame, frame, iters=20, test_mode=True)
            if not viz_video(prev_frame, flow_up, 30):
                break

            # Update previous frame 
            prev_frame = frame
        
        return 0


if __name__ == '__main__':
    ######## To run: ####################################################
    # python viz_clip.py --model=models/raft-things.pth --path=data/kimera 
    #####################################################################

    path_to_raft = '/home/jrached/cv_project_code/RAFT/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    args.model = path_to_raft + args.model 

    # path_to_video = "./juan_code/messi.mp4"
    # frames = load_video(path_to_video, play=True)

    # avg_fps = demo_video(args) 
    avg_fps = demo_kimera(args, split_index=3)
    print(f"Average fps: {avg_fps}")

