import torch 
import imageio.v3 as iio
from cotracker.utils.visualizer import Visualizer

# Download the video and extract frames 
# url = 'https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4'
url = './data/out/flow_masked_image.mp4'
frames = iio.imread(url)[:136, ...]
# frames = iio.imread(url, plugin="FFMPEG") 

# print(frames.shape)
# print(type(frames[0]))

# Convert video to tensor 
device = 'cuda' 
grid_size = 10
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device) # B T CH W

# Run Offline CoTracker: 
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2, B T N 1

# Visualize tracks 
vis = Visualizer(save_dir="./saved_videos/project_videos", pad_value=120, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)
