import torch
import numpy as np
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path, read_images_from_path
import argparse
from config import *

parser = argparse.ArgumentParser(description="Synthetic image generation")

parser.add_argument('-S', type=int, default=1,
                        help='Number of images want to be generated')

parser.add_argument('-N', type=int, default=20,
                        help='Number of images want to be generated')

args = parser.parse_args()
N = args.N
S = args.S
#N = 20
def mode(S, N):
    video = read_images_from_path(Input_ImgDir, S, N)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    # queries = torch.tensor([
    #     [0., 400., 350.],  # point tracked from the first frame
    #     [10., 600., 500.], # frame number 10
    #     [20., 750., 600.], # ...
    #     [30., 900., 200.]
    # ])
    # print(queries.size())
    queries = torch.load('q.pt')
    model = CoTrackerPredictor(checkpoint='cotracker/ckpt/cotracker_stride_4_wind_8.pth')

    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
        queries = queries.cuda()

    pred_tracks, pred_visibility = model(video, queries=queries.unsqueeze(0))
    print(pred_tracks.size())
    torch.save(pred_tracks, 'tracked.pt')
    
mode(S, N)