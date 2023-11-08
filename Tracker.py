import torch
import numpy as np
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path, read_images_from_path
import argparse
from config.config import *

parser = argparse.ArgumentParser(description="Synthetic image generation")

parser.add_argument('-S', type=int, default=100,
                        help='Number of the starting')

parser.add_argument('-N', type=int, default=150,
                        help='Number of images want to be generated')

parser.add_argument('-T', type=int, default=300,
                        help='Total images')

args = parser.parse_args()
N = args.N
S = args.S
T = args.T
#N = 20
def mode(S, N, T):
    video = read_images_from_path(Input_ImgDir, S, N)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    # queries = torch.tensor([
    #     [0., 400., 350.],  # point tracked from the first frame
    #     [10., 600., 500.], # frame number 10
    #     [20., 750., 600.], # ...
    #     [30., 900., 200.]
    # ])
    # print(queries.size())
    queries = torch.load('tracked/q.pt')
    model = CoTrackerPredictor(checkpoint='cotracker/ckpt/cotracker_stride_4_wind_8.pth')

    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
        queries = queries.cuda()

    pred_tracks, pred_visibility = model(video, queries=queries.unsqueeze(0))

    torch.save(pred_tracks, 'tracked/tracked.pt')

    try:
        trackedDict = torch.load('tracked/tracked_Dict.pt')
    except:
        trackedDict = {}
        trackedDict.update({"img{}".format(i+1) : np.array([]) for i in range(T)})

    for i in range(pred_tracks.size()[1]):
        trackedDict["img{}".format(S+i)] = pred_tracks[0,i,:,:].cpu().numpy().astype(np.int16)

    torch.save(trackedDict,'tracked/tracked_Dict.pt')

mode(S, N, T)