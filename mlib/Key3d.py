import torch
import numpy as np
from . import knownWorld3dBoxPoints, projectedKnown3dBoxPoints, pointsOnALine, cat_points

def keypoints3d(cat_order, num_est_classes=11,num_keys=32,device='cuda:0', dtype=torch.float32):

    Kw3D = torch.empty((num_est_classes, num_keys, 3),device=device, dtype=dtype)
    for i in range(num_est_classes):
        if cat_order[i]==0:
            cat_order[i] = 1
        P = cat_points(cat_order[i])
        Pw, Pw_C, Pw_I, Pw_axis, Pw_shifted_axis = knownWorld3dBoxPoints(P, points_only=0)
        Kw3D[i] = torch.from_numpy(Pw)
    return Kw3D
