import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import subprocess
import torch
import pandas as pd
from .CSV import CSV
from .plot import *
from .esti import *
from .Cat import *
from .newKey import *
from .config import *

def requiredkeys():
    df = pd.read_excel('pw.xlsx',header=0, sheet_name='All')
    csv = CSV(df)
    total = int(csv.requried_keys[0])
    modified = int(csv.modified_keys[0])
    return total, modified

def Warning(pc_tracked, Tk, Nk, keyCount, msg):
    if pc_tracked.shape == (0,):
       if keyCount < Tk:
          msg = "   Please select {} points".format(Tk - keyCount)
       elif keyCount == Tk:
          msg = "   Selection Complete"
       else:
          msg = "   Warning : Over Selected \n"

    elif Nk > 0:
      if keyCount < Nk:
        msg = " Please select {} points".format(Nk - keyCount)
      elif keyCount == Nk:
        msg = " Selection Complete"
      else:
        msg = " Warning : Over Selected "
        
    return msg   

def markPointWin(pc, image,  win = "Window", color = (120,255,50)):
    for i in range(pc.shape[0]):
       cv2.circle(image, (pc[i][0],pc[i][1]) , 1,color, -1)
    scaledImg= image.copy()
    cv2.imshow(win,scaledImg)
    return scaledImg

def markPoint(pc, image, color = (120,255,50), color2 = (120,255,50)):
    for i in range(pc.shape[0]):
       cv2.circle(image, (pc[i][0],pc[i][1]) , 0, (0,0,0), -1)
       cv2.circle(image, (pc[i][0]-1,pc[i][1]), 0, color, -1)
       cv2.circle(image, (pc[i][0]+1,pc[i][1]), 0, color, -1)
       cv2.circle(image, (pc[i][0],pc[i][1]+1), 0, color, -1)
       cv2.circle(image, (pc[i][0],pc[i][1]-1), 0, color, -1)
       cv2.circle(image, (pc[i][0]-1,pc[i][1]-1), 0, color2, -1)
       cv2.circle(image, (pc[i][0]+1,pc[i][1]+1), 0, color2, -1)
       cv2.circle(image, (pc[i][0]-1,pc[i][1]+1), 0, color2, -1)
       cv2.circle(image, (pc[i][0]+1,pc[i][1]-1), 0, color2, -1)

def draw_markerTracked(pc_tracked, image):
    try:
       markPoint(pc_tracked[:,1:3], image, (0,120,255), (0,120,185))
    except:
       pass

def draw_markerSelected(pc_selected, image):
    try:
       markPoint(pc_selected, image, (120,255,50),(120,185,50))
    except:
       pass

def draw_tracked_(image, No, color = (0,120,255), color2 = (0,120,185)):
    x = torch.load(track_dir + 'tracked.pt')
    p = x.cpu().numpy().astype(np.int16)
    for i in range(p.shape[2]):
        (x,y) = p[0][No-1][i,:].tolist()
        cv2.circle(image, (x,y) , 0, (0,0,0), -1)
        cv2.circle(image, (x-1,y), 0, color, -1)
        cv2.circle(image, (x+1,y), 0, color, -1)
        cv2.circle(image, (x,y+1), 0, color, -1)
        cv2.circle(image, (x,y-1), 0, color, -1)
        cv2.circle(image, (x-1,y-1), 0, color2, -1)
        cv2.circle(image, (x+1,y+1), 0, color2, -1)
        cv2.circle(image, (x-1,y+1), 0, color2, -1)
        cv2.circle(image, (x+1,y-1), 0, color2, -1)

    return image, p[0][No-1]

def draw_tracked(image, No, color = (0,120,255), color2 = (0,120,185)):
    trackedDict = torch.load(track_dir + 'tracked_Dict.pt')
    p = trackedDict["img{}".format(No)]
    if p != np.array([]):
        for i in range(p.shape[0]):
            (x,y) = p[i].tolist()
            cv2.circle(image, (x,y) , 0, (0,0,0), -1)
            cv2.circle(image, (x-1,y), 0, color, -1)
            cv2.circle(image, (x+1,y), 0, color, -1)
            cv2.circle(image, (x,y+1), 0, color, -1)
            cv2.circle(image, (x,y-1), 0, color, -1)
            cv2.circle(image, (x-1,y-1), 0, color2, -1)
            cv2.circle(image, (x+1,y+1), 0, color2, -1)
            cv2.circle(image, (x-1,y+1), 0, color2, -1)
            cv2.circle(image, (x+1,y-1), 0, color2, -1)
    else:
       print("Img No: {} There is no tracked keypoints history".format(No))
       p = np.array([])

    return image, p

def draw_saved(image, p, color = (40,40,255), color2 = (0,30,255)):
    
    for i in range(p.shape[0]):
        (x,y) = p[i,:].tolist()
        cv2.circle(image, (x,y) , 0, (0,0,0), -1)
        cv2.circle(image, (x-1,y), 0, color, -1)
        cv2.circle(image, (x+1,y), 0, color, -1)
        cv2.circle(image, (x,y+1), 0, color, -1)
        cv2.circle(image, (x,y-1), 0, color, -1)
        cv2.circle(image, (x-1,y-1), 0, color2, -1)
        cv2.circle(image, (x+1,y+1), 0, color2, -1)
        cv2.circle(image, (x-1,y+1), 0, color2, -1)
        cv2.circle(image, (x+1,y-1), 0, color2, -1)

    return image, p

def draw_crosshair(x,y,scaledImg, f):
   cv2.line(scaledImg, (x-3,y),(x-8-f,y), (0,20,255), 1)
   cv2.line(scaledImg, (x+3,y),(x+8+f,y), (0,20,255), 1)
   cv2.line(scaledImg, (x,y-3),(x,y-8-f), (0,20,255), 1)
   cv2.line(scaledImg, (x,y+3),(x,y+8+f), (0,20,255), 1)
   cv2.circle(scaledImg, (x-1,y), 0, (0,0,0), -1)
   cv2.circle(scaledImg, (x+1,y), 0, (0,0,0), -1)
   cv2.circle(scaledImg, (x,y+1), 0, (0,0,0), -1)
   cv2.circle(scaledImg, (x,y-1), 0, (0,0,0), -1)