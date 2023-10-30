bl_info = {
    "name": "Making Synthetic Dataset using Ship 3D CAD model",
    "author": "Maneesh",
    "version": (3, 0, 0),
    "blender": (3, 4, 1),
    "date": '17.03.2023'}

import sys
import os
import bpy
import numpy as np
import random
from random import randint 
import math
from math import radians
from math import degrees
import mathutils
from mathutils import Vector
from mathutils import Matrix
from numpy import linalg as LA

import time

sys.path.insert(0, os.getcwd()+"/BlenderCyclesRender")

from mylib import * #Dir, RenDir, Sea, Shipskin, Sky, Landingpad, Markings, OfflineDir, tracker, fileio
from mylib.prob import cal_pdf, cal_ni, chooseRanPosVal, getRanPosVal_around_point
from mylib.camera import cameraLocFocus2Hom, get_calibration_matrix_K_from_blender, randomCamLoc
from mylib.Add import adding
from mylib.misc import make_a_scene
from mylib.moveSun import rotate_sun
from mylib.moveOcclusions import *
from mylib.moveCamera import *
from mylib.env import *
from mylib.pose import Pose

#numbering
R_i = 0  #Rendering frame number
S_i = 0 #Different Environment = Scene No
j = 0

jpg_i = 0 #Jpg No
cameraFocusPoint = []
cameraLocation = []


RotPass = True

Q = [[]] #np.loadtxt(Dir + RenDir + "Q.txt")

    
def main(scene):
    global S_i,j,Q,RotPass #Different Environment = Scene No
    #TRACKER = False
    # if TRACKER:
    #     FIXCAM = False
    

    gbp = Pose(k) #get best pose / k is camera calibration matrix
    #TRACKER = True
    
    if TRACKER:
        Hw_track = np.loadtxt(tracker + "Hw_gt.txt")
    
    shift_occlusions()
    rotate_sun()
  
    if S_i%4 == 0 and RotPass: 
        bpy.ops.file.make_paths_absolute()
        changeEnvironments(scene)
        try:
            img_no = np.loadtxt(OfflineDir + "NoOfTrainingImgs.txt") #number of images already in the training tadaset
        except:
            img_no = np.array([[0],[0]])
        
        if FIXED_CAM:
            # When camera location is fixed
            camLoc= Camera_Location
            # When focucing point is fixed
            camFocus = Camera_Focusing_point
            
            _,Hw_t = rotateCamera(scene,cF=camFocus,cL=camLoc)
#            x  = Hw_track[0,:]
#            Hw_t = np.eye(4)
#            Hw_t[0:3,0:4] =np.reshape(x, (3,4))

        elif TRACKER:
            x  = Hw_track[nImg,:]
            Hw_t = np.eye(4)
            Hw_t[0:3,0:4] =np.reshape(x, (3,4))
            rotateCamera(scene,Hw=Hw_t)
        else:
            _, Hw, _, _, _, _ = gbp.get_best_pose()
            _,Hw_t = rotateCamera(scene,Hw=Hw)        

        bpy.context.scene.render.film_transparent = False
        bpy.data.objects["Wave"].hide_render = False
        bpy.data.objects["Dome"].hide_render = True
        ship = bpy.context.view_layer.active_layer_collection
        ship.holdout = False
        bpy.context.scene.render.filepath = RenDir +"######"
        img_no = img_no+1
        np.savetxt(OfflineDir+"NoOfTrainingImgs.txt",img_no, fmt='%i')
        
        j = j +1
    elif S_i%4 == 1:
        print("----------- Rendering --> SynMaskImage_{:6d}...".format(S_i+1))
        bpy.context.scene.render.film_transparent = True
        bpy.data.objects["Wave"].hide_render = False
        bpy.data.objects["Dome"].hide_render = False
        ship = bpy.context.view_layer.active_layer_collection
        ship.holdout = True
        bpy.context.scene.render.filepath = RenDir +"######mask1"
        print("Generated    {:06d}.png   and   {:6d}mask1.png  ".format(S_i,S_i+1))
        print("\n")
    elif S_i%4 == 2:
        print("----------- Rendering --> SynMaskImage_{:6d}...".format(S_i+1))
        bpy.context.scene.render.film_transparent = True
        bpy.data.objects["Wave"].hide_render = False
        bpy.data.objects["Dome"].hide_render = True
        ship = bpy.context.view_layer.active_layer_collection
        ship.holdout = True
        bpy.context.scene.render.filepath = RenDir +"######mask2"
        print("Generated    {:06d}.png   and   {:6d}mask2.png  ".format(S_i,S_i+1))
        print("\n") 
    elif S_i%4 == 3:
        print("----------- Rendering --> SynMaskImage_{:6d}...".format(S_i+1))
        bpy.context.scene.render.film_transparent = True
        bpy.data.objects["Wave"].hide_render = False
        bpy.data.objects["Dome"].hide_render = True
        ship = bpy.context.view_layer.active_layer_collection
        ship.holdout = False
        bpy.context.scene.render.filepath = RenDir +"######mask3"
        print("Generated    {:06d}.png   and   {:6d}mask3.png  ".format(S_i,S_i+1))
        print("\n")
        RotPass = False   
    S_i = S_i + 1
    


def createDataset():
    # clear old handler
    bpy.app.handlers.frame_change_pre.clear()
    # register new handler
    #bpy.ops.file.make_paths_relative()
    
    bpy.app.handlers.frame_change_pre.append(main)
   
k = get_calibration_matrix_K_from_blender()
np.savetxt(OfflineDir+"K.txt",k)
   
createDataset()

#start animation
bpy.context.scene.frame_start = 1
if MASKON:
    bpy.context.scene.frame_end = 4
else:
    bpy.context.scene.frame_end = 1
bpy.context.scene.render.film_transparent = True
bpy.data.objects["Wave"].hide_render = True

"#### Render settings ######"
bpy.context.scene.render.resolution_x = 640
bpy.context.scene.render.resolution_y = 480
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 1
#bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.image_settings.compression = 0
#bpy.context.scene.cycles.preview_samples = 64

bpy.ops.render.render(animation=True,use_viewport = True, write_still=False)



