import numpy as np
import bpy
import mathutils
from mathutils import Vector
from mathutils import Matrix
from numpy import linalg as LA
from .prob import getRanPosVal_around_point, cal_pdf, cal_ni
from . import *

def randomCamLoc(ri,dr,shift_from_center,mu,sigma,Nc,N,A):
    theta_start = A[0]
    theta_end = A[1]
    theta = A[2]
    phi_start = A[3]
    phi_end = A[4]
    phi = A[5]
    
    Sum = 0
    for i in range(N):
        ri = i * dr
        Sum = cal_pdf(ri,mu,sigma) + Sum

    n = np.zeros(N)
    for i in range(N):
        ri = i * dr
        n[i] = int(cal_ni(ri,mu,sigma,Nc,Sum))

    ri = []
    thetai = []
    phii = []
    for i in range(N):
        ri = np.append(ri,(np.random.random(int(n[i]))*dr + i*dr)) 
        thetai = np.append(thetai,(np.random.random(int(n[i]))*theta + theta_start))
        phii = np.append(phii,(np.pi/2 - np.random.random(int(n[i]))*phi + phi_start))
    #ri = ri + shift_from_center
    #Cartesian coordinates
    x = ri * np.sin(phii) * np.cos(thetai)
    y = -ri * np.sin(phii) * np.sin(thetai)
    z = ri * np.cos(phii) + 0.1
    
    RCL = np.zeros((3,z.shape[0]))
    RCL[0,:] = x
    RCL[1,:] = y - shift_from_center
    RCL[2,:] = z
    
    return RCL

def cameraLocFocus2Hom(cameraFocusPoint,cameraLocation):
    inv = 1
    
    OC = cameraLocation - cameraFocusPoint

    theta = rotAngleX(np.array([1,0,0]),np.array([OC[0],OC[1],0]),np.array([0,0,1]))
    Rz = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    x_hat = np.array([1,0,0]) @ Rz.T
    phi =  rotAngleZ(np.array([0,0,1]),OC,x_hat) 
    Rx = np.array([[1,0,0],[0, np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])

    theta_deg =  float(getRanPosVal_around_point(0,10,1000,0,0.08,0))
    theta2 = np.pi*theta_deg/180 # in radian
    
    if TRACKER:
        R = Rz @ Rx
    if FIXED_CAM:
        R = Rz @ Rx
    else:
        Rz_uncertinity = np.array([[np.cos(theta2),-np.sin(theta2),0],[np.sin(theta2),np.cos(theta2),0],[0,0,1]]) # camera roll as a noise
        R = Rz @ Rx@ Rz_uncertinity #Eular angles
     
    T = cameraLocation
    
    Hw = np.array([[R[0,0],R[0,1],R[0,2],T[0]],
                   [R[1,0],R[1,1],R[1,2],T[1]],
                   [R[2,0],R[2,1],R[2,2],T[2]],
                   [     0,     0,     0,  1]])
    return Hw
    
    
def get_calibration_matrix_K_from_blender(mode='simple'):
    # https://mcarletti.github.io/articles/blenderintrinsicparams/
    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale # px
    height = scene.render.resolution_y * scale # px

    camdata = scene.camera.data

    if mode == 'simple':

        aspect_ratio = width / height
        K = np.zeros((3,3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()
    
    if mode == 'complete':

        focal = camdata.lens # mm
        sensor_width = camdata.sensor_width # mm
        sensor_height = camdata.sensor_height # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal), 
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio 
            s_v = height / sensor_height
        else: # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal), 
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0 # only use rectangular pixels

        K = np.array([
            [alpha_u,    skew, u_0],
            [      0, alpha_v, v_0],
            [      0,       0,   1]
        ], dtype=np.float32)
    
    return K
    
def rotAngleX(a,b,n):
    sintheta = np.dot(n,np.cross(a, b))/(LA.norm(a)*LA.norm(b))
    costheta= np.dot(a,b)/(LA.norm(a)*LA.norm(b))
    if sintheta > 0 and costheta < 0:
        return np.arccos(costheta) +np.pi/2
    elif sintheta < 0 and costheta < 0:
        return -np.arccos(costheta)+np.pi/2   
    elif sintheta < 0 and costheta >= 0:
        return -np.arccos(costheta) + np.pi/2
    else:
        return np.arccos(costheta)+np.pi/2
    
    
def rotAngleZ(a,b,n):
    sintheta = np.dot(n,np.cross(a, b))/(LA.norm(a)*LA.norm(b))
    costheta= np.dot(a,b)/(LA.norm(a)*LA.norm(b))
    if sintheta > 0 and costheta < 0:
        return np.arccos(costheta) +np.pi/2
    elif sintheta < 0 and costheta < 0:
        return -np.arccos(costheta)   
    elif sintheta < 0 and costheta >= 0:
        return -np.arccos(costheta) + np.pi
    else:
        return np.arccos(costheta)


def cameraLocFocus2Hom2(cameraFocusPoint,cameraLocation):
    R_cam_w = get_CameraLocFocus2Rot(cameraLocation, cameraFocusPoint)

    theta_deg =  float(getRanPosVal_around_point(0,10,1000,0,0.08,0))
    theta2 = np.pi*theta_deg/180 # in radian
    Rz_uncertinity = np.array([[np.cos(theta2),-np.sin(theta2),0],[np.sin(theta2),np.cos(theta2),0],[0,0,1]]) # camera roll as a noise
    if FIXED_CAM:
        Rz_uncertinity = np.eye(3)
    Hw = np.eye(4)
    R_cw = R_cam_w @ Rz_uncertinity #Eular angles
    Hw[0:3,0:3] = R_cw 
    Hw[0:3,3] = cameraLocation

    return Hw
    

def yawangle_form_vector(camLoc_orig, camFoc_orig, Degree = True):
    Foc_Loc = camFoc_orig - camLoc_orig
    Foc_Loc[2] = 0
    y = np.array([0,1,0])

    if Foc_Loc[0] == 0 and Foc_Loc[1] == 0:
        return 0.0
    unit_vector_Foc_Loc = Foc_Loc / np.linalg.norm(Foc_Loc)
    dot_product = np.dot(unit_vector_Foc_Loc, y)
    
    yawangle = np.arccos(dot_product) * 180 / np.pi

    if Foc_Loc[0] >= 0 and Foc_Loc[1] >= 0:
        yawangle = -yawangle
    elif Foc_Loc[0] > 0 and Foc_Loc[1] < 0:
        yawangle = -yawangle
    if Degree:
        return yawangle
    return yawangle * np.pi / 180

def get_CameraLocFocus2Rot(camLoc, camFoc):
    yawangle = yawangle_form_vector(camLoc, camFoc, Degree = False)
    Rz = np.array([[np.cos(yawangle),-np.sin(yawangle),0],[np.sin(yawangle),np.cos(yawangle),0],[0,0,1]])

    Foc_Loc = camFoc - camLoc
    Foc_Loc = np.expand_dims(Foc_Loc, axis=0)
    Foc_Loc_x0 = Rz.T @ Foc_Loc.T
    rollangle = np.arctan2(Foc_Loc_x0[2], Foc_Loc_x0[1])[0] + np.pi/ 2
    Rx = np.array([[1,0,0],[0, np.cos(rollangle),-np.sin(rollangle)],[0,np.sin(rollangle),np.cos(rollangle)]])

    Rw = Rz @ Rx

    return Rw
