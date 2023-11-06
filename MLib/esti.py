import cv2
import numpy as np
from .misc_fucntions import *
from . import *

def Epnp2H(PW, Pc, K, dist = None):
    try:
        Pw = PW
        _, Rvec, Tvec = cv2.solvePnP(Pw.astype(float), Pc.astype(float), K, distCoeffs=dist, flags=cv2.SOLVEPNP_EPNP)
    except:
        print("pc and pw are not matched. try touse same number of 2d keypoints as for 3d keypoints")
        try:
            Pw = PW[:Pc.shape[0],:]
            _, Rvec, Tvec = cv2.solvePnP(Pw.astype(float), Pc.astype(float), K, distCoeffs=dist, flags=cv2.SOLVEPNP_EPNP)
        except:
            print("pc and pw are not matched please select corresponding keypoints")

    T = Tvec
    R, _ = cv2.Rodrigues(Rvec)
    h_pnp = np.concatenate((R,T),axis = 1)
    """correction """
    tt = -180
    tt = tt * np.pi / 180
    R0 = np.array([[1,0,0],[0,np.cos(tt),-np.sin(tt)],[0,np.sin(tt),np.cos(tt)]])
    Hc_est = R0@h_pnp
    Hw_est = Hc2Hw(Hc_est)
    return Hc_est, Hw_est

def est_pc(Hc, K, cat_id = 1):
    P = cat_points(cat_id=cat_id)
    Pc, Pc_C, Pc_I, Pc_axis, Pc_shifted_axis = projectedKnown3dBoxPoints(P, Hc,K,points_only = 0)
    return Pc, Pc_axis