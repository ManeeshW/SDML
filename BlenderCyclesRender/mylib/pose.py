from random import randint, uniform
import numpy as np
from numpy import linalg as L2

from .tools import *
from .tools.rule import *
from .camera import *
from .misc import *
from .prob import *

class Pose:
    def __init__(self, K):
        self.K = K
        # Camera location handling parameters
        self.Nc = 4000 #Total number of camera locations
        self.sigma = 200 # standard deviation of camera location distribution
        self.mu = 20 # mean of camera location distribution

        self.min_L = 30
        self.max_L = 30
        self.L = uniform(self.min_L, self.max_L) #Max Radius Length 50
        self.dr = 0.01 #step size 
        self.N = int(self.L/self.dr) #number of steps

        self.dangle = uniform(1, 3) * np.pi/12
        self.theta_start = 0 - np.pi/18 + self.dangle
        self.theta_end = np.pi  + np.pi/18 - self.dangle
        self.theta = self.theta_end-self.theta_start

        self.div = uniform(2,8) 
        self.phi_start = 0 + np.pi/72
        self.phi_end = np.pi/self.div
        self.phi = self.phi_start + self.phi_end

        self.cam_angles = np.array([self.theta_start,self.theta_end,self.phi_start,self.phi_end])
        self.cam_shift = np.array([0.4, 0, 0.40]) # shift origin of a camera location distribution

        # Camera focusing locations handling parameters
        # Create Uncertinity in forcusing direction
        self.Nf = 2000 # Total number of camera focusing locations
        self.f_sigma = 10 # standard deviation of camera focusing location distribution
        self.f_mu = 4 # mean of camera focusing location distribution

        self.f_L = 8 #random.randint(1, 4)#2 #Max Length
        self.f_dr = 0.01 #step size
        self.f_N = int(self.f_L/self.f_dr) #number of steps

        self.f_theta_start = 0
        self.f_theta_end = 2*np.pi
        self.f_theta = self.f_theta_end-self.f_theta_start

        self.f_phi_start = 0#np.pi
        self.f_phi_end = 2*np.pi
        self.f_phi = self.f_phi_start+self.f_phi_end

        self.foc_angles = np.array([self.f_theta_start,self.f_theta_end,self.f_phi_start,self.f_phi_end])
        self.F_shift=np.array([0, 6, -0.5]) # shift origin of a camera focusing point distribution
        
        self.foc_dist_radius_factor = 0.9
        # constrain for min catid sizes
        self.Thr_x = 6 # object size threshold x
        self.Thr_y = 6 # object size threshold y

    def get_best_pose(self):
        cameraLocations = generatePointDist(self.L,self.dr,self.cam_shift,self.mu,self.sigma,self.Nc,self.cam_angles).T
        np.random.shuffle(cameraLocations)
        Bool = False
        for j in range(cameraLocations.shape[0]):
            #Random camera locations
            
            camLoc = cameraLocations[j]
            rx = L2.norm(camLoc)
            f_L = self.f_L + self.foc_dist_radius_factor*rx # random.randint(1, 2)#2 #Max Length

            if camLoc[2] > 6:
                continue
                
            if camLoc[2] < 1:
                continue
            if camLoc[1] > -4 and (camLoc[0] > -4 or camLoc[0] < 4):
                continue

            if rx>12:
                self.f_phi_end = 2*np.pi
                self.f_phi = self.f_phi_start+self.f_phi_end
                cat_id_i = randint(1, 2)
            else:
                self.f_phi_end= np.pi
                self.foc_angles = np.array([self.f_theta_start,self.f_theta_end,self.f_phi_start,self.f_phi_end])
                cat_id_i = 1

            cameraFocusPoints = generatePointDist(f_L,self.f_dr,self.F_shift,self.f_mu,self.f_sigma,self.Nf,self.foc_angles).T
            np.random.shuffle(cameraFocusPoints)

            for i in range(cameraFocusPoints.shape[0]):
                camFocus = cameraFocusPoints[i]
                Hw=cameraLocFocus2Hom2(camFocus,camLoc)
                Hc = Hw2Hc(Hw)
                P = cat_points(cat_id=cat_id_i)
                Pc, _, _, _, _ = projectedKnown3dBoxPoints(P, Hc, self.K)
                count, Bool = is_object_visible(Pc, self.Thr_x, self.Thr_y)

                if Bool:
                    break

            if Bool:
                break
        print((cat_id_i, count, Bool))
        print("r = ",L2.norm(camLoc))

        return Hc, Hw, camLoc, cameraLocations, camFocus, cameraFocusPoints
