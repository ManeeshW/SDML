from random import randint 
import random
import numpy as np
from . import  *

def make_a_scene(freq = 1,  feedback = "", texture = TEXTURE):
    if freq == 1:
        jpg_N = len(os.listdir(Sky)) #number of jpg files in highfrequency folder
        frequncy = "sky"
    elif freq == 2:
        jpg_N = len(os.listdir(Sea)) #number of jpg files in lowfrequency folder
        frequncy = "sea"
    elif freq == 3:
        jpg_N = len(os.listdir(Shipskin)) #number of jpg files in lowfrequency folder
        frequncy = "shipskin"
    elif freq == 4:
        jpg_N = len(os.listdir(Landingpad)) #number of jpg files in lowfrequency folder
        frequncy = "landingpad"
    elif freq == 5:
        jpg_N = len(os.listdir(Markings)) #number of jpg files in lowfrequency folder
        frequncy = "markings"
    if texture["FIXED_TEXTURE"]: 
        jpg_i = texture[frequncy]
    else:  
        jpg_i = random.randint(1,jpg_N)
        

        
    jpg = '{:d}'.format(jpg_i)+".jpg"
    #print(frequncy," ",jpg,"  loaded")
    return jpg

def H_AB_to_H_BA(H_AB):  
    # A->B  to B->A
    # relative to camera frame
    # Pc = Rw^TPw + (-Rw^T Tw)
    # Hc = [Rw^T | -Rw^T Tw] = [Rc | Tc]
    H_BA = np.zeros((4,4))
    H_BA[0:3,0:3] = H_AB[0:3,0:3].T
    H_BA[0:3,3] = -H_AB[0:3,0:3].T @ H_AB[0:3,3]
    H_BA[3,3] = 1
    return H_BA

def Hw2Hc(Hw):
    Hc = np.zeros((4,4))
    Hc[0:3,0:3] = Hw[0:3,0:3].T
    Hc[0:3,3] = -Hw[0:3,0:3].T @ Hw[0:3,3]
    Hc[3,3] = 1
    return Hc

def Hc2Hw(Hc):
    Hw = np.zeros((4,4))
    Hw[0:3,0:3] = Hc[0:3,0:3].T
    Hw[0:3,3] = -Hc[0:3,0:3].T @ Hc[0:3,3]
    Hw[3,3] = 1
    return Hw
