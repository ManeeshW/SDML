import bpy
import mathutils
from mathutils import Vector
from mathutils import Matrix
import numpy as np
import random
from random import randint 
from .camera import *
from .misc import Hw2Hc, Hc2Hw
from . import *


#Camera motion
def rotateCamera(scene,**kwargs):
    # relative to world frame
    # Pw = RwPc + Tw
    # Hw = [Rw | Tw]

    try:
        cameraFocusPoint=kwargs['cF']
        cameraLocation=kwargs['cL']
        Hw = cameraLocFocus2Hom2(cameraFocusPoint,cameraLocation)
        Hc = Hw2Hc(Hw)
    except:
        try:
            Hw=kwargs['Hw']
            Hc = Hw2Hc(Hw)
        except:
            try:
                Hc=kwargs['Hc']
                Hw = Hw2Hc(Hc)
            except:
                try:
                    Hw = np.eye(4)
                    Hw[0:3,0:3] = kwargs['Rw']
                    Hw[0:3,3] = kwargs['Tw']
                    Hc = Hw2Hc(Hw)
                except:
                    try:
                        Hc = np.eye(4)
                        Hc[0:3,0:3] = kwargs['Rc']
                        Hc[0:3,3] = kwargs['Tc']
                        Hw = Hw2Hc(Hc)
                    except:
                        print("Invalid name")

    
    tt = 2
    tt = tt * np.pi / 180
    #Hw = np.array([[1,0,0,0],[0,np.cos(tt),-np.sin(tt),0],[0,np.sin(tt),np.cos(tt),5],[0,0,0,1]]) #x axis
    #Hw = np.array([[np.cos(tt),0,np.sin(tt),0],[0,1,0,0],[-np.sin(tt),0,np.cos(tt),5],[0,0,0,1]]) #y axis
    #Hw = np.array([[np.cos(tt),-np.sin(tt),0,0],[np.sin(tt),np.cos(tt),0,0],[0,0,1,10],[0,0,0,1]])  #z axis

    # relative to camera frame
    # Pc = Rw^TPw + (-Rw^T Tw)
    # Hc = [Rw^T | -Rw^T Tw] = [Rc | Tc]
    
    # Hc = np.zeros((4,4))
    # Hc[0:3,0:3] = Hw[0:3,0:3].T
    # Hc[0:3,3] = -Hw[0:3,0:3].T @ Hw[0:3,3]
    # Hc[3,3] = 1
    
    #Hc = Hw2Hc(Hw)

    #Camera Location + Rotation
    M = Matrix(((Hw[0,0],  Hw[0,1], Hw[0,2], Hw[0,3]),
                (Hw[1,0],  Hw[1,1], Hw[1,2], Hw[1,3]),
                (Hw[2,0],  Hw[2,1], Hw[2,2], Hw[2,3]),
                (Hw[3,0],  Hw[3,1], Hw[3,2], Hw[3,3])))
         
    camera = bpy.data.objects['Camera']
    camera.matrix_world = M

    np.savetxt(RenDir+"Hw.txt",Hw)
    np.savetxt(RenDir+"Hc.txt",Hc)
    return M, Hw
