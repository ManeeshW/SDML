import bpy
import mathutils
from mathutils import Vector
from mathutils import Matrix
from .prob import *


def rotate_sun():
    pdf = getRanPosVal_around_point(0,100,1000,0,0.2,0)
    x = pdf
    y = getRanPosVal_around_point(0,100,1000,0,0.2,0)
    z = getRanPosVal_around_point(100,40,1000,0,0.2,0) 
    Hw = np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])
    M = Matrix(((Hw[0,0],  Hw[0,1], Hw[0,2], Hw[0,3]),
                (Hw[1,0],  Hw[1,1], Hw[1,2], Hw[1,3]),
                (Hw[2,0],  Hw[2,1], Hw[2,2], Hw[2,3]),
                (Hw[3,0],  Hw[3,1], Hw[3,2], Hw[3,3])))

    SUN = bpy.data.objects['SUN']
    SUN.matrix_world = M