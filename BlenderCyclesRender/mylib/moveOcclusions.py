import bpy
import mathutils
from mathutils import Vector
from mathutils import Matrix
import numpy as np
import random
from random import randint 
from .prob import *
from . import *

def shift_occlusions():
    
    pdf1 = getRanPosVal_around_point(0,0.4,1000,0,0.15,0)
    x1 = -(pdf1 + 1.3)
    y1 = pdf1 + 4.85
    z1 = 1.63

    pdf2 = getRanPosVal_around_point(0,0.2,1000,0,0.15,0)
    x2 = -(pdf2 + 1.7)
    y2 = pdf2+ 2.35
    z2 = 0
    
    pdf3 = getRanPosVal_around_point(0.6,1,1000,0,10,0)
    xP = -0.3+pdf3
    yP = 2.47
    zP = 0.12 
#    x = -(pdf + 1.75 )
#    y = pdf + 2.3
    Th1 = random.randint(0, 9)
    Th2 = random.randint(0, 9)
    Th3 = random.randint(0, 9)

    if Th1 < 5:
        z1 = -5
    if Th2 < 4:
        z2 = -5
    if Th3 < 5:
        zP = -5
        
    if HUMAN["Movement"] is False:
        x1 = -1.25
        y1 = 4.85
        z1 = 1.63
        x2 = -1.9
        y2 = 2.7
        z2 = 0
        
    if PELICAN["Movement"] is False:
        xP = 0.352026
        yP = 2.51339 
        zP = 0.12 
        
    bpy.data.objects['human.Body'].location = Vector((x1,y1,z1))

    bpy.data.objects['human.Body2'].location = Vector((x2,y2,z2))
    
#    xP = 0.352026
#    yP = 2.51339 
#    zP = 0.12 
    bpy.data.objects['pelican'].location = Vector((xP,yP,zP))
#    pelican = bpy.data.objects['pelican']
#    pelican.matrix_world = M3

