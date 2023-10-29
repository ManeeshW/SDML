import numpy as np
from misc_fucntions import twist2Hom
import argparse

TauDir = '/home/maneesh/Desktop/Syn_Data_Genarate_from_ShipCAD_model/CyclesRenderOutput/offline_saved/Tau.txt'
Tau = np.loadtxt(TauDir)

parser = argparse.ArgumentParser(description="Read Pose Text")
parser.add_argument('-i', type=int, default=1,
                        help='Read Tau.txt specific line')
args = parser.parse_args()
i = args.i # Render N number of images online 

H = twist2Hom(Tau[i-1,:])
print(H)
print(H[0:3,3])

