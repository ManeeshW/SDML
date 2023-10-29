import os
import time
import cv2
import numpy as np
from IPython.display import Image
from PIL import Image
import matplotlib.pyplot as plt
from randimage import get_random_image
from mylib.overlay import *
from tqdm import tqdm
from configparser import ConfigParser
import argparse
import asyncio
'''
https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
'''

# instantiate
config = ConfigParser()

#Dir = "/home/maneesh/Desktop/Syn_Data_Genarate_from_ShipCAD_model/"
Dir = os.getcwd()+"/"

# parse existing file
config.read(Dir+'config/DataGen.ini')


fimg = config.get('Output', 'Merge_Sub_dir')+'Train/'
fmask = config.get('Output', 'Merge_Sub_dir')+'TrainMask/'
overlay = config.get('Output', 'Merge_Sub_dir')+'TrainOverlay/'
genTexture = texture_dir = config.get('texture', 'textures_dir')+'GeneratedTexture/'

try: 
    os.mkdir(overlay)
    os.mkdir(genTexture)
except:
    pass

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

N_Masks = num_of_img_files_in_folder(fmask) #520
parser = argparse.ArgumentParser(description="Synthetic image generation")
parser.add_argument('-N', type=int, default=30,
                        help='Number of images want to be generated')
args = parser.parse_args()
N = args.N
N_img = num_of_img_files_in_folder(overlay)

print("No. imgs : ",N_img)

@background
def runOverlay(ibatch, nbatch, N_img, N):
    b = N//nbatch
    Modulus = 0
    if ibatch+1 == nbatch:
        Modulus = (N_img+N) % nbatch
    for i in tqdm(range(ibatch*b+1+N_img, N_img + (ibatch+1)*b+1+Modulus)):
        if i > N_Masks:
            break
        imgN = cv2.imread(fimg+'{:06d}'.format(i)+'.jpg')
        frame = cv2.imread(fmask+'{:06d}'.format(i)+'.png')
        background, rand_img_resized_up = genRandomShip(imgN, frame, ibatch)
        background.save(overlay + "{:06d}.jpg".format(i)) 
        cv2.imwrite(genTexture+"{:d}.jpg".format(i), rand_img_resized_up)

loop = asyncio.get_event_loop()  # Have a new event loop

nbatch = 1
looper = asyncio.gather(*[runOverlay(ibatch, nbatch, N_img, N) for ibatch in range(nbatch)]) # Run the loop

results = loop.run_until_complete(looper)