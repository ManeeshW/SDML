import os
import time
import cv2
import numpy as np
from PIL import Image, ImageFilter 
import matplotlib
import matplotlib.pyplot as plt
import random
from randimage import get_random_image
'''
randimage - create random images in python
https://github.com/nareto/randimage/tree/master
'''


def MaskIt(pred_mask):
    '''
    (128, 128, 192),
    (128, 0, 64),
    (128, 96, 0),
    '''
    pred_label = np.where(pred_mask != 192, pred_mask, 255)
    im = np.zeros(pred_label.shape, np.uint8)
    ship = np.where(pred_mask != 192, pred_mask, 255)
    ret, im[:,:,0] = cv2.threshold(ship[:,:,0], 250, 255, cv2.THRESH_BINARY)

    sea = np.where(pred_mask != 64, pred_mask, 255)
    ret, im[:,:,1] = cv2.threshold(sea[:,:,0], 250, 255, cv2.THRESH_BINARY)
    sky = np.where(pred_mask != 0, pred_mask, 255)
    ret, im[:,:,2] = cv2.threshold(sky[:,:,0], 250, 255, cv2.THRESH_BINARY)
    return im

def genRandomShip(imgN, frame, i):
    pred_mask = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    mask = MaskIt(pred_mask)
    img = get_random_image((480,640))
    matplotlib.use('agg')
    fig = plt.figure(frameon=False)
    fig.set_size_inches(12.8,9.6)
    plt.imshow(img)
    plt.axis('off')
    fig.savefig('tmp_{}.png'.format(i), bbox_inches='tight', pad_inches=0)
    img2 = cv2.imread('tmp_{}.png'.format(i))
    os.remove('tmp_{}.png'.format(i))
    rand_img_resized_up = cv2.resize(img2, (640, 480), interpolation= cv2.INTER_LINEAR)
    masked = cv2.bitwise_and(imgN, imgN, mask=mask[:,:,0])
    foreground = Image.fromarray(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    background = Image.fromarray(cv2.cvtColor(rand_img_resized_up, cv2.COLOR_BGR2RGB))

    smask = Image.fromarray(mask[:,:,0])
    background.paste(foreground , (0,0), mask = smask)
    rand = random.randint(0,1)
    background=background.filter(ImageFilter.GaussianBlur(radius = rand)) 
    return background, rand_img_resized_up

def num_of_img_files_in_folder(idir):
    list = os.listdir(idir) # dir is your directory path
    number_files = len(list)
    return number_files

