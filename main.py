# Author: Maneesha Wickramasuriya
# Company: Flight Dynamics and Control Lab (FDCL)
# License: This code is free to use, modify, and distribute as long as credit and citation are given to the first author and FDCL.

import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import subprocess
import torch
import pandas as pd

from config.config import *
from MLib.plot import *
from MLib.esti import *
from MLib.Cat import *
from MLib.newKey import *
from MLib.CSV import CSV
from MLib.misc import *
#from llib import *
#from llib.Status import *

#Input_ImgDir = ImgDir + "input_data/Test_{}/".format(Test_no)
images = os.listdir(Input_ImgDir)
No_imgs_in_folder = len(images)
curr_img_no = 0

img_No = 1
PH = 1 #prediction horizon
pc_selected = np.array([])
pc_tracked = np.array([])
pcs_added = np.array([])
rc = np.array([])

catID = 1
msg = ""
Hws = np.zeros((No_imgs_in_folder,12))
Hcs = np.zeros((No_imgs_in_folder,12))

# Lists to store the bounding box coordinates
top_left_corner=[]
bottom_right_corner=[]
top_left_corner_scale=[]
bottom_right_corner_scale=[]
Mouse_Move = []

maxScaleUp = 100
scaleFactor = 0
keyCount = 0
trackbarValue = "Scale"
factor = 0.5
    
# function which will be called on mouse input
def selectKeyPoints(action, x, y, flags, *userdata):
  # Referencing global variables 
  global  scaledImg, image, pc_selected, pc_tracked, pcs_added, rc,  curr_img_no, PT, Tk, Nk, msg, keyCount

  # Mark the top left corner when left mouse button is pressed
  scaleValue = cv2.getTrackbarPos('Scale', 'Window')
  scaleFactor = 1+ scaleValue/100.0
  #cv2.displayOverlay("Window", str(1.34445))
  
  if action == cv2.EVENT_LBUTTONDOWN:
   pc_selected = cat(pc_selected, [x,y])
   pcs_added = pc_selected
   keyCount = pc_selected.shape[0]
   rc = np.array([])
   draw_markerSelected(pc_selected, image)
   scaledImg= image.copy()
   cv2.imshow("Window",scaledImg)

  elif action == cv2.EVENT_MOUSEMOVE:
     draw_crosshair(x,y,scaledImg, int(scaleValue/5))
     cv2.imshow("Window",scaledImg)
     scaledImg = cv2.resize(image, None, fx=scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
     #cv2.imshow("Window",scaledImg)
  if curr_img_no != img_No:
     curr_img_no = img_No
     scaleValue = 0
     scaleFactor = 1
     cv2.setTrackbarPos('Scale', 'Window', 0)
     cv2.imshow("Window",scaledImg)

  Tk,Nk = requiredkeys()
  msg = Warning(pc_tracked, Tk, Nk, keyCount, msg)
  
  cv2.displayStatusBar("Window", "Img No. {:03d} [{:03d},{:03d}] | Keys [Tot. : {:02d} | New : {:02d} | Picked : {:02d}]".format(img_No,int(x/scaleFactor),int(y/scaleFactor),Tk,Nk, keyCount)+ msg )


# Create the function for the trackbar since its mandatory but we wont be using it so pass.
def scaleIt(x):
    global scaledImg
    scaledImg = scaleImage(x)
    cv2.imshow("Window",scaledImg)
    pass
   
def scaleImage(value=0):
    global scaledImg
    # Get the scale factor from the trackbar 
    scaleFactor = 1+ value/100.0

    # Resize the image
    scaledImg = cv2.resize(image, None, fx=scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
    return scaledImg
    
def Next(*args):
    global img_No, image, scaledImg, pc_selected, pc_tracked, pcs_added, PH, rc, No_imgs_in_folder, Tk,Nk, msg, keyCount
    img_no = img_No+1
    if img_No < No_imgs_in_folder:
        PH = PH+1
    if img_no <= No_imgs_in_folder:
       img_No = img_no
    else:
       img_No = No_imgs_in_folder
    #cv2.circle(image, (200,200), 10,(255,255,0), -1) 
    #image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    #
    scaledImg= image.copy()
    pc_selected = np.array([])
    cv2.setTrackbarPos('Scale', 'Window', 0)
    cv2.setTrackbarPos('Img No', 'Window', img_No)
    
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    try:
        image, pcs_added = draw_tracked(image, PH)
    except:
        print("There is no keypoints to draw")
    try:
        pc_tracked[:,1:3] = pcs_added
    except:
        pass
    scaledImg= image.copy()
    Tk,Nk = requiredkeys()
    msg = Warning(pc_tracked, Tk, Nk, keyCount, msg)
    cv2.displayStatusBar("Window", "Img No. {:03d} [{:03d},{:03d}] | Keys [Tot. : {:02d} | New : {:02d} | Picked : {:02d}]".format(img_No,0,0,Tk,Nk, keyCount)+ msg) 

    cv2.imshow("Window",scaledImg)

def Back(*args):
    global img_No, image, scaledImg, pc_selected, pc_tracked, pcs_added, PH, Tk, Nk, msg, keyCount
    img_no = img_No-1
    PH = PH-1
    if img_no < 1:
       img_No = 1
    else:
       img_No = img_no
    
    scaledImg = image.copy()
    pc_selected = np.array([])
    cv2.setTrackbarPos('Scale', 'Window', 0)
    cv2.setTrackbarPos('Img No', 'Window', img_No)

    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    image, pcs_added = draw_tracked(image, PH)
    pc_tracked[:,1:3] = pcs_added
    scaledImg= image.copy()
    Tk,Nk = requiredkeys()
    msg = Warning(pc_tracked, Tk, Nk, keyCount, msg)
    cv2.displayStatusBar("Window", "Img No. {:03d} [{:03d},{:03d}] | Keys [Tot. : {:02d} | New : {:02d} | Picked : {:02d}]".format(img_No,0,0,Tk,Nk, keyCount)+ msg) 

    cv2.imshow("Window",scaledImg) 
    

# Create the function for the trackbar since its mandatory but we wont be using it so pass.
def imageScroll(x):
    global img_No, image, scaledImg, curr_img_no, PT
    img_No = cv2.getTrackbarPos('Img No', 'Window')
    if img_No == 0:
       img_No = 1
    curr_img_no = img_No
    PT = 1
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    scaledImg= image.copy()
    cv2.imshow("Window",scaledImg) 

def status(*args):
    global img_No, pc_selected, pc_tracked, PH
    print("Points pre Selected : \n", pc_selected)
    #pw, pc_tracked = observeKeysDict(pc_tracked, pc)
    print("img_No : ", img_No)
    print("Points Tracked : \n", pc_tracked)
    #print("Points Tracked : \n", pw)

def ShowTracked(*args):
    global img_No, image, scaledImg, pc_selected, pc_tracked, pcs_added, PH
    #im =  cv2.imread("/home/maneesh/Desktop/LAB2.0/my_Git/E_Test_6_2023.06.26/{:06}.jpg".format(img_No), 1)
    try:
        image, pcs_added = draw_tracked(image, PH)
    except:
        print("There is no tracked keypoints to show")

    try:
        pc_tracked[:,1:3] = pcs_added
    except:
        pc_selected = pcs_added
    #pw, pc_tracked = observeKeysDict(pc_tracked, pc_cotracked)
    scaledImg= image.copy()
    cv2.imshow("Window",scaledImg)

def Track(*args):
    global img_No, image, scaledImg, pc_selected, pc_tracked,pcs_added, PH, No_imgs_in_folder, Tk,Nk, msg, keyCount
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    pw, pc_tracked, msg = observeKeysDict(pc_tracked, pc_selected)
    PH = 1
    imN = torch.ones(pc_tracked.shape[0]) * (PH-1)
    quaries = torch.cat((imN.unsqueeze(1),torch.from_numpy(pc_tracked[:,1:3])), dim=1)
    print(quaries.size())
    torch.save(quaries, 'q.pt')
    H = 100
    if img_No+H > No_imgs_in_folder:
        img_Tracked = No_imgs_in_folder
    else:
       img_Tracked = img_No+H
       
    subprocess.check_output(["python3 Tracker.py -S {} -N {}".format(img_No, img_Tracked)],shell=True)
    #msg = ">>>> Img : {} to {}  Tracked <<<<<".format(img_No,img_No+H)
    #draw_markerTracked(pc_tracked, image)
    image, pcs_added = draw_tracked(image, PH)
    pc_selected = []
    scaledImg= image.copy()
    Tk,Nk = requiredkeys()
    msg = Warning(pc_tracked, Tk, Nk, keyCount, msg)
    cv2.displayStatusBar("Window", "Img No. {:03d} [{:03d},{:03d}] | Keys [Tot. : {:02d} | New : {:02d} | Picked : {:02d}]".format(img_No,0,0,Tk,Nk, keyCount)+ msg) 
    cv2.imshow("Window",scaledImg)


def Refresh(*args):
    global img_No, image, scaledImg, pc_selected, pc_tracked, Tk,Nk, msg, keyCount
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    scaledImg= image.copy()
    pc_selected = np.array([])
    pc_tracked  = np.array([])
    keyCount = 0
    try:
        subprocess.check_output(["rm "+out_bimg+"{:06}.jpg".format(img_No)],shell=True)
    except:
        pass
    cv2.displayStatusBar("Window", "Img No. {:03d} [{:03d},{:03d}] | Keys [Tot. : {:02d} | New : {:02d} | Picked : {:02d}]".format(img_No,0,0,Tk,Nk, keyCount)+ msg) 
    cv2.imshow("Window",scaledImg)

def Undo(*args):
    global img_No, image, scaledImg, pc_selected, pc_tracked, rc, Tk,Nk, msg, keyCount

    try:
        rc = cat(rc, pc_selected[-1])
        pc_selected = pc_selected[:-1]
    except:
        pass

    keyCount = pc_selected.shape[0]
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))

    draw_markerSelected(pc_selected, image)

    scaledImg= image.copy()
    Tk,Nk = requiredkeys()
    msg = Warning(pc_tracked, Tk, Nk, keyCount, msg)
    cv2.displayStatusBar("Window", "Img No. {:03d} [{:03d},{:03d}] | Keys [Tot. : {:02d} | New : {:02d} | Picked : {:02d}]".format(img_No,0,0,Tk,Nk, keyCount)+ msg) 
    cv2.imshow("Window",scaledImg)

def Redo(*args):
    global img_No, image, scaledImg, pc_selected, pc_tracked, rc, Tk, Nk, msg, keyCount
    try:
        pc_selected = cat(pc_selected,rc[-1])
        rc = rc[:-1]
    except:
        pass

    keyCount = pc_selected.shape[0]
    image = cv2.imread(Input_ImgDir + "{:06}.jpg".format(img_No))

    draw_markerSelected(pc_selected, image)

    scaledImg= image.copy()
    Tk,Nk = requiredkeys()
    msg = Warning(pc_tracked, Tk, Nk, keyCount, msg)
    cv2.displayStatusBar("Window", "Img No. {:03d} [{:03d},{:03d}] | Keys [Tot. : {:02d} | New : {:02d} | Picked : {:02d}]".format(img_No,0,0,Tk,Nk, keyCount)+ msg) 
    cv2.imshow("Window",scaledImg)

def Save(*args):
    global img_No, image, pc_selected, Hws, Hcs
    cv2.imwrite(out_img+'/{}.png'.format(img_No), image)
    #np.savetxt(out_annot+"Hc_gt.txt",Hcs)
    np.savetxt(out_annot+"Hw_gt_saved.txt",Hws)   
    
def OpenImgLabel(*args):
    global img_No, pcs_added, Hws, Hcs, PW
    df = pd.read_excel('pw.xlsx',header=0, sheet_name='All')
    csv = CSV(df)
    PW = csv.Pw

    #fig = plt.figure(figsize=(6.4,4.8),dpi = 150)
    Hc, Hw = Epnp2H(PW, pcs_added, K, dist = None)
    Hcs[img_No-1,:] = Hc.reshape(1,12)
    Hws[img_No-1,:] = Hw.reshape(1,12)

    #np.savetxt(out_annot+"Hc_label_{}.txt".format(Test_no),Hcs)
    np.savetxt(out_annot+"Hw_gt.txt".format(Test_no),Hws)
    
    global catID
    if catID == 11:
       catID = 1
    # im = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    box_pc = est_pc(Hc, K, cat_id = catID)
    catID = catID+1
    #plot_box(pc, clr_lines = "green", clr_corners = "red", clr_int_corners = "blue")
    #plot_box(box_pc, clr_lines = "orange", corners = False, int_corners = False, linewidth=1.3,  label = 'TNN-PnP Est. for Real imgs')
    #image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    im = image.copy()
    cv2draw_boxLines(im, box_pc, 1)

    # imshow(im)
    # plt.savefig(out_label+"{:06}.jpg".format(img_No))
    # im = cv2.imread(out_label+"{:06}.jpg".format(img_No))
    cv2.imshow("Window",im)
    cv2.waitKey(5000)  
    im = image.copy()
    catID = 1

def blendTrackBar(*args):
    global factor
    

def blenderOut(*args):
    global img_No, image, scaledImg, bImg, factor, Hws
    image_pil = Image.fromarray(image)
    factor = cv2.getTrackbarPos('Blend', 'Window')
    try:
        Hws = np.loadtxt(out_annot+"Hw_gt.txt")
    except:
        pass

    try:

        list = os.listdir(out_bimg)

        if np.sum(Hws[img_No-1,:])!=0:
            if "{:06}.jpg".format(img_No) in list:
                bImg = cv2.imread(out_bimg + "{:06}.jpg".format(img_No))
            else:
                bImg = np.zeros_like(image)
                subprocess.check_output(["python3 BlenderCyclesRender/RunShipblenderCycles.py -N {}".format(img_No)],shell=True)
                bImg = cv2.imread(out_bimg + "{:06}.jpg".format(img_No))
        else:
            factor = 90
            bImg = np.zeros_like(image)
            try:
                subprocess.check_output(["rm "+out_bimg+"{:06}.jpg".format(img_No)],shell=True)
            except:
                pass
        #bImg_pil = Image.fromarray(bImg)
    except:
        bImg = np.zeros_like(image)

        #bImg_pil = Image.fromarray(bImg)

    bImg_pil = Image.fromarray(bImg)
    #x = cv2.getTrackbarPos('Blend', 'Window')
    blend_image = Image.blend(image_pil,bImg_pil,factor/100)
    #grey = cv2.cvtColor(sImg, cv2.COLOR_RGB2GRAY)
    # Make the grey scale image have three channels
    #grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    bimage = np.asarray(blend_image)
    sImg = np.hstack((image, bimage))
    cv2.imshow("Window",sImg)
    cv2.imwrite(out_comp+ "{:06}.jpg".format(img_No), bimage)

    sImg = scaledImg 

# Read Images
image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))

scaledImg= image.copy()
# Make a temporary image, will be useful to clear the drawing
temp = image.copy()
# Create a named window
cv2.namedWindow("Window")
#cv2.namedWindow("Label")
# highgui function called when mouse events occur
cv2.setMouseCallback("Window", selectKeyPoints)
# Create trackbar and associate a callback function / Attach mouse call back to a window and a method
#cv2.setMouseCallback('Window', draw_circle)
# cv2.createButton("Window",back)

cv2.createButton("<-Back",Back,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Next->",Next,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Status",status,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Refresh",Refresh,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Undo",Undo,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Redo",Redo,None,cv2.QT_PUSH_BUTTON,1)

cv2.createButton("ShowTracked",ShowTracked,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Track",Track,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Img-Label",OpenImgLabel,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Save",Save,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Blend",blenderOut,None,cv2.QT_PUSH_BUTTON,1)

# Create trackbar and associate a callback function / create trackbars Named Radius with the range of 150 and starting position of 5.
cv2.createTrackbar('Scale', 'Window', 0, 200, scaleIt) 
cv2.createTrackbar('Img No', 'Window', 1, No_imgs_in_folder, imageScroll) 
cv2.createTrackbar('Blend', 'Window', 50, 100, blenderOut) 

# Create trackbar and associate a callback function
#cv2.createTrackbar(trackbarValue, windowName, scaleFactor, maxScaleUp, scaleImage)

k=0
# Close the window when key q is pressed
while k!=113:
  
  # Display the image
  scaledImg = scaleImage()
#  cv2.imshow(windowName, scaledImage) 
  cv2.imshow("Window",scaledImg)
  k = cv2.waitKey(0)
  # If c is pressed, clear the window, using the dummy image

  
  if (k == 99):
    print('reset')
    cv2.setTrackbarPos('Scale', 'Window', 0)
    image= temp.copy()
    cv2.imshow("Window", image)
    
c = cv2.waitKey(0)
cv2.destroyAllWindows() 


