import cv2
import os
import numpy as np
from mlib.plot import *
from mlib.esti import *
from mlib.Cat import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import subprocess
import torch
from config import *
import pandas as pd
from pd.CSV import CSV


img_No = 1
#Input_ImgDir = ImgDir + "input_data/Test_{}/".format(Test_no)
images = os.listdir(Input_ImgDir)
No_imgs_in_folder = len(images)
curr_img_no = 0

# Create the function for the trackbar since its mandatory but we wont be using it so pass.
def imageScroll(x):
    global img_No, image, scaledImg, curr_img_no
    img_No = cv2.getTrackbarPos('Img No', 'Window')
    curr_img_no = img_No
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    scaledImg= image.copy()
    cv2.imshow("Window",scaledImg) 
    pass

pc = np.array([])
pc_tracked = np.array([])
rc = np.array([])
pcs_added = np.array([])

catID = 1

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
#windowName = "Resize Image"
trackbarValue = "Scale"

PH = 1 #prediction horizon

def markPointWin(pc, image,  win = "Window", color = (120,255,50)):
    for i in range(pc.shape[0]):
       cv2.circle(image, (pc[i][0],pc[i][1]) , 1,color, -1)
    scaledImg= image.copy()
    cv2.imshow(win,scaledImg)
    return scaledImg

def markPoint(pc, image, color = (120,255,50)):
    for i in range(pc.shape[0]):
       cv2.circle(image, (pc[i][0],pc[i][1]) , 1,color, -1)

# function which will be called on mouse input
def selectKeyPoints(action, x, y, flags, *userdata):
  # Referencing global variables 
  global  scaledImg, image, pc, pc_tracked, rc,  curr_img_no
  # Mark the top left corner when left mouse button is pressed
  scaleValue = cv2.getTrackbarPos('Scale', 'Window')
  scaleFactor = 1+ scaleValue/100.0
  
  #cv2.displayOverlay("Window", str(1.34445))
  cv2.displayStatusBar("Window", "Image No : {}      {}|{}".format(img_No,int(x/scaleFactor),int(y/scaleFactor))) 

  if action == cv2.EVENT_LBUTTONDOWN:
   pc = cat(pc, [x,y])
   rc = np.array([])
   markPoint(pc, image)
   markPoint(pc_tracked, image, (255,120,50))
   scaledImg= image.copy()
   cv2.imshow("Window",scaledImg)

  elif action == cv2.EVENT_MOUSEMOVE:
     cv2.line(scaledImg, (x-3,y),(x-6,y), (255,20,200), 1)
     cv2.line(scaledImg, (x+3,y),(x+6,y), (255,20,200), 1)
     cv2.line(scaledImg, (x,y-3),(x,y-6), (255,20,200), 1)
     cv2.line(scaledImg, (x,y+3),(x,y+6), (255,20,200), 1)

     cv2.circle(scaledImg, (x,y), 1, (255,20,200), -1)
     
     markPoint(pc_tracked, scaledImg, (255,120,0))
     cv2.imshow("Window",scaledImg)
     scaledImg = cv2.resize(image, None, fx=scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
     #cv2.imshow("Window",scaledImg)
  if curr_img_no != img_No:
     curr_img_no = img_No
     scaleValue = 0
     scaleFactor = 1
     cv2.setTrackbarPos('Scale', 'Window', 0)
     cv2.imshow("Window",scaledImg)

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
    global img_No, image, scaledImg, pc, PH, rc, No_imgs_in_folder
    img_no = img_No+1
    if img_No < No_imgs_in_folder:
        PH = PH+1
    if img_no <= No_imgs_in_folder:
       img_No = img_no
    else:
       img_No = No_imgs_in_folder

    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    scaledImg= image.copy()

    pc = np.array([])
 
    cv2.setTrackbarPos('Scale', 'Window', 0)
    cv2.setTrackbarPos('Img No', 'Window', img_No)
    cv2.imshow("Window",scaledImg)

def status(*args):
    global img_No, pc, pc_tracked, PH
    print("img_No : ", img_No)
    print("Points Selected : \n", pc)
    print("Points Tracked : \n", pc_tracked)

def ShowTracked(*args):
    global img_No, image, scaledImg, pc, pc_tracked, PH
    im =  cv2.imread("/home/maneesh/Desktop/LAB2.0/my_Git/E_Test_6_2023.06.26/{:06}.jpg".format(img_No), 1)
    im, pc_tracked = trackedImg(im, PH)
    print("tracked pc_t : ", pc_tracked)
    cv2.imshow("Window",im)

def trackedImg(im, No):
    x = torch.load('tracked.pt')
    p = x.cpu().numpy().astype(np.int16)
    for i in range(p.shape[2]):
        print(p[0][No-1][i,:].tolist())
        cv2.circle(im, p[0][No-1][i,:].tolist(), 1,(200,100,205), -1)
    return im, p[0][No-1]

def Track(*args):
    global img_No, image, scaledImg, pc, PH, No_imgs_in_folder
    PH = 1
    imN = torch.ones(pc.shape[0]) * (PH-1)
    quaries = torch.cat((imN.unsqueeze(1),torch.from_numpy(pc)), dim=1)
    #print(quaries.size())
    torch.save(quaries, 'q.pt')
    if img_No+100 > No_imgs_in_folder:
        img_Tracked = No_imgs_in_folder
    else:
       img_Tracked = img_No+100
       
    subprocess.check_output(["python3 Tracker.py -S {} -N {}".format(img_No, img_Tracked)],shell=True)
    print("Img : ", img_No, " - ", img_No + 100, "  Tracked")

def Refresh(*args):
    global img_No, image, scaledImg, pc
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    scaledImg= image.copy()
    pc = np.array([])
    cv2.imshow("Window",scaledImg)

def Undo(*args):
    global img_No, image, scaledImg, pc, rc
    # print("pc before : ", pc)
    # print("rc before : ", rc)
    try:
        rc = cat(rc, pc[-1])
        pc = pc[:-1]
        # print("pc after : ", pc)
        # print("rc after : ", rc)
        # print("--------------")
    except:
        pass
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    for i in range(pc.shape[0]):
       cv2.circle(image, (pc[i][0],pc[i][1]) , 1,(255,255,0), -1)
    scaledImg= image.copy()
    cv2.imshow("Window",scaledImg)

def Redo(*args):
    global img_No, image, scaledImg, pc, rc
    # print("pc before : ", pc)
    # print("rc before : ", rc)
    try:
        pc = cat(pc,rc[-1])
        rc = rc[:-1]
    except:
        pass
    # print("pc after : ", pc)
    # print("rc after : ", rc)
    # print("------rc--------")
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    for i in range(pc.shape[0]):
       cv2.circle(image, (pc[i][0],pc[i][1]) , 1,(255,255,0), -1)
    scaledImg= image.copy()
    cv2.imshow("Window",scaledImg)

def Back(*args):
    global img_No, image, scaledImg, pc, PH
    img_no = img_No-1
    PH = PH-1
    if img_no < 1:
       img_No = 1
    else:
       img_No = img_no
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    scaledImg = image.copy()
    pc = np.array([])

    cv2.setTrackbarPos('Scale', 'Window', 0)
    cv2.setTrackbarPos('Img No', 'Window', img_No)
    cv2.imshow("Window",scaledImg) 
    
def Save(*args):
    global img_No, image, pc, Hws, Hcs
    cv2.imwrite(out_img+'/{}.png'.format(img_No), image)
    np.savetxt(out_annot+"Hc_Test_{}_gt.txt".format(Test_no),Hcs)
    np.savetxt(out_annot+"Hw_Test_{}_gt.txt".format(Test_no),Hws)   
    
def OpenImgLabel(*args):
    global img_No, pc, Hws, Hcs, PW
    print(pc)
    df = pd.read_excel('pw.xlsx',header=0, sheet_name='All')
    csv = CSV(df)
    PW = csv.Pw

    fig = plt.figure(figsize=(6.4,4.8),dpi = 150)
    Hc, Hw = Epnp2H(PW, pc, K, dist = None)
    Hcs[img_No-1,:] = Hc.reshape(1,12)
    Hws[img_No-1,:] = Hw.reshape(1,12)

    np.savetxt(out_annot+"Hc_Test_{}_gt.txt".format(Test_no),Hcs)
    np.savetxt(out_annot+"Hw_Test_{}_gt.txt".format(Test_no),Hws)

    global catID
    if catID == 11:
       catID = 1
    im = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    box_pc = est_pc(Hc, K, cat_id = catID)
    catID = catID+1
    #plot_box(pc, clr_lines = "green", clr_corners = "red", clr_int_corners = "blue")
    plot_box(box_pc, clr_lines = "orange", corners = False, int_corners = False, linewidth=1.3,  label = 'TNN-PnP Est. for Real imgs')
    imshow(im)
    plt.savefig(out_label+"{:06}.jpg".format(img_No))
    im = cv2.imread(out_label+"{:06}.jpg".format(img_No))
    cv2.imshow("Label",im)
    cv2.waitKey(5000)  
    cv2.destroyWindow("Label") 
    fig.canvas.draw()
    fig.canvas.flush_events()
    catID = 1

# Read Images
image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))

scaledImg= image.copy()
# Make a temporary image, will be useful to clear the drawing
temp = image.copy()
# Create a named window
cv2.namedWindow("Window")
cv2.namedWindow("Label")
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

# Create trackbar and associate a callback function / create trackbars Named Radius with the range of 150 and starting position of 5.
cv2.createTrackbar('Scale', 'Window', 0, 200, scaleIt) 
cv2.createTrackbar('Img No', 'Window', 1, No_imgs_in_folder, imageScroll) 
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


