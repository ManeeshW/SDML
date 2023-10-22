import cv2
import os
import numpy as np
from lib.plot import *
from lib.esti import *
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
print("No. Images : ",No_imgs_in_folder)
curr_img_no = 0
Pc = np.array([])
Pcs = np.array([])
arry = []
catID = 1

Hws = np.zeros((No_imgs_in_folder,12))
Hcs = np.zeros((No_imgs_in_folder,12))

# Lists to store the bounding box coordinates
top_left_corner=[]
bottom_right_corner=[]
top_left_corner_scale=[]
bottom_right_corner_scale=[]
Mouse_Move = []

TK = np.zeros(PW_label.shape[0])
PW = PW_label.copy()

maxScaleUp = 100
scaleFactor = 0
#windowName = "Resize Image"
trackbarValue = "Scale"

PH = 1 #prediction horizon

# def fourPointsOnLine(P_A, P_B):
#    P_C = P_A + (P_B-P_A) / 3
#    P_D = P_A + 2* (P_B-P_A) / 3
#    P = np.concatenate((P_C, P_D), axis=0).reshape(2,P_A.shape[0])
#    P_all = np.concatenate((P_A, P_C, P_D, P_B), axis=0).reshape(4,P_A.shape[0])
#    return P_all, P

# def BipartiteInterpolation(P):
#     P_I = []
#     for i in range(P.shape[0]):
#         for j in range(i+1,P.shape[0]):
#             _ , p_i = fourPointsOnLine(P[i], P[j])
#             P_I = np.append(P_I,p_i.reshape(-1), axis=0)

#     P_Is = P_I.reshape(-1,P.shape[1])
#     Pc  = np.concatenate((P,P_Is), axis = 0)
#     return Pc
        
# function which will be called on mouse input
def drawLine(action, x, y, flags, *userdata):
  # Referencing global variables 
  global MMD, top_left_corner,top_left_corner_scale, bottom_right_corner,bottom_right_corner_scale, scaledImg, image, arry, Pc, curr_img_no, scaleFactor
  # Mark the top left corner when left mouse button is pressed
  scaleValue = cv2.getTrackbarPos('Scale', 'Window')
  scaleFactor = 1+ scaleValue/100.0
  
  #cv2.displayOverlay("Window", str(1.34445))
  cv2.displayStatusBar("Window", "Image No : {}      {}|{}".format(img_No,int(x/scaleFactor),int(y/scaleFactor))) 
  #cv2.circle(scaledImg, (x,y), 10, (255,0,0), -1)
  #cv2.imshow("Window",scaledImg)
  #
#  scaleImg = scaleImage(radius)
  if action == cv2.EVENT_LBUTTONDOWN:
   top_left_corner = [(x,y)]
   top_left_corner_scale = [(int(x/scaleFactor),int(y/scaleFactor))]
   
   #print(np.array([x,y]))
   arry.append([x,y])
   Pc = np.array(arry)

    #arr = np.append([arr,], axis=0)
    # When left mouse button is released, mark bottom right corner
  elif action == cv2.EVENT_LBUTTONUP:
   #  bottom_right_corner = [(x,y)]  
   #  bottom_right_corner_scale = [(int(x/scaleFactor),int(y/scaleFactor))]
    # Draw the rectangle
    cv2.circle(scaledImg, top_left_corner[0], 1, (2,255,0), -1)
    cv2.circle(image, top_left_corner_scale[0], 1, (2,255,0), -1)
    # cv2.rectangle(image, top_left_corner_scale[0], bottom_right_corner_scale[0], (0,255,0),2, 8)
    # cv2.rectangle(scaledImg, top_left_corner[0], bottom_right_corner[0], (0,255,0),2, 8)
    cv2.imshow("Window",scaledImg)

  elif action == cv2.EVENT_MOUSEMOVE:
     cv2.circle(scaledImg, (x,y), 1, (255,20,200), -1)
     cv2.imshow("Window",scaledImg)
     scaledImg = cv2.resize(image, None, fx=scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)

  elif action == cv2.EVENT_MBUTTONDOWN:
     MMD = np.array([x,y])

  elif action == cv2.EVENT_MBUTTONUP:
     for i in range(Pc.shape[0]):
       if Pc[i][0] ==  MMD[0] and Pc[i][1] ==  MMD[1]:
         cv2.circle(image, (MMD[0],MMD[1]), 1, (255,255,0), -1)

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
    global img_No, image, scaledImg, Pc, arry, PH
    img_no = img_No+1
    PH = PH+1
    if img_no <= No_imgs_in_folder:
       img_No = img_no
    else:
       img_No = No_imgs_in_folder

    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    scaledImg= image.copy()
    imN = torch.ones(Pc.shape[0]) * (img_no-1)
   #  print(torch.ones(Pc.shape[0]))
   #  print(torch.from_numpy(Pc).size())
   #  print(torch.cat((imN.unsqueeze(1),torch.from_numpy(Pc)), dim=1))
    p = torch.cat((imN.unsqueeze(1),torch.from_numpy(Pc)), dim=1)
    trackedImg(scaledImg, PH)
    Pc = np.array([])
    arry = []    
    cv2.setTrackbarPos('Scale', 'Window', 0)
    cv2.imshow("Window",scaledImg)

def ShowTracked(*args):
    global img_No, image, scaledImg, Pc, arry, PH
    im =  cv2.imread("/home/maneesh/Desktop/LAB2.0/my_Git/E_Test_6_2023.06.26/{:06}.jpg".format(img_No), 1)
    im, Pc = trackedImg(im, PH)
    cv2.imshow("Window",im)

def trackedImg(im, No):
    x = torch.load('tracked.pt')
    p = x.cpu().numpy().astype(np.int16)
    for i in range(p.shape[2]):
        cv2.circle(im, p[0][No-1][i,:].tolist(), 1,(200,100,205), -1)
    return im, p[0][No-1]

def Track(*args):
    global img_No, image, scaledImg, Pc, arry, PH
    PH = 1
    imN = torch.ones(Pc.shape[0]) * (PH-1)
    quaries = torch.cat((imN.unsqueeze(1),torch.from_numpy(Pc)), dim=1)
    torch.save(quaries, 'q.pt')
    subprocess.check_output(["python3 Tracker.py -S {} -N {}".format(img_No, img_No + 100)],shell=True)
    print("Img : ", img_No, " - ", img_No + 100, "  Tracked")

def Refresh(*args):
    global img_No, image, scaledImg, Pc, arry
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    scaledImg= image.copy()
    Pc = np.array([])
    arry = []
    cv2.imshow("Window",scaledImg)

def Back(*args):
    global img_No, image, scaledImg, Pc, arry, PH
    img_no = img_No-1
    PH = PH-1
    if img_no < 1:
       img_No = 1
    else:
       img_No = img_no
    image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    scaledImg = image.copy()
    Pc = np.array([])
    arry = []
    cv2.setTrackbarPos('Scale', 'Window', 0)
    cv2.imshow("Window",scaledImg) 
    
def Save(*args):
    global img_No, image, Pc, Hws, Hcs
    cv2.imwrite(out_img+'/{}.png'.format(img_No), image)
    np.savetxt(out_annot+"Hc_Test_{}_gt.txt".format(Test_no),Hcs)
    np.savetxt(out_annot+"Hw_Test_{}_gt.txt".format(Test_no),Hws)   
    
def OpenImgLabel(*args):
    global img_No, Pc, Hws, Hcs, PW

    df = pd.read_excel('pw.xlsx',header=0, sheet_name='All')
    csv = CSV(df)
    PW = csv.Pw

    fig = plt.figure(figsize=(6.4,4.8),dpi = 150)
   #  Pws = BipartiteInterpolation(PW)
   #  Pcs = BipartiteInterpolation(Pc)
    Hc, Hw = Epnp2H(PW, Pc, K, dist = None)
    Hcs[img_No-1,:] = Hc.reshape(1,12)
    Hws[img_No-1,:] = Hw.reshape(1,12)

    np.savetxt(out_annot+"Hc_Test_{}_gt.txt".format(Test_no),Hcs)
    np.savetxt(out_annot+"Hw_Test_{}_gt.txt".format(Test_no),Hws)

    global catID
    if catID == 11:
       catID = 1
    im = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    pc = est_pc(Hc, K, cat_id = catID)
    catID = catID+1
    #plot_box(pc, clr_lines = "green", clr_corners = "red", clr_int_corners = "blue")
    plot_box(pc, clr_lines = "orange", corners = False, int_corners = False, linewidth=1.3,  label = 'TNN-PnP Est. for Real imgs')
    imshow(im)
    plt.savefig(out_label+"{:06}.jpg".format(img_No))
    im = cv2.imread(out_label+"{:06}.jpg".format(img_No))
    cv2.imshow("Label",im)
    cv2.waitKey(5000)  
    cv2.destroyWindow("Label") 
    fig.canvas.draw()
    fig.canvas.flush_events()
    catID = 1

# def undo():
#     global Pc
# def deletePw(TK, PW_label):
#     PW
#    Del = []
#    for i in range(TK.shape[0]):
#       if TK[i] == 0:
#          Del = np.append(Del, i).astype(int)
#    PW = PW_label.copy()
#    PW = np.delete(PW,Del,axis=0)


# def click(j=1): 
#    i = j-1
#    global TK
#    if TK[i]:
#       TK[i] = 0
#       deletePw(TK, PW_label)
#    else:
#       TK[i] = 1
#       deletePw(TK, PW_label)


# def Tick1(*args): click(1)
# def Tick2(*args): click(2)
# def Tick3(*args): click(3)
# def Tick4(*args): click(4)
# def Tick5(*args): click(5)
# def Tick6(*args): click(6)
# def Tick7(*args): click(7)
# def Tick8(*args): click(8)
# def Tick9(*args): click(9)
# def Tick10(*args): click(10)
# def Tick11(*args): click(11)
# def Tick12(*args): click(12)
# def Tick13(*args): click(13)
# def Tick14(*args): click(14)
# def Tick15(*args): click(15)
# def Tick16(*args): click(16)
# def Tick17(*args): click(17)
# def Tick18(*args): click(18)
# def Tick19(*args): click(19)
# def Tick20(*args): click(20)
# def Tick21(*args): click(21)
# def Tick22(*args): click(22)
# def Tick23(*args): click(23)
# def Tick24(*args): click(24)
# def Tick25(*args): click(25)
# def Tick26(*args): click(26)
# def Tick27(*args): click(27)
# def Tick28(*args): click(28)
# def Tick29(*args): click(29)
# def Tick30(*args): click(30)
# def Tick31(*args): click(31)
# def Tick32(*args): click(32)
# def Tick33(*args): click(33)
# def Tick34(*args): click(34)
# def Tick(*args): pass

# Read Images
image = cv2.imread(Input_ImgDir+ "{:06}.jpg".format(img_No))

scaledImg= image.copy()
# Make a temporary image, will be useful to clear the drawing
temp = image.copy()
# Create a named window
cv2.namedWindow("Window")
cv2.namedWindow("Label")
# highgui function called when mouse events occur
cv2.setMouseCallback("Window", drawLine)
# Create trackbar and associate a callback function / Attach mouse call back to a window and a method
#cv2.setMouseCallback('Window', draw_circle)
# cv2.createButton("Window",back)

cv2.createButton("Back",Back,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Refresh",Refresh,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Next",Next,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("ShowTracked",ShowTracked,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Track",Track,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Img-Label",OpenImgLabel,None,cv2.QT_PUSH_BUTTON,1)
cv2.createButton("Save",Save,None,cv2.QT_PUSH_BUTTON,1)

# cv2.createButton("Dog house F1",Tick1,None,cv2.QT_CHECKBOX|cv2.QT_NEW_BUTTONBAR,0) # create a push button in a new row
# cv2.createButton("Window 1",Tick6,None,cv2.QT_CHECKBOX,1)
# cv2.createButton("LandingMark A 1",Tick10,None,cv2.QT_CHECKBOX,1)
# cv2.createButton("LandingMark B 1",Tick14,None,cv2.QT_CHECKBOX,1)
# cv2.createButton("LandingMark C 1",Tick18,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("LandingMark D 1",Tick22,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("Landing 1",Tick26,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("Light               ",Tick30,None,cv2.QT_CHECKBOX,0) #Light 1    
# cv2.createButton("Back 1",Tick33,None,cv2.QT_CHECKBOX,0)

# cv2.createButton("Dog house F2",Tick2,None,cv2.QT_CHECKBOX|cv2.QT_NEW_BUTTONBAR,1) # create a push button in a new row
# cv2.createButton("Window 2",Tick7,None,cv2.QT_CHECKBOX,1)
# cv2.createButton("LandingMark A 2",Tick11,None,cv2.QT_CHECKBOX,1)
# cv2.createButton("LandingMark B 2",Tick15,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("LandingMark C 2",Tick19,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("LandingMark D 2",Tick23,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("Landing 2",Tick27,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("Dog house B 1",Tick31,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("Back 2",Tick34,None,cv2.QT_CHECKBOX,0)

# cv2.createButton("Dog house F3",Tick3,None,cv2.QT_CHECKBOX|cv2.QT_NEW_BUTTONBAR,1) # create a push button in a new row
# cv2.createButton("Window 3",Tick8,None,cv2.QT_CHECKBOX,1)
# cv2.createButton("LandingMark A 3",Tick12,None,cv2.QT_CHECKBOX,1)
# cv2.createButton("LandingMark B 3",Tick16,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("LandingMark C 3",Tick20,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("LandingMark D 3",Tick24,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("Landing 3",Tick28,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("Dog house B 2",Tick32,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("    2 ",Tick,None,cv2.QT_CHECKBOX,0)

# cv2.createButton("Dog house F4",Tick4,None,cv2.QT_CHECKBOX|cv2.QT_NEW_BUTTONBAR,1) # create a push button in a new row
# cv2.createButton("Window 4",Tick9,None,cv2.QT_CHECKBOX,1)
# cv2.createButton("LandingMark A 4",Tick13,None,cv2.QT_CHECKBOX,1)
# cv2.createButton("LandingMark B 4",Tick17,None,cv2.QT_CHECKBOX,1)
# cv2.createButton("LandingMark C 4",Tick21,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("LandingMark D 4",Tick25,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("Landing 4",Tick29,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("     3       ",Tick,None,cv2.QT_CHECKBOX,0)
# cv2.createButton("   4  ",Tick,None,cv2.QT_CHECKBOX,0)

# cv2.createButton("Dog house F5",Tick5,None,cv2.QT_CHECKBOX|cv2.QT_NEW_BUTTONBAR,1) # create a push button in a new row




#createButton("button6",callbackButton2,NULL,QT_PUSH_BUTTON|QT_NEW_BUTTONBAR) # create a push button in a new row

# Create trackbar and associate a callback function / create trackbars Named Radius with the range of 150 and starting position of 5.
cv2.createTrackbar('Scale', 'Window', 0, 200, scaleIt) 
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


