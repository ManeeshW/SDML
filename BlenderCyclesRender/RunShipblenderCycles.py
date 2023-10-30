import os
import cv2
import numpy as np
from PIL import Image , ImageOps
import time
from tqdm import tqdm
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt
#from IPython.display import display
from matplotlib.pyplot import imshow
from matplotlib.animation import FuncAnimation, PillowWriter
import argparse
from datetime import datetime
#from mylib import Dir
from configparser import ConfigParser, ExtendedInterpolation

# instantiate
config = ConfigParser()

#Dir = "/home/maneesh/Desktop/Syn_Data_Genarate_from_ShipCAD_model/"
Dir = os.getcwd()+"/"

# parse existing file
config.read(Dir+'config/DataGen.ini')

blender_fp = config.get('Blender', 'models_dir') #Dir + 'Blender/ShipBlenderCycles.blend'
code = Dir + 'BlenderCyclesRender/ShipBlenderCycles.py'
blender = config.get('Blender', 'blender')
enableGPU = Dir + 'BlenderCyclesRender/GPU.py'
renderer = "CYCLES"

#imgdir = config.get('Output', 'Img_dir') #Dir + "CyclesRenderOutput/offline_saved/Train"
Idir = "/home/maneesh/Desktop/Syn_Data_Genarate_from_ShipCAD_model/CyclesRenderOutput/"

def pltShowImg(image):
    fig = plt.figure(dpi=100)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([]) 
    plt.axis('off')
    plt.gcf().canvas.draw()
    
def cv2topilconverter(Idir,Tdir):
    dir1 = Idir + "tmp/000002mask1.png" 
    dir2 = Idir + "tmp/000003mask2.png"  
    dir3 = Idir + "tmp/000004mask3.png"
    Img1 = cv2.imread(dir1,cv2.IMREAD_UNCHANGED)
    ret,Img1 = cv2.threshold(Img1[:,:,3],120,128,cv2.THRESH_BINARY_INV) #THRESH_BINARY
    Img1 = cv2.cvtColor(Img1, cv2.COLOR_GRAY2RGB)
    ret,Img1[:,:,0] = cv2.threshold(Img1[:,:,0],120,192,cv2.THRESH_BINARY)
    #pltShowImg(Img1)

    Img2 = cv2.imread(dir2,cv2.IMREAD_UNCHANGED)
    ret,Img2 = cv2.threshold(Img2[:,:,3],120,128,cv2.THRESH_BINARY) #THRESH_BINARY
    Img2 = cv2.cvtColor(Img2, cv2.COLOR_GRAY2RGB)
    Img2[:,:,1]=0
    ret,Img2[:,:,2] = cv2.threshold(Img2[:,:,2],120,64,cv2.THRESH_BINARY)
    #pltShowImg(Img2)

    Img3 = cv2.imread(dir3,cv2.IMREAD_UNCHANGED)
    ret,Img3 = cv2.threshold(Img3[:,:,3],120,128,cv2.THRESH_BINARY_INV) #THRESH_BINARY
    Img3 = cv2.cvtColor(Img3, cv2.COLOR_GRAY2RGB)
    ret,Img3[:,:,1] = cv2.threshold(Img3[:,:,1],120,64,cv2.THRESH_BINARY)
    Img3[:,:,2]=0
    #pltShowImg(Img3)

    img = Img1+Img2+Img3
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #pltShowImg(img)

    # Existing palette as nested list
    palette = [
        [192, 128, 128],
        [64, 0, 128],
        [0, 64, 128],
    ]
    h, w = img.shape[:2]

    # Generate grayscale output image with replaced values
    img_pal = np.zeros((h, w), np.uint8)
    for i_p, p in enumerate(palette):
        img_pal[np.all(img == p, axis=2)] = i_p
    
    img_pil = Image.fromarray(img_pal)

    # Convert to mode 'P', and apply palette as flat list
    img_pil = img_pil.convert('P')
    palette = [value for color in palette for value in color]
    img_pil.putpalette(palette)

    # Save indexed image for comparison
    img_pil.save(Idir + Tdir)

def showimg(image1,image2,scene_i):
    ch = len(np.shape(image1))    
    fig, (ax1, ax2) = plt.subplots(ncols=2,dpi=250)
    
    if ch==3:
        #plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        #ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        #ax2.imshow(image2)
        ax2.imshow(image2,cmap='gray')
        #print(image2.shape)
    elif ch==2:
        ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        #ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        ax2.imshow(image2,cmap='gray')
        #plt.imshow(image, cmap='hot')
    else:
        print('Unsuported image size')
        raise
    ax1.set_title('Scene No : {:06d}'.format(scene_i),fontsize=4)
    #ax1.title("Scene No : ", scene_i)
    plt.xticks([]), plt.yticks([]) 
    ax1.axis('off')
    ax2.axis('off')
    plt.gcf().canvas.draw()

def num_of_img_files_in_folder(idir):
    list = os.listdir(idir) # dir is your directory path
    number_files = len(list)
    return number_files

def renderRandomScenes(N, img_total, Dir, sDir, bimg,  tmp, Track):
    
    #sDir = config.get('Output', 'Sub_dir')
    ext = '.jpg'
    # Dir = "/home/maneesh/Desktop/Syn_Data_Genarate_from_ShipCAD_model/"
    
    Script = Dir + 'BlenderCyclesRender/ShipBlenderCycles.py'
    GPU_fp = Dir + 'BlenderCyclesRender/GPU.py'
    # renderer = "CYCLES"
    
    # Render synthetic images using blender cycles
    ### !blender -P $enableGPU -noaudio -b $filename -P $code --python-use-system-env -E 'CYCLES' 1> nul
    call = blender + " -P "+ GPU_fp + " -b "+ blender_fp + " -P "+ Script + " --python-use-system-env  -E 'CYCLES' 1> nul -- -tmp " + tmp + " -nImg {}".format(N) + " -track " + Track
    os.system(call) 
    shipImg = cv2.imread(outputFolder + 'tmp/{:06d}.png'.format(1))
    cv2.imwrite(bimg +'{:06d}'.format(N) + ext ,shipImg) 

    # Maskon = config.getboolean('Mask', 'Maskon')
    # if Maskon:
    #     Tdir = "TrainMask/{:06d}.png".format(N) 
    #     cv2topilconverter(sDir,Tdir)


    Hw = np.loadtxt(tmp + "Hw.txt")
    #Hc = np.loadtxt(tmp + "Hc.txt")

    try:
        Hw_all= np.loadtxt(sDir + "Hw_b.txt") 
    except:
        Hw_gt = np.loadtxt(sDir + "Hw_gt.txt")
        Hw_all = np.zeros_like(Hw_gt)
    #Hc_all = np.loadtxt(sDir + "Hc_b.txt")  

    Hw_all[N-1,:] = np.reshape(Hw[0:3,0:4], 12)
    #Hc_all[N-1,:] = np.reshape(Hc[0:3,0:4], 12)
    np.savetxt(sDir + "Hw_b.txt",Hw_all)
    #np.savetxt(sDir + "Hc_b.txt",Hc_all)

def timeProp(N, T, disp = True):
    print("Avg, loop time : ", sum(T)/N, "  (", int(3600/(sum(T)/N)), " images/hour )")
    if disp:
        for scene_i in range(1,N+1):
            print("Scene No ", scene_i, " : ", T[scene_i-1] )
    return 



config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(Dir+"config/config.ini")
outputF = config.get('Dir', 'OUT_DIR')
Test_no = config.getint('Test', 'TEST_NO')
outputFolder = outputF +"Test_{}/".format(Test_no)
out_bimg = outputFolder+"bimg/"
out_annot = outputFolder+"Annotations/"
tmp = outputFolder+'tmp/'

parser = argparse.ArgumentParser(description="Synthetic image generation")
parser.add_argument('-N', type=int, default=1,
                        help='Images Number')
args = parser.parse_args()
N = args.N # Render N number of images online 

Idir = Dir + "CyclesRenderOutput/"

try:
    os.mkdir(config.get('Output', 'Dataset_dir'))
except:
    pass

img_total = 200

try:
    os.mkdir(tmp)
    Hw_all = np.zeros((img_total, 12))
    np.savetxt(out_annot + "Hw_b.txt",Hw_all)
    #np.savetxt(out_annot + "Hc_b.txt",Hw_all)
except:
    pass

T = renderRandomScenes(N,img_total, Dir, out_annot, out_bimg, tmp, out_annot)
