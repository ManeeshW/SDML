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
from configparser import ConfigParser

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

# def resetTimeLog():
#     #Reset Time.txt log 
#     T = []
#     np.savetxt("Time.txt",T)

# def resetHLog(sDir):
#     #Reset Time.txt log 
#     Hw = []
#     np.savetxt(sDir + "tmp/Hw.txt",Hw)

def renderRandomScenes(N, Dir, sDir):
    
    #sDir = config.get('Output', 'Sub_dir')
    ext = '.jpg'
    # Dir = "/home/maneesh/Desktop/Syn_Data_Genarate_from_ShipCAD_model/"
    
    Script = Dir + 'BlenderCyclesRender/ShipBlenderCycles.py'
    GPU_fp = Dir + 'BlenderCyclesRender/GPU.py'
    # renderer = "CYCLES"
    
    #Dir = "/home/maneesh/Desktop/Syn_Data_Genarate_from_ShipCAD_model/CyclesRenderOutput/"
    imgdir = sDir + "Train/"
    #imgdirtr = rDir + "offline_saved/Train/"
    N_img = num_of_img_files_in_folder(imgdir)
    if N_img == 0:
        print("Start generating images for the first time because there is no images in the training dataset")
    if N_img > 0:
        print(N_img, " images already in the training dataset and keep adding images")

    Time = []
    start_time = time.time()
    for scene_i in tqdm(range(1,N+1)):
        scene_i = scene_i + N_img
        
        fileio = open(sDir + "DescriptionLog.txt", "a")  # write
        fileio.write("{:06d}".format(scene_i))
        fileio.close()
        
        subscript = sDir
        # Render synthetic images using blender cycles
        ### !blender -P $enableGPU -noaudio -b $filename -P $code --python-use-system-env -E 'CYCLES' 1> nul
        call = blender + " -P "+ GPU_fp + " -b "+ blender_fp + " -P "+ Script + " --python-use-system-env  -E 'CYCLES' 1> nul -- -sub " + subscript 
        os.system(call) 
        shipImg = cv2.imread(sDir + 'tmp/{:06d}.png'.format(1))
        cv2.imwrite(sDir +'Train/{:06d}'.format(scene_i) + ext ,shipImg) 

        Maskon = config.getboolean('Mask', 'Maskon')
        if Maskon:
            Tdir = "TrainMask/{:06d}.png".format(scene_i) 
            cv2topilconverter(sDir,Tdir)


        Hw = np.loadtxt(sDir + "tmp/Hw.txt")
        Hc = np.loadtxt(sDir + "tmp/Hc.txt")
        # K = np.loadtxt(rDir + "offline_saved/K.txt")
        hw_save = np.reshape(Hw[0:3,0:4], 12)
        hw_save = np.expand_dims( hw_save, axis=0)
        hc_save = np.reshape(Hc[0:3,0:4], 12)
        hc_save = np.expand_dims( hc_save, axis=0)
        #tau = HomogMatrix2twist(Hw)[np.newaxis]

        '''
        #3d world cordinates
        Pw = np.array([[0,0,0,1],[0,0,1,1],[0,0,0,1],[0,1,0,1],[0,0,0,1],[1,0,0,1],[0,-3,0,1],[-3.65,-3,0,1],[3.65,-3,0,1]])

        #Perspective projection matrix M
        M = K @ Hc[0:3,0:4]

        #perspective projection keypoints (image coordinates)
        Pc = M @ Pw.T
        pc = Pc[0:2,:]/Pc[2,:]
        pc[0,:] = 640-pc[0:1]
        uv_save = np.reshape(pc, pc.shape[1]*2)
        uv_save = np.expand_dims( uv_save, axis=0)
        '''

        if N_img == 0:
            if scene_i == 1:
                np.savetxt(sDir + "Hw.txt",hw_save)
                np.savetxt(sDir + "Hc.txt",hc_save)
                # np.savetxt(sDir + "Tau.txt",tau)
                # np.savetxt(sDir + "UV.txt",uv_save) 
                Hw_save = hw_save
                Hc_save = hc_save
                # Tau = tau
                # UV_save = uv_save
                
            else:
                
                Hw_save = np.concatenate((Hw_save,hw_save),axis = 0)
                Hc_save = np.concatenate((Hc_save,hc_save),axis = 0)
                # Tau = np.concatenate((Tau,tau),axis = 0)
                # UV_save = np.concatenate((UV_save,uv_save),axis = 0)
                
                np.savetxt(sDir + "Hw.txt",Hw_save)
                np.savetxt(sDir + "Hc.txt",Hc_save)
                # np.savetxt(rDir + "offline_saved/Tau.txt",Tau)
                # np.savetxt(rDir + "offline_saved/UV.txt",UV_save)
        if N_img > 0:
            if scene_i == 1+N_img:
                Hw_save = np.loadtxt(sDir + "Hw.txt") 
                Hc_save = np.loadtxt(sDir + "Hc.txt")   
                # Tau = np.loadtxt(rDir + "offline_saved/Tau.txt") 
                # UV_save = np.loadtxt(rDir + "offline_saved/UV.txt")   
            
            Hw_save = np.concatenate((Hw_save,hw_save),axis = 0)
            Hc_save = np.concatenate((Hc_save,hc_save),axis = 0)
            # Tau = np.concatenate((Tau,tau),axis = 0)
            # UV_save = np.concatenate((UV_save,uv_save),axis = 0)
            
            np.savetxt(sDir + "Hw.txt",Hw_save)
            np.savetxt(sDir + "Hc.txt",Hc_save)
            # np.savetxt(sDir + "Tau.txt",Tau)
            # np.savetxt(sDir + "UV.txt",UV_save)
        end_time = time.time()
        dt = end_time - start_time
        Time = np.append(Time,dt)
        start_time = end_time

    N_img = num_of_img_files_in_folder(imgdir)
    print("Total ", N_img, " images in the training dataset folders")
    
    np.savetxt("Time.txt",Time)
    return Time

def timeProp(N, T, disp = True):
    print("Avg, loop time : ", sum(T)/N, "  (", int(3600/(sum(T)/N)), " images/hour )")
    if disp:
        for scene_i in range(1,N+1):
            print("Scene No ", scene_i, " : ", T[scene_i-1] )
    return 


parser = argparse.ArgumentParser(description="Synthetic image generation")
parser.add_argument('-N', type=int, default=2,
                        help='Number of images want to be generated')
parser.add_argument('--ss', type=str, default="",
                        help='Number of images want to be generated')
args = parser.parse_args()
N = args.N # Render N number of images online 
subscript = args.ss
if not subscript == "":
    subscript = "_" + subscript + "/"
    sDir = config.get('Output', 'Sub_dir')[:-1] + subscript
else:
    sDir = config.get('Output', 'Sub_dir')
start_time = time.time()
#Idir = Dir + "CyclesRenderOutput/"

#Create Dataset description log file
#sDir = config.get('Output', 'Sub_dir')[:-1] + subscript

try:
    os.mkdir(config.get('Output', 'Dataset_dir'))
except:
    pass

try:
    os.mkdir(sDir)
    os.mkdir(sDir+'/Train/')
    os.mkdir(sDir+'/TrainMask/')
    os.mkdir(sDir+'/tmp/')
except:
    pass

try:
    fileio = open(sDir + "DescriptionLog.txt", "a")  # write mode
    fileio.write("******** Start Date and time :")
    fileio.write(datetime.isoformat(datetime.now()))
    fileio.write("*********************************\n")
    fileio.close()
    
except :
    fileio = open(sDir + "DescriptionLog.txt", "w")  # write mode
    fileio.write("################################ \n")
    fileio.write("File : Maneesha \n")
    fileio.write("Date and time :")
    fileio.write(datetime.isoformat(datetime.now()))
    fileio.write("\n")
    fileio.write("################################\n \n")
    fileio.close()

# resetTimeLog()
# resetHLog(sDir)
T = renderRandomScenes(N,Dir,sDir)

T = np.loadtxt('Time.txt')
if N > 1:
    imgs = len(T)
else:
    imgs = 1
print("Total time taken for rendering ", imgs, " images : ", np.sum(T)/3600 )
TimeB = np.sum(T)/imgs
print("Avg.Time taken for rendering : ", TimeB)
print("Generated total synthetic images : ", num_of_img_files_in_folder(sDir + "Train/"))
