import os
import numpy as np
from tqdm import tqdm
import shutil
from configparser import ConfigParser

# instantiate
config = ConfigParser()

#os.rename(dir2 + filename, dir2 + '{:d}'.format(n) + ".png")
#os.rename(Input_ImgDir + filename, Output_ImgDir + filename)

def bindDatasets(Input= os.getcwd()+"/Input_Dataset/", Output=os.getcwd()+"/Output_Dataset/"):
    files = os.listdir(Input)
    Output_ImgDir_Train = Output+"offline_saved/Train/"
    Output_ImgDir_TrainMask = Output+"offline_saved/TrainMask/"
    M=len(files)
    Hw = np.array([])
    Hc = np.array([])
    #Tau = np.array([])
    #UV = np.array([])

    r = 0
    for m in tqdm(range(1, M+1)):
        Input_ImgDir_Train = Input+"offline_saved_{:d}/Train/".format(m)
        Input_ImgDir_TrainMask = Input+"offline_saved_{:d}/TrainMask/".format(m)
        K = np.loadtxt(Input+"offline_saved_{:d}/K.txt".format(m)) 
        Hw_m = np.loadtxt(Input+"offline_saved_{:d}/Hw.txt".format(m)) 
        Hc_m = np.loadtxt(Input+"offline_saved_{:d}/Hc.txt".format(m))
        #Tau_m = np.loadtxt(Input+"offline_saved_{:d}/Tau.txt".format(m))
        #UV_m = np.loadtxt(Input+"offline_saved_{:d}/UV.txt".format(m))
        
        images = os.listdir(Input_ImgDir_Train)
        N=len(images)
        for n in tqdm(range(1, N+1)):
            in_filename_Train  = "{:06d}.jpg".format(n)
            out_filename_Train  = "{:06d}.jpg".format(r+n)
            shutil.copy2(Input_ImgDir_Train + in_filename_Train, Output_ImgDir_Train + out_filename_Train)
            
            in_filename_TrainMask  = "{:06d}.png".format(n)
            out_filename_TrainMask  = "{:06d}.png".format(r+n)
            shutil.copy2(Input_ImgDir_TrainMask + in_filename_TrainMask, Output_ImgDir_TrainMask + out_filename_TrainMask)   
        r += N
        
        if m == 1:
            Hw = Hw_m
            Hc = Hc_m
            #Tau = Tau_m
            #UV = UV_m
        else:
            Hw = np.concatenate((Hw, Hw_m), axis = 0)
            Hc = np.concatenate((Hc, Hc_m), axis = 0)
            #Tau = np.concatenate((Tau, Tau_m), axis = 0)
            #UV = np.concatenate((UV, UV_m), axis = 0)
            
    np.savetxt(Output+"offline_saved/K.txt",K)        
    np.savetxt(Output+"offline_saved/Hw.txt",Hw)
    np.savetxt(Output+"offline_saved/Hc.txt",Hc)
    # np.savetxt(Output+"offline_saved/Tau.txt",Tau)
    # np.savetxt(Output+"offline_saved/UV.txt",UV)


def renamefile(imgNo = [1,1], intag="tmp_", outtag="tmp_", Output=os.getcwd()+"/Output_Dataset/"):

    Output_ImgDir_Train = Output+"offline_saved/Train/"
    Output_ImgDir_TrainMask = Output+"offline_saved/TrainMask/"
    
    in_filename_Train  = "{:06d}.jpg".format(imgNo[0])
    out_filename_Train  = "{:06d}.jpg".format(imgNo[1])
    in_filename_TrainMask  = "{:06d}.png".format(imgNo[0])
    out_filename_TrainMask  = "{:06d}.png".format(imgNo[1])
    
    os.rename(Output_ImgDir_Train + intag + in_filename_Train, Output_ImgDir_Train + outtag + out_filename_Train)
    os.rename(Output_ImgDir_TrainMask + intag + in_filename_TrainMask, Output_ImgDir_TrainMask + outtag + out_filename_TrainMask)
        
def renamefiles(intag="tmp_", outtag="tmp_", Output=os.getcwd()+"/Output_Dataset/", shuffleidx=[]):
    Output_ImgDir_Train = Output+"offline_saved/Train/"
    Output_ImgDir_TrainMask = Output+"offline_saved/TrainMask/"
    images = os.listdir(Output_ImgDir_Train)
    N=len(images)
    np.savetxt(Output+"/offline_saved/Total_Images_{}.txt".format(N), np.array([N]), fmt='%s')
    print("N ",N)
    for n in tqdm(range(1, N+1)):
        renamefile([n, n], intag, outtag, Output)
        
    if len(shuffleidx) == 0:
        print("Not shuffled")
    else:
        intag="tmp_"
        outtag=""
        for i in tqdm(range(N)):
            renamefile([shuffleidx[i],i+1], intag, outtag, Output)
            
def shuffleDataset(Output=os.getcwd()+"/Output_Dataset/"):
    Output_ImgDir_Train = Output+"offline_saved/Train/"
    images = os.listdir(Output_ImgDir_Train)
    l = len(images)
    idx = np.arange(1,l+1)
    print(idx)
    np.random.shuffle(idx)
    print(idx)
    
    Hw = np.loadtxt(Output+"/offline_saved/Hw.txt") 
    Hc = np.loadtxt(Output+"/offline_saved/Hc.txt")
    #Tau = np.loadtxt(Output+"/offline_saved/Tau.txt")
    #UV = np.loadtxt(Output+"/offline_saved/UV.txt")
    
    Hw = Hw[idx-1]
    Hc = Hc[idx-1]
    #Tau = Tau[idx-1]
    #UV = UV[idx-1]
    np.savetxt(Output+"/offline_saved/idx.txt", idx, fmt='%i')
    np.savetxt(Output+"/offline_saved/Hw.txt",Hw)
    np.savetxt(Output+"/offline_saved/Hc.txt",Hc)
    #np.savetxt(Output+"/offline_saved/_Tau.txt",Tau)
    #np.savetxt(Output+"/offline_saved/UV.txt",UV)
    
    renamefiles(intag="", outtag="tmp_", Output=Output, shuffleidx=idx) 

Dir = os.getcwd()+"/"
# parse existing file
config.read(Dir+'config/DataGen.ini')
output = config.get('Output', 'Merge_dir') 
input = config.get('Output', 'Dataset_dir') 
try:
    os.mkdir(output)
except:
    pass
try:
    os.mkdir(output+"/offline_saved/")
except:
    pass
try:
    os.mkdir(output+"/offline_saved/Train/")
    os.mkdir(output+"/offline_saved/TrainMask/")
except:
    pass

bindDatasets(Input=input, Output=output)  
shuffleDataset(Output=output)    


 

