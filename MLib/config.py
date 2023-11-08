from configparser import ConfigParser, ExtendedInterpolation
import os
import numpy as np

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config/config.ini")

Input_ImgDir = config.get('Dir', 'INPUT_DIR')
CamProp = config.get('Dir', 'CAMERA')
outputF = config.get('Dir', 'OUT_DIR')
Test_no = config.getint('Test', 'TEST_NO')
K = np.loadtxt(CamProp+"Blender_camera/K.txt")
outputFolder = outputF +"Test_{}/".format(Test_no)

out_img = outputFolder+"imgs/"
out_label = outputFolder+"label/"
out_annot = outputFolder+"Annotations/"
out_bimg = outputFolder+"bimg/"
out_comp = outputFolder+"comparison/"
out_keys = outputFolder+"keypoints/"
track_dir = outputFolder+"tracked/"

try:
   os.mkdir(outputFolder)
except:
   pass

try:
   os.mkdir(out_img)
except:
   pass

try:
   os.mkdir(out_label)
except:
   pass

try:
   os.mkdir(out_annot)
except:
   pass

try:
   os.mkdir(out_bimg)
except:
   pass

try:
   os.mkdir(out_comp)
except:
   pass

try:
   os.mkdir(out_keys)
except:
   pass

try:
   os.mkdir(track_dir)
except:
   pass

