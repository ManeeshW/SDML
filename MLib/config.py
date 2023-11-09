from configparser import ConfigParser, ExtendedInterpolation
import os
import numpy as np

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config/config.ini")

MAIN_DIR = config.get('Auto_Dir', 'MAIN_DIR')
CamProp = config.get('Main_dir', 'CAMERA')
MAIN_DIR_Data = config.get('Auto_Dir', 'MAIN_DIR_Data')
MAIN_DIR_Labels = config.get('Auto_Dir', 'MAIN_DIR_Labels')
COMBINED_DATA = config.get('Auto_Dir', 'COMBINED_DATA')
TEST = config.get('Auto_Dir', 'TEST')
TEST_L = config.get('Auto_Dir', 'TEST_L')
REAL_IMG_DIR = config.get('Test', 'REAL_IMG_DIR')
INPUT_IMG = config.get('Auto_Dir', 'INPUT_IMG_DIR')

try:
   os.mkdir(MAIN_DIR)
except:
   pass

try:
   os.mkdir(MAIN_DIR_Data)
except:
   pass

try:
   os.mkdir(MAIN_DIR_Labels)
except:
   pass

try:
   os.mkdir(COMBINED_DATA)
except:
   pass

try:
   os.mkdir(TEST)
except:
   pass

try:
   os.mkdir(TEST_L)
except:
   pass

try:
   os.mkdir(INPUT_IMG)
except:
   pass


#outputF = config.get('Dir', 'OUT_DIR')
Test_no = config.getint('Test', 'TEST_NO')

K = np.loadtxt(CamProp+"Blender_camera/K.txt")
np.savetxt(TEST+"K.txt", K)
#outputFolder = outputF +"Test_{}/".format(Test_no)
#out_label = outputFolder+"label/"


out_annot_test = TEST
out_img = TEST_L+"imgs/"
out_bimg = TEST_L+"imgs_blender/"
out_comp = TEST_L+"imgs_comparison/"
out_keys = TEST_L+"keypoints/"
out_annot = TEST_L+"blender_annot/"
track_dir = TEST_L+"tracked/"
INPUT_IMG_DIR = config.get('Auto_Dir', 'INPUT_IMG_DIR')


try:
   os.mkdir(out_img)
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
   os.mkdir(out_annot)
except:
   pass

try:
   os.mkdir(track_dir)
except:
   pass

