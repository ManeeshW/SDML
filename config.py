from configparser import ConfigParser, ExtendedInterpolation
import os
import numpy as np

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")

Input_ImgDir = config.get('Dir', 'INPUT_DIR')
CamProp = config.get('Dir', 'CAMERA')
outputF = config.get('Dir', 'OUT_DIR')
Test_no = config.getint('Test', 'TEST_NO')
K = np.loadtxt(CamProp+"Blender_camera/K.txt")
outputFolder = outputF +"Test_{}/".format(Test_no)
out_img = outputFolder+"imgs/"
out_label = outputFolder+"label/"
out_annot = outputFolder+"Annotations/"

try:
   os.mkdir(outputFolder)
   os.mkdir(outputFolder+"imgs/")
   os.mkdir(outputFolder+"label/")
   os.mkdir(outputFolder+"Annotations/")
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

