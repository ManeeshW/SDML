import os
import numpy as np

from configparser import ConfigParser
from .blenderArg import ArgumentParserForBlender

parser = ArgumentParserForBlender()

parser.add_argument('-sub', type=str, default="",
                        help='Number of images want to be generated')
args = parser.parse_args()
OfflineDir = args.sub

# with open("Output.txt", "w") as text_file:
#     print(OfflineDir, file=text_file)
#os.chdir('../')
Dir = os.getcwd()+"/"

# instantiate
config = ConfigParser()

# parse existing file
config.read(Dir + 'config/DataGen.ini')

texture_dir = config.get('texture', 'textures_dir')
#texture_dir = os.getcwd()+"/textures/"
tdir = "/textures/"


HighFreq = texture_dir + "high_frequency/"
LowFreq = texture_dir + "low_frequency/"
Sea = texture_dir + "sea/"
Shipskin = texture_dir + "shipskin/"
Sky = texture_dir + "sky/"
Landingpad = texture_dir + "landingpad/"
Markings = texture_dir + "markings/"


RenDir = OfflineDir + "tmp/" #config.get('Output', 'tmp')
#OfflineDir = config.get('Output', 'Sub_dir')
tracker = "Tracker/"

fileio = open(OfflineDir + "DescriptionLog.txt", "a")

TRACKER =True
FIXED_CAM =False
MASKON= config.getboolean('Mask', 'Maskon')
Camera_Location = np.array([0,-5,3])
Camera_Focusing_point =  np.array([0,0,1])

#33, 39, 53, 202
TEXTURE = {
"FIXED_TEXTURE" : True,
          "sky" :     89, 
          "sea" :     246,
     "shipskin" :     10,
   "landingpad" :     1,
     "markings" :     55
}

LIGHT = {
  "FIXED_LIGHT"       : True,
  "atmosphere_energy" :     1,
   "sun_energy"       :     10,
   "sun_shadow_diffuse" :     2,
     "sun_rot0_deg"   :     20,
     "sun_rot1_deg"   :     0,
}

HUMAN = {
   "Movement" : False
}

PELICAN = {
   "Movement" : False
}

