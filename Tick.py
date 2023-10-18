import cv2
import numpy as np
from lib import *

TK = np.zeros(PW_label.shape[0])
PW = PW_label.copy()

def deletePw(TK, PW_label):
   global PW, x
   x = 2
   Del = []
   for i in range(TK.shape[0]):
      if TK[i] == 0:
         Del = np.append(Del, i).astype(int)
   PW = PW_label.copy()
   PW = np.delete(PW,Del,axis=0)



def click(*args,j=1): 
   i = j-1
   global TK
   if TK[i]:
      TK[i] = 0
      deletePw(TK, PW_label)

   else:
      TK[i] = 1
      deletePw(TK, PW_label)


def Tick1(*args): click(*args,1)
def Tick2(*args): click(*args,2)
def Tick3(*args): click(*args,3)
def Tick4(*args): click(*args,4)
def Tick5(*args): click(*args,5)
def Tick6(*args): click(*args,6)
def Tick7(*args): click(*args,7)
def Tick8(*args): click(*args,8)
def Tick9(*args): click(*args,9)
def Tick10(*args): click(*args,10)
def Tick11(*args): click(*args,11)
def Tick12(*args): click(*args,12)
def Tick13(*args): click(*args,13)
def Tick14(*args): click(*args,14)
def Tick15(*args): click(*args,15)
def Tick16(*args): click(*args,16)
def Tick17(*args): click(*args,17)
def Tick18(*args): click(*args,18)
def Tick19(*args): click(*args,19)
def Tick20(*args): click(*args,20)
def Tick21(*args): click(*args,21)
def Tick22(*args): click(*args,22)
def Tick23(*args): click(*args,23)
def Tick24(*args): click(*args,24)
def Tick25(*args): click(*args,25)
def Tick26(*args): click(*args,26)
def Tick27(*args): click(*args,27)
def Tick28(*args): click(*args,28)
def Tick29(*args): click(*args,29)
def Tick30(*args): click(*args,30)
def Tick31(*args): click(*args,31)
def Tick32(*args): click(*args,32)
def Tick33(*args): click(*args,33)
def Tick(*args): pass