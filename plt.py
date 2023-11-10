#!/usr/bin/env python3

from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')

# x = [0, 2, 0,0]
# y = [0, 2, 0,2]
# z = [0, 2, 2,0]

# scatter = ax.scatter(x,y,z,picker=True)

# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')


def onMouseMotion(event):
  print(event)

def chaos_onclick(event, ax, x, y, z):

    #print(dir(event.mouseevent))
    xx=event.mouseevent.x
    yy=event.mouseevent.y
    
    #magic from https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
    x2, y2, z2=proj3d.proj_transform(x[0], y[0], z[0], plt.gca().get_proj())
    x3, y3 = ax.transData.transform((x2, y2))
    #the distance
    d=np.sqrt ((x3 - xx)**2 + (y3 - yy)**2)
    print(xx, yy)
    print ("distance=",d)
    
    
    # #find the closest by searching for min distance.
    # #All glory to https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
    # imin=0
    # dmin=10000000
    # for i in range(len(x)):
    #   #magic from https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
    #   x2, y2, z2=proj3d.proj_transform(x[i], y[i], z[i], plt.gca().get_proj())
    #   x3, y3 = ax.transData.transform((x2, y2))
    #   #the distance magic from https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
    #   d=np.sqrt ((x3 - xx)**2 + (y3 - yy)**2)
    #   #We find the distance and also the index for the closest datapoint
    #   if d< dmin:
    #     dmin=d
    #     imin=i
        
    #   #print ("i=",i," d=",d, " imin=",imin, " dmin=",dmin)
    
    # # gives the incorrect data point index
    # point_index = int(event.ind)

    # print("Xfixed=",x[imin], " Yfixed=",y[imin], " Zfixed=",z[imin], " PointIdxFixed=", imin)
    # print("Xbroke=",x[point_index], " Ybroke=",y[point_index], " Zbroke=",z[point_index], " PointIdx=", point_index)

# fig.canvas.mpl_connect('pick_event', chaos_onclick)
# #fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion


# plt.show()