#!/usr/bin/env python3

from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from MLib.ship import *


import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection = '3d')
ax.set_axis_off()
x = [-1, 0, 2, 2, 0]
y = [-1, 0, 2, -8, 2]
z = [-1, 0, 2, 0, 0]
ship3D(ax)
scatter = ax.scatter(x,y,z,picker=True)
zoom = 10
ax.set_xlim(-zoom,zoom)
ax.set_ylim(-zoom,zoom)
ax.set_zlim(-zoom,zoom)
ax.set_aspect('equal', adjustable='box')
ax.view_init(azim=0, elev=0)

def chaos_onclick(event):

    point_index = int(event.ind)
    print(point_index)

    #proj = ax.get_proj()
    #x_p, y_p, _ = proj3d.proj_transform(x[point_index], y[point_index], z[point_index], proj)
    #plt.annotate(str(point_index), xy=(x_p, y_p))
    
    print("X=",x[point_index], " Y=",y[point_index], " Z=",z[point_index], " PointIdx=", point_index)


fig.canvas.mpl_connect('pick_event', chaos_onclick)
plt.show()

