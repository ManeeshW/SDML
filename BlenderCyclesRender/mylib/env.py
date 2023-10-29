import bpy
import mathutils
from mathutils import Vector
from mathutils import Matrix
import random
from random import randint 
import numpy as np
from .prob import *
from . import  *
from .misc import make_a_scene


def changeEnvironments(scene):
    world = bpy.data.worlds['World']
    world.use_nodes = True

    # remove default light    
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete(use_global=False)
    
    if LIGHT["FIXED_LIGHT"]:
        light_data = bpy.data.lights.new(name="SUN", type='SUN')
        light_data.energy = LIGHT["sun_energy"]
        light_data.angle = LIGHT["sun_shadow_diffuse"]*np.pi/180
        light_object = bpy.data.objects.new(name="SUN", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = (0, 0, 25)
        light_object.rotation_euler[0] = LIGHT["sun_rot0_deg"]*np.pi/180
        light_object.rotation_euler[2] = LIGHT["sun_rot1_deg"]*np.pi/180
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = LIGHT["atmosphere_energy"]
    else:
        # Create random lighting
        light_data = bpy.data.lights.new(name="SUN", type='SUN')
        alpha = abs(getRanPosVal_around_point(3,4,2000,0,0.2,0))
        light_data.energy = abs(getRanPosVal_around_point(alpha*3,alpha*8,1000,0,0.2,0))
        light_data.angle = (randint(2, 20)+randint(0, 100)/100)*np.pi/180 #np.pi * getRanPosVal_around_point(0,60,1000,0,0.05,0)/100
        light_object = bpy.data.objects.new(name="SUN", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = (0, 0, 25)
        light_object.rotation_euler[0] = getRanPosVal_around_point(0,75,1000,0,0.2,0)*np.pi/180
        light_object.rotation_euler[2] = getRanPosVal_around_point(0,180,1000,0,0.2,0)*np.pi/180
        
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = alpha
    
    
    # Create new light
#    alpha = chooseRanPosVal(4,1000,0.1,0.35,0.00001,False)
#    light_data = bpy.data.lights.new(name="SUN", type='SUN')
#    light_data.energy = getRanPosVal_around_point(1.5,0.7,1000,0,0.2,0)
#    light_data.angle = np.pi * getRanPosVal_around_point(0,1,1000,0,0.05,0)
#    light_object = bpy.data.objects.new(name="SUN", object_data=light_data)
#    bpy.context.collection.objects.link(light_object)
#    #light_object.location = (0, 100, 20)
#    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = alpha*2
    
    "Environment Texture sky"
    bpy.context.view_layer.objects.active = bpy.data.objects['Ship_with_helipad']
    env_img = bpy.context.scene.world.node_tree.nodes['Environment Texture'].image
    bpy.ops.image.open(filepath=Sky+"1.jpg", directory=Sky, files=[{"name":"1.jpg", "name":"1.jpg"}], relative_path=True, show_multiview=False)
    env_jpg = make_a_scene(1,"Environment Texture")
    env_img.filepath = Sky+env_jpg
    

    "Ship skin Texture" 
    bpy.context.view_layer.objects.active = bpy.data.objects['Ship_with_helipad']
    Ship_skin = bpy.data.materials['Shipskin'].node_tree.nodes['Environment Texture'].image
    Ship_skin_jpg = make_a_scene(3,"Ship skin Texture")
    bpy.ops.image.open(filepath=Shipskin+"1.jpg", directory=Shipskin, files=[{"name":"1.jpg", "name":"1.jpg"}], show_multiview=False)

    Ship_skin.filepath = Shipskin+Ship_skin_jpg
    
    bpy.data.materials["Shipskin"].node_tree.nodes["Environment Texture"].projection =  'EQUIRECTANGULAR' #'MIRROR_BALL' #

    
    "Ship Helipad Texture" 
    Helipad = bpy.data.materials['Helipad'].node_tree.nodes['Environment Texture'].image
    Helipad_jpg = make_a_scene(4,"landingpad")
    bpy.ops.image.open(filepath=Landingpad+"1.jpg", directory=Landingpad, files=[{"name":"1.jpg", "name":"1.jpg"}], show_multiview=False)
    
    Helipad.filepath = Landingpad+Helipad_jpg
    bpy.data.materials["Helipad"].node_tree.nodes["Environment Texture"].projection =  'EQUIRECTANGULAR'
    
    "Ship Helipad Marking Texture" 
    markings = bpy.data.materials['Helipad_Markings'].node_tree.nodes['Environment Texture'].image
    markings_jpg = make_a_scene(5,"markings")
    bpy.ops.image.open(filepath=Markings+"1.jpg", directory=Markings, files=[{"name":"1.jpg", "name":"1.jpg"}], show_multiview=False)

    markings.filepath = Markings+markings_jpg
    bpy.data.materials["Helipad_Markings"].node_tree.nodes["Environment Texture"].projection =  'EQUIRECTANGULAR'

    "Wave Texture" 
    bpy.context.view_layer.objects.active = bpy.data.objects['Wave']
    #wave_env = bpy.data.materials['Wave'].node_tree.nodes['Environment Texture'].image
    wave_env = bpy.data.objects['Wave'].active_material.node_tree.nodes['Image Texture'].image
    wave_env_jpg = make_a_scene(2,"Wave Texture")
    bpy.ops.image.open(filepath=Sea+"1.jpg", directory=Sea, files=[{"name":"1.jpg", "name":"1.jpg"}], show_multiview=False)

    "Wave2 Texture" 
    bpy.context.view_layer.objects.active = bpy.data.objects['Wave2']
    #wave_env = bpy.data.materials['Wave'].node_tree.nodes['Environment Texture'].image
    wave_env2 = bpy.data.objects['Wave2'].active_material.node_tree.nodes['Image Texture'].image

    wave_env.filepath = Sea+wave_env_jpg
    wave_env2.filepath = Sea+wave_env_jpg
    
    #bpy.data.objects['Wave'].modifiers['Ocean'].time = chooseRanPosVal(100,1000,-0.1,0.4,0.00001,False)
    #bpy.data.objects['Wave'].location = Vector((-10, -150, -chooseRanPosVal(1.7,1000,-0.1,0.4,0.00001,False)-1))
    bpy.data.objects['Wave'].location = Vector((-10, -150, -2.6))
    
    fileio.write(" >> ShipSkin: ")
    fileio.write(Ship_skin_jpg)
    fileio.write(" | Sky: ")
    fileio.write(env_jpg)
    fileio.write(" | Ocean: ")
    fileio.write(wave_env_jpg)
    fileio.write(" | Helipad: ")
    fileio.write(Helipad_jpg)
    fileio.write(" | Markings: ")
    fileio.write(markings_jpg)
    fileio.write("\n")
    #fileio.write("ShipSkin :"+Ship_skin_jpg+" | Sky :"+env_jpg+" | Ocean :"+wave_env_jpg " | Helipad :"+ Helipad_jpg+"| Markings :"+markings_jpg+"\n")
