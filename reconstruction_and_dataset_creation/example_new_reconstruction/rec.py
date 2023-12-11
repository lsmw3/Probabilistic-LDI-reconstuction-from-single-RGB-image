import argparse
import os
import numpy as np
import cv2
import bpy
import bmesh
from mathutils import Vector
from mathutils import Matrix
from mathutils.bvhtree import BVHTree
import bpy
import bmesh
from mathutils.bvhtree import BVHTree
import math
from mathutils import Vector
from mathutils import Matrix
import numpy
from mathutils import Euler
import mathutils
import math
import _cycles
import os
import sys
import random
import cv2
import string
import numpy as np


# Create a sphere mesh
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.005)
sphere_mesh = bpy.context.active_object.data

def CreateSphere(hit_location):
    global sphere_mesh
    # create a sphere at the hit location with size of 0.01
    sphere_object = bpy.data.objects.new("Sphere", sphere_mesh.copy())
    sphere_object.location = hit_location
    bpy.context.scene.collection.objects.link(sphere_object)
            

def ConstructFromLDI(all_hits, cam, cam_data, image_width, image_height, write_path=None):
     
    # Get the intrinsic parameters
    focal_length = cam_data.lens # 7.5 # 
    sensor_width = cam_data.sensor_width
    sensor_height = cam_data.sensor_height
    # fov_x = 2 * np.arctan(sensor_width / (2 * focal_length))
    # fov_y = 2 * np.arctan(sensor_height / (2 * focal_length))
    
    # Calculate the aspect ratio of the camera
    aspect_ratio = cam_data.sensor_fit == 'VERTICAL' and cam_data.sensor_height / cam_data.sensor_width or cam_data.sensor_width / cam_data.sensor_height

    # Calculate the image height based on the aspect ratio
    image_height2 = image_width / aspect_ratio

    # Calculate the scale factors to convert from pixels to blender units
    scale_x = sensor_width / image_width
    scale_y = sensor_height / image_height2
    
    # we have all_hits, now we need create objects for each pixel in the image
    
    point_cloud = []
    
    # For each pixel in the image
    for i in range(0, image_width, 1):
        for j in range(0, image_height, 1):
            print(i,j)
            for c in range(all_hits.shape[2]):

                if all_hits[j,i,c] == 0:
                    continue

                # Convert from pixel coordinates to blender units
                x = (i - image_width / 2) * scale_x
                y = ((image_height)/ 2 - j) * scale_y
                
                # x = (i - image_width / 2) * np.tan(fov_x / 2) / (image_width / 2)
                # y = (j - image_height / 2) * np.tan(fov_y / 2) / (image_height / 2)


                # Calculate the direction of the ray in camera space
                ray_direction_camera_space = Vector((x, y, -focal_length)).normalized()

                # Transform the direction to world space
                ray_direction = cam.matrix_world.to_quaternion() @ ray_direction_camera_space

                # Define the ray origin and direction
                ray_origin = cam.location

                
                hit_distance = -all_hits[j,i,c]
                actual_hit_distance = hit_distance / ray_direction_camera_space.z
                # hit distance is calculated in camera's local z axis -> hit_dist = -(cam.matrix_world.inverted() @ hit_loc)[2]
                # now calculate the location using the reverse of this formula
                #hit_location = ray_origin + ray_direction * hit_distance
                hit_location = ray_origin + ray_direction * actual_hit_distance
                
                point_cloud.append(hit_location)
                
                if write_path is None:
                    CreateSphere(hit_location)
    
    if write_path is not None:
        # save point cloud to a file as .obj
        with open(write_path, 'w') as f:
            for point in point_cloud:
                f.write(f"v {point.x} {point.y} {point.z}\n")
 


def GenerateLDI(loaded_objects, gbvh, cam, cam_data, image_width, image_height, total_max_hits, out_path = None):
        
    bvh, face_to_object_index = gbvh
    
    face_to_object_index = GetFaceToBinary(loaded_objects, face_to_object_index)
    
    # Get the intrinsic parameters
    focal_length = cam_data.lens # 7.5 # 
    sensor_width = cam_data.sensor_width
    sensor_height = cam_data.sensor_height
    # fov_x = 2 * np.arctan(sensor_width / (2 * focal_length))
    # fov_y = 2 * np.arctan(sensor_height / (2 * focal_length))
    
    # Calculate the aspect ratio of the camera
    aspect_ratio = cam_data.sensor_fit == 'VERTICAL' and cam_data.sensor_height / cam_data.sensor_width or cam_data.sensor_width / cam_data.sensor_height

    # Calculate the image height based on the aspect ratio
    image_height2 = image_width / aspect_ratio

    # Calculate the scale factors to convert from pixels to blender units
    scale_x = sensor_width / image_width
    scale_y = sensor_height / image_height2
    
    
    all_hits = np.zeros((image_height, image_width, total_max_hits), dtype=np.float32)

    # For each pixel in the image
    for i in range(image_width):
        for j in range(image_height):
            
            # Convert from pixel coordinates to blender units
            x = (i - image_width / 2) * scale_x
            y = ((image_height)/ 2 - j) * scale_y
            
            # x = (i - image_width / 2) * np.tan(fov_x / 2) / (image_width / 2)
            # y = (j - image_height / 2) * np.tan(fov_y / 2) / (image_height / 2)


            # Calculate the direction of the ray in camera space
            ray_direction_camera_space = Vector((x, y, -focal_length)).normalized()

            # Transform the direction to world space
            ray_direction = cam.matrix_world.to_quaternion() @ ray_direction_camera_space

            # Define the ray origin and direction
            ray_origin = cam.location

            #print("ray_origin", ray_origin)
            #print("ray_direction", ray_direction)

            # Cast the ray and collect all hits
            hit_index = 0
            while hit_index < total_max_hits:
                hit_data = bvh.ray_cast(ray_origin, ray_direction)
                #print(hit_data)
                if hit_data[0] is not None:
                    face_index = hit_data[2]
                    
                    
                    hit_loc = hit_data[0]
                    
                    hit_dist = -(cam.matrix_world.inverted() @ hit_loc)[2]
                    
                    
                    all_hits[j,i, hit_index] = hit_dist
                    
                    if face_to_object_index[face_index] == 0:
                        break
                    
                    ray_origin = hit_loc + ray_direction*0.00001
                    hit_index+=1
                else:
                    break

    if out_path is not None:
        # save all hits to a file
        np.save(out_path, all_hits)

    return all_hits

def GetBVH():
    
    face_to_object_index = []
    
    object_index = 0
    prev_face_count = 0
    
    bm = bmesh.new()
    for obj in bpy.context.scene.objects:
        # Check if the object is a mesh
        if obj.type == 'MESH':
            # Get the evaluated mesh (with modifiers applied)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            
            # Create a temporary bmesh to add the object's transformed mesh
            temp_bm = bmesh.new()
            temp_bm.from_mesh(mesh)
            
            # Apply the object's transformation to the temporary bmesh
            temp_bm.transform(obj.matrix_world)
            
            # Add the temporary bmesh to the main bmesh
            temp_bm.to_mesh(mesh)
            temp_bm.free()
            bm.from_mesh(mesh)
            
            
            face_list = list(bm.faces)
            
            for i in range(prev_face_count, len(face_list)):
                face_to_object_index.append(object_index)
                
            prev_face_count = len(face_list)
        
        object_index += 1
    
    bvh = BVHTree.FromBMesh(bm)

    return bvh, face_to_object_index




def GetFaceToBinary(loaded_objects, face_to_object_index):
    
        
    indices = []
    
    obj_names = []
        
    for obj in loaded_objects:
        
        # get get_all_cps
        custom_properties = obj.get_all_cps()
        # print("Custom Properties: ", custom_properties)
        
        blender_obj  = obj.blender_obj # blender object

        # get name
        name = blender_obj.name
        

        if "model_path" in custom_properties:
            obj_names.append(name)
        
        # # get euler angles
        # euler = blender_obj.rotation_euler
        # euler = np.array([euler.x, euler.y, euler.z])

        # # get translation
        # translation = blender_obj.location
        # translation = np.array([translation.x, translation.y, translation.z])

        # # get scale
        # scale = blender_obj.scale
        # scale = np.array([scale.x, scale.y, scale.z])


    
    object_index = 0
    for obj in bpy.context.scene.objects:
        
        if obj.name in obj_names:
            
            indices.append(object_index)
        
        object_index += 1
    
    
    for i in range(len(face_to_object_index)):
        
        if face_to_object_index[i] in indices:
            face_to_object_index[i] = 1
        else:
            face_to_object_index[i] = 0
            
    return face_to_object_index        

def NormalizeLDI(ldi):
    
    ldi_max = np.max(ldi)

    # remove zeros
    ldi2 = np.where(ldi <= 0, ldi_max, ldi)

    ldi_min = np.min(ldi2)

    normalized_ldi = (ldi) / (ldi_max - ldi_min)
    
    return normalized_ldi

def CleanLDI(ldi):

    for i in range(1, ldi.shape[2]):
        ldi[:,:,i] = np.maximum(ldi[:,:,i], ldi[:,:,i-1])

    return ldi

all_hits_path = r"C:\Users\CR\Desktop\Praktikum\new_reconstruction\out_sil\ldi.npy"
all_hits = np.load(all_hits_path)

all_hits = NormalizeLDI(all_hits)

# image width and height
image_width = 1024
image_height = 1024

# total_max_hits for ldi
total_max_hits = 10

# get camera named "Camera"
camera = bpy.data.objects["Camera"]

# set field of view to 50 degrees
camera.data.angle = np.deg2rad(50)

bpy.context.view_layer.update()

camera_data = camera.data

ConstructFromLDI(all_hits, camera, camera_data, image_width, image_height)


    



