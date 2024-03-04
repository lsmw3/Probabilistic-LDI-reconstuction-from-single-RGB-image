import blenderproc as bproc
import argparse
import os
import numpy as np
import cv2
import bpy
import bmesh
from mathutils import Vector
from mathutils import Matrix
from mathutils.bvhtree import BVHTree

parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument("output_dir", help="Path to where the data should be saved")
args = parser.parse_args()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                  transmission_bounces=200, transparent_max_bounces=200)

# load the front 3D objects
loaded_objects = bproc.loader.load_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)

    


# Init sampler for sampling locations inside the loaded front3D house
point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

poses = 0
tries = 0


def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed"]:
        if category_name in name.lower():
            return True
    return False


# filter some objects from the loaded objects, which are later used in calculating an interesting score
special_objects = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name())]

proximity_checks = {"min": 1.0, "avg": {"min": 2.5, "max": 3.5}, "no_background": True}
while tries < 50000 and poses < 1:
    # Sample point inside house
    height = np.random.uniform(1.4, 1.8)
    location = point_sampler.sample(height)
    # Sample rotation (fix around X and Y axis)
    rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

    # Check that obstacles are at least 1 meter away from the camera and have an average distance between 2.5 and 3.5
    # meters and make sure that no background is visible, finally make sure the view is interesting enough
    if bproc.camera.scene_coverage_score(cam2world_matrix, special_objects, special_objects_weight=10.0) > 0.8 \
            and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
        bproc.camera.add_camera_pose(cam2world_matrix)
        poses += 1
    tries += 1


def GenerateLDI(loaded_objects, gbvh, cam, cam_data, image_width, image_height, total_max_hits):
        
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

    there_is_furniture = False
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
                    else:
                        there_is_furniture = True
                    
                    ray_origin = hit_loc + ray_direction*0.00001
                    hit_index+=1
                else:
                    break

    return all_hits, there_is_furniture

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
    
    
if poses == 1:
    # if folder does not exist, create it
    if not os.path.exists(os.path.join(os.getcwd(), args.output_dir)):
        os.makedirs(os.path.join(os.getcwd(), args.output_dir))

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

    ldi_output_path = os.path.join(os.getcwd(), args.output_dir, "ldi.npy")

    all_hits, there_is_furniture = GenerateLDI(loaded_objects, GetBVH(), camera, camera_data, image_width, image_height, total_max_hits)

    if there_is_furniture == True:
        # save all hits to a file
        np.save(ldi_output_path, all_hits)


        bpy.context.scene.render.resolution_x = image_width
        bpy.context.scene.render.resolution_y = image_height
        bpy.context.scene.cycles.samples = 256
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPTIX'

        bpy.context.scene.render.filepath = os.path.join(os.getcwd(), args.output_dir, "rgb.png")
        bpy.ops.render.render(write_still = True)
            

        # for i in range(total_max_hits):
        #     img = all_hits[:,:,i]
        #     img = img - np.min(img)
        #     img = img / np.max(img)
        #     img = img * 255
        #     img = img.astype(np.uint8)
            
        #     save_path = os.path.join(os.getcwd(), args.output_dir, f"ldi{i}.png")
        #     cv2.imwrite(save_path, img)


            



