import json
import glob
import copy
import os
import pathlib

import numpy as np

def simplify_mesh(data):

    # Convert to a numpy array for easier manipulation
    vertices = np.array(data['xyz']).reshape(-1, 3)

    # Identify the constant axis
    constant_axis = np.where(vertices.min(axis=0) == vertices.max(axis=0))[0]
   
    # If there is no constant axis, return the original data
    if len(constant_axis) == 0:
        return data
    
    # Create the new data deep copy
    new_data = copy.deepcopy(data)

    constant_value = float(vertices[0, constant_axis])

    if constant_axis == 0:
        min_y = vertices.min(axis=0)[1]
        max_y = vertices.max(axis=0)[1]
        min_z = vertices.min(axis=0)[2]
        max_z = vertices.max(axis=0)[2]

        # calculate corner points and new 'xyz', 'normal', 'uv', and 'faces' values
        corner_points = [(constant_value, min_y, min_z), (constant_value, min_y, max_z), (constant_value, max_y, max_z), (constant_value, max_y, min_z)]
        new_data['xyz'] = [coord for cp in corner_points for coord in cp]
        new_data['normal'] = [1, 0, 0]*4
        new_data['uv'] = [0, 0,  1, 0,  1, 1,  0, 1]
        new_data['faces'] = [0, 1, 2,  2, 3, 0]

    elif constant_axis == 1:
        min_x = vertices.min(axis=0)[0]
        max_x = vertices.max(axis=0)[0]
        min_z = vertices.min(axis=0)[2]
        max_z = vertices.max(axis=0)[2]

        # calculate corner points and new 'xyz', 'normal', 'uv', and 'faces' values
        corner_points = [(min_x, constant_value, min_z), (min_x, constant_value, max_z), (max_x, constant_value, max_z), (max_x, constant_value, min_z)]
        new_data['xyz'] = [coord for cp in corner_points for coord in cp]
        new_data['normal'] = [0, 1, 0]*4
        new_data['uv'] = [0, 0,  1, 0,  1, 1,  0, 1]
        new_data['faces'] = [0, 1, 2,  2, 3, 0]

    elif constant_axis == 2:
        min_x = vertices.min(axis=0)[0]
        max_x = vertices.max(axis=0)[0]
        min_y = vertices.min(axis=0)[1]
        max_y = vertices.max(axis=0)[1]

        # calculate corner points and new 'xyz', 'normal', 'uv', and 'faces' values
        corner_points = [(min_x, min_y, constant_value), (min_x, max_y, constant_value), (max_x, max_y, constant_value), (max_x, min_y, constant_value)]
        new_data['xyz'] = [coord for cp in corner_points for coord in cp]
        new_data['normal'] = [0, 0, 1]*4
        new_data['uv'] = [0, 0,  1, 0,  1, 1,  0, 1]
        new_data['faces'] = [0, 1, 2,  2, 3, 0]

    return new_data

def SplitRooms(data):

    rooms = []

    rooms_refs = []
    rooms_instanceids = []

    for room in data["scene"]["room"]:
        
        room_refs = []
        room_instanceids = []
        children = room["children"]

        for child in children:
            room_refs.append(child["ref"])
            room_instanceids.append(child["instanceid"])

        rooms_refs.append(room_refs)
        rooms_instanceids.append(room_instanceids)

    for i in range(len(rooms_refs)):
            
        room = copy.deepcopy(data)

        room["scene"]["room"] = []
        room["scene"]["room"].append(data["scene"]["room"][i])

        room["mesh"] = []

        banned_meshes = ["WallOuter", "WallBottom", "WallTop", "Pocket", "SlabSide", "SlabBottom", "SlabTop", "Front", "Back", "Baseboard", "Door", "Window", "BayWindow", "Hole", "Beam"]

        # add meshes which appear in rooms_refs[i]
        for mesh in data["mesh"]:
            if mesh["uid"] in rooms_refs[i] and \
                mesh["type"] not in banned_meshes:

                new_mesh = simplify_mesh(mesh)

                room["mesh"].append(new_mesh)

        room["furniture"] = []

        # add furniture which appear in rooms_refs[i]
        for furniture in data["furniture"]:
            if furniture["uid"] in rooms_refs[i]:
                room["furniture"].append(furniture)

        rooms.append(room)

    return rooms



source_scenes_folder = "Data/3D-FRONT"
target_rooms_folder = "Data/3D-FRONT-ROOMS"

# read all json files in source_scenes_folder
json_files = glob.glob(f"{source_scenes_folder}/*.json")
json_files.sort()

saved_rooms = 0

# for each json file
for json_file in json_files:

    json_file_name_without_extension = pathlib.Path(json_file).stem
    
    # load json file
    with open(json_file) as f:
        data = json.load(f)

    try:

        rooms = SplitRooms(data)

        for i in range(len(rooms)):
            room = rooms[i]

            # if file does not exist
            if not os.path.exists(f"{target_rooms_folder}/{json_file_name_without_extension}_{i}.json"):
                # save room to a file
                with open(f"{target_rooms_folder}/{json_file_name_without_extension}_{i}.json", "w") as f:
                    json.dump(room, f)

                saved_rooms += 1

        print(f"Saved {saved_rooms} rooms")
    except:
        pass

# # load json file
# with open("new_reconstruction/x.json") as f:
#     data = json.load(f)

# rooms = SplitRooms(data)
# room = rooms[1]

# with open("deneme/x.json", "w") as f:
#     json.dump(room, f, indent=4)

