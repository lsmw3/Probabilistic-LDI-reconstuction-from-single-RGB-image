import os
import glob
import pathlib
import subprocess
import multiprocessing
import numpy as np
import shutil

def check_if_ok(room_path):

    ldi_path = os.path.join(room_path, "ldi.npy")
    rgb_path = os.path.join(room_path, "rgb.png")

    if not os.path.exists(ldi_path):
        return False
    
    if not os.path.exists(rgb_path):
        return False
    
    ldi = np.load(ldi_path)

    if np.sum(ldi) <= 0.1:
        return False

    return True

def process(part_nums, render_path, pre_command, room_part):
    
    for i in range(len(room_part)):
        room = room_part[i]

        # get name of room without extension
        room_name = pathlib.Path(room).stem

        if check_if_ok(f"{render_path}/{room_name}") == True:
            continue

        script_command = f"""blenderproc run script.py "Data/3D-FRONT-ROOMS/{room_name}.json" "Data/3D-FUTURE-model" "Data/3D-FRONT-texture" {render_path}/{room_name}"""

        try:
            # rerun with timeout = 1min
            subprocess.call(pre_command + script_command, shell=True, timeout=90)
        except:
            pass    
    
        if check_if_ok(f"{render_path}/{room_name}") == False:
            try:
                # delete folder with its files shutil
                shutil.rmtree(f"{render_path}/{room_name}")
            except:
                pass


        print(f"Process {part_nums} - Rendered {i+1}/{len(room_part)}")


def FilterRooms(start_room):

    rooms2 = []
    for room in rooms:
        if room >= start_room:
            rooms2.append(room)

    return rooms2

if __name__ == "__main__":

    part_nums = 2
    render_path = "renders"
    pre_command = ""
    pre_command += "cd \"C:\\Users\\CR\\Desktop\\Praktikum\" & "
    pre_command += "activate blenderdeneme & "

    # get names of all json files in Data/3D-FRONT Data\3D-FRONT-ROOMS
    rooms = glob.glob(os.path.join("Data", "3D-FRONT-ROOMS", "*.json"))
    rooms = list(set(rooms))
    rooms.sort()

    # split rooms into part_nums parts, each part will be rendered by a different process
    room_parts = [rooms[i::part_nums] for i in range(part_nums)]

    # create a process for each part
    processes = []

    for i in range(part_nums):
        p = multiprocessing.Process(target=process, args=(i, render_path, pre_command, room_parts[i]))
        processes.append(p)
        p.start()

    # wait for all processes to finish
    for p in processes:
        p.join()

    print("Finished rendering all rooms")
