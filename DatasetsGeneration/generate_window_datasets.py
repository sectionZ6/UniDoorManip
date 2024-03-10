'''
window dataset generation
'''
import os
import json
import torch
import numpy as np
import trimesh
import shutil
import argparse

class Generator:

    def __init__(self, args) -> None:
        self.args = args
    
    def mkdir(self, path):
        # 去除首位空格
        path = path.strip()
        # 去除尾部 \ 符号
        path = path.rstrip("\\")
        # 判断路径是否存在
        isExists = os.path.exists(path)
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(path)
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            return False

    def generate_datasets(self, door_path, handle_path, door_data, handle_data, mesh):
        # process data
        board_min_points, board_max_points = mesh.bounds
        name = handle_path.split('/')[-1]
        door_name = door_path.split('/')[-1]
        save_path = os.path.join(self.args.save_path,(door_name + name))
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        self.mkdir(save_path)
        save_texture_path = os.path.join(save_path, "texture_dae")
        self.mkdir(save_texture_path)
        urdf_save_path = os.path.join(save_path, 'mobility.urdf')
        for file in os.listdir(os.path.join(door_path, "texture_dae")):
            shutil.copy(os.path.join(door_path, "texture_dae", file), save_texture_path)
        
        # 计算scaler 重写bounding_box
        door_height_scaler = 1
        if door_name == "9969004" or door_name == "9969008" or door_name == "9969012" or door_name == "9969013":
            door_height_scaler = 0.5
            print("window_name:{}-scaler:{}".format(door_name, door_height_scaler))
        elif door_name == "9969017":
            door_height_scaler = 0.8
            print("window_name:{}-scaler:{}".format(door_name, door_height_scaler))

        link_bis = str(-board_min_points[0] * door_height_scaler) + ' ' + str(-board_max_points[2] * door_height_scaler) + ' 0'
        joint_bis = str(board_min_points[0] * door_height_scaler) + ' ' + str(board_max_points[2] * door_height_scaler) + ' 0'
        joint_bis_2 = str(board_max_points[0] * door_height_scaler  - board_min_points[0] * door_height_scaler - 0.03) + " " + str(-board_max_points[2] * door_height_scaler) + " " + str(board_max_points[1] * door_height_scaler)

        door_bounding_box = {}
        door_bounding_box["min"] = [door_height_scaler * item for item in door_data["min"]]
        door_bounding_box["max"] = [door_height_scaler * item for item in door_data["max"]]

        with open(os.path.join(save_path, "bounding_box.json"), 'w') as f:
            json.dump(door_bounding_box, f)
        
        handle_length_scaler = self.args.lever_handle_length / (-handle_data["handle"][0][2] + handle_data["handle"][1][2])
        handle_bounding_box = {}
        handle_bounding_box["handle_min"] = [handle_length_scaler * item for item in handle_data["handle"][0]]
        handle_bounding_box["handle_max"] = [handle_length_scaler * item for item in handle_data["handle"][1]]
        if "lock" in handle_data.keys():
            handle_bounding_box["lock_min"] = [handle_length_scaler * item for item in handle_data["lock"][0]]
            handle_bounding_box["lock_max"] = [handle_length_scaler * item for item in handle_data["lock"][1]]
        handle_bounding_box["goal_pos"] = [handle_length_scaler * item for item in torch.load(os.path.join(handle_path, "goal_pos.npy")).tolist()]
        # ipdb.set_trace()
        with open(os.path.join(save_path, "handle_bounding.json"), 'w') as f:
            json.dump(handle_bounding_box, f)
        # 拼一个sacle
        handle_scale = str(handle_length_scaler) + " " + str(handle_length_scaler) + " " + str(handle_length_scaler)
        window_scale = str(door_height_scaler) + " " + str(door_height_scaler) + " " + str(door_height_scaler)

        for file in os.listdir(handle_path):
            if not "bounding" in file and not "goal" in file:
                shutil.copy(os.path.join(handle_path, file), save_texture_path)
        with open(urdf_save_path, 'w') as f:
            # 写入urdf文件头部
            f.write('<?xml version="1.0" ?>\n')
            f.write('<robot name="window">\n')
            f.write('\t<link name="base"/>\n')

            # link-0
            f.write('\t<link name="link_0">\n')

            f.write('\t\t<visual name="out-frame">\n')
            f.write('\t\t\t<origin xyz="0 0 0"/>\n')
            f.write('\t\t\t<geometry>\n')
            f.write('\t\t\t\t<mesh filename="texture_dae/untitled.dae" scale="{}"/>\n'.format(window_scale))
            f.write('\t\t\t</geometry>\n')
            f.write('\t\t</visual>\n')

            f.write('\t\t<collision>\n')
            f.write('\t\t\t<origin xyz="0 0 0"/>\n')
            f.write('\t\t\t<geometry>\n')
            f.write('\t\t\t\t<mesh filename="texture_dae/untitled.dae" scale="{}"/>\n'.format(window_scale))
            f.write('\t\t\t</geometry>\n')
            f.write('\t\t</collision>\n')

            f.write('\t</link>\n')

            # joint-0
            f.write('\t<joint name="joint_0" type="fixed">\n')
            
            f.write('\t\t<origin rpy="1.570796326794897 0 -1.570796326794897" xyz="0 0 0"/>\n')
            f.write('\t\t<child link="link_0"/>\n')
            f.write('\t\t<parent link="base"/>\n')

            f.write('\t</joint>\n')

            # link-1 handle left board right

            f.write('\t<link name="link_1">\n')

            f.write('\t\t<visual name="surf-board">\n')
            f.write('\t\t\t<origin xyz="{}"/>\n'.format(link_bis))
            f.write('\t\t\t<geometry>\n')
            f.write('\t\t\t\t<mesh filename="texture_dae/untitled1.dae" scale="{}"/>\n'.format(window_scale))
            f.write('\t\t\t</geometry>\n')
            f.write('\t\t</visual>\n')

            f.write('\t\t<collision>\n')
            f.write('\t\t\t<origin xyz="{}"/>\n'.format(link_bis))
            f.write('\t\t\t<geometry>\n')
            f.write('\t\t\t\t<mesh filename="texture_dae/untitled1.dae" scale="{}"/>\n'.format(window_scale))
            f.write('\t\t\t</geometry>\n')
            f.write('\t\t</collision>\n')

            f.write('\t</link>\n')
            # joint-1
            f.write('\t<joint name="joint_1" type="revolute">\n')
            
            f.write('\t\t<origin xyz="{}"/>\n'.format(joint_bis))

            f.write('\t\t<axis xyz="0 -1 0"/>\n')

            f.write('\t\t<child link="link_1"/>\n')
            f.write('\t\t<parent link="link_0"/>\n')
            f.write('\t\t<limit lower="0" upper="1.5079644737231006"/>\n')
            
            f.write('\t</joint>\n')

            # link-2
            f.write('\t<link name="link_2">\n')
            if name + "-lock.dae" in os.listdir(handle_path):
                f.write('\t\t<visual name="handle">\n')
                f.write('\t\t\t<origin xyz="0 0 0"/>\n')
                f.write('\t\t\t<geometry>\n')
                f.write('\t\t\t\t<mesh filename="texture_dae/{}-handle.dae" scale="{}"/>\n'.format(name, handle_scale))
                f.write('\t\t\t</geometry>\n')
                f.write('\t\t</visual>\n')

                f.write('\t\t<collision>\n')
                f.write('\t\t\t<origin xyz="0 0 0"/>\n')
                f.write('\t\t\t<geometry>\n')
                f.write('\t\t\t\t<mesh filename="texture_dae/{}-handle.dae" scale="{}"/>\n'.format(name, handle_scale))
                f.write('\t\t\t</geometry>\n')
                f.write('\t\t</collision>\n')

                f.write('\t</link>\n')

                # joint-2
                f.write('\t<joint name="joint_2" type="revolute">\n')

                f.write('\t\t<origin xyz="0 0 0"/>\n')
                f.write('\t\t<child link="link_2"/>\n')
                f.write('\t\t<parent link="link_3"/>\n')
                f.write('\t\t<axis xyz="0 0 1"/>\n')
                f.write('\t\t<limit lower="0" upper="1.5079644737231006"/>\n')
                
                f.write('\t</joint>\n')
                # link-3
                f.write('\t<link name="link_3">\n')
                f.write('\t\t<visual name="lock">\n')
                f.write('\t\t\t<origin xyz="0 0 0"/>\n')
                f.write('\t\t\t<geometry>\n')
                f.write('\t\t\t\t<mesh filename="texture_dae/{}-lock.dae" scale="{}"/>\n'.format(name, handle_scale))
                f.write('\t\t\t</geometry>\n')
                f.write('\t\t</visual>\n')

                f.write('\t\t<collision>\n')
                f.write('\t\t\t<origin xyz="0 0 0"/>\n')
                f.write('\t\t\t<geometry>\n')
                f.write('\t\t\t\t<mesh filename="texture_dae/{}-lock.dae" scale="{}"/>\n'.format(name, handle_scale))
                f.write('\t\t\t</geometry>\n')
                f.write('\t\t</collision>\n')

                f.write('\t</link>\n')

                # joint-3
                f.write('\t<joint name="joint_3" type="fixed">\n')
                
                f.write('\t\t<origin xyz="{}"/>\n'.format(joint_bis_2))
                f.write('\t\t<child link="link_3"/>\n')
                f.write('\t\t<parent link="link_1"/>\n')
                f.write('\t</joint>\n')
            else:

                f.write('\t\t<visual name="handle">\n')
                f.write('\t\t\t<origin xyz="0 0 0"/>\n')
                f.write('\t\t\t<geometry>\n')
                f.write('\t\t\t\t<mesh filename="texture_dae/{}.dae" scale="{}"/>\n'.format(name, handle_scale))
                f.write('\t\t\t</geometry>\n')
                f.write('\t\t</visual>\n')

                f.write('\t\t<collision>\n')
                f.write('\t\t\t<origin xyz="0 0 0"/>\n')
                f.write('\t\t\t<geometry>\n')
                f.write('\t\t\t\t<mesh filename="texture_dae/{}.dae" scale="{}"/>\n'.format(name, handle_scale))
                f.write('\t\t\t</geometry>\n')
                f.write('\t\t</collision>\n')

                f.write('\t</link>\n')
                # joint-2
                f.write('\t<joint name="joint_2" type="revolute">\n')
                f.write('\t\t<origin xyz="{}"/>\n'.format(joint_bis_2))
                f.write('\t\t<child link="link_2"/>\n')
                f.write('\t\t<parent link="link_1"/>\n')
                f.write('\t\t<axis xyz="0 0 1"/>\n')
                f.write('\t\t<limit lower="0" upper="1.5079644737231006"/>\n')
                
                f.write('\t</joint>\n')
            # 写入urdf文件尾部
            f.write('</robot>\n')

if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser(description='compose objects from safe and handle')
    parser.add_argument('--window_dae_path', type=str, default='./assets/3dware_window', help='Path to the safe dae file')
    parser.add_argument('--handle_path', type=str, default='./assets/window_handle', help='Path to the safe handle dae file')
    parser.add_argument('--save_path', type=str, default='./generated_datasets/window_datasets/', help='Path to the urdf file')
    parser.add_argument('--lever_handle_length', type=float, default=0.15, help='Path to the urdf file')
    parser.add_argument('--window_list', nargs='+', required=True, help="window body parts list")
    parser.add_argument('--window_handle_list', nargs='+', required=True, help="window handle parts list")

    args = parser.parse_args()

    generator = Generator(args)
    
    for window in args.window_list:
        # 拿到bounding box
        window_path = os.path.join(args.window_dae_path, window)
        mesh = trimesh.load(os.path.join(window_path, "texture_dae", "untitled1.dae"))
        with open(os.path.join(window_path, 'bounding_box.json'), 'r') as f:
            window_bounding_data = json.load(f)
        
        for handle in args.window_handle_list:
            handle_path = os.path.join(args.handle_path, handle)
            with open(os.path.join(handle_path,'bounding_box.json'), 'r') as f:
                handle_bounding_data = json.load(f)
            generator.generate_datasets(window_path, handle_path, window_bounding_data, handle_bounding_data, mesh)