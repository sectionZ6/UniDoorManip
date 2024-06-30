# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import os
import operator
from copy import deepcopy
import random
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym import gymapi
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples
import math
import numpy as np
import torch
import ipdb
from random import shuffle
import json


# Base class for RL tasks
class BaseEnv():

    def _cam_pose(self) :

        # cam_pos = gymapi.Vec3(1.5, 1.5, 1.2)
        # cam_target = gymapi.Vec3(0.0, 0.0, 0.7)
        cam_pos = gymapi.Vec3(1.0, 1.0, 1.5)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)

        return cam_pos, cam_target

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, enable_camera_sensors=False):
        
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.up_axis = 'z'
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless
        self.device = "cpu"
        self.seed = cfg["seed"]
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        
        self.figure = cfg["env"]["figure"]
        if self.figure:
            self.camera_handle_front_list = []
            self.camera_handle_right_list = []
            self.camera_handle_left_list = []
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]  # 192

        self.env_num_train = cfg["env"]["numTrain"]
        self.env_num_val = cfg["env"]["numVal"]
        self.num_envs = self.env_num_train + self.env_num_val
        self.asset_root = cfg["env"]["asset"]["assetRoot"]
        self.door_num_train = cfg["env"]["asset"]["DoorAssetNumTrain"]  # 1
        self.door_num_val = cfg["env"]["asset"]["DoorAssetNumVal"]  # 0
        self.door_num = self.door_num_train + self.door_num_val
        self.load_block = cfg["env"]["asset"]["load_block"]
        door_train_list_len = len(cfg["env"]["asset"]["trainAssets"][self.load_block])  # 1

        self.task = cfg["task"]["task_name"]
        self.action_type = cfg["task"]["action_type"]

        self.collectData = cfg["env"]["collectData"]
        self.collectALL = cfg["env"]["collect_all"]
        self.start_index = cfg["env"]["start_index"]
        self.collectPC = cfg["env"]["collectPC"]
        self.modelTest = cfg["env"]["model_test"]
        self.gapartnet_baseline = cfg["env"]["gapartnet_baseline"]
        self.collectForce = cfg["env"]["collectForce"]

        self.mobile = cfg["model"]["mobile"]
        self.exp_name = cfg['env']["env_name"]  # franka_drawer_state_open_handle
        print("Simulator: number of doors", self.door_num)
        print("Simulator: number of environments", self.num_envs)

        if self.door_num_train:
            assert (self.env_num_train % self.door_num_train == 0)
        if self.door_num_val:
            assert (self.env_num_val % self.door_num_val == 0)
        assert (self.env_num_train * self.door_num_val == self.env_num_val * self.door_num_train)
        assert (self.door_num_train <= door_train_list_len)  # the number of used length must less than real length
        # assert (self.door_num_val <= door_val_list_len)  # the number of used length must less than real length
        assert (self.num_envs % self.door_num == 0)  # each cabinet should have equal number envs
        self.env_per_door = self.num_envs // self.door_num

        # 两个自由度
        self.door_dof_lower_limits_tensor = torch.zeros((self.door_num, 2), device=self.device)
        self.door_dof_upper_limits_tensor = torch.zeros((self.door_num, 2), device=self.device)
        
        self.handle_door_min_tensor = torch.zeros((self.door_num, 3), device=self.device)
        self.handle_door_max_tensor = torch.zeros((self.door_num, 3), device=self.device)

        self.goal_pos_offset_tensor = torch.zeros((self.door_num, 3), device=self.device)
        # 环境指针数组 存放所有环境句柄
        self.env_ptr_list = []
        self.obj_loaded = False
        self.franka_loaded = False
        
        if self.cfg["env"]["enableForceSensors"]:
            self.hand_sensor_list = []
            self.num_sensor = 0
        
        self.camera_handle_list = []
        self.camera_tensor_list = []
        self.camera_view_list = []
        self.camera_vinv_list = []
        self.camera_proj_list = []
        self.env_origin_list = []

        self.d_5 = 0
        self.d_15 = 0
        self.d_30 = 0
        self.d_45 = 0

        self.gym = gymapi.acquire_gym()

        # double check!
        self.graphics_device_id = self.device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]

        self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        # self.original_props = {}
        # self.dr_randomizations = {}
        # self.first_randomization = True
        # self.actor_params_generator = None
        # self.extern_actor_params = {}
        # for env_id in range(self.num_envs):
        #     self.extern_actor_params[env_id] = None

        # self.last_step = -1
        # self.last_rand_step = -1

        # create envs, sim and viewer

        self.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        self.need_update = False

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            cam_pos, cam_tar = self._cam_pose()
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_tar)

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        
        self.sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
        self._create_ground_plane()
        self._place_agents(self.cfg["env"]["numTrain"]+self.cfg["env"]["numVal"], self.cfg["env"]["envSpacing"])

    def _create_ground_plane(self) :
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0.1
        plane_params.dynamic_friction = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _place_agents(self, env_num, spacing):

        print("Simulator: creating agents")
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.space_middle = torch.zeros((env_num, 3), device=self.device)
        self.space_range = torch.zeros((env_num, 3), device=self.device)
        self.space_middle[:, 0] = self.space_middle[:, 1] = 0
        self.space_middle[:, 2] = spacing/2
        self.space_range[:, 0] = self.space_range[:, 1] = spacing
        self.space_middle[:, 2] = spacing/2
        num_per_row = int(np.sqrt(env_num))
        

        for env_id in range(env_num):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.env_ptr_list.append(env_ptr)
            self._load_franka(env_ptr, env_id)
            self._load_obj(env_ptr, env_id)
            if self.collectData | self.modelTest:
                self.create_camera(env_ptr, env_id)
    
    def _load_franka(self, env_ptr, env_id):
        if self.cfg["env"]["enableForceSensors"]:
            hand_sensor_idx = 0

        if self.franka_loaded == False:

            self.franka_actor_list = []

            asset_root = self.asset_root
            if self.mobile:
                asset_file = "franka_description/robots/franka_panda_slider.urdf"
            else:
                asset_file = "franka_description/robots/franka_panda.urdf"
            self.gripper_length = 0.11
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
            asset_options.flip_visual_attachments = True
            asset_options.armature = 0.01
            self.franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            
            if self.cfg["env"]["enableForceSensors"]:
                hand_sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
                sensor_props = gymapi.ForceSensorProperties()
                sensor_props.enable_forward_dynamics_forces = True
                sensor_props.enable_constraint_solver_forces = True
                sensor_props.use_world_frame = False
                hand_idx = self.gym.find_asset_rigid_body_index(self.franka_asset, "panda_hand")
                hand_sensor_idx = self.gym.create_asset_force_sensor(self.franka_asset, hand_idx, hand_sensor_pose, sensor_props)

            self.franka_loaded = True
        # 拿到franka panda自由度的配置 封装成为tensor
        franka_dof_max_torque, franka_dof_lower_limits, franka_dof_upper_limits = self._get_dof_property(
            self.franka_asset)
        self.franka_dof_max_torque_tensor = torch.tensor(franka_dof_max_torque, device=self.device)
        
        self.franka_dof_lower_limits_tensor = torch.tensor(franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits_tensor = torch.tensor(franka_dof_upper_limits, device=self.device)

        dof_props = self.gym.get_asset_dof_properties(self.franka_asset)

        # use position drive for all dofs 设置控制模式
        if self.cfg["env"]["driveMode"] in ["pos", "ik"]:
            dof_props["driveMode"][:-2].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][:-3].fill(400.0)
            dof_props["damping"][:-3].fill(40.0)
            dof_props["stiffness"][-3].fill(400.0)
            dof_props["damping"][-3].fill(10.0)
        else:  # osc
            dof_props["driveMode"][:-2].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"][:-2].fill(0.0)
            dof_props["damping"][:-2].fill(0.0)
        # grippers
        # dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_EFFORT)
        # dof_props["stiffness"][-2:].fill(0.0)
        # dof_props["damping"][-2:].fill(0.0)
        dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"][-2:].fill(400.0)
        dof_props["damping"][-2:].fill(40.0)
        dof_props["friction"][-2:].fill(400.0)

        # root pose
        initial_franka_pose = self._franka_init_pose(self.mobile)

        # set start dof
        self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        default_dof_pos[:-2] = (franka_dof_lower_limits + franka_dof_upper_limits)[:-2] * 0.3
        # grippers open
        default_dof_pos[-2:] = franka_dof_upper_limits[-2:]
        franka_dof_state = np.zeros_like(franka_dof_max_torque, gymapi.DofState.dtype)
        franka_dof_state["pos"] = default_dof_pos

        franka_actor = self.gym.create_actor(
            env_ptr,
            self.franka_asset,
            initial_franka_pose,
            "franka",
            env_id,
            2,
            0)

        # rigid props
        franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_actor)
        for shape in franka_shape_props:
            shape.friction = 20
        # franka_shape_props[-1].friction = 2.15
        self.gym.set_actor_rigid_shape_properties(env_ptr, franka_actor, franka_shape_props)
        self.gym.set_actor_dof_properties(env_ptr, franka_actor, dof_props)
        self.gym.set_actor_dof_states(env_ptr, franka_actor, franka_dof_state, gymapi.STATE_ALL)
        self.franka_actor_list.append(franka_actor)

        if self.cfg["env"]["enableForceSensors"]:
            hand_sensor = self.gym.get_actor_force_sensor(env_ptr, franka_actor, hand_sensor_idx)
            self.hand_sensor_list.append(hand_sensor)
            self.num_sensor += self.gym.get_actor_force_sensor_count(env_ptr, franka_actor)

    def _load_obj(self, env_ptr, env_id):
        pass
    
    def _load_obj_asset(self):

        self.door_asset_name_list = []
        self.door_asset_list = []
        self.door_pose_list = []
        self.door_actor_list = []
        # self.door_pc = []

        train_len = len(self.cfg["env"]["asset"]["trainAssets"][self.load_block].items())
        # val_len = len(self.cfg["env"]["asset"]["testAssets"][self.load_block].items())
        train_len = min(train_len, self.door_num_train)
        # val_len = min(val_len, self.door_num_val)
        # total_len = train_len + val_len
        # used_len = min(total_len, self.door_num)

        random_asset = self.cfg["env"]["asset"]["randomAsset"]
        select_train_asset = [i for i in range(train_len)]
        # select_val_asset = [i for i in range(val_len)]
        if random_asset:  # if we need random asset list from given dataset, we shuffle the list to be read
            shuffle(select_train_asset)
            # shuffle(select_val_asset)
        select_train_asset = select_train_asset[:train_len]  # [0]
        # select_val_asset = select_val_asset[:val_len]  # []

        cur = 0

        asset_list = []

        # prepare the assets to be used
        for id, (name, val) in enumerate(self.cfg["env"]["asset"]["trainAssets"][self.load_block].items()):
            # id就是第几个资源文件 name是资源文件名 val是所有属性
            if id in select_train_asset:
                asset_list.append((id, (name, val)))
        # for id, (name, val) in enumerate(self.cfg["env"]["asset"]["testAssets"][self.load_block].items()):
        #     if id in select_val_asset:
        #         asset_list.append((id, (name, val)))
        dataset_path = self.cfg["env"]["asset"]["datasetPath"] #datset/door
        for id, (name, val) in asset_list:

            self.door_asset_name_list.append(val["name"])

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.collapse_fixed_joints = True  # 合并由固定关节连接的刚体
            asset_options.use_mesh_materials = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 2048

            door_asset = self.gym.load_asset(self.sim, self.asset_root, os.path.join(dataset_path, val["path"]), asset_options)
            self.door_asset_list.append(door_asset)
            # 记录下来加载数据集的大小
            with open(os.path.join(self.asset_root, dataset_path, val["bounding_box"]), "r") as f:
                door_bounding_box = json.load(f)
                min_dict = door_bounding_box["min"]
                max_dict = door_bounding_box["max"]

            dof_dict = self.gym.get_asset_dof_dict(door_asset)
            # print(list(dof_dict.keys()))
            self.door_dof_name = list(dof_dict.keys())[0]
            self.handle_dof_name = list(dof_dict.keys())[1] #handle的joint_name

            rig_dict = self.gym.get_asset_rigid_body_dict(door_asset)
            assert (len(rig_dict) == 3) #fixed的刚体已经合并
            self.handle_rig_name = list(rig_dict.keys())[2]
            self.door_rig_name = list(rig_dict.keys())[1]
            self.door_base_rig_name = list(rig_dict.keys())[0]
            assert (self.door_rig_name != "base")
            assert (self.handle_rig_name != "base")
            # 初始pose
            self.door_pose_list.append(self._obj_init_pose(min_dict, max_dict, val["name"][3:7]))

            max_torque, lower_limits, upper_limits = self._get_dof_property(door_asset)
            self.door_dof_lower_limits_tensor[cur, :] = torch.tensor(lower_limits, device=self.device)
            self.door_dof_upper_limits_tensor[cur, :] = torch.tensor(upper_limits, device=self.device)

            # with open(os.path.join(self.asset_root, dataset_path, str(name), "handle.yaml"), "r") as f:
            with open(os.path.join(self.asset_root, dataset_path, str(val["name"]), "handle_bounding.json"), "r") as f:
                handle_dict = json.load(f)
                self.handle_door_min_tensor[cur][0] = handle_dict["handle_min"][0]
                self.handle_door_min_tensor[cur][1] = handle_dict["handle_min"][2]
                self.handle_door_min_tensor[cur][2] = handle_dict["handle_max"][1]
                self.handle_door_max_tensor[cur][0] = handle_dict["handle_max"][0]
                self.handle_door_max_tensor[cur][1] = handle_dict["handle_max"][2]
                self.handle_door_max_tensor[cur][2] = - handle_dict["handle_min"][1]
                self.goal_pos_offset_tensor[cur][0] = handle_dict["goal_pos"][0]
                self.goal_pos_offset_tensor[cur][1] = handle_dict["goal_pos"][1]
                self.goal_pos_offset_tensor[cur][2] = handle_dict["goal_pos"][2]
            #点云部分
            # self.door_pc.append(
            #     torch.load(os.path.join(self.asset_root, dataset_path, name, "point_clouds", "pointcloud_tensor"),
            #                map_location=self.device))
            cur += 1

        # self.cabinet_pc = torch.stack(self.cabinet_pc).float()

    def init_all_tensor(self):
        # acquire tensors
        # actor state[num_actors,13]
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        # dofs state [num_dofs,2]
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        # rigid body state [num_rigid_bodies,13]
        self.rigid_body_tensor = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        # TODO:pos control also need jacobian
        if self.cfg["env"]["driveMode"] == "ik":  # inverse kinetic needs jacobian tensor, other drive mode don't need
            self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "franka"))

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)
        self.dof_state_tensor = self.dof_state_tensor.view(self.num_envs, -1, 2)
        self.rigid_body_tensor = self.rigid_body_tensor.view(self.num_envs, -1, 13)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.initial_dof_states = self.dof_state_tensor.clone()
        self.initial_root_states = self.root_tensor.clone()
        self.initial_rigid_body_states = self.rigid_body_tensor.clone()

        # precise slices of tensors
        env_ptr = self.env_ptr_list[0]
        franka1_actor = self.franka_actor_list[0]
        door1_actor = self.door_actor_list[0]

        if self.cfg["env"]["driveMode"] == "osc":  # inverse kinetic needs jacobian tensor, other drive mode don't need
            # 刚度
            joints = self.franka_num_dofs - 2
            self.kp = 200
            # 阻尼
            self.kv = 2 * math.sqrt(self.kp)
            self.kp_null = 10.
            self.kd_null = 2.0 * np.sqrt(self.kp_null)
            self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "franka"))
            self.mm_tensor = gymtorch.wrap_tensor(self.gym.acquire_mass_matrix_tensor(self.sim, "franka"))
            hand_index = self.gym.get_asset_rigid_body_dict(self.franka_asset)["panda_hand"]
            self.j_eef = self.jacobian_tensor[:, hand_index - 1, :, :joints]
            self.mm = self.mm_tensor[:, :joints, :joints]

        self.hand_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            franka1_actor,
            "panda_hand",
            gymapi.DOMAIN_ENV
        )
        self.hand_lfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            franka1_actor,
            "panda_leftfinger",
            gymapi.DOMAIN_ENV
        )
        self.hand_rfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            franka1_actor,
            "panda_rightfinger",
            gymapi.DOMAIN_ENV
        )
        # self.basex_rigid_body_index = self.gym.find_actor_rigid_body_index(
        #     env_ptr,
        #     franka1_actor,
        #     "basex",
        #     gymapi.DOMAIN_ENV
        # )
        # print("basex", self.basex_rigid_body_index)
        self.door_handle_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            door1_actor,
            self.handle_rig_name,
            gymapi.DOMAIN_ENV
        )
        self.door_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            door1_actor,
            self.door_rig_name,
            gymapi.DOMAIN_ENV
        )
        self.door_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr,
            door1_actor,
            self.door_base_rig_name,
            gymapi.DOMAIN_ENV
        )
        self.handle_dof_index = self.gym.find_actor_dof_index(
            env_ptr,
            door1_actor,
            self.handle_dof_name,
            gymapi.DOMAIN_ENV
        )
        self.door_dof_index = self.gym.find_actor_dof_index(
            env_ptr,
            door1_actor,
            self.door_dof_name,
            gymapi.DOMAIN_ENV
        )
        self.lfinger_dof_index = self.gym.find_actor_dof_index(
            env_ptr,
            franka1_actor,
            "panda_finger_joint1",
            gymapi.DOMAIN_ENV
        )
        self.rfinger_dof_index = self.gym.find_actor_dof_index(
            env_ptr,
            franka1_actor,
            "panda_finger_joint2",
            gymapi.DOMAIN_ENV
        )
        self.hand_rigid_body_tensor = self.rigid_body_tensor[:, self.hand_rigid_body_index, :]
        self.lfinger_rigid_body_tensor = self.rigid_body_tensor[:, self.hand_lfinger_rigid_body_index, :]
        self.rfinger_rigid_body_tensor = self.rigid_body_tensor[:, self.hand_rfinger_rigid_body_index, :]
        self.franka_dof_tensor = self.dof_state_tensor[:, :self.franka_num_dofs, :]
        #拿到转轴的position 和 velocity
        self.door_handle_dof_tensor = self.dof_state_tensor[:, self.handle_dof_index, :]
        #门轴的position 和 velocity
        self.door_dof_tensor = self.dof_state_tensor[:, self.door_dof_index, :]
        self.lfinger_dof_tensor = self.dof_state_tensor[:, self.lfinger_dof_index, :]
        self.rfinger_dof_tensor = self.dof_state_tensor[:, self.rfinger_dof_index, :]

        self.door_handle_dof_tensor_spec = self._detailed_view(self.door_handle_dof_tensor)  # [cabinet_num,each cabinet envs,2]
        #门把手刚体的state
        self.door_handle_rigid_body_tensor = self.rigid_body_tensor[:, self.door_handle_rigid_body_index, :]
        #门刚体的state
        self.door_rigid_body_tensor = self.rigid_body_tensor[:, self.door_rigid_body_index, :]
        self.franka_root_tensor = self.root_tensor[:, 0, :]  # [num_envs,13]
        self.door_root_tensor = self.root_tensor[:, 1, :]

        self.door_actor_dof_max_torque_tensor = torch.zeros((self.num_envs, 2), device=self.device)
        self.door_actor_dof_upper_limits_tensor = torch.zeros((self.num_envs, 2), device=self.device)
        self.door_actor_dof_lower_limits_tensor = torch.zeros((self.num_envs, 2), device=self.device)
        self.get_obj_dof_property_tensor()

        #self.cabinet_dof_target = self.initial_dof_states[:, self.cabinet_dof_index, 0]  # pos
        self.dof_dim = self.franka_num_dofs + 2  # 2DoF的物体，限制了门板
        self.pos_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.eff_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.stage = torch.zeros((self.num_envs,2), device=self.device)
        self.open_door_stage = torch.zeros((self.num_envs), device=self.device)
        self.open_door_flag = torch.zeros((self.num_envs, 3), device=self.device) == 1

        # params of randomization
        self.door_reset_position_noise = self.cfg["env"]["reset"]["door"]["resetPositionNoise"]
        self.door_reset_rotation_noise = self.cfg["env"]["reset"]["door"]["resetRotationNoise"]
        self.door_reset_dof_pos_interval = self.cfg["env"]["reset"]["door"]["resetDofPosRandomInterval"]
        self.door_reset_dof_vel_interval = self.cfg["env"]["reset"]["door"]["resetDofVelRandomInterval"]
        self.franka_reset_position_noise = self.cfg["env"]["reset"]["franka"]["resetPositionNoise"]
        self.franka_reset_rotation_noise = self.cfg["env"]["reset"]["franka"]["resetRotationNoise"]
        self.franka_reset_dof_pos_interval = self.cfg["env"]["reset"]["franka"]["resetDofPosRandomInterval"]
        self.franka_reset_dof_vel_interval = self.cfg["env"]["reset"]["franka"]["resetDofVelRandomInterval"]

        self.action_speed_scale = self.cfg["env"]["actionSpeedScale"]

        # flags for switching between training and evaluation mode
        self.train_mode = True
        # self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.adjust_hand_pose = self.get_adjust_hand_pose()
        self.PointDownSampleNum = self.cfg["env"]["PointDownSampleNum"]
        if self.collectData:
            self.pc_list = []
            self.hand_sensor_force_list = []
            self.impedance_force_list = []
            self.proprioception_info_list = []
            self.action_list = []
            self.finger_action_list = []
            self.rotate_martix = []
            self.proprioception_ik = []
            self.gt_state_list = []
            self.collect_data_path = self.cfg["env"]["collectDataPath"]
            self.num_episode = 0
        self.init_state_index = torch.zeros((self.num_envs, ), device=self.device)
        self.gripper = False
        if self.cfg["env"]["enableForceSensors"]:
            print("num_envs", self.num_envs)
            print("num_sensor: ", self.num_sensor)
            print("num_sensor: ", self.gym.get_sim_force_sensor_count(self.sim))
        
        if self.cfg["env"]["visualizePointcloud"] == True :
            import open3d as o3d
            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else :
            self.pointCloudVisualizer = None
        
        if self.figure:
            pass
            # self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(2, 2, 2))

    def get_obj_dof_property_tensor(self):
        env_id = 0
        for env, actor in zip(self.env_ptr_list, self.door_actor_list):
            dof_props = self.gym.get_actor_dof_properties(env, actor)
            dof_num = self.gym.get_actor_dof_count(env, actor)
            # print(dof_num)
            for i in range(dof_num):
                self.door_actor_dof_max_torque_tensor[env_id, i] = torch.tensor(dof_props['effort'][i],
                                                                                device=self.device)
                self.door_actor_dof_upper_limits_tensor[env_id, i] = torch.tensor(dof_props['upper'][i],
                                                                                  device=self.device)
                self.door_actor_dof_lower_limits_tensor[env_id, i] = torch.tensor(dof_props['lower'][i],
                                                                                  device=self.device)
            env_id += 1

    def _get_dof_property(self, asset):
        dof_props = self.gym.get_asset_dof_properties(asset)
        dof_num = self.gym.get_asset_dof_count(asset)
        dof_lower_limits = []
        dof_upper_limits = []
        dof_max_torque = []
        for i in range(dof_num) :
            dof_max_torque.append(dof_props['effort'][i])
            dof_lower_limits.append(dof_props['lower'][i])
            dof_upper_limits.append(dof_props['upper'][i])
        dof_max_torque = np.array(dof_max_torque)
        dof_lower_limits = np.array(dof_lower_limits)
        dof_upper_limits = np.array(dof_upper_limits)
        return dof_max_torque, dof_lower_limits, dof_upper_limits

    def step(self, actions):
        self.progress_buf += 1
        self._perform_actions(actions)
        # if self.collectData:
        #     if (self.progress_buf[0] > 80) & (self.progress_buf[0]%10==1):
        #         print("------------------------------{}".format(self.progress_buf[0]))
        #         self.collect_data()

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        if not self.headless:
            self.render()
        if self.cfg["env"]["enableCameraSensors"] == True:
            self.gym.step_graphics(self.sim)

        self._refresh_observation()

        self.open_door_flag = ((self.door_handle_dof_tensor[:, 0] >= 0.75 * (self.door_actor_dof_upper_limits_tensor[:, 1] - self.door_actor_dof_lower_limits_tensor[:, 1]))).unsqueeze(1).repeat_interleave(3, dim=-1) | self.open_door_flag
        self.cal_success()
        
        done = self.reset_buf.clone()
        # self._partial_reset(self.reset_buf)
        return self.obs_buf, done, None

    def get_states(self):
        return self.states_buf

    def render(self, sync_frame_time=False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def get_actor_params_info(self, dr_params, env):
        """Returns a flat array of actor params, their names and ranges."""
        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name+'_'+str(prop_idx)+'_'+attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0*float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name+'_'+str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs
    
    def _refresh_observation(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg["env"]["enableForceSensors"]:
            self.gym.refresh_dof_force_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
        if self.cfg["env"]["enableNetContact"]:
            self.gym.refresh_net_contact_force_tensor(self.sim)
        if self.cfg["env"]["enableCameraSensors"] == True:
            self.gym.render_all_camera_sensors(self.sim)
        if self.cfg["env"]["driveMode"] == "ik":
            self.gym.refresh_jacobian_tensors(self.sim)
        if self.cfg["env"]["driveMode"] == "osc":
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)

    def reset(self, to_reset="all"):

        self._partial_reset(to_reset)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if not self.headless:
            self.render()
        if self.cfg["env"]["enableCameraSensors"] == True:
            self.gym.step_graphics(self.sim)

        self._refresh_observation()
        return self.obs_buf, self.reset_buf, None

    def _partial_reset(self, to_reset="all"):

        """
        reset those need to be reseted
        """

        if to_reset == "all":
            to_reset = np.ones((self.num_envs,))
        reseted = False
        for env_id, reset in enumerate(to_reset):
            # is reset:
            if reset.item():
                # franka和doorknob初始pose和dof的reset
                reset_dof_states = self.initial_dof_states[env_id].clone()
                reset_root_states = self.initial_root_states[env_id].clone()
                # ipdb.set_trace()
                franka_reset_pos_tensor = reset_root_states[0, :3]
                franka_reset_rot_tensor = reset_root_states[0, 3:7]
                franka_reset_dof_pos_tensor = reset_dof_states[:self.franka_num_dofs, 0]
                franka_reset_dof_vel_tensor = reset_dof_states[:self.franka_num_dofs, 1]
                door_reset_pos_tensor = reset_root_states[1, :3]
                door_reset_rot_tensor = reset_root_states[1, 3:7]
                door_reset_dof_pos_tensor = reset_dof_states[self.franka_num_dofs:, 0]
                door_reset_dof_vel_tensor = reset_dof_states[self.franka_num_dofs:, 1]

                # self.goal_pos_tensor[env_id, :] = self.get_goal_pos(env_id)

                # 确定是哪个门，8897或者9393，这里需要注意一下
                door_type = env_id // self.env_per_door

                # 给reset的pos添加噪声扰动
                self.intervaledRandom_(franka_reset_pos_tensor[:2], self.franka_reset_position_noise) 
                self.intervaledRandom_(franka_reset_rot_tensor, self.franka_reset_rotation_noise)
                self.intervaledRandom_(franka_reset_dof_pos_tensor, self.franka_reset_dof_pos_interval,
                                       self.franka_dof_lower_limits_tensor, self.franka_dof_upper_limits_tensor)
                self.intervaledRandom_(franka_reset_dof_vel_tensor, self.franka_reset_dof_vel_interval)
                self.intervaledRandom_(door_reset_pos_tensor, self.door_reset_position_noise)
                self.intervaledRandom_(door_reset_rot_tensor, self.door_reset_rotation_noise)
                self.intervaledRandom_(door_reset_dof_pos_tensor, self.door_reset_dof_pos_interval,
                                       self.door_dof_lower_limits_tensor[door_type],
                                       self.door_dof_upper_limits_tensor[door_type])
                self.intervaledRandom_(door_reset_dof_vel_tensor, self.door_reset_dof_vel_interval)

                # ipdb.set_trace()
                self.dof_state_tensor[env_id].copy_(reset_dof_states)
                self.root_tensor[env_id].copy_(reset_root_states)
                reseted = True
                self.progress_buf[env_id] = 0
                self.reset_buf[env_id] = 0

        if reseted:
            self.gym.set_dof_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state_tensor)
            )
            self.gym.set_actor_root_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.root_tensor)
            )

    def draw_line_all(self, src, dst, color, cpu=False):
        if not cpu:
            line_vec = np.concatenate([src.cpu().numpy(), dst.cpu().numpy()], axis=1).astype(np.float32)
        else:
            line_vec = np.concatenate([src, dst], axis=1).astype(np.float32)
        # ipdb.set_trace()
        self.gym.clear_lines(self.viewer)
        for env_id in range(self.num_envs):
            self.gym.add_lines(
                self.viewer,
                self.env_ptr_list[env_id],
                1,
                line_vec[env_id, :],
                color
            )

    def intervaledRandom_(self, tensor, dist, lower=None, upper=None):
        tensor += torch.rand(tensor.shape, device=self.device) * dist * 2 - dist
        if lower is not None and upper is not None:
            torch.clamp_(tensor, min=lower, max=upper)

    def cal_success(self):
        self.open_handle_success = (torch.abs(self.door_handle_dof_tensor[:, 0] - self.door_lower_limits_tensor[:, 1]) > 0.1)
        self.open_door = (torch.abs(self.door_dof_tensor[:, 0]) > 0.1)
        self.open_door_success_15 = (torch.abs(self.door_dof_tensor[:, 0]) > math.pi / 180 * 15)
        self.open_door_success_30 = (torch.abs(self.door_dof_tensor[:, 0]) > math.pi / 180 * 30)
        self.open_door_success_45 = (torch.abs(self.door_dof_tensor[:, 0]) > math.pi / 180 * 45)
        # self.open_door_success_60 = (torch.abs(self.door_dof_tensor[:, 0]) > math.pi / 180 * 60)

        mean_open_handle_success = (self.open_handle_success * 1.0).mean()
        mean_open_door = (self.open_door * 1.0).mean()
        mean_success_15 = (self.open_door_success_15 * 1.0).mean()
        mean_success_30 = (self.open_door_success_30 * 1.0).mean()
        mean_success_45 = (self.open_door_success_45 * 1.0).mean()
        # mean_success_60 = (self.open_door_success_60 * 1.0).mean()
        if(mean_open_door>self.d_5):
            self.d_5 = mean_open_door
        if(mean_success_15>self.d_15):
            self.d_15 = mean_success_15
        # if(mean_success_60>self.d_60):
        #     self.d_60 = mean_success_60
        if(mean_success_30>self.d_30):
            self.d_30 = mean_success_30
        if(mean_success_45>self.d_45):
            self.d_45 = mean_success_45
        print("mean open handle success", mean_open_handle_success)
        print("mean open door", mean_open_door)
        print("mean 15 success", mean_success_15)

        print("mean 30 success", mean_success_30)
        print("mean 45 success", mean_success_45)

        # print("mean 60 success", mean_success_60)
        print()

    # Apply randomizations only on resets, due to current PhysX limitations
    def apply_randomizations(self, dr_params):
        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        for actor, actor_properties in dr_params["actor_params"].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue
                    if prop_name == 'scale':
                        attr_randomization_params = prop_attrs
                        sample = generate_random_samples(attr_randomization_params, 1,
                                                         self.last_step, None)
                        og_scale = 1
                        if attr_randomization_params['operation'] == 'scaling':
                            new_scale = og_scale * sample
                        elif attr_randomization_params['operation'] == 'additive':
                            new_scale = og_scale + sample
                        self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], p, attr)
                                apply_random_samples(
                                    p, og_p, attr, attr_randomization_params,
                                    self.last_step, smpl)
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            smpl = None
                            if self.actor_params_generator is not None:
                                smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                    extern_sample, extern_offsets[env_id], prop, attr)
                            apply_random_samples(
                                prop, self.original_props[prop_name], attr,
                                attr_randomization_params, self.last_step, smpl)

                    setter = param_setters_map[prop_name]
                    default_args = param_setter_defaults_map[prop_name]
                    setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def _detailed_view(self, tensor):
        #[num_env,2]
        shape = tensor.shape
        return tensor.view(self.door_num, self.env_per_door, *shape[1:])


    def post_physics_step(self):
        raise NotImplementedError


def get_attr_val_from_sample(sample, offset, prop, attr):
    """Retrieves param value for the given prop and attr from the sample."""
    if sample is None:
        return None, 0
    if isinstance(prop, np.ndarray):
        smpl = sample[offset:offset+prop[attr].shape[0]]
        return smpl, offset+prop[attr].shape[0]
    else:
        return sample[offset], offset+1
