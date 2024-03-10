
from base_env import BaseEnv
import torch
import numpy as np
import os
import math
import json
import random
from isaacgym.torch_utils import *
from random import shuffle
from isaacgym import gymutil, gymtorch, gymapi
import ipdb


def quat_axis(q, axis=0):
    """ ?? """
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device, dpose, num_envs):

    # Set controller parameters
    # IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u
# franka panda tensor / hand rigid body tensor
def relative_pose(src, dst) :

    shape = dst.shape
    p = dst.view(-1, shape[-1])[:, :3] - src.view(-1, src.shape[-1])[:, :3]
    ip = dst.view(-1, shape[-1])[:, 3:]
    ret = torch.cat((p, ip), dim=1)
    return ret.view(*shape)

class FrankaSliderCabinet(BaseEnv):

    def __init__(self,cfg, sim_params, physics_engine, device_type, device_id, headless,is_multi_agent=False, log_dir=None):
        # 环境配置文件
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.is_multi_agent = is_multi_agent
        self.log_dir = log_dir
        self.up_axis = 'z'
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless
        self.figure = cfg["env"]["figure"]
        self.device = "cpu"
        self.seed = cfg["seed"]
        self.use_handle = False
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]  # 192

        self.env_num_train = cfg["env"]["numTrain"]
        self.env_num_val = cfg["env"]["numVal"]
        self.env_num = self.env_num_train + self.env_num_val
        self.asset_root = cfg["env"]["asset"]["assetRoot"]
        self.door_num_train = cfg["env"]["asset"]["DoorAssetNumTrain"]  # 1
        self.door_num_val = cfg["env"]["asset"]["DoorAssetNumVal"]  # 0
        self.door_num = self.door_num_train + self.door_num_val
        self.load_block = cfg["env"]["asset"]["load_block"]
        door_train_list_len = len(cfg["env"]["asset"]["trainAssets"][self.load_block])  # 1
        # door_val_list_len = len(cfg["env"]["asset"]["testAssets"])  # 0

        self.task = cfg["task"]["task_name"]
        self.action_type = cfg["task"]["action_type"]

        self.collectData = cfg["env"]["collectData"]
        self.collectALL = cfg["env"]["collect_all"]
        self.start_index = cfg["env"]["start_index"]
        self.collectPC = cfg["env"]["collectPC"]
        self.modelTest = cfg["env"]["model_test"]
        self.gapartnet_baseline = cfg["env"]["gapartnet_baseline"]
        self.collectForce = cfg["env"]["collectForce"]

        # 存放训练和测试用的资源的名字（id）
        # self.door_train_name_list = []
        # self.door_val_name_list = []
        self.mobile = cfg["model"]["mobile"]
        self.exp_name = cfg['env']["env_name"]  # franka_drawer_state_open_handle
        print("Simulator: number of doors", self.door_num)
        print("Simulator: number of environments", self.env_num)
        if self.door_num_train:
            assert (self.env_num_train % self.door_num_train == 0)
        if self.door_num_val:
            assert (self.env_num_val % self.door_num_val == 0)
        assert (self.env_num_train * self.door_num_val == self.env_num_val * self.door_num_train)
        assert (self.door_num_train <= door_train_list_len)  # the number of used length must less than real length
        # assert (self.door_num_val <= door_val_list_len)  # the number of used length must less than real length
        assert (self.env_num % self.door_num == 0)  # each cabinet should have equal number envs
        self.env_per_door = self.env_num // self.door_num

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
        self.row_length = 0.15
        self.col_length = 0.1
        self.x_init = 0.7
        self.y_init = 0

        if self.cfg["env"]["enableForceSensors"]:
            self.hand_sensor_list = []
            self.num_sensor = 0
        if self.figure:
            self.camera_handle_front_list = []
            self.camera_handle_right_list = []
            self.camera_handle_left_list = []

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
        # self.d_60 = 0

        super().__init__(cfg=self.cfg, enable_camera_sensors=cfg["env"]["enableCameraSensors"])

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
        self.door_reset_position_noise = cfg["env"]["reset"]["door"]["resetPositionNoise"]
        self.door_reset_rotation_noise = cfg["env"]["reset"]["door"]["resetRotationNoise"]
        self.door_reset_dof_pos_interval = cfg["env"]["reset"]["door"]["resetDofPosRandomInterval"]
        self.door_reset_dof_vel_interval = cfg["env"]["reset"]["door"]["resetDofVelRandomInterval"]
        self.franka_reset_position_noise = cfg["env"]["reset"]["franka"]["resetPositionNoise"]
        self.franka_reset_rotation_noise = cfg["env"]["reset"]["franka"]["resetRotationNoise"]
        self.franka_reset_dof_pos_interval = cfg["env"]["reset"]["franka"]["resetDofPosRandomInterval"]
        self.franka_reset_dof_vel_interval = cfg["env"]["reset"]["franka"]["resetDofVelRandomInterval"]

        self.action_speed_scale = cfg["env"]["actionSpeedScale"]

        # params for success rate
        self.success = torch.zeros((self.env_num,), device=self.device)
        self.success_rate = torch.zeros((self.env_num,), device=self.device)
        self.success_buf = torch.zeros((self.env_num,), device=self.device).long()

        self.average_reward = None
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
        
        if cfg["env"]["visualizePointcloud"] == True :
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

    def create_camera(self, env_ptr, env_id):
        if self.figure:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 2560
            camera_props.height = 2560
            camera_handle_front = self.gym.create_camera_sensor(env_ptr, camera_props)
            camera_handle_left = self.gym.create_camera_sensor(env_ptr, camera_props)
            camera_handle_right = self.gym.create_camera_sensor(env_ptr, camera_props)
            #正面camera
            self.gym.set_camera_location(camera_handle_front, env_ptr, gymapi.Vec3(1.2, 0.0, 2.0), gymapi.Vec3(0.01, 0.0, 0.8))
            # 左侧camera
            self.gym.set_camera_location(camera_handle_right, env_ptr, gymapi.Vec3(1.5, 1.2, 2.0), gymapi.Vec3(0.01, 0.0, 1.4))
            # 右侧camera
            self.gym.set_camera_location(camera_handle_left, env_ptr, gymapi.Vec3(1.2, -1.0, 2.0), gymapi.Vec3(0.01, 0.0, 0.8))

            self.camera_handle_front_list.append(camera_handle_front)
            self.camera_handle_left_list.append(camera_handle_left)
            self.camera_handle_right_list.append(camera_handle_right)

        camera_props = gymapi.CameraProperties()
        camera_props.width = 256
        camera_props.height = 256
        # camera_props.enable_tensors = True
        # ipdb.set_trace()
        camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
        # print("camera_handle", camera_handle)

        # local_transform = gymapi.Transform()
        # #正对偏右视角
        # local_transform.p = gymapi.Vec3(-0.12, 0.05, -0.0)
        # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(-60))
        # #左上视角
        # # local_transform.p = gymapi.Vec3(0.0, 0.20, 0.06)
        # # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 1), np.deg2rad(-90))
        # hand_handle = self.gym.get_actor_rigid_body_handle(env_ptr, self.gripper_actor_list[env_id], 6)
        # # print("hand_handle", hand_handle)
        # # print("local", local_transform.p, local_transform.r)
        # self.gym.attach_camera_to_body(camera_handle, env_ptr, hand_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
        # # self.gym.set_camera_transform(camera_handle, env_ptr, local_transform)

        self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(1.8, 0.0, 1.8), gymapi.Vec3(0.01, 0.0, 1.5))

        # _cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
        # cam_tensor = gymtorch.wrap_tensor(_cam_tensor)
        cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
        cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)
        origin = self.gym.get_env_origin(env_ptr)
        env_origin = torch.zeros((1, 3), device=self.device)
        env_origin[0, 0] = origin.x
        env_origin[0, 1] = origin.y
        env_origin[0, 2] = origin.z
        self.camera_handle_list.append(camera_handle)
        # self.camera_tensor_list.append(cam_tensor)
        self.camera_vinv_list.append(cam_vinv)
        self.camera_proj_list.append(cam_proj)
        self.env_origin_list.append(env_origin)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
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
        self.gym.set_actor_rigid_shape_properties(env_ptr, franka_actor, franka_shape_props)
        self.gym.set_actor_dof_properties(env_ptr, franka_actor, dof_props)
        self.gym.set_actor_dof_states(env_ptr, franka_actor, franka_dof_state, gymapi.STATE_ALL)
        self.franka_actor_list.append(franka_actor)

        if self.cfg["env"]["enableForceSensors"]:
            hand_sensor = self.gym.get_actor_force_sensor(env_ptr, franka_actor, hand_sensor_idx)
            self.hand_sensor_list.append(hand_sensor)
            self.num_sensor += self.gym.get_actor_force_sensor_count(env_ptr, franka_actor)

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

    def _franka_init_pose(self, mobile):
        # random init franka pos  # cabinet 0-1
        if mobile:
            random_num = np.clip(np.random.normal(loc=0.0, scale=0.5, size=None), -1, 1)
            x = self.x_init + self.col_length * random_num
            y = self.y_init + self.row_length * random_num
        else:
            x = self.x_init
            y = self.y_init
        initial_franka_pose = gymapi.Transform()
        # print("x_{},y_{}".format(x,y))
        initial_franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        # 侧向门把手那一侧
        initial_franka_pose.p = gymapi.Vec3(x, y, 0.1)

        return initial_franka_pose

    def get_adjust_hand_pose(self):
        doorhandle_pos = self.door_handle_rigid_body_tensor[:, :3].clone()
        doorhandle_rot = self.door_handle_rigid_body_tensor[:, 3:7].clone()
        # rotate 偏移（Y轴偏移）
        if self.task in ["leverdoor", "rounddoor"]:
            down_q = torch.stack(self.num_envs * [torch.tensor([0, 1, 0, 0])]).to(self.device).view((self.num_envs, 4))
        # down_q = torch.stack(self.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.device).view((self.num_envs, 4))
        if not self.modelTest:
            random_num = np.clip(np.random.normal(loc=0.0, scale=0.5, size=(self.num_envs,)), -1, 1)
            random_num = torch.tensor(random_num, device=self.device)
            if self.task == "leverdoor":
                range = self.handle_door_max_tensor[:, 0] - self.handle_door_min_tensor[:, 0]
                scale = range/5
                y_b = scale * random_num
            
                self.goal_pos_offset_tensor[:, 0] = self.goal_pos_offset_tensor[:, 0] + y_b

            rotate_tensor = torch.zeros((self.num_envs, 3), device=self.device)
            rotate_tensor[:,2] += torch.pi/18 * random_num
            
            phi = rotate_tensor[:, 0]
            theta = rotate_tensor[:, 1]
            psi = rotate_tensor[:, 2]
            qx = torch.sin(phi/2) * torch.cos(theta/2) * torch.cos(psi/2) - torch.cos(phi/2) * torch.sin(theta/2) * torch.sin(psi/2)
            qy = torch.cos(phi/2) * torch.sin(theta/2) * torch.cos(psi/2) + torch.sin(phi/2) * torch.cos(theta/2) * torch.sin(psi/2)
            qz = torch.cos(phi/2) * torch.cos(theta/2) * torch.sin(psi/2) - torch.sin(phi/2) * torch.sin(theta/2) * torch.cos(psi/2)
            qw = torch.cos(phi/2) * torch.cos(theta/2) * torch.cos(psi/2) + torch.sin(phi/2) * torch.sin(theta/2) * torch.sin(psi/2)
            # ipdb.set_trace()
            r_noise = torch.stack((qx, qy, qz, qw), dim=1)
        
        if self.gapartnet_baseline:
            random_num = np.clip(np.random.normal(loc=0.0, scale=0.5, size=(self.num_envs,)), -1, 1)
            random_num = torch.tensor(random_num, device=self.device)
            if self.task == "leverdoor":
                range_x = self.handle_door_max_tensor[:, 0] - self.handle_door_min_tensor[:, 0]
                range_z = self.handle_door_max_tensor[:, 1] - self.handle_door_min_tensor[:, 1]
                range_y = self.handle_door_max_tensor[:, 2] - self.handle_door_min_tensor[:, 2]
                self.goal_pos_offset_tensor[:, 2] = range_z - 0.01
                self.goal_pos_offset_tensor[:, 0] = -range_x/2
                self.goal_pos_offset_tensor[:, 1] = 0
            elif self.task == "rounddoor":
                range_x = self.handle_door_max_tensor[:, 0] - self.handle_door_min_tensor[:, 0]
                range_z = self.handle_door_max_tensor[:, 1] - self.handle_door_min_tensor[:, 1]
                range_y = self.handle_door_max_tensor[:, 2] - self.handle_door_min_tensor[:, 2]

                self.goal_pos_offset_tensor[:, 2] = range_z*4/5
                self.goal_pos_offset_tensor[:, 0] = 0
                self.goal_pos_offset_tensor[:, 1] = 0
            rotate_tensor = torch.zeros((self.num_envs, 3), device=self.device)
            rotate_tensor[:,2] += torch.pi/18 * random_num
            
            phi = rotate_tensor[:, 0]
            theta = rotate_tensor[:, 1]
            psi = rotate_tensor[:, 2]
            qx = torch.sin(phi/2) * torch.cos(theta/2) * torch.cos(psi/2) - torch.cos(phi/2) * torch.sin(theta/2) * torch.sin(psi/2)
            qy = torch.cos(phi/2) * torch.sin(theta/2) * torch.cos(psi/2) + torch.sin(phi/2) * torch.cos(theta/2) * torch.sin(psi/2)
            qz = torch.cos(phi/2) * torch.cos(theta/2) * torch.sin(psi/2) - torch.sin(phi/2) * torch.sin(theta/2) * torch.cos(psi/2)
            qw = torch.cos(phi/2) * torch.cos(theta/2) * torch.cos(psi/2) + torch.sin(phi/2) * torch.sin(theta/2) * torch.sin(psi/2)
            # ipdb.set_trace()
            r_noise = torch.stack((qx, qy, qz, qw), dim=1)
        goal_pos_tensor = quat_apply(doorhandle_rot, self.goal_pos_offset_tensor) + doorhandle_pos
        self.global_goal_pos = goal_pos_tensor.clone()
        # print(self.global_goal_pos)
        goal_pos_tensor[:, 0] += self.gripper_length*2
        # print(y_b)
        goal_rot_tensor = quat_mul(doorhandle_rot, down_q)
        if not self.modelTest:
            goal_rot_tensor = quat_mul(goal_rot_tensor, r_noise)
        # goal_rot_tensor_noise = quat_mul(goal_rot_tensor, r_noise)
        goal_pose = torch.cat([goal_pos_tensor, goal_rot_tensor], dim=-1)
        return goal_pose

    def _load_obj(self, env_ptr, env_id):

        if self.obj_loaded == False :

            self._load_obj_asset()
            #这个是handle的bounding box
            self.handle_door_min_tensor = self.handle_door_min_tensor.repeat_interleave(self.env_per_door, dim=0)
            self.handle_door_max_tensor = self.handle_door_max_tensor.repeat_interleave(self.env_per_door, dim=0)
            self.door_lower_limits_tensor = self.door_dof_lower_limits_tensor.repeat_interleave(self.env_per_door,dim=0)
            self.door_upper_limits_tensor = self.door_dof_upper_limits_tensor.repeat_interleave(self.env_per_door,dim=0)

            self.goal_pos_offset_tensor = self.goal_pos_offset_tensor.repeat_interleave(self.env_per_door, dim=0)
            
            self.initial_goal_pos_offser_tensor = self.goal_pos_offset_tensor.clone()

            self.random_numbers = random.sample(range(32), 16)
            self.dof_type = torch.zeros((self.num_envs,), device=self.device)

            self.obj_loaded = True
        
        door_type = env_id // self.env_per_door
        subenv_id = env_id % self.env_per_door
        obj_actor = self.gym.create_actor(
            env_ptr,
            self.door_asset_list[door_type],
            self.door_pose_list[door_type],
            "door{}-{}".format(door_type, subenv_id),
            env_id,
            1,
            0)
        
        door_dof_props = self.gym.get_asset_dof_properties(self.door_asset_list[door_type])

        
            
        #set door props
        # random_phy = (0.2 * np.random.rand() + 0.8)
        # door_dof_props['stiffness'][0] = 30.0 * random_phy
        # door_dof_props['damping'][0] = 2.0 * random_phy
        # door_dof_props['effort'][0] = 0.5 * random_phy

        #set handle props, randomlization
        #random handle dof [0.3, 0.6]*dof
        # random_upper = ((1/6)*np.random.rand()+0.5)* door_dof_props['upper'][1]
        door_dof_props['upper'][1] = math.pi / 4
        
        # random physics props [0.8, 1.0]
        # random_phy = (0.2*np.random.rand()+0.5)
        # door_dof_props['stiffness'][1] = 3.5 * random_phy
        # door_dof_props['friction'][1] = 1.0 * random_phy
        # door_dof_props['effort'][1] = 1.0
        door_dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT
        # door_dof_props['upper'][0] = 0.0
        # door_dof_props['lower'][0] = -1.5707963267948966
        
        self.gym.set_actor_dof_properties(env_ptr, obj_actor, door_dof_props)
        # obj_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, obj_actor)
        # for shape in obj_shape_props :
        #     shape.friction = 1
        # self.gym.set_actor_rigid_shape_properties(env_ptr, obj_actor, obj_shape_props)
        self.door_actor_list.append(obj_actor)
        # ipdb.set_trace()

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

    def _obj_init_pose(self, min_dict, max_dict, name):
        # {"min": [-0.687565, -0.723071, -0.373959], "max": [0.698835, 0.605562, 0.410705]}
        cabinet_start_pose = gymapi.Transform()
        height = 0.1
        if name == "1307" or name == "1314" or name == "":
            height = 0.3
        elif name == "1320" or name == "1137":
            height = 0.3
        elif name == "1319" or name == "" :
            height = 0.5
        elif name == "1304" or name == "1308":
            height = 0.4
        elif name == "" or name == "1314" or name == "1316":
            height = 0.2
        elif name == "1302" or name == "1305":
            height = 0.2
        elif name == "1340":
            height = 0.5
        cabinet_start_pose.p = gymapi.Vec3(0.0, 0.0, -min_dict[2] + height)
        cabinet_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        return cabinet_start_pose

    def _detailed_view(self, tensor):
        #[num_env,2]
        shape = tensor.shape
        return tensor.view(self.door_num, self.env_per_door, *shape[1:])

    def reset(self, to_reset="all"):

        self._partial_reset(to_reset)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if not self.headless:
            self.render()
        if self.cfg["env"]["enableCameraSensors"] == True:
            self.gym.step_graphics(self.sim)

        self._refresh_observation()
        success = self.success.clone()

        self.extras["successes"] = success
        self.extras["success_rate"] = self.success_rate
        return self.obs_buf, self.rew_buf, self.reset_buf, None

    def _partial_reset(self, to_reset="all"):

        """
        reset those need to be reseted
        """

        if to_reset == "all":
            to_reset = np.ones((self.env_num,))
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
                self.success_buf[env_id] = 0

        if reseted:
            self.gym.set_dof_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state_tensor)
            )
            self.gym.set_actor_root_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.root_tensor)
            )

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
        success = self.success.clone()
        # self._partial_reset(self.reset_buf)

        self.extras["successes"] = success
        self.extras["success_rate"] = self.success_rate
        return self.obs_buf, self.rew_buf, done, None

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
    
    def _perform_actions(self, actions):
        self.actions = actions.clone()
        # Deploy control based on pos i.e. ik
        if self.cfg["env"]["driveMode"] == "ik":
            dof_pos = self.franka_dof_tensor[:, :, 0]
            joints = self.franka_num_dofs - 2
            target_pos = actions[:, :3]*self.space_range + self.space_middle
            pos_err = target_pos - self.hand_rigid_body_tensor[:, :3]

            target_rot = actions[:, 3:7] / torch.sqrt((actions[:, 3:7]**2).sum(dim=-1)+1e-8).view(-1, 1)
            rot_err = orientation_error(target_rot, self.hand_rigid_body_tensor[:, 3:7])
            
            # self._draw_line(target_pos[0], target_pos[0] + quat_axis(target_rot, 0)[0])
            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.jacobian_tensor[:, self.hand_rigid_body_index - 1, :, :-2], self.device, dpose, self.num_envs)
            #转化到关节上
            self.pos_act[:, :joints] = dof_pos.squeeze(-1)[:, :joints] + delta
        elif self.cfg["env"]["driveMode"] == "pos":
            joints = self.franka_num_dofs - 2
            self.pos_act[:, :joints] = self.pos_act[:, :joints] + actions[:, 0:joints] * self.dt * self.action_speed_scale
            self.pos_act[:, :joints] = tensor_clamp(
                self.pos_act[:, :joints], self.franka_dof_lower_limits_tensor[:joints],
                self.franka_dof_upper_limits_tensor[:joints])
        else:  # osc
            joints = self.franka_num_dofs - 2
            pos_err = actions[:,:3] - self.hand_rigid_body_tensor[:, :3]
            # self.draw_line_all(self.hand_rigid_body_tensor[:, :3],actions[:,:3]*4,color = np.array([1,0,0], dtype=np.float32))
            orn_cur = self.hand_rigid_body_tensor[:, 3:7]
            # orn_des = quat_from_euler_xyz(actions[:,-1],actions[:,-2],actions[:,-3])
            orn_err = orientation_error(actions[:,3:7], orn_cur)
            dpose = torch.cat([pos_err, orn_err], -1)
            # print(dpose)
            mm_inv = torch.inverse(self.mm)
            m_eef_inv = self.j_eef @ mm_inv @ torch.transpose(self.j_eef, 1, 2)
            m_eef = torch.inverse(m_eef_inv)
            # ipdb.set_trace()
            self.impedance_force = (self.kp * dpose) - (self.kv * (self.j_eef @ self.franka_dof_tensor[:, :joints, 1].unsqueeze(-1))).squeeze()
            u = torch.transpose(self.j_eef, 1, 2) @ m_eef @ (self.kp * dpose.unsqueeze(-1) - self.kv * self.hand_rigid_body_tensor[:, 7:].unsqueeze(-1))
            # u = torch.transpose(self.j_eef, 1, 2) @ m_eef @ (self.kp * dpose).unsqueeze(-1) - self.kv * self.mm @ self.franka_dof_tensor[:,:joints,1].unsqueeze(-1)
            j_eef_inv = m_eef @ self.j_eef @ mm_inv
            u_null = self.kd_null * -self.franka_dof_tensor[:,:,1].unsqueeze(-1) + self.kp_null * (
                    (self.initial_dof_states[:,:self.franka_num_dofs, 0].view(self.num_envs,-1,1) - self.franka_dof_tensor[:,:,0].unsqueeze(-1) + np.pi) % (2 * np.pi) - np.pi)
            u_null = u_null[:,:joints]
            u_null = self.mm @ u_null
            u += (torch.eye(joints, device=self.device).unsqueeze(0) - torch.transpose(self.j_eef, 1, 2) @ j_eef_inv) @ u_null
            # print(u)
            self.eff_act[:,:joints] = u.squeeze(-1)
            # self.eff_act[:,:joints] = tensor_clamp(u.squeeze(-1), -self.franka_dof_max_torque_tensor[:joints],self.franka_dof_max_torque_tensor[:joints])
            # if self.collectData & (self.progress_buf[0]%10==1):
            #     self.u = self.eff_act[:, :joints]
            # print(self.u)
        
        close = 0.0*torch.ones((self.num_envs, 2), device=self.device)
        open = 0.04*torch.ones((self.num_envs, 2), device=self.device)

        # 根据panda hand和goal pos的距离来判断是否关闭
        doorhandle_pos = self.door_handle_rigid_body_tensor[:, :3]
        doorhandle_rot = self.door_handle_rigid_body_tensor[:, 3:7]
        hand_pos = self.hand_rigid_body_tensor[:, 0:3]
        goal_pos = quat_apply(doorhandle_rot, self.goal_pos_offset_tensor) + doorhandle_pos
        
        dist = torch.norm(hand_pos - goal_pos, p=2, dim=-1).unsqueeze(1).repeat_interleave(2, dim=-1)
        # ipdb.set_trace()
        
        self.pos_act[:,-4:-2] = torch.where((dist<0.11), close, open)
       
        
        self.open_door_stage = ((self.door_handle_dof_tensor[:, 0] >= 0.65 * (self.door_actor_dof_upper_limits_tensor[:, 1] - self.door_actor_dof_lower_limits_tensor[:, 1])) |
                                (torch.abs(self.door_dof_tensor[:, 0]) > 0.01))
        self.eff_act[:, -2] = torch.where(self.open_door_stage, -3*self.door_dof_tensor[:, 0], -150)
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_act.view(-1)))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.eff_act.view(-1)))

    def intervaledRandom_(self, tensor, dist, lower=None, upper=None):
        tensor += torch.rand(tensor.shape, device=self.device) * dist * 2 - dist
        if lower is not None and upper is not None:
            torch.clamp_(tensor, min=lower, max=upper)

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