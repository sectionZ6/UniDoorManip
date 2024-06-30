
from base_env import BaseEnv
import torch
import numpy as np
import os
import math
import json
import yaml
import random
from isaacgym.torch_utils import *
from random import shuffle
from isaacgym import gymutil, gymtorch, gymapi
import ipdb


def quat_axis(q, axis=0):
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

class FrankaSliderCar(BaseEnv):

    def __init__(self,cfg, sim_params, physics_engine, device_type, device_id, headless, log_dir=None):
        
        self.log_dir = log_dir
        self.use_handle = False

        self.row_length = 0.1
        self.col_length = 0.1
        self.x_init = 0.6
        self.y_init = -0.1
        
        super().__init__(cfg, sim_params, physics_engine, 
                         device_type, device_id, headless, 
                         enable_camera_sensors=cfg["env"]["enableCameraSensors"])

        self.init_all_tensor()

    def _franka_init_pose(self, mobile):
        # random init franka pos
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
        down_q = torch.stack(self.num_envs * [torch.tensor([0.0, 1.0, 0, 0])]).to(self.device).view((self.num_envs, 4))
        if not self.modelTest:
            random_num = np.clip(np.random.normal(loc=0.0, scale=0.5, size=(self.num_envs,)), -1, 1)
            random_num = torch.tensor(random_num, device=self.device)
            if self.task == "open_car":
                range = self.handle_door_max_tensor[:, 0] - self.handle_door_min_tensor[:, 0]
                scale = range/6
                y_b = scale * random_num
            
                self.goal_pos_offset_tensor[:, 0] = self.goal_pos_offset_tensor[:, 0] + y_b

            rotate_tensor = torch.zeros((self.num_envs, 3), device=self.device)
            rotate_tensor[:,2] += torch.pi/18 * random_num
            # quat = tf.matrix_to_quaternion(tf.euler_angles_to_matrix(rotate_tensor, 'XYZ'))
            # print(quat)
            # print(rotate_tensor)
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

            range_x = self.handle_door_max_tensor[:, 0] - self.handle_door_min_tensor[:, 0]
            range_z = self.handle_door_max_tensor[:, 1] - self.handle_door_min_tensor[:, 1]
            range_y = self.handle_door_max_tensor[:, 2] - self.handle_door_min_tensor[:, 2]
            self.goal_pos_offset_tensor[:, 2] = range_z - 0.01
            self.goal_pos_offset_tensor[:, 0] = range_x/2
            self.goal_pos_offset_tensor[:, 1] = 0
            
            rotate_tensor = torch.zeros((self.num_envs, 3), device=self.device)
            rotate_tensor[:,2] += torch.pi/18 * random_num
            # quat = tf.matrix_to_quaternion(tf.euler_angles_to_matrix(rotate_tensor, 'XYZ'))
            # print(quat)
            # print(rotate_tensor)
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
        # door_dof_props['upper'][1] = math.pi / 4
        
        # random physics props [0.8, 1.0]
        # random_phy = (0.2*np.random.rand()+0.5)
        # door_dof_props['stiffness'][1] = 3.5 * random_phy
        # door_dof_props['friction'][1] = 1.0 * random_phy
        # door_dof_props['effort'][1] = 0.0
        # door_dof_props['stiffness'][1] = 0.0
        # door_dof_props['friction'][1] = 0.0
        # door_dof_props['effort'][1] = 0.0
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

    def _obj_init_pose(self, min_dict, max_dict, name):
        # {"min": [-0.687565, -0.723071, -0.373959], "max": [0.698835, 0.605562, 0.410705]}
        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(0.0, 0.0, -min_dict[2]+0.1)
        cabinet_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        return cabinet_start_pose
    
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
        self.pos_act[:,-4:-2] = torch.where((dist<0.12), close, open)

        self.open_door_stage = ((self.door_handle_dof_tensor[:, 0] >= 0.9 * (self.door_actor_dof_upper_limits_tensor[:, 1] - self.door_actor_dof_lower_limits_tensor[:, 1])) |
                                (torch.abs(self.door_dof_tensor[:, 0]) > 0.01))
        # -1 ours
        self.eff_act[:, -2] = torch.where(self.open_door_stage, -3*self.door_dof_tensor[:, 0], -150)
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_act.view(-1)))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.eff_act.view(-1)))