import os
import sys
import torch
import torch.utils.data as data
import random
import torch.nn.functional as F
import numpy as np
import json
import math
from progressbar import ProgressBar
from argparse import ArgumentParser
import open3d as o3d

import ipdb

class multimodaldata(data.Dataset):
    def __init__(self, cfg, data_features, train_mode=True) -> None:
        super().__init__()
        self.device = cfg.device
        # self.residual = cfg.residual
        self.only_one_step = cfg.only_one_step
        self.open_door_multistep = cfg.open_door_multistep
        self.pointnum = cfg.pointnum
        self.data_features = data_features
        self.train_mode = train_mode
        self.train_ratio = cfg.train_ratio

        self.critic_score = None
    
    def load_process_data(self, path):
        # load all data
        self.pc = torch.load(os.path.join(path,'pc.pth'), map_location=self.device)
        self.proprioception_info = torch.load(os.path.join(path,'proprioception.pth'), map_location=self.device)
        self.action = torch.load(os.path.join(path,'action.pth'), map_location=self.device)
        self.state = torch.load(os.path.join(path,'state.pth'), map_location=self.device)
        self.rot_mat = torch.load(os.path.join(path,'rot.pth'), map_location=self.device)
        data_num = self.action.shape[0]
        print("---------all data num {}-----------".format(data_num))
        index = int(data_num * self.train_ratio)
        if self.train_mode:
            self.pc = self.pc[:index,:,:]
            self.proprioception_info = self.proprioception_info[:index,:]
            self.action = self.action[:index,:]
            self.state = self.state[:index,:]
            self.rot_mat = self.rot_mat[:index,:,:]
        else:
            self.pc = self.pc[index:,:,:]
            self.proprioception_info = self.proprioception_info[index:,:]
            self.action = self.action[index:,:]
            self.state = self.state[index:,:]
            self.rot_mat = self.rot_mat[index:,:,:]
        
        print("load data num:", self.state.shape[0])

    def load_affordance_data(self, path):
        pc_list = []
        state_list = []
        contact_point_list = []
        contact_rot_list = []
        finger_pos_list = []

        data_num = 10
        seed_num = 1
        for i in range(data_num):
            for seed in range(seed_num):
                data_path = os.path.join(path, "_seed{}/universal_rule/window/data_{}.pt".format(seed, 30+i))
                obs = torch.load(data_path, map_location=self.device)
                # flag = obs["collect_flag"]
                pc = obs["pc"][:,0,:,:]
                finger_pos_tensor = obs["finger_pos_tensor"][:,:]
                state = obs["state"][:,-1,4:5]
                contact_point = obs["goal_pos_global_tensor"][:,:]
                contact_rot = obs["goal_rot_global_tensor"][:,:,:]

                pc_list.append(pc)
                finger_pos_list.append(finger_pos_tensor)
                state_list.append(state)
                contact_point_list.append(contact_point)
                contact_rot_list.append(contact_rot)
        
        self.pc = torch.cat(pc_list, dim=0).view(-1, 4096, 4)
        self.pc[:, :, :3] = self.pc_normalize(self.pc[:, :, :3])
        self.state = torch.cat(state_list, dim=0).view(-1, 1)
        print(self.state.mean())
        self.state = torch.where(self.state < 0, 0, self.state)
        self.finger_pos = torch.cat(finger_pos_list, dim=0).view(-1, 3)
        self.contact_point = torch.cat(contact_point_list, dim=0).view(-1, 3)
        self.contact_rot = torch.cat(contact_rot_list, dim=0).view(-1, 3, 3)

        round = False
        if round:
            self.pc = torch.cat([self.pc,self.pc], dim=0).view(-1, 4096, 4)
            self.state = torch.cat([self.state,self.state], dim=0).view(-1, 1)
            self.finger_pos = torch.cat([self.finger_pos[:,:3], self.finger_pos[:,3:6]], dim=0)
            self.contact_rot = torch.cat([self.contact_rot,self.contact_rot], dim=0)
            self.contact_point = torch.cat([self.contact_point,self.contact_point], dim=0)

            print(self.pc.shape)


        # self.pc = torch.cat(pc_list, dim=0)
        # self.proprioception_info = torch.cat(proprioception_info_list, dim=0)
        # self.state = torch.cat(state_list, dim=0)  # door dof + handle dof
        # self.contact_point = torch.cat(contact_point_list, dim=0)
        # self.contact_rot = torch.cat(contact_rot_list, dim=0)

        # score = self.state.clone()
        # # for i in range(data_num):
        # #     score[:, i] = self.state[:, i]*0.5 + torch.mean(self.state, dim=1)*0.5
        # self.pc = self.pc.view(-1, self.pointnum, 4)
        # self.proprioception_info = self.proprioception_info.view(-1, 9)
        # self.state = score.view(-1, 1)
        # self.contact_point = self.contact_point.view(-1, 3)
        # self.contact_rot = self.contact_rot.view(-1, 3, 3)

        train_num = int(self.state.shape[0] * self.train_ratio)
        index = np.arange(self.state.shape[0])
        random.shuffle(index)
        if self.train_mode:
            ind = index[:train_num]
        else:
            ind = index[train_num:]
        self.pc = self.pc[ind]
        # self.proprioception_info = self.proprioception_info[ind]
        self.state = self.state[ind]
        if round:
            self.finger_pos = self.finger_pos[ind]
        else:
            self.contact_point = self.contact_point[ind]
        self.contact_rot = self.contact_rot[ind]
        #
        # ipdb.set_trace()
        print("load data num:", ind.shape[0])

    def load_data(self, path, num, model="actor"):
        pc_list = []
        hand_sensor_force_list = []
        impedance_force_list = []
        proprioception_info_list = []
        action_list = []
        proprioception_ik_list = []
        state_list = []
        rot_mat_list = []
        index_list = []
        train_num = int(self.train_ratio*num)
        if self.train_mode:
            ran = range(train_num)
        else:
            ran = range(train_num, num)
        for i in ran:
            data_path = os.path.join(path, "_seed{}/handle_right_pull_door_residual/data_{}.pt".format(i, i))
            obs = torch.load(data_path, map_location=self.device)
            pc = obs["pc"]
            hand_sensor_force = obs["hand_sensor_force"]
            impedance_force = obs["impedance_force"]
            proprioception_info = obs["proprioception_info"]
            action = obs["action"]
            proprioception_ik = obs["proprioception_ik"]
            state = obs["state"]
            rot_mat = obs["rotate_matrix"]
            index = obs["index"]
            index[index == 15] = 14
            # index[index == 14] = 13
            index = index.int()
            collect_flag = obs["collect_flag"]
            
            if model == "actor":
                '''
                这边的collect_flag要并上index不为零
                '''
                flag = collect_flag & (index != 0) & (index != 14)
                select_num = (flag*1.0).sum().int()
                pc = pc[flag].view(select_num, -1, self.pointnum, 3)
                hand_sensor_force = hand_sensor_force[flag].view(select_num, -1, 6)
                proprioception_info = proprioception_info[flag].view(select_num, -1, 24)
                action = action[flag].view(select_num, -1, 3)
                proprioception_ik = proprioception_ik[flag].view(select_num, -1, 9)
                state = state[flag].view(select_num, -1, 4)
                rot_mat = rot_mat[flag].view(select_num, -1, 3, 3)
                index = index[flag]

            pc_list.append(pc)
            hand_sensor_force_list.append(hand_sensor_force)
            impedance_force_list.append(impedance_force)
            proprioception_info_list.append(proprioception_info)
            action_list.append(action)
            proprioception_ik_list.append(proprioception_ik)
            state_list.append(state)
            rot_mat_list.append(rot_mat)
            index_list.append(index)

        self.pc = torch.cat(pc_list, dim=0)
        self.hand_sensor_force = torch.cat(hand_sensor_force_list, dim=0)
        self.impedance_force = torch.cat(impedance_force_list, dim=0)
        self.proprioception_info = torch.cat(proprioception_info_list, dim=0) #
        self.action = torch.cat(action_list, dim=0)
        
        self.proprioception_ik = torch.cat(proprioception_ik_list, dim=0) #eff_act
        self.state = torch.cat(state_list, dim=0) #door dof + handle dof
        self.rot_mat = torch.cat(rot_mat_list, dim=0)
        self.index = torch.cat(index_list, dim=0)
        
        if self.only_one_step:
            self.pc = self.get_index_tensor(self.pc, self.index)
            self.hand_sensor_force = self.get_index_tensor(self.hand_sensor_force, self.index)
            self.impedance_force = self.get_index_tensor(self.impedance_force, self.index)
            self.proprioception_info = self.get_index_tensor(self.proprioception_info, self.index)
            self.action = self.get_index_tensor(self.action, self.index)
            self.proprioception_ik = self.get_index_tensor(self.proprioception_ik, self.index)
            self.state = self.get_index_tensor(self.state, self.index + 1)
            self.rot_mat =self.get_index_tensor(self.rot_mat, self.index)
            print("load data num:", self.state.shape[0])
        elif self.open_door_multistep:
            # 构造index_bool tensor
            index_mask = (torch.zeros(self.index.shape[0], self.pc.shape[1]-1, device=self.device) == 0)
            for i in range(self.index.shape[0]):
                for j in range(self.pc.shape[1]-1):
                    if j < self.index[i]:
                        index_mask[j] = False
                    else:
                        index_mask[j] = True
            
            self.pc = self.get_index_mask_tensor(self.pc[:,:-1,:,:], index_mask)
            self.proprioception_info = self.get_index_mask_tensor(self.proprioception_info[:,:-1,:], index_mask)
            self.action = self.get_index_mask_tensor(self.action[:,:-1,:], index_mask)
            self.rot_mat =self.get_index_mask_tensor(self.rot_mat[:,:-1,:,:], index_mask)
            # 将index_bool索引向后挪一
            self.state = self.get_index_mask_tensor(self.state[:,1:,:], index_mask)
            print("load data num:", self.state.shape[0]*self.state.shape[1])

    def get_index_tensor(self, tensor, index):
        index = index.long()
        shape_size = len(tensor.shape)
        if shape_size == 3:
            # return tensor[torch.arange(tensor.shape[0]), index].reshape(-1, tensor.shape[-1])
            index=index.view(-1, 1, 1).repeat_interleave(tensor.shape[-1], 2)
        elif shape_size == 4:
            # return tensor[torch.arange(tensor.shape[0]), index].reshape(-1, tensor.shape[-2], tensor.shape[-1])
            index = index.view(-1, 1, 1, 1).repeat_interleave(tensor.shape[-2], 2).repeat_interleave(tensor.shape[-1], 3)
        # ipdb.set_trace()
        return torch.gather(tensor, dim=1, index=index)

    def get_index_mask_tensor(self, tensor, index_mask):
        
        shape_size = len(tensor.shape)
        if shape_size == 3:
            index_mask = index_mask.view(index_mask.shape[0], index_mask.shape[1], 1).repeat_interleave(tensor.shape[-1], 2)
            return tensor[index_mask].reshape(-1, tensor.shape[-1])
        elif shape_size == 4:
            index_mask = index_mask.view(index_mask.shape[0], index_mask.shape[1], 1, 1).repeat_interleave(tensor.shape[-2], 2).repeat_interleave(tensor.shape[-1], 3)
            return tensor[index_mask].reshape(-1, tensor.shape[-2], tensor.shape[-1])

    def process_data(self):
        #pc normalization
        self.pc = self.pc_normalize(self.pc)
        self.critic_score = self.state[..., 0]  # 直接用度数监督

    def pc_normalize(self, pc):
        print("normalize pc", pc.shape)
        center = torch.mean(pc, dim=-2, keepdim=True)
        pc = pc - center
        m = torch.max(torch.norm(pc, p=2, dim=-1)).unsqueeze(-1)
        pc = pc / m

        return pc
    
    def residual_action(self):
        return self.action - self.proprioception_info[..., -3:]
    
    def __len__(self):
        
        return self.pc.shape[0]
    
    def __getitem__(self, index):
        data_feat = ()
        for feat in self.data_features:
            if feat == 'pc':
                data_feat = data_feat + (self.pc.view(self.pc.shape[0], self.pointnum, -1)[index],)
            # elif feat == 'hand_sensor_force':
            #     data_feat = data_feat + (self.hand_sensor_force.view(-1, 6)[index],)
            elif feat == 'franka_state':
                data_feat = data_feat + (self.proprioception_info.view(-1, 32)[index, 0:22],)
            elif feat == 'hand_pos':
                data_feat = data_feat + (self.proprioception_info.view(-1, 32)[index, 25:28],)
            elif feat == 'root_pos':
                data_feat = data_feat + (self.proprioception_info.view(-1, 32)[index, 22:25],)
            elif feat == 'action':
                data_feat = data_feat + (self.action.view(-1, 3)[index],)
            elif feat == 'finger_pos':
                data_feat = data_feat + (self.finger_pos.view(-1, 3)[index],)
            elif feat == 'contact_rot':
                data_feat = data_feat + (self.contact_rot.view(-1, 3, 3)[index],)
            elif feat == 'rotmat':
                data_feat = data_feat + (self.rot_mat.view(-1, 3, 3)[index],)
            elif feat == 'critic_score':
                data_feat = data_feat + (self.critic_score.view(-1, 1)[index],)
            elif feat == 'affordance_score':
                data_feat = data_feat + (self.state.view(-1, 1)[index],)
            elif feat == 'contact_point':
                data_feat = data_feat + (self.contact_point.view(-1, 3)[index],)
            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feat
    
