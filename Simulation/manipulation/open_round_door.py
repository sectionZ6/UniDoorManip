'''
Open door manipulation task
time 2023/7/22
'''
from manipulation.base_manipulation import BaseManipulation
from env.base_env import BaseEnv
from manipulation.transform import *
from logging import Logger
import numpy as np
import pytorch3d.transforms as tf
import ipdb

class OpenRoundDoorManipulation(BaseManipulation) :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    '''
    用于收集全程gt data --- 训练Open Door Stage
    '''
    def plan_pathway_gt_multi_dt(self, pose, eval=False):
        batch_size = pose.shape[0]
        pose[:,2] += 0.005
        # move to the handle
        self.env.reset()
        for i in range(50):
            self.env.step(pose)
            print("step_{}".format(i+1))
        
        # grasp the handle
        pose[:, 0] -= self.env.gripper_length + 0.012
        for i in range(30):
            self.env.step(pose)
            print("step_{}".format(i+1+50))
        
        down_q = torch.stack(self.env.num_envs * [torch.tensor([0.0, 1.0, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
        
        step_size = 0.05
        
        rot_quat = torch.tensor([[ 0, 0, -0.1736482, 0.9848078]]*batch_size, device=self.env.device) #每次旋转30°
        for i in range(18):
            
            print("step_{}".format(i))
            handle_q = self.env.door_handle_rigid_body_tensor[:, 3:7]
            # 定义两个方向
            # rotate_dir = quat_axis(handle_q, axis=1) # y-up
            open_dir = quat_axis(handle_q, axis=2)
            cur_p = self.env.hand_rigid_body_tensor[:, :3]
            cur_q = self.env.hand_rigid_body_tensor[:, 3:7]
            pred_p = torch.where(self.env.open_door_flag, cur_p + open_dir * step_size, cur_p)
            open_door_flag = self.env.open_door_flag[:,0].unsqueeze(1).repeat_interleave(4, dim=-1)
            pred_q = torch.where(open_door_flag, quat_mul(handle_q, down_q), quat_mul(cur_q, rot_quat)) 
            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
            for j in range(10):
                self.env.step(pred_pose)
    