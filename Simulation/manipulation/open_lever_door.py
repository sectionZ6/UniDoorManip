'''
Open door manipulation task
time 2023/7/22
'''
from manipulation.base_manipulation import BaseManipulation
from env.base_env import BaseEnv
from manipulation.transform import *
from logging import Logger
import numpy as np
import ipdb

class OpenLeverDoorManipulation(BaseManipulation) :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)

    '''
    center:grasp pos
    axis:
    '''
    def plan_pathway_gt_multi_dt(self, pose, eval=False) :
        print("stage two lever right pull")
    
        # move to the handle
        
        self.env.reset()
        for i in range(50):
            self.env.step(pose)
        
        #grasp the handle
        pose[:, 0] -= self.env.gripper_length + 0.012
        for i in range(30):
            self.env.step(pose)
            print("step_{}".format(i+50))
        
        down_q = torch.stack(self.env.num_envs * [torch.tensor([0.0, 1.0, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
        
        step_size = 0.06
        
        for i in range(18):
            print("step_{}".format(i))
            handle_q = self.env.door_handle_rigid_body_tensor[:, 3:7]
            # 定义两个方向
            rotate_dir = quat_axis(handle_q, axis=1) # y-up
            open_dir = quat_axis(handle_q, axis=2)
            cur_p = self.env.hand_rigid_body_tensor[:, :3]
            pred_p = torch.where(self.env.open_door_flag, cur_p + open_dir * step_size, cur_p - rotate_dir * step_size)
            pred_q = quat_mul(handle_q, down_q)
            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
            for j in range(10):
                self.env.step(pred_pose)