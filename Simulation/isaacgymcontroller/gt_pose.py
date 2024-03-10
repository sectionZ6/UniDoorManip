from env.base_env import BaseEnv
from manipulation.base_manipulation import BaseManipulation
from isaacgymcontroller.base_controller import BaseController
from logging import Logger
import numpy as np

class GtPoseController(BaseController) :

    def __init__(self, env : BaseEnv, manipulation : BaseManipulation, cfg : dict, logger : Logger):
        super().__init__(env, manipulation, cfg, logger)

    def run(self, eval=False) :
        '''
        Run the controller.
        '''
        goal_hand_pose = self.env.adjust_hand_pose
        self.manipulation.plan_pathway_gt_multi_dt(goal_hand_pose, eval)
    