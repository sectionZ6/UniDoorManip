# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch
import ipdb


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!")

def warn_algorithm_name():
    raise Exception(
                "Unrecognized algorithm!\nAlgorithm should be one of: [ppo, happo, hatrpo, mappo]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

def retrieve_cfg(args):

    log_dir = None
    task_cfg = None

    print(args.task)
    if args.task == "FrankaDoorBase":
        log_dir = os.path.join(args.logdir, "franka_door_base/")
        task_cfg = "cfg/franka_door_base.yaml"
    elif args.task == "FrankaSliderDoor":
        log_dir = os.path.join(args.logdir, "franka_slider_door/")
        task_cfg = "cfg/franka_slider_door.yaml"
    elif args.task == "FrankaSliderSafe":
        log_dir = os.path.join(args.logdir, "franka_slider_safe/")
        task_cfg = "cfg/franka_slider_safe.yaml"
    elif args.task == "FrankaSliderCar":
        log_dir = os.path.join(args.logdir, "franka_slider_car/")
        task_cfg = "cfg/franka_slider_car.yaml"
    elif args.task == "FrankaSliderWindow":
        log_dir = os.path.join(args.logdir, "franka_slider_window/")
        task_cfg = "cfg/franka_slider_window.yaml"
    elif args.task == "FrankaSliderSDoor":
        log_dir = os.path.join(args.logdir, "franka_slider_sdoor/")
        task_cfg = "cfg/franka_open_sdoor.yaml"
    elif args.task == "FrankaSliderFridge":
        log_dir = os.path.join(args.logdir, "franka_slider_fridge/")
        task_cfg = "cfg/franka_open_fridge.yaml"
    elif args.task == "FrankaSliderCabinet":
        log_dir = os.path.join(args.logdir, "franka_slider_cabinet/")
        task_cfg = "cfg/franka_open_cabinet.yaml"
    else:
        warn_task_name()
    
    return log_dir, task_cfg


def load_cfg(args):

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numTrain"] = args.num_envs
        cfg["env"]["numVal"] = 0
    
    if args.num_envs_val > 0:
        cfg["env"]["numVal"] = args.num_envs_val
    
    if args.num_objs > 0:
        cfg["env"]["asset"]["cabinetAssetNumTrain"] = args.num_objs
        cfg["env"]["asset"]["cabinetAssetNumVal"] = 0
        cfg["env"]["asset"]["assetNumTrain"] = args.num_objs
        cfg["env"]["asset"]["assetNumVal"] = 0
    
    if args.num_objs_val > 0 :
        cfg["env"]["asset"]["cabinetAssetNumVal"] = args.num_objs_val
        cfg["env"]["asset"]["assetNumVal"] = args.num_objs_val

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    cfg["name"] = args.task
    cfg["headless"] = args.headless
    cfg["seed"] = args.seed

    logdir = args.logdir

    return cfg, logdir


def parse_sim_params(args, cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads
    return sim_params


def get_args():
    custom_parameters = [
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--task", "type": str, "default": "Humanoid",
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--controller", "type": str, "default": "BaseController"},
        {"name": "--manipulation", "type": str, "default": "BaseManipulation"},
        {"name": "--logdir", "type": str, "default": "../logs/"},
        {"name": "--cfg_env", "type": str, "default": "Base"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to train - override config file"},
        {"name": "--num_envs_val", "type": int, "default": 0,
            "help": "Number of environments to validate - override config file"},
        {"name": "--num_objs", "type": int, "default": 0,
            "help": "Number of objects to train - override config file"},
        {"name": "--num_objs_val", "type": int, "default": 0,
            "help": "Number of objects to validate - override config file"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "help": "Random seed"}]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # allignment with examples
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    logdir, cfg_env = retrieve_cfg(args)

    # use custom parameters if provided by user
    if args.logdir == "logs/":
        args.logdir = logdir

    if args.cfg_env == "Base":
        args.cfg_env = cfg_env

    return args
