# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from base_env import BaseEnv
from franka_slider_door import FrankaSliderDoor
from franka_slider_safe import FrankaSliderSafe
from franka_slider_car import FrankaSliderCar
from franka_slider_window import FrankaSliderWindow
from franka_slider_fridge import FrankaSliderFridge
from franka_slider_cabinet import FrankaSliderCabinet
from utils.config import warn_task_name

import json


def parse_env(args, cfg, sim_params, log_dir):

    # create native task and pass custom config
    device_id = args.device_id

    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]


    log_dir = log_dir + "_seed{}".format(cfg_task["seed"])

    try:
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless,
            log_dir=log_dir)
        print(task)
    except NameError as e:
        print(e)
        warn_task_name()
    return task
