from isaacgymcontroller.gt_pose import GtPoseController
from isaacgymcontroller.base_controller import BaseController

from manipulation.base_manipulation import BaseManipulation
from manipulation.open_lever_door import OpenLeverDoorManipulation
from manipulation.open_round_door import OpenRoundDoorManipulation
from manipulation.open_safe import OpenSafeManipulation
from manipulation.open_car import OpenCarManipulation
from manipulation.open_window import OpenWindowManipulation
from manipulation.open_cabinet import OpenCabinetManipulation
from manipulation.open_fridge import OpenFridgeManipulation


def parse_controller(args, env, manipulation, cfg, logger):
    try:
        controller = eval(args.controller)(env, manipulation, cfg, logger)
    except NameError as e:
        print(e)
    return controller

def parse_manipulation(args, env, cfg, logger):
    try:
        # print(args.manipulation)
        manipulation = eval(args.manipulation)(env, cfg, logger)
    except NameError as e:
        print(e)
    return manipulation