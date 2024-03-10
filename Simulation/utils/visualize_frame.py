"""
visualize dataset
"""

import math
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
from isaacgym import gymtorch
import yaml
import os
import ipdb

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")
# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

sim_params.use_gpu_pipeline = False
# if args.use_gpu_pipeline:
#     print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(0, 0, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0
# create the ground plane
gym.add_ground(sim, plane_params)

# add cartpole urdf asset
asset_root = "../assets"
door_frame_list = ['8903','8966','8983','8994','9065','9127','9277','9280','9281','9288','9393','9410']
door_frame_asset_list = []
# 遍历yaml文件
for door_name in door_frame_list:
    asset_name = 'dataset/door_datasets/'+door_name+'plus1551894098/mobility.urdf'
    door_frame_asset_list.append(asset_name)

print(len(door_frame_asset_list))

# set up the env grid
num_envs = len(door_frame_asset_list)
spacing = 2
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

print(door_frame_asset_list)
# Load asset with default control type of position for all joints
asset_list = []
for i in range(num_envs):
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.vhacd_enabled = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.collapse_fixed_joints = True
    asset_options.vhacd_params = gymapi.VhacdParams()
    asset_options.vhacd_params.resolution = 2048
    print("Loading asset '%s'" % (door_frame_asset_list[i]))
    asset = gym.load_asset(sim, asset_root, door_frame_asset_list[i], asset_options)
    asset_list.append(asset)
# print(asset_list)
# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 1)
initial_pose.r = gymapi.Quat(0, 0.0, 1.0, 0)

# Create environment
env_list = []
camera_handles = []
# ipdb.set_trace()
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, 2)
    env_list.append(env)
    door = gym.create_actor(env, asset_list[i], initial_pose, 'door', i, 1)
    # 创建sensor
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 75.0
    camera_props.width = 1080
    camera_props.height = 1920
    camera_door = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(camera_door, env, gymapi.Vec3(1,0,1), gymapi.Vec3(0,0,1))
    camera_handles.append(camera_door)
    # 把手sensor
    # TODO 创建这个sensor的时候需要根据把手不同位置进行判断
    '''
    
    '''

    # camera_props = gymapi.CameraProperties()
    # camera_props.width = 512
    # camera_props.height = 512
    # camera_handle = gym.create_camera_sensor(env, camera_props)
    # if '9393' in door_name_list[i] or '8994' in door_name_list[i]:
    #     gym.set_camera_location(camera_handle, env, gymapi.Vec3(0.2,-0.26,1.15), gymapi.Vec3(0,-0.26,1.0))
    # else:
    #     gym.set_camera_location(camera_handle, env, gymapi.Vec3(0.2,0.26,1.15), gymapi.Vec3(0,0.26,1.0))
    # # image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    # camera_handles.append(camera_handle)



cam_pos = gymapi.Vec3(3, 0, 2)
cam_target = gymapi.Vec3(0, 0, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.render_all_camera_sensors(sim)

    flag = False
    if flag == False:
        for env_id, env_ptr in enumerate(env_list):
            rgb_filename = os.path.join("../dataset_images/door_%s.png" % (door_frame_list[env_id]))
            gym.write_camera_image_to_file(sim, env_ptr, camera_handles[env_id], gymapi.IMAGE_COLOR, rgb_filename)
            # rgb_filename = os.path.join("dataset_images/handle_%s.png" % (door_name_list[env_id]))
            # gym.write_camera_image_to_file(sim, env_ptr, camera_handles[env_id*2+1], gymapi.IMAGE_COLOR, rgb_filename)
            flag = True
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)