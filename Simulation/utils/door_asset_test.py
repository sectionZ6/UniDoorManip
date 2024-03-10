"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""

import math
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
from isaacgym import gymtorch
import json
import torch

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
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

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

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# add cartpole urdf asset
asset_root = "../../../Datasets/door_datasets"
asset_file = "99650089960011/mobility.urdf"

# Load asset with default control type of position for all joints
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
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

with open("../../../Datasets/door_datasets/99650089960001/bounding_box.json", "r") as f:
    data = json.load(f)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, -data["min"][2] + 0.1)
initial_pose.r = gymapi.Quat(0, 0.0, 1.0, 0)

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)
door = gym.create_actor(env0, cartpole_asset, initial_pose, 'door', 0, 1)
state_tensor = gym.get_actor_rigid_body_states(env0, door, gymapi.STATE_ALL)

# state_tensor = gymtorch.wrap_tensor(state_tensor)

# Configure DOF properties
# props = gym.get_actor_dof_properties(env0, door)
# props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
# props["stiffness"] = (5000.0, 5000.0)
# props["damping"] = (100.0, 100.0)
# gym.set_actor_dof_properties(env0, door, props)
# dict = gym.get_actor_dof_dict(env0,door)
# print(dict)
# # Set DOF drive targets
# surface = gym.find_actor_dof_handle(env0, door, 'joint_2')
# surface_1 = gym.find_actor_dof_handle(env0, door, 'joint_1')

# gym.set_dof_target_position(env0, surface, 1)
# gym.set_dof_target_position(env0, surface_1, 1)

def _draw_line(src, dst):
    line_vec = np.stack([src, dst]).flatten().astype(np.float32)
    color = np.array([1,0,0], dtype=np.float32)
    gym.clear_lines(viewer)
    gym.add_lines(
        viewer,
        env0,
        1,
        line_vec,
        color
    )


# Look at the first env
cam_pos = gymapi.Vec3(3, 0, 1.5)
cam_target = gymapi.Vec3(0, 0, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 构造goal_pos
with open("../../../Datasets/door_datasets/99650089960001/handle_bounding.json", "r") as f:
    bounding_box = json.load(f)
    # max_x = bounding_box["bounding_box"]['max_x']
    # min_x = bounding_box["bounding_box"]['min_x']

    # max_y = bounding_box["bounding_box"]['max_y']
    # min_y = bounding_box["bounding_box"]['min_y']

    # max_z = bounding_box["bounding_box"]['max_z']
    # min_z = bounding_box["bounding_box"]['min_z']

    x = bounding_box["goal_pos"][0]
    z = bounding_box["goal_pos"][2]

def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

local_handle_pos = np.zeros(3, dtype=np.float32)
local_handle_pos_s = np.zeros(3, dtype=np.float32)
# # x = (max_x - min_x)/2
# # print(x)
# # z = (max_z - min_z)
# # print(z)
# # local_handle_pos[0] -= x
# # local_handle_pos[2] += z
local_handle_pos[0] = x
local_handle_pos[2] = z
local_handle_pos_s[0] = x
local_handle_pos_s[2] = z + 0.1

local_handle_pos = torch.tensor(local_handle_pos)
local_handle_pos_s = torch.tensor(local_handle_pos_s)
doorhandle_rot = state_tensor["pose"]['r'][2].tolist()
doorhandle_pos = state_tensor["pose"]['p'][2].tolist()
doorhandle_rot = torch.tensor(doorhandle_rot)
doorhandle_pos = torch.tensor(doorhandle_pos)
print(doorhandle_pos)
print(doorhandle_rot)

goal_pos = quat_apply(doorhandle_rot, local_handle_pos) + doorhandle_pos
src_pos =  quat_apply(doorhandle_rot, local_handle_pos_s) + doorhandle_pos
print(goal_pos)

_draw_line(src_pos,goal_pos)

# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Nothing to be done for env 0

    # Nothing to be done for env 1
    '''
    # Update env 2: reverse cart target velocity when bounds reached
    pos = gym.get_dof_position(env2, cart_dof_handle2)
    if pos >= 0.5:
        gym.set_dof_target_velocity(env2, cart_dof_handle2, -1.0)
    elif pos <= -0.5:
        gym.set_dof_target_velocity(env2, cart_dof_handle2, 1.0)

    # Update env 3: apply an effort to the pole to keep it upright
    pos = gym.get_dof_position(env3, pole_dof_handle3)
    gym.apply_dof_effort(env3, pole_dof_handle3, -pos * 50)
    '''
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)