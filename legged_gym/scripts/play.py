# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import os
import cv2
import numpy as np
from isaacgym import gymapi
from legged_gym import LEGGED_GYM_ROOT_DIR

# import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime

import pygame
from threading import Thread

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 3.0

joystick_use = True
joystick_opened = False

if joystick_use:

    pygame.init()

    try:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"cannot open joystick device:{e}")

    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd
        
        while not exit_flag:
            pygame.event.get()

            x_vel_cmd = -joystick.get_axis(1) * x_vel_max
            y_vel_cmd = -joystick.get_axis(0) * y_vel_max
            yaw_vel_cmd = -joystick.get_axis(3) * yaw_vel_max

            pygame.time.delay(100)

        # launch gamepad thread
    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False 
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.continuous_push = False 
    env_cfg.domain_rand.randomize_base_mass = False 
    env_cfg.domain_rand.randomize_base_com = False 
    env_cfg.domain_rand.randomize_pd_gains = False 
    env_cfg.domain_rand.randomize_calculated_torque = False 
    env_cfg.domain_rand.randomize_link_mass = False 
    env_cfg.domain_rand.randomize_motor_zero_offset = False 
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_cmd_action_latency = False
    env_cfg.domain_rand.range_cmd_action_latency = [5, 5]
    env_cfg.domain_rand.add_obs_latency = True
    env_cfg.domain_rand.randomize_obs_motor_latency = True
    env_cfg.domain_rand.range_obs_motor_latency = [5, 5]
    env_cfg.domain_rand.randomize_obs_imu_latency = True
    env_cfg.domain_rand.range_obs_imu_latency = [5, 5]
    env_cfg.noise.curriculum = False
    env_cfg.commands.heading_command = False

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, '0_exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env_cfg.sim.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 5 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir,exist_ok=True)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir,exist_ok=True)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))
    
    obs = env.get_observations()

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    for i in range(stop_state_log):
        actions = policy(obs.detach()) # * 0.
    
        if FIX_COMMAND:
            env.commands[:, 0] = 0.5    # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.
            
        else:
            env.commands[:, 0] = x_vel_cmd
            env.commands[:, 1] = y_vel_cmd
            env.commands[:, 2] = yaw_vel_cmd
            env.commands[:, 3] = 0.
        
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, 0].item() * env.cfg.control.action_scale,
                    'dof_pos_target[0]': actions[robot_index, 0].item() * env.cfg.control.action_scale,
                    'dof_pos_target[1]': actions[robot_index, 1].item() * env.cfg.control.action_scale,
                    'dof_pos_target[2]': actions[robot_index, 2].item() * env.cfg.control.action_scale,
                    'dof_pos_target[3]': actions[robot_index, 3].item() * env.cfg.control.action_scale,
                    'dof_pos_target[4]': actions[robot_index, 4].item() * env.cfg.control.action_scale,
                    'dof_pos_target[5]': actions[robot_index, 5].item() * env.cfg.control.action_scale,
                    'dof_pos_target[6]': actions[robot_index, 6].item() * env.cfg.control.action_scale,
                    'dof_pos_target[7]': actions[robot_index, 7].item() * env.cfg.control.action_scale,
                    'dof_pos_target[8]': actions[robot_index, 8].item() * env.cfg.control.action_scale,
                    'dof_pos_target[9]': actions[robot_index, 9].item() * env.cfg.control.action_scale,
                    'dof_pos_target[10]': actions[robot_index,10].item() * env.cfg.control.action_scale,
                    'dof_pos_target[11]': actions[robot_index, 11].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, 0].item(),
                    'dof_pos[0]': env.dof_pos[robot_index, 0].item(),
                    'dof_pos[1]': env.dof_pos[robot_index, 1].item(),
                    'dof_pos[2]': env.dof_pos[robot_index, 2].item(),
                    'dof_pos[3]': env.dof_pos[robot_index, 3].item(),
                    'dof_pos[4]': env.dof_pos[robot_index, 4].item(),
                    'dof_pos[5]': env.dof_pos[robot_index, 5].item(),
                    'dof_pos[6]': env.dof_pos[robot_index, 6].item(),
                    'dof_pos[7]': env.dof_pos[robot_index, 7].item(),
                    'dof_pos[8]': env.dof_pos[robot_index, 8].item(),
                    'dof_pos[9]': env.dof_pos[robot_index, 9].item(),
                    'dof_pos[10]': env.dof_pos[robot_index, 10].item(),
                    'dof_pos[11]': env.dof_pos[robot_index, 11].item(),
                    'dof_torque': env.torques[robot_index, 0].item(),
                    'dof_torque[0]': env.torques[robot_index, 0].item(),
                    'dof_torque[1]': env.torques[robot_index, 1].item(),
                    'dof_torque[2]': env.torques[robot_index, 2].item(),
                    'dof_torque[3]': env.torques[robot_index, 3].item(),
                    'dof_torque[4]': env.torques[robot_index, 4].item(),
                    'dof_torque[5]': env.torques[robot_index, 5].item(),
                    'dof_torque[6]': env.torques[robot_index, 6].item(),
                    'dof_torque[7]': env.torques[robot_index, 7].item(),
                    'dof_torque[8]': env.torques[robot_index, 8].item(),
                    'dof_torque[9]': env.torques[robot_index, 9].item(),
                    'dof_torque[10]': env.torques[robot_index, 10].item(),
                    'dof_torque[11]': env.torques[robot_index, 11].item(),
                    'dof_vel': env.dof_vel[robot_index, 0].item(),
                    'dof_vel[0]': env.dof_vel[robot_index, 0].item(),
                    'dof_vel[1]': env.dof_vel[robot_index, 1].item(),
                    'dof_vel[2]': env.dof_vel[robot_index, 2].item(),
                    'dof_vel[3]': env.dof_vel[robot_index, 3].item(),
                    'dof_vel[4]': env.dof_vel[robot_index, 4].item(),
                    'dof_vel[5]': env.dof_vel[robot_index, 5].item(),
                    'dof_vel[6]': env.dof_vel[robot_index, 6].item(),
                    'dof_vel[7]': env.dof_vel[robot_index, 7].item(),
                    'dof_vel[8]': env.dof_vel[robot_index, 8].item(),
                    'dof_vel[9]': env.dof_vel[robot_index, 9].item(),
                    'dof_vel[10]': env.dof_vel[robot_index, 10].item(),
                    'dof_vel[11]': env.dof_vel[robot_index, 11].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
                )
            
        elif i == stop_state_log:
            logger.plot_states()

        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)

    # logger.print_rewards()
    
    while True:
        True

    if RENDER:
        video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = False
    FIX_COMMAND = False # whether to use joystick to control the robot
    args = get_args()
    play(args)
