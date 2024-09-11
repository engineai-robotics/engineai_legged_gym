from legged_gym.envs import ZqSA01Cfg
import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
# from legged_gym.envs import *
from legged_gym.utils import  Logger
import torch
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
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_vel_cmd
        
        
        while not exit_flag:
            pygame.event.get()

            x_vel_cmd = -joystick.get_axis(1) * x_vel_max
            y_vel_cmd = -joystick.get_axis(0) * y_vel_max
            yaw_vel_cmd = -joystick.get_axis(3) * yaw_vel_max

            pygame.time.delay(100)

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()
class cmd:
    vx = 1.0
    vy = 0.0
    dyaw = 0.0
    
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data,model):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = q[:3]
    foot_positions = []
    foot_forces = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if '6_link' in body_name: 
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces.append(data.cfrc_ext[i][2].copy().astype(np.double)) 
    
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q ) * kp + (target_dq - dq)* kd
    return torque_out


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    
    model.opt.timestep = cfg.sim_config.dt
    
    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions  # This should match the number of actuated joints in your model
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos

    mujoco.mj_step(model, data)
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -45
    viewer.cam.lookat[:] =np.array([0.0,-0.25,0.824])

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
   
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)
    
    stop_state_log = 4000

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    for _ in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):

        # Obtain an observation
        q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data,model)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]
        
        base_z = base_pos[2]
        foot_z = foot_positions
        foot_force_z = foot_forces

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / cfg.rewards.cycle_time)
            obs[0, 2] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel
            obs[0, 5:17] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:41] = action
            obs[0, 41:44] = omega
            obs[0, 44:47] = eu_ang

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action * cfg.control.action_scale

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        
        data.ctrl = tau
        applied_tau = data.actuator_force

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
        idx = 5
        dof_pos_target = target_q + cfg.robot_config.default_dof_pos
        if _ < stop_state_log:
            logger.log_states(
                {   
                    'base_height' : base_z,
                    'foot_z_l' : foot_z[0],
                    'foot_z_r' : foot_z[1],
                    'foot_forcez_l' : foot_force_z[0],
                    'foot_forcez_r' : foot_force_z[1],
                    'base_vel_x': v[0],
                    'command_x': x_vel_cmd,
                    'base_vel_y': v[1],
                    'command_y': y_vel_cmd,
                    'base_vel_z': v[2],
                    'base_vel_yaw': omega[2],
                    'command_yaw': yaw_vel_cmd,
                    'dof_pos_target': dof_pos_target[idx] ,
                    'dof_pos': q[idx],
                    'dof_vel': dq[idx],
                    'dof_torque': applied_tau[idx],
                    'cmd_dof_torque': tau[idx],
                    'dof_pos_target[0]': dof_pos_target[0].item(),
                    'dof_pos_target[1]': dof_pos_target[1].item(),
                    'dof_pos_target[2]': dof_pos_target[2].item(),
                    'dof_pos_target[3]': dof_pos_target[3].item(),
                    'dof_pos_target[4]': dof_pos_target[4].item(),
                    'dof_pos_target[5]': dof_pos_target[5].item(),
                    'dof_pos_target[6]': dof_pos_target[6].item(),
                    'dof_pos_target[7]': dof_pos_target[7].item(),
                    'dof_pos_target[8]': dof_pos_target[8].item(),
                    'dof_pos_target[9]': dof_pos_target[9].item(),
                    'dof_pos_target[10]': dof_pos_target[10].item(),
                    'dof_pos_target[11]': dof_pos_target[11].item(),
                    'dof_pos':    q[0].item(),
                    'dof_pos[0]': q[0].item(),
                    'dof_pos[1]': q[1].item(),
                    'dof_pos[2]': q[2].item(),
                    'dof_pos[3]': q[3].item(),
                    'dof_pos[4]': q[4].item(),
                    'dof_pos[5]': q[5].item(),
                    'dof_pos[6]': q[6].item(),
                    'dof_pos[7]': q[7].item(),
                    'dof_pos[8]': q[8].item(),
                    'dof_pos[9]': q[9].item(),
                    'dof_pos[10]': q[10].item(),
                    'dof_pos[11]': q[11].item(),
                    'dof_torque': applied_tau[0].item(),
                    'dof_torque[0]': applied_tau[0].item(),
                    'dof_torque[1]': applied_tau[1].item(),
                    'dof_torque[2]': applied_tau[2].item(),
                    'dof_torque[3]': applied_tau[3].item(),
                    'dof_torque[4]': applied_tau[4].item(),
                    'dof_torque[5]': applied_tau[5].item(),
                    'dof_torque[6]': applied_tau[6].item(),
                    'dof_torque[7]': applied_tau[7].item(),
                    'dof_torque[8]': applied_tau[8].item(),
                    'dof_torque[9]': applied_tau[9].item(),
                    'dof_torque[10]': applied_tau[10].item(),
                    'dof_torque[11]': applied_tau[11].item(),
                    'dof_vel': dq[0].item(),
                    'dof_vel[0]': dq[0].item(),
                    'dof_vel[1]': dq[1].item(),
                    'dof_vel[2]': dq[2].item(),
                    'dof_vel[3]': dq[3].item(),
                    'dof_vel[4]': dq[4].item(),
                    'dof_vel[5]': dq[5].item(),
                    'dof_vel[6]': dq[6].item(),
                    'dof_vel[7]': dq[7].item(),
                    'dof_vel[8]': dq[8].item(),
                    'dof_vel[9]': dq[9].item(),
                    'dof_vel[10]': dq[10].item(),
                    'dof_vel[11]': dq[11].item(),
                }
                )
        
        elif _== stop_state_log:
            logger.plot_states()

    viewer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True, help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(ZqSA01Cfg):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/zq_humanoid/urdf/zq_sa01.xml'
            sim_duration = 120.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = np.array([50, 50, 70, 70, 20, 20, 50, 50, 70, 70, 20, 20], dtype=np.double)
            kds = np.array([5.0, 5.0, 7.0, 7.0, 2, 2, 5.0, 5.0, 7.0, 7.0, 2, 2], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)
            default_dof_pos = np.array([0.0, 0.0, -0.24, 0.48, -0.24, 0.0, 0.0, 0.0, -0.24, 0.48, -0.24, 0.0])

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
