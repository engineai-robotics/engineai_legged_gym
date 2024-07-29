
import torch
import numpy as np
from matplotlib import pyplot as plt


class ActionGenerator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = 1
        self.num_dof = 12
        self.device = "cpu"
        self.episode_length_buf = np.zeros(
            self.num_envs, dtype=np.double)
        self.dt = 0.005
        self.target_joint_pos_scale = 0.2
        # default_dof_pos： 所有电机初始位置都是0
        # self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = np.array(self.cfg.env.default_dof_pos)
        self.dof_pos = np.zeros(self.num_dof, dtype=np.double)
        self.ref_dof_pos = np.zeros_like(self.dof_pos)
        self.actions = np.zeros(self.num_dof, dtype=np.double)


    def _get_phase(self):
        cycle_time = self.cfg.env.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        stance_mask[torch.abs(sin_pos) < 0.1] = 1  # 双脚同时接地的时刻
        # print('phase={:.3f} sin={:.3f} step= {:.1f} left= {:.1f}  right= {:.1f}'.format(phase[0], sin_pos[0], self.episode_length_buf[0], stance_mask[0, 0], stance_mask[0, 1]))
        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.target_joint_pos_scale
        scale_2 = 2 * scale_1
        self.ref_dof_pos[0] = self.default_dof_pos[0]
        self.ref_dof_pos[1] = self.default_dof_pos[1]
        self.ref_dof_pos[6] = self.default_dof_pos[6]
        self.ref_dof_pos[7] = self.default_dof_pos[7]
        # left swing
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[2] = - sin_pos_l * scale_1 + self.default_dof_pos[2]
        self.ref_dof_pos[3] = sin_pos_l * scale_2 + self.default_dof_pos[3]
        self.ref_dof_pos[4] = -sin_pos_l * scale_1 + self.default_dof_pos[4]
        self.ref_dof_pos[5] = 0
        # right
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[8] = sin_pos_r * scale_1 + self.default_dof_pos[8]
        self.ref_dof_pos[9] = - sin_pos_r * scale_2 + self.default_dof_pos[9]
        self.ref_dof_pos[10] = sin_pos_r * scale_1 + self.default_dof_pos[10]
        self.ref_dof_pos[11] = 0

        # self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0.

    def calibrate(self, joint_pos):
        # 将所有电机缓缓重置到初始状态
        target = joint_pos - self.default_dof_pos
        # joint_pos[target > 0.03] -= 0.03
        # joint_pos[target < -0.03] += 0.03
        joint_pos[target > 0.01] -= 0.01
        joint_pos[target < -0.01] += 0.01
        return 4 * joint_pos

    def step(self):
        self.compute_ref_state()
        self.episode_length_buf += 1
        actions = 4 * self.ref_dof_pos
        # delay = torch.rand((self.num_envs, 1), device=self.device)
        # actions = (1 - delay) * actions + delay * self.actions
        return actions


if __name__ == '__main__':
    # ref_dof_pos = torch.zeros((1, 12), dtype=torch.float, device='cpu', requires_grad=False)  # 步态生成器-生成的参考姿势
    # ref_count = torch.zeros(1, device='cpu', dtype=torch.long)  # 步态生成器--计数器
    # for i in range(200):
    #
    #     phase = (ref_count * 0.01 / 0.66) % 1.
    #     print(i, phase, end='')
    #     sin_pos = torch.sin(2 * torch.pi * phase)
    #     sin_pos_r = sin_pos.clone()
    #     sin_pos_l = sin_pos.clone()
    #     scale_1 = 0.3
    #     scale_2 = 2 * scale_1
    #
    #
    #     sin_pos_r[sin_pos_r < 0] = 0
    #     ref_dof_pos[:, 2] = sin_pos_r * scale_1 + 0.
    #     ref_dof_pos[:, 3] = -sin_pos_r * scale_2 + 0.
    #     ref_dof_pos[:, 4] = sin_pos_r * scale_1 + 0.
    #     # left foot stance phase set to default joint pos
    #     sin_pos_l[sin_pos_l > 0] = 0
    #     ref_dof_pos[:, 8] = -sin_pos_l * scale_1 + 0.
    #     ref_dof_pos[:, 9] = sin_pos_l * scale_2 + 0.
    #     ref_dof_pos[:, 10] = -sin_pos_l * scale_1 + 0.
    #
    #     ref_dof_pos[torch.abs(sin_pos) < 0.1, :] = 0.
    #     # ref_dof_pos[:, :] +=
    #     print(sin_pos[0], ref_dof_pos[0, [2, 3, 4, 8, 9, 10]])
    #     ref_count += 1
    #
    # target_q = np.array([-1.5, 1.25, -5.15, 3.2, -2.5, 1.8, -3.5, 6.28, -7.15, 4.2, -9.8, -0.5], dtype=np.float32)
    # joint_limit_min = np.array([-0.5, -0.25, -1.15, -2.2, -0.5, -0.8, -0.5, -0.28, -1.15, -2.2, -0.8, -0.5], dtype=np.float32)
    # joint_limit_max = np.array([0.5, 0.25, 1.15, -0.05, 0.8, 0.5, 0.5, 0.28, 1.15, -0.05, 0.5, 0.8], dtype=np.float32)
    # print(np.clip(target_q, joint_limit_min, joint_limit_max))


    # 定义phase的取值范围
    i_values = torch.linspace(0, 1000, 1000)
    phase = i_values * 0.01 * 1.5 * 2.

    # 生成mask_right和mask_left
    mask_right = (torch.floor(phase) + 1) % 2
    mask_left = torch.floor(phase) % 2

    # 计算cos_pos
    cos_pos = (1 - torch.cos(2 * torch.pi * phase)) / 2

    # 生成r和l
    vr = cos_pos * mask_right
    vl = cos_pos * mask_left

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(i_values, vr.numpy(), label='r')
    plt.plot(i_values, vl.numpy(), label='l')
    plt.title('曲线图')
    plt.xlabel('i')
    plt.ylabel('值')
    plt.legend()
    plt.grid(True)
    plt.show()


