from datetime import datetime
import os


def get_title_short():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(59):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 23:
            label = 'n_vel'
            i = 0
        elif k == 35:
            label = 'n_act'
            i = 0
        elif k == 47:
            label = 'sc_act'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title


def get_title_long():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(95):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 23:
            label = 'n_vel'
            i = 0
        elif k == 35:
            label = 'n_act'
            i = 0
        elif k == 47:
            label = 'r_pos'
            i = 0
        elif k == 59:
            label = 'r_vel'
            i = 0
        elif k == 71:
            label = 'r_act'
            i = 0
        elif k == 83:
            label = 'uf_act'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title

def get_title_ankle_follow_check():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(103):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 23:
            label = 'n_vel'
            i = 0
        elif k == 35:
            label = 'n_act'
            i = 0
        elif k == 47:
            label = 'r_pos'
            i = 0
        elif k == 59:
            label = 'r_vel'
            i = 0
        elif k == 71:
            label = 'r_act'
            i = 0
        elif k == 83:
            label = 'uf_act'
            i = 0
        elif k == 95:
            label = 'ankle_joint_cmd'
            i = 0
        elif k == 99:
            label = 'ankle_joint_state'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title

def get_title_delay():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(94):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 23:
            label = 'n_vel'
            i = 0
        elif k == 35:
            label = 'n_act'
            i = 0
        elif k == 47+0:
            label = 'd_sin'
            i = 0
        elif k == 47+2:
            label = 'd_cmd'
            i = 0
        elif k == 47+5:
            label = 'd_omg'
            i = 0
        elif k == 47+8:
            label = 'd_eul'
            i = 0
        elif k == 47+11:
            label = 'd_pos'
            i = 0
        elif k == 47+23:
            label = 'd_vel'
            i = 0
        elif k == 47+35:
            label = 'd_act'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title

def get_title_tau_mapping():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(127):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 23:
            label = 'n_vel'
            i = 0
        elif k == 35:
            label = 'n_act'
            i = 0
        elif k == 47:
            label = 'r_pos'
            i = 0
        elif k == 59:
            label = 'r_vel'
            i = 0
        elif k == 71:
            label = 'r_act'
            i = 0
        elif k == 83:
            label = 'uf_act'
            i = 0
        elif k == 95:
            label = 'tau_joint_state'
            i = 0
        elif k == 107:
            label = 'tau_joint_cmd'
            i = 0
        elif k == 119:
            label = 'ankle_joint_pos_cmd'
            i = 0
        elif k == 123:
            label = 'ankle_joint_pos'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title

def get_title_tau_mapping_5dof():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(127):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 21:
            label = 'n_vel'
            i = 0
        elif k == 31:
            label = 'n_act'
            i = 0
        elif k == 41:
            label = 'r_pos'
            i = 0
        elif k == 53:
            label = 'r_vel'
            i = 0
        elif k == 65:
            label = 'r_act'
            i = 0
        elif k == 77:
            label = 'uf_act'
            i = 0
        elif k == 89:
            label = 'tau_joint_state'
            i = 0
        elif k == 101:
            label = 'tau_joint_cmd'
            i = 0
        elif k == 113:
            label = 'ankle_joint_pos_cmd'
            i = 0
        elif k == 117:
            label = 'ankle_joint_pos'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title

def get_title_5dof_deploy():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(101):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 21:
            label = 'n_vel'
            i = 0
        elif k == 31:
            label = 'n_act'
            i = 0
        elif k == 41:
            label = 'r_pos'
            i = 0
        elif k == 53:
            label = 'r_vel'
            i = 0
        elif k == 65:
            label = 'r_act'
            i = 0
        elif k == 77:
            label = 'tau_joint_state'
            i = 0
        elif k == 89:
            label = 'tau_joint_cmd'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title

def get_title_6dof_deploy():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(107):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 23:
            label = 'n_vel'
            i = 0
        elif k == 35:
            label = 'n_act'
            i = 0
        elif k == 47:
            label = 'r_pos'
            i = 0
        elif k == 59:
            label = 'r_vel'
            i = 0
        elif k == 71:
            label = 'r_act'
            i = 0
        elif k == 83:
            label = 'tau_joint_state'
            i = 0
        elif k == 95:
            label = 'tau_joint_cmd'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title
def get_title_5dof_play():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(81):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 21:
            label = 'n_vel'
            i = 0
        elif k == 31:
            label = 'n_act'
            i = 0
        elif k == 41:
            label = 'scaled_act'
            i = 0
        elif k == 51:
            label = 'n_vel2'
            i = 0
        elif k == 61:
            label = 'tau_cmd'
            i = 0
        elif k == 71:
            label = 'tau_state'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title
def get_title_6dof_play():
    i = 0
    label = 'No,Time,'
    title = ''
    title += f'{label}'
    for k in range(95):
        if k == 0:
            label = 'sin'
            i = 0
        elif k == 2:
            label = 'cmd'
            i = 0
        elif k == 5:
            label = 'omg'
            i = 0
        elif k == 8:
            label = 'eul'
            i = 0
        elif k == 11:
            label = 'n_pos'
            i = 0
        elif k == 23:
            label = 'n_vel'
            i = 0
        elif k == 35:
            label = 'n_act'
            i = 0
        elif k == 47:
            label = 'scaled_act'
            i = 0
        elif k == 59:
            label = 'n_vel2'
            i = 0
        elif k == 71:
            label = 'tau_cmd'
            i = 0
        elif k == 83:
            label = 'tau_state'
            i = 0
        title += f'{label}_{i},'
        i += 1
    return title

class SimpleLogger:
    def __init__(self, path, title):
        now = datetime.now()
        # 将当前时间格式化为字符串
        formatted_time = now.strftime('%Y-%m-%d_%H:%M:%S')
        filename = f"{path}/log_{formatted_time}.csv"
        if not os.path.exists(path):
            os.mkdir(path)
        self.file = open(filename, "w")
        print(f"Saving log! Path: {filename}")
        self.file.write(f'{title}\n')

    def save(self, obs, step, time):
        for row in obs:
            k = 0
            self.file.write('%d,%d,' % (step, int(time * 10 ** 3)))  # us
            for index, item in enumerate(row):
                self.file.write(' %.4f,' % item)
                k += 1

            self.file.write('\n')
            break

    def close(self):
        self.file.close()
