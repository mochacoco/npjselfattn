import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import math
import torch

def make_task(count, tasknum, freq):

    if tasknum == 1:
        return [-0.1 + 0.1 * np.cos(1 * count / freq), 0.1 * np.sin(1 * count / freq), 0]
    elif tasknum == 2:
        return [0, -0.1 + 0.1 * np.cos(1 * count / freq), 0.1 * np.sin(1 * count / freq)]
    elif tasknum == 3:
        return [-0.1 + 0.1 * np.cos(1 * count / freq), 0, 0.1 * np.sin(1 * count / freq)]
    elif tasknum == 4:
        return [-0.3 + 0.3 * np.cos(1 * count / freq), - 0.3 + 0.3 * np.cos(1 * count / freq),
                     -0.25 + 0.25 * np.cos(1 * count / freq)]
    elif tasknum == 5:
        return [-0.3 + 0.3 * np.cos(1 * count / freq), 0.3 - 0.3 * np.cos(1 * count / freq),
                     -0.25 + 0.25 * np.cos(1 * count / freq)]
    elif tasknum == 11:
        return [-0.6, - 0.6, -0.5]
    else:
        return [0,0,0]
        print('Zero Regulation')

def compensate_angle(xi, num_output):
    if num_output == 6:
        if xi[3] >= 0:
            xi[3] -= 6.28
        if xi[5] >= 0:
            xi[5] -= 6.28
        xi[3:6] = xi[3:6]/ 20 #20
    #xi[0:3] *= 10
    return xi

def plot_result(tasknum, data_log, count):
    if tasknum == 1:
        plt.plot(data_log[0:count, 0], data_log[0:count, 1], 'red')
        plt.plot(data_log[0:count, 3], data_log[0:count, 4], 'blue')
    elif tasknum == 2:
        plt.plot(data_log[0:count, 1], data_log[0:count, 2], 'red')
        plt.plot(data_log[0:count, 4], data_log[0:count, 5], 'blue')
    elif tasknum == 3:
        plt.plot(data_log[0:count, 0], data_log[0:count, 2], 'red')
        plt.plot(data_log[0:count, 3], data_log[0:count, 5], 'blue')
    elif tasknum >= 4:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(data_log[0:count, 0], data_log[0:count, 1], data_log[0:count, 2], 'red')
        ax.plot(data_log[0:count, 3], data_log[0:count, 4], data_log[0:count, 5], 'blue')
    plt.show()


def getcurrentJointPosVel(robot):
    cur_joint_states = p.getJointStates(robot.kukaUid, [0,1,2,3,4,5,6])
    cur_joint_pos = [cur_joint_states[i][0] for i in range(7)]
    cur_joint_vel = [cur_joint_states[i][1] for i in range(7)]
    cur_joint_force = [cur_joint_states[i][2] for i in range(7)]
    return cur_joint_pos, cur_joint_vel, cur_joint_force