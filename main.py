import pybullet as p
from kuka import Kuka
from ddp_agent import DGPS
from baseline_SAC.py import SAC_agent
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
from parsing import make_parser
from sklearn.metrics import mean_squared_error
from utils import *
import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch.autograd import Variable


def initialize_engine():
    args = make_parser()
    physicsClient = p.connect(p.GUI) if args.gui else p.connect(p.DIRECT)
    np.random.seed(0)
    p.setGravity(0,0,-9.81)
    p.setTimeStep(0.001)
    p.setPhysicsEngineParameter(fixedTimeStep = 0.001)
    p.setRealTimeSimulation(0)

def initialize_robot(robot):
    for i in range(7):
        p.enableJointForceTorqueSensor(robot.kukaUid, i)
    p.enableJointForceTorqueSensor(robot.kukaUid, robot.kukaEndEffectorIndex)

def main():
    args = make_parser()
    device = args.device
    initialize_engine()
    agent = DGPS()
    robot = Kuka()
    initialize_robot(robot)


    data_log = torch.empty((200000,40), device = device)
    time_start = time.time()
    EF_init = (np.array(robot.endEffectorPos) + [0.163, 0, -0.1]).tolist()
    print('Desired Init:', EF_init)
    jointPoses = p.calculateInverseKinematics(robot.kukaUid, robot.kukaEndEffectorIndex, EF_init, p.getQuaternionFromEuler([0, -3.14, 0]))
    u0 = np.array(jointPoses[0:6])
    u0[4:6] = 0
    u0 = torch.from_numpy(u0).to(device)
    u = torch.zeros(6).to(device)
    pert = torch.zeros(6).to(device)

    print('Init Joint:', u0)

    for t in range(5000):
        for i in range(6):
            p.setJointMotorControl2(robot.kukaUid, i, p.POSITION_CONTROL, u0[i].item(), positionGain=1.5, velocityGain=1.0, force = 500)
        p.setJointMotorControl2(robot.kukaUid, 6, p.POSITION_CONTROL, 0, positionGain=1.5, velocityGain=1.0, force = 500)
        p.stepSimulation()
    u0 = []
    for i in range(7):
        _, _, _, u0_ = p.getJointState(robot.kukaUid, i)
        u0.append(u0_)
    print('Now the input forces are:',u0)
    p.setJointMotorControlArray(robot.kukaUid, [0,1,2,3,4,5,6], p.VELOCITY_CONTROL, forces = 0.0*np.ones(7))

    for t in range(5000):
        p.setJointMotorControlArray(robot.kukaUid, [0,1,2,3,4,5,6] ,p.TORQUE_CONTROL, forces = u0)
        p.stepSimulation()

    x = torch.zeros((args.output_dim)).to(device)
    xdes = torch.zeros(args.output_dim).to(device)

    [cur_joint_pos, vraw, _] = getcurrentJointPosVel(robot)
    q0 = torch.from_numpy(np.array(cur_joint_pos[0:6]).ravel()).to(torch.float32).to(device)
    agent.make_initial_qdes(q0.reshape([-1,1]))
    qv = torch.from_numpy(np.array(vraw[0:6]).ravel()).to(torch.float32).to(device)

    _, _, _, _, x00, x01 = p.getLinkState(robot.kukaUid, robot.kukaEndEffectorIndex, computeForwardKinematics=True)
    x02 = p.getEulerFromQuaternion(x01)
    x0 = torch.from_numpy(np.array(x00 + x02)).to(torch.float32).to(device)

    x0[3:6] = x0[3:6] / 20
    v = torch.zeros(6).to(device)
    x = torch.zeros(6).to(device)
    q = torch.zeros(6).to(device)
    q_prev = torch.zeros(6).to(device)
    f = torch.zeros(6).to(device)


    print('Initial Joint Torque Info.:',u0)
    print('Start')
    error = torch.zeros((args.output_dim,1)).to(device)
    sf = args.sfinit if not args.w_reuse else 0.05
    count = -1


    while 1:
        count += 1
        time_now = time.time() - time_start
        if count % 1000 == 0:
            print('Simul_count:',count,'Safety factor:{:.3f}'.format(sf))
        if count >= args.simcount:
            break
        if args.control == 'admittance':
            if torch.norm(f) <= 0.1:
                try:
                    xref = xdes
                except:
                    xref = torch.zeros(6)
            else:
                xref = q
            xdes = xref
        else:
            xdes[0:3] = torch.tensor(make_task(count, args.tasknum, freq=400)).to(device)

        u, sf, attn_u, attn_q, attnloss, func_j = agent.generate_action(u, q.view(-1,1), x[0:args.output_dim].view(-1,1), qv.view(-1,1),
                                      xdes[0:args.output_dim].view(-1,1), v.view(-1,1), f.view(-1,1), sf)

        for i in range(6):
            p.setJointMotorControl2(robot.kukaUid, i, p.TORQUE_CONTROL, force = u0[i] + u[i].item())
        p.stepSimulation()
        [xraw, vraw, fraw] = getcurrentJointPosVel(robot)
        f = torch.tanh(torch.from_numpy(np.array(fraw[6]).ravel()).to(torch.float32).to(device)) - 0.5
        q_2prev = q_prev
        q_prev = q
        q = torch.from_numpy(np.array(xraw[0:6]).ravel()).to(torch.float32).to(device)
        q = q - q0
        v_prev = qv
        qv = torch.tanh((q-q_prev) * 1000) - 0.5
        qa = torch.tanh((q - 2*q_prev + q_2prev) * 1000) - 0.5
        _, _, imsi, _, x00, x01 = p.getLinkState(robot.kukaUid, robot.kukaEndEffectorIndex, computeForwardKinematics=True)
        x02 = p.getEulerFromQuaternion(x01)
        R01 = p.getMatrixFromQuaternion(x01)
        xi = torch.from_numpy(np.array(x00 + x02)).to(torch.float32).to(device)
        xii = xi
        xi = compensate_angle(xi, args.output_dim)
        x_prev = x
        x = (xi - x0)
        v = torch.tanh((x - x_prev)*1000) - 0.5

        func_j_comp, _ = p.calculateJacobian(robot.kukaUid, robot.kukaEndEffectorIndex, imsi, xraw, [0,0,0,0,0,0,0], [0,0,0,0,0,0,0])
        func_j_comp = FT(np.array(func_j_comp)).to(device)


        agent.save_buffer(agent.Ksignal, qv.view(-1), qa.view(-1), torch.tanh(u.view(-1)), x.view(-1), q.view(-1), f.view(-1), torch.tensor(R01).to(device), xii.view(-1))
        if count % 50 == 0:
            xk, lossA, lossB, costerror_, costerror_kin = agent.train(args.learn, args.control)


        #xp = predictor.train(q.ravel(), x.ravel())

        if count % 10 == 0:
            data_log[count // 10, 0:3] = x[0:3]
            data_log[count // 10, 3:6] = xdes[0:3]
            data_log[count // 10, 6] = sf
            data_log[count // 10, 7] = attn_u
            data_log[count // 10, 8] = lossA
            mse_loss = nn.MSELoss()
            data_log[count // 10, 9] = mse_loss(data_log[count // 10, 0:3], data_log[count // 10, 3:6])
            data_log[count // 10, 10] = attn_q
            data_log[count // 10, 12] = count * 0.001
            data_log[count // 10, 15] = lossB
            data_log[count // 10, 16:19] = agent.q_des[0:3].view(-1)

            data_log[count // 10, 19:22] = xk[-1,0:3]
            data_log[count // 10, 22:28] = q
            data_log[count // 10, 29] = costerror_
            data_log[count // 10, 30] = mse_loss(func_j[0:3,:], func_j_comp[:,0:6])
            print(data_log[count // 10, 30])
    print('Simulation Done')

    time.sleep(1)
    p.disconnect()
    data_log = data_log.cpu().detach().numpy()
    plot_result(args.tasknum, data_log, count)

    fig = plt.figure()
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2)
    ax3 = fig.add_subplot(4,1,3)
    ax4 = fig.add_subplot(4,1,4)
    """
    ax1.plot(data_log[0:count, 12], data_log[0:count,0], color = 'tab:blue')
    ax1.plot(data_log[0:count, 12], data_log[0:count, 16], color='tab:red')
    ax1.plot(data_log[0:count, 12], data_log[0:count, 19], color='tab:green')
    ax2.plot(data_log[0:count, 12], data_log[0:count, 1], color='tab:blue')
    ax2.plot(data_log[0:count, 12], data_log[0:count, 17], color='tab:red')
    ax2.plot(data_log[0:count, 12], data_log[0:count, 20], color='tab:green')
    ax3.plot(data_log[0:count, 12], data_log[0:count, 2], color='tab:blue')
    ax3.plot(data_log[0:count, 12], data_log[0:count, 18], color='tab:red')
    ax3.plot(data_log[0:count, 12], data_log[0:count, 21], color='tab:green')
    ax4.plot(data_log[0:count, 12], data_log[0:count, 9], color='black')
    """
    ax1.plot(data_log[0:count//10, 12], data_log[0:count//10, 7], color = 'tab:blue')
    #ax1.plot(data_log[10000:count, 12], data_log[10000:count, 14], color='skyblue')
    ax2.plot(data_log[0:count//10, 12], data_log[0:count//10, 10], color = 'tab:red')
    ax3.plot(data_log[0:count//10, 12], data_log[0:count//10, 6], color = 'tab:green')
    #ax3.plot(data_log[10000:count, 12], data_log[10000:count, 11], color='lime')
    ax4.plot(data_log[0:count//10, 12], data_log[0:count//10, 9], color='black')

    plt.show()


    #print('Error:',mean_squared_error(data_log[count-10000:count, 0:3], data_log[count-10000:count, 3:6]))
    pd.DataFrame(data_log[0:count//10,:]).to_csv("datalog/230516_jcss.csv")


    #agent.post_train(data_log[0:count, 0:3], data_log[0:count, 28:37])




if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        time.sleep(1)
        p.disconnect()
