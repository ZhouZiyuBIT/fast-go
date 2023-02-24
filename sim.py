import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import csv

from quadrotor import QuadrotorModel, QuadrotorSim
from tracker import TrackerOpt, TrackerOpt2

from trajectory import Trajectory, TrajLog
from gates.gates import Gates

import os, sys
BASEPATH = os.path.abspath(__file__).split('script', 1)[0]+'script/fast-go/'

traj = Trajectory(BASEPATH+"results/res_t_n6.csv")
traj_log = TrajLog(BASEPATH+"results/res_track_n6.csv")
gates = Gates(BASEPATH+"gates/gates_n6.yaml")
quad = QuadrotorModel(BASEPATH+'quad/quad.yaml')
tracker = TrackerOpt(quad)
q_sim = QuadrotorSim(quad)
q_sim._X[:3] = gates._pos[0]-1
tracker.define_opt()
tracker.reset_xul()
# 10s
plot_quad_xy = np.zeros((2,1000))
for t in range(1000):
    plot_quad_xy[0,t] = q_sim._X[0]
    plot_quad_xy[1,t] = q_sim._X[1]

    trjp = traj.sample(tracker._trj_N, q_sim._X[:3]).reshape(-1)
    if t>4990:
        print(trjp)
    res = tracker.solve(q_sim._X, trjp, 20)
    x = res['x'].full().flatten()
    
    # u = np.zeros(4)
    # u[0] = 1.0*(x[tracker._Herizon*13+0]+x[tracker._Herizon*13+1]+x[tracker._Herizon*13+2]+x[tracker._Herizon*13+3])/4
    # u[1] = x[10]
    # u[2] = x[11]
    # u[3] = x[12]
    # q_sim.step10ms(u)
    # q_sim.step10ms(u)
    # traj_log.log(t*0.01, q_sim._X[:13], u)

    q_sim._T[0] = x[10*13+0]
    q_sim._T[1] = x[10*13+1]
    q_sim._T[2] = x[10*13+2]
    q_sim._T[3] = x[10*13+3]
    for _ in range(10):
        q_sim.step1ms()
    traj_log.log(t*0.01, q_sim._X[:13], q_sim._T)
    
    print(x[-tracker._Herizon:])
    
plt.plot(traj._pos[:,0], traj._pos[:,1])
plt.plot(plot_quad_xy[0,:], plot_quad_xy[1,:])
plt.show()

