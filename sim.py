import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from quadrotor import QuadrotorModel, QuadrotorSim
from tracker import TrackerOpt

# trajector
trjyaw = np.zeros(20)

from plotting import Trajectory
traj = Trajectory("./res.csv")

quad = QuadrotorModel('quad.yaml')
tracker = TrackerOpt(quad)
q_sim = QuadrotorSim(quad)
tracker.define_opt()
tracker.reset_xul()

# 10s
plot_quad_xy = np.zeros((2,1000))
for t in range(1000):
    plot_quad_xy[0,t] = q_sim._X[0]
    plot_quad_xy[1,t] = q_sim._X[1]
    
    trjp = traj.sample(tracker._trj_N, q_sim._X[:3]).reshape(-1)
    res = tracker.solve(q_sim._X, trjp, trjyaw)
    x = res['x'].full().flatten()
    
    u = np.zeros(4)
    u[0] = (x[tracker._Herizon*13+0]+x[tracker._Herizon*13+1]+x[tracker._Herizon*13+2]+x[tracker._Herizon*13+3])/4
    u[1] = x[10]
    u[2] = x[11]
    u[3] = x[12]
    q_sim.step10ms(u)

    # q_sim._T[0] = x[10*13+0]
    # q_sim._T[1] = x[10*13+1]
    # q_sim._T[2] = x[10*13+2]
    # q_sim._T[3] = x[10*13+3]
    # for _ in range(10):
    #     q_sim.step1ms()
    
    print(x[-tracker._Herizon:])
    
ax = plt.gca()
traj.plot_pos_xy(ax)
plt.plot(plot_quad_xy[0,:], plot_quad_xy[1,:])
plt.show()

