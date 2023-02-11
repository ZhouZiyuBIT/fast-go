import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

from quadrotor import QuadrotorModel, QuadrotorSim
from tracker import TrackerOpt

# trajector
trjp = np.zeros(20*3)
trjyaw = np.zeros(20)
plot_trj_x = np.zeros(20)
plot_trj_y = np.zeros(20)
for i in range(20):
    trjp[3*i+0] = np.sin(i/15)*8
    trjp[3*i+1] = np.cos(i/3)*9-8
    trjp[3*i+2] = 0
    plot_trj_x[i] = trjp[3*i+0]
    plot_trj_y[i] = trjp[3*i+1]

quad = QuadrotorModel('quad.yaml')
tracker = TrackerOpt(quad)
q_sim = QuadrotorSim(quad)
tracker.define_opt()
tracker.reset_xul()

# 10s
plot_quad_xy = np.zeros((2,300))
for t in range(300):
    plot_quad_xy[0,t] = q_sim._X[0]
    plot_quad_xy[1,t] = q_sim._X[1]
    res = tracker.solve(q_sim._X, trjp, trjyaw)
    x = res['x'].full().flatten()
    u = np.zeros(4)
    u[0] = (x[tracker._Herizon*13+0]+x[tracker._Herizon*13+1]+x[tracker._Herizon*13+2]+x[tracker._Herizon*13+3])/4
    u[1] = x[10]
    u[2] = x[11]
    u[3] = x[12]
    q_sim.step10ms(u)
    q_sim.step10ms(u)
    # q_sim._T[0] = x[10*13+0]
    # q_sim._T[1] = x[10*13+1]
    # q_sim._T[2] = x[10*13+2]
    # q_sim._T[3] = x[10*13+3]
    # for _ in range(10):
    #     q_sim.step1ms()
    print(x[-tracker._Herizon:])
    
plt.plot(plot_trj_x, plot_trj_y)
plt.plot(plot_quad_xy[0,:], plot_quad_xy[1,:])
plt.show()

