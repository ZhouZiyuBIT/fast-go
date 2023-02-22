import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import csv

from quadrotor import QuadrotorModel, QuadrotorSim
from tracker import TrackerOpt

from plotting import Trajectory, Gates

class TrajLog():
    def __init__(self, path):
        self._fd = open(path, 'w')
        self._traj_writer = csv.writer(self._fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        labels = ['t',
                  "p_x", "p_y", "p_z",
                  "v_x", "v_y", "v_z",
                  "q_w", "q_x", "q_y", "q_z",
                  "w_x", "w_y", "w_z",
                  "u_1", "u_2", "u_3", "u_4"]
        self._traj_writer.writerow(labels)
    def log(self, t, s, u):
        self._traj_writer.writerow([t, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], u[0], u[1], u[2], u[3]])

    def __del__(self):
        self._fd.close()

traj = Trajectory("./results/res_t_n8.csv")
traj_log = TrajLog("./results/res_track_n8.csv")
gates = Gates("./gates/gates_n8.yaml")
quad = QuadrotorModel('quad.yaml')
tracker = TrackerOpt(quad)
q_sim = QuadrotorSim(quad)
q_sim._X[:3] = gates._pos[0]-1
tracker.define_opt()
tracker.reset_xul()
# 10s
plot_quad_xy = np.zeros((2,5000))
for t in range(5000):
    plot_quad_xy[0,t] = q_sim._X[0]
    plot_quad_xy[1,t] = q_sim._X[1]

    trjp = traj.sample(tracker._trj_N, q_sim._X[:3]).reshape(-1)
    if t>4990:
        print(trjp)
    res = tracker.solve(q_sim._X, trjp)
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
    
ax = plt.gca()
traj.plot_pos_xy(ax)
plt.plot(plot_quad_xy[0,:], plot_quad_xy[1,:])
plt.show()

