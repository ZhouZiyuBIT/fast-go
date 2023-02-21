import numpy as np
import csv
import yaml
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Trajectory():
    def __init__(self, csv_f):
        t = []
        pos = []
        vel = []
        quaternion = []
        omega = []
        with open(csv_f, 'r') as f:
            traj_reader = csv.DictReader(f)
            for s in traj_reader:
                t.append(float(s['t']))
                pos.append([ float(s["p_x"]), float(s["p_y"]), float(s["p_z"]) ])
                vel.append([ float(s["v_x"]), float(s["v_y"]), float(s["v_z"]), np.sqrt(float(s["v_x"])*float(s["v_x"])+float(s["v_y"])*float(s["v_y"])+float(s["v_z"])*float(s["v_z"])) ])
                quaternion.append([ float(s["q_w"]), float(s["q_x"]), float(s["q_y"]), float(s["q_z"]) ])
                omega.append([ float(s["w_x"]), float(s["w_y"]), float(s["w_z"]) ])
        self._t = np.array(t)
        self._pos = np.array(pos)
        self._vel = np.array(vel)
        self._quaternion = np.array(quaternion)
        self._omega = np.array(omega)
        self._N = self._pos.shape[0]-1

    def sample(self, n, pos):
        idx = 0
        lmin = 1000000
        for i in range(self._N):
            l = np.linalg.norm(pos-self._pos[i])
            if l < lmin:
                lmin = l
                idx=i
        
        traj_seg = np.zeros((n, 3))
        for i in range(n):
            traj_seg[i,:] = self._pos[(idx+int(i*1.0))%self._N]
        return traj_seg
    
    def plot_pos_xy(self, axis, with_colorbar=False):
        if with_colorbar:
            traj_linesegment = np.stack((self._pos[:-1, :2], self._pos[1:, :2]), axis=1)
            vel_min = np.min(self._vel[:-1,3])
            vel_max = np.max(self._vel[:-1,3])
            norm = plt.Normalize(vel_min, vel_max)
            traj_linecollection = LineCollection(traj_linesegment, cmap="jet", linewidth=3, norm=norm)
            traj_linecollection.set_array(traj._vel[:-1, 3])
            line = axis.add_collection(traj_linecollection)
            fig2.colorbar(line)
        else:
            axis.plot(self._pos[:,0], self._pos[:,1])
        axis.set_xlim([-10, 10])
        axis.set_ylim([-10, 10])
        axis.set_aspect("equal")

class GatesShape():
    def __init__(self, yaml_f):
        with open(yaml_f, 'r') as f:
            gf = yaml.load(f, Loader=yaml.FullLoader)
            self._pos = np.array(gf["pos"])
            self._rot = np.array(gf["rot"])/180*np.pi
        self._N = self._pos.shape[0]
        
        R = 0.5
        self._shapes = []
        angles = np.linspace(0, 2*np.pi, 50)

        for idx in range(self._N):
            x = R*np.cos(angles)*np.cos(self._rot[idx])
            y = R*np.cos(angles)*np.sin(self._rot[idx])
            z = R*np.sin(angles)
            self._shapes.append([x+self._pos[idx][0], y+self._pos[idx][1], z+self._pos[idx][2]])
    
    def plot3d(self, axes):
        for idx in range(self._N):
            axes.plot(self._shapes[idx][0], self._shapes[idx][1], self._shapes[idx][2], linewidth=3, color="firebrick")
    
    def plot2d(self, axes):
        for idx in range(self._N):
            axes.plot(self._shapes[idx][0], self._shapes[idx][1], linewidth=3, color="firebrick")

if __name__ == "__main__":
    traj = Trajectory("./res.csv")
    gates = GatesShape("./gates.yaml")

    fig = plt.figure("3d")
    ax_3d = fig.add_subplot(projection="3d")
    ax_3d.set_xlim((-10,10))
    ax_3d.set_ylim((-10,10))
    ax_3d.set_zlim((-4,0))
    ax_3d.set_zticks([-4, -3, -2, -1, 0])
    ax_3d.set_xlabel("X[m]")
    ax_3d.set_ylabel("Y[m]")
    ax_3d.set_zlabel("Z[m]")
    ax_3d.view_init(elev=200, azim=-15)

    gates.plot3d(ax_3d)
    
    ax_3d.plot(traj._pos[:,0], traj._pos[:,1], traj._pos[:,2])

    ax_3d.set_aspect("equal")
    fig.savefig("test.eps", dpi=600)

    fig2 = plt.figure("gate")
    ax_xy = fig2.add_subplot()
    gates.plot2d(ax_xy)

    traj.plot_pos_xy(ax_xy, with_colorbar=True)
    
    plt.show()
