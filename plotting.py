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
    
    def plot_pos_xy(self, axes:plt.Axes, with_colorbar=False, other_traj=0, linestyle="", label=""):
        if with_colorbar:
            traj_linesegment = np.stack((self._pos[:-1, :2], self._pos[1:, :2]), axis=1)
            vel_min = np.min(self._vel[:-1,3])
            vel_max = np.max(self._vel[:-1,3])
            norm = plt.Normalize(vel_min, vel_max)
            traj_linecollection = LineCollection(traj_linesegment, cmap="jet", linewidth=3, norm=norm, linestyles='-', label="Traj: Time-optimal")
            traj_linecollection.set_array(self._vel[:-1, 3])
            line = axes.add_collection(traj_linecollection)
            if other_traj != 0:
                other_traj_linesegment = np.stack((other_traj._pos[:-1, :2], other_traj._pos[1:, :2]), axis=1)
                traj_linecollection = LineCollection(other_traj_linesegment, cmap="jet", linewidth=3, norm=norm, linestyles=":", label="Traj: Warm-up")
                traj_linecollection.set_array(other_traj._vel[:-1, 3])
                line = axes.add_collection(traj_linecollection)
            plt.legend(loc="upper right")
            plt.colorbar(line, label='Velocity [m/s]')
        else:
            axes.plot(self._pos[:,0], self._pos[:,1], linewidth=3, linestyle=linestyle, label=label)
            plt.legend()
        axes.set_xlim([-10, 10])
        axes.set_ylim([-10, 10])
        axes.set_xlabel("X [m]", labelpad=5)
        axes.set_ylabel("Y [m]", labelpad=-10)
        axes.set_aspect("equal")
    
    def plot_pos_3d(self, ax_3d):
        ax_3d.set_xlim((-10.5,10.5))
        ax_3d.set_ylim((-10.5,10.5))
        ax_3d.set_zlim((-4.5,0.5))
        # ax_3d.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        ax_3d.set_zticks([-4.0, -2.0, 0.0])
        ax_3d.tick_params(pad=0, labelrotation=0)
        ax_3d.set_xlabel("X [m]", labelpad=8, rotation=0)
        ax_3d.set_ylabel("Y [m]", labelpad=8, rotation=0)
        ax_3d.set_zlabel("Z [m]", labelpad=-2, rotation=0)
        ax_3d.view_init(elev=200, azim=-15)

        ax_3d.plot(self._pos[:,0], self._pos[:,1], self._pos[:,2])
        ax_3d.set_aspect("equal")

class Gates():
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
            axes.plot(self._shapes[idx][0], self._shapes[idx][1], self._shapes[idx][2], linewidth=3, color="dimgray")

    def plot2d(self, axes:plt.Axes):
        for idx in range(self._N):
            axes.plot(self._shapes[idx][0], self._shapes[idx][1], linewidth=3, color="dimgray")

if __name__ == "__main__":

    traj = Trajectory("./results/res_n8.csv")
    traj_t = Trajectory("./results/res_t_n8.csv")
    traj_track = Trajectory("./results/res_track_n8.csv")
    rpg_n6 = Trajectory("./rpg_results/result_n6.csv")
    gates = Gates("./gates/gates_n8.yaml")

    fig = plt.figure("3d", figsize=(10,6))
    ax_3d = fig.add_subplot([0,0,1,1], projection="3d")

    gates.plot3d(ax_3d)
    # traj.plot_pos_3d(ax_3d)
    traj_t.plot_pos_3d(ax_3d)

    # fig.savefig("test.eps", dpi=600)

    fig2 = plt.figure("gate")
    ax_xy = fig2.add_subplot()
    gates.plot2d(ax_xy)
    # traj_t.plot_pos_xy(ax_xy, with_colorbar=True, other_traj=traj)
    traj_t.plot_pos_xy(ax_xy, linestyle="--", label="Planner")
    # rpg_n6.plot_pos_xy(ax_xy, linestyle="--", label="CPC")
    # traj_t.plot_pos_xy(ax_xy)
    traj_track.plot_pos_xy(ax_xy, with_colorbar=True)

    plt.show()
