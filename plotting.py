import numpy as np
import csv
import yaml
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
# sns.set()

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
        ax_3d.set_aspect("equal")

        ax_3d.plot(self._pos[:,0], self._pos[:,1], self._pos[:,2])

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

def plot_gates_3d(axes:plt.Axes, gates:Gates):
    for idx in range(gates._N):
        axes.plot(gates._shapes[idx][0], gates._shapes[idx][1], gates._shapes[idx][2], linewidth=3, color="dimgray")

def plot_gates_2d(axes:plt.Axes, gates:Gates):
    for idx in range(gates._N):
        axes.plot(gates._shapes[idx][0], gates._shapes[idx][1], linewidth=3, color="dimgray")

def plot_traj_xy(axes:plt.Axes, traj:Trajectory, linewidth=1, linestyle="-", label=""):
    axes.plot(traj._pos[:,0], traj._pos[:,1], linewidth=linewidth, linestyle=linestyle, label=label)

def plot_traj_3d(axes3d, gates:Gates):
    axes3d.plot(gates._pos[:,0], gates._pos[:,1], gates._pos[:,2])

def plot_tracked(gates:Gates, traj_planned:Trajectory, traj_tracked:Trajectory):
    fig = plt.figure()

    ax = fig.add_subplot()
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("X [m]", labelpad=5)
    ax.set_ylabel("Y [m]", labelpad=-10)
    ax.set_aspect("equal")

    plot_gates_2d(ax, gates)
    
    plot_traj_xy(ax, traj_tracked, linewidth=2, linestyle="-", label="Traj: Tracked")
    plot_traj_xy(ax, traj_planned, linewidth=3, linestyle="--", label="Traj: Planned")

        # inset axes....
    axins = ax.inset_axes([0.56, 0.22, 0.28, 0.28])
    plot_traj_xy(axins, traj_tracked, linewidth=2, linestyle="-", label="Traj: Tracked")
    plot_traj_xy(axins, traj_planned, linewidth=3, linestyle="--", label="Traj: Planned")
    plot_gates_2d(axins, gates)
    # subregion of the original image
    x1, x2, y1, y2 = 7.7, 8.8, 5.2, 6.3
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="gray")
    ax.legend()

def plot_track_vel(gates:Gates, traj_planned:Trajectory, traj_tracked:Trajectory, first_gate_pos):
    loop_idx = []
    flag1 = 0
    flag2 = 0
    for i in range(traj_tracked._N):
        if np.linalg.norm(traj_tracked._pos[i]-first_gate_pos)< 0.5:
            if flag1 == 0:
                l = np.linalg.norm(traj_tracked._pos[i] - first_gate_pos)
                flag1 = 1
            else:
                if l<np.linalg.norm(traj_tracked._pos[i] - first_gate_pos):
                    if flag2 == 0:
                        loop_idx.append(i)
                        flag2 = 1
            l = np.linalg.norm(traj_tracked._pos[i] - first_gate_pos)
        else:
            flag1 = 0
            flag2 = 0
    
    gs = GridSpec(20,21)
    fig = plt.figure()

    ax1 = fig.add_subplot(gs[:,:10])
    ax1.set_xlim([-10, 10])
    ax1.set_ylim([-10, 10])
    ax1.set_xlabel("X [m]", labelpad=5)
    ax1.set_ylabel("Y [m]", labelpad=-10)
    ax1.set_aspect("equal")
    plot_gates_2d(ax1, gates)
    plot_traj_xy(ax1, traj_planned, linestyle='--', linewidth=3, label="Traj: Planned")

    ax2 = fig.add_subplot(gs[:,10:])
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([-10, 10])
    ax2.set_xlabel("X [m]", labelpad=5)
    ax2.set_ylabel("Y [m]", labelpad=-10)
    ax2.set_aspect("equal")
    plot_gates_2d(ax2, gates)
    plot_traj_xy(ax2, traj_planned, linestyle='--', linewidth=3, label="Traj: Planned")
    
    vel_min = np.min(traj_tracked._vel[:-1,3])
    vel_max = np.max(traj_tracked._vel[:-1,3])
    norm = plt.Normalize(vel_min, vel_max)

    traj1_seg = np.stack((traj_tracked._pos[:loop_idx[1]-1, :2], traj_tracked._pos[1:loop_idx[1], :2]), axis=1)
    traj2_seg = np.stack((traj_tracked._pos[loop_idx[1]:loop_idx[2], :2], traj_tracked._pos[loop_idx[1]+1:loop_idx[2]+1, :2]), axis=1)

    traj1_collection = LineCollection(traj1_seg, cmap="jet", linewidth=3, norm=norm, linestyles='-', label="Traj: Tracked 1 loop")
    traj1_collection.set_array(traj_tracked._vel[:loop_idx[1]-1,3])
    traj2_collection = LineCollection(traj2_seg, cmap="jet", linewidth=3, norm=norm, linestyles='-', label="Traj: Tracked 2 loop")
    traj2_collection.set_array(traj_tracked._vel[loop_idx[1]:loop_idx[2],3])

    line1 = ax1.add_collection(traj1_collection)
    line2 = ax2.add_collection(traj2_collection)


    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(line2, cax=cax, label='Velocity [m/s]')
    fig.tight_layout()
    ax1.legend()
    ax2.legend()
    print(loop_idx)

def plot_3d(gates:Gates):
    fig = plt.figure("3d", figsize=(10,6))
    ax_3d = fig.add_subplot([0,0,1,1], projection="3d")
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
    ax_3d.set_aspect("equal")
    
    plot_gates_3d(ax_3d, gates)

if __name__ == "__main__":

    traj = Trajectory("./results/res_n8.csv")
    traj_t = Trajectory("./results/res_t_n8.csv")
    traj_track = Trajectory("./results/res_track_n8.csv")
    # rpg_n6 = Trajectory("./rpg_results/result_n8.csv")
    gates = Gates("./gates/gates_n8.yaml")

    plot_track_vel(gates, traj_t, traj_track, gates._pos[0])
    plot_tracked(gates, traj_t, traj_track)
    # plot_3d(gates)

    # plt.tight_layout(pad=0)
    plt.show()
