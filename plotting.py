import csv
import numpy as np
import yaml

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gates.gates import Gates
from trajectory import Trajectory

# sns.set()

def plot_gates_3d(axes:plt.Axes, gates:Gates):
    for idx in range(gates._N):
        axes.plot(gates._shapes[idx][0], gates._shapes[idx][1], gates._shapes[idx][2], linewidth=3, color="dimgray")

def plot_gates_2d(axes:plt.Axes, gates:Gates):
    for idx in range(gates._N):
        axes.plot(gates._shapes[idx][0], gates._shapes[idx][1], linewidth=3, color="dimgray")

def plot_traj_xy(axes:plt.Axes, traj:Trajectory, linewidth=1, linestyle="-", label="", color=None, alpha=1):
    axes.plot(traj._pos[:,0], traj._pos[:,1], linewidth=linewidth, linestyle=linestyle, color=color, alpha=alpha, label=label)

def plot_traj_3d(axes3d, gates:Gates):
    axes3d.plot(gates._pos[:,0], gates._pos[:,1], gates._pos[:,2])

def plot_tracked(gates:Gates, traj_planned:Trajectory, traj_tracked:Trajectory):
    fig = plt.figure(figsize=(6,5.5))

    ax = fig.add_subplot(111)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("X [m]", labelpad=0)
    ax.set_ylabel("Y [m]", labelpad=-5)

    ax.set_aspect("equal")

    plot_gates_2d(ax, gates)
    
    ax.scatter(traj_tracked._pos[0,0], traj_tracked._pos[0,1], marker="*")
    plot_traj_xy(ax, traj_tracked, linewidth=2, linestyle="-", color="#2878B5", label="Traj: Tracked")
    plot_traj_xy(ax, traj_planned, linewidth=3, linestyle="--", color="#F8AC8C", label="Traj: Planned")

        # inset axes....
    axins = ax.inset_axes([0.56, 0.22, 0.28, 0.28])
    plot_traj_xy(axins, traj_tracked, linewidth=2, linestyle="-", color="#2878B5", label="Traj: Tracked")
    plot_traj_xy(axins, traj_planned, linewidth=3, linestyle="--", color="#F8AC8C", label="Traj: Planned")
    plot_gates_2d(axins, gates)
    # subregion of the original image
    x1, x2, y1, y2 = 7.7, 8.8, 5.2, 6.3
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="gray")
    ax.legend()
    fig.tight_layout(pad=0, rect=[0.01, 0.0, 0.99, 1])
    plt.savefig("track.eps")

def plot_track_vel(gates:Gates, traj_planned:Trajectory, traj_tracked:Trajectory, first_gate_pos):

    loops = traj_tracked.divide_loops(first_gate_pos)
    print(len(loops))
    gs = GridSpec(20,21)
    fig = plt.figure(figsize=(6, 2.8))

    ax1 = fig.add_subplot(121)
    ax1.set_xlim([-10, 10])
    ax1.set_ylim([-10, 10])
    ax1.set_xlabel("X [m]", labelpad=0)
    ax1.set_ylabel("Y [m]", labelpad=-5)
    ax1.set_title("Loop 1")
    ax1.set_aspect("equal")
    plot_gates_2d(ax1, gates)
    plot_traj_xy(ax1, traj_planned, linestyle='--', linewidth=3, color="gray", label="Traj: Planned")

    ax2 = fig.add_subplot(122)
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([-10, 10])
    ax2.set_xlabel("X [m]", labelpad=0)
    ax2.set_ylabel("Y [m]", labelpad=-5)
    ax2.set_aspect("equal")
    ax2.set_title("Loop 2")
    plot_gates_2d(ax2, gates)
    plot_traj_xy(ax2, traj_planned, linestyle='--', linewidth=3, color="gray", label=None)
    
    vel_min = np.min(traj_tracked._vel[:-1,3])
    vel_max = np.max(traj_tracked._vel[:-1,3])
    norm = plt.Normalize(vel_min, vel_max)

    traj1_seg = np.stack((loops[0]._pos[:-1, :2], loops[0]._pos[1:, :2]), axis=1)
    traj2_seg = np.stack((loops[1]._pos[:-1, :2], loops[1]._pos[1:, :2]), axis=1)

    traj1_collection = LineCollection(traj1_seg, cmap="jet", linewidth=2, norm=norm, linestyles='-', label="Traj: Tracked")
    traj1_collection.set_array(loops[0]._vel[:,3])
    traj2_collection = LineCollection(traj2_seg, cmap="jet", linewidth=2, norm=norm, linestyles='-')
    traj2_collection.set_array(loops[1]._vel[:,3])

    ax1.scatter(loops[0]._pos[0,0], loops[0]._pos[0,1], marker="*")
    line1 = ax1.add_collection(traj1_collection)
    line2 = ax2.add_collection(traj2_collection)
    # plot_traj_xy(ax1, traj_planned, linestyle=':', linewidth=2, color="black", label="Traj: Planned")
    # plot_traj_xy(ax2, traj_planned, linestyle='--', linewidth=2, color="gray", label="Traj: Planned")

    # divider1 = make_axes_locatable(ax1)
    # cax1 = divider1.append_axes("right", size="5%", pad="5%")
    # fig.colorbar(line1, cax=cax1, label='Velocity [m/s]')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad="10%")
    fig.colorbar(line2, cax=cax2, label='Velocity [m/s]')
    fig.legend(ncol=2, loc="upper center")
    fig.tight_layout(pad=0, rect=[0.0, 0, 0.99, 0.85])
    plt.savefig("track_loop.eps")
    # ax1.margins(x=0,y=0)
    # ax1.margins(x=0,y=0)
    # fig.tight_layout()
    # ax1.legend()
    # ax2.legend()

def plot_3d(gates:Gates):
    fig = plt.figure("3d", figsize=(6,3))
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

    import os, sys
    BASEPATH = os.path.abspath(__file__).split('script', 1)[0]+'script/fast-go/'

    traj = Trajectory(BASEPATH+"results/res_n6.csv")
    traj_t = Trajectory(BASEPATH+"results/res_t_n6.csv")
    traj_track = Trajectory(BASEPATH+"results/res_track_n6.csv")
    # rpg_n6 = Trajectory("./rpg_results/result_n8.csv")
    gates = Gates(BASEPATH+"gates/gates_n6.yaml")

    plot_track_vel(gates, traj_t, traj_track, gates._pos[0])
    plot_tracked(gates, traj_t, traj_track)
    # plot_3d(gates)

    # plt.tight_layout(pad=0)
    plt.show()
