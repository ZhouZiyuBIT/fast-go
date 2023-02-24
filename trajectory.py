import numpy as np
import csv

class Trajectory():
    def __init__(self, csv_f="", t=np.array([]), pos=np.array([]), vel=np.array([]), quaternion=np.array([]), omega=np.array([]), N=0):
        if csv_f != "":
            t = []
            pos = []
            vel = []
            quaternion = []
            omega = []
            self._N = 0
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
        else:
            self._t = t
            self._pos = pos
            self._vel = vel
            self._quaternion = quaternion
            self._omega = omega
            self._N = N

    def __getitem__(self, idx):
        return Trajectory(t=self._t[idx], pos=self._pos[idx], vel=self._vel[idx], quaternion=self._quaternion[idx], omega=self._omega[idx], N=self._pos[idx].shape[0])

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
    
    def divide_loops(self, pos):
        loop_idx = []
        flag1 = 0
        flag2 = 0
        for i in range(self._N):
            if np.linalg.norm(self._pos[i]-pos)< 0.5:
                if flag1 == 0:
                    l = np.linalg.norm(self._pos[i] - pos)
                    flag1 = 1
                else:
                    if l<np.linalg.norm(self._pos[i] - pos):
                        if flag2 == 0:
                            loop_idx.append(i)
                            flag2 = 1
                l = np.linalg.norm(self._pos[i] - pos)
            else:
                flag1 = 0
                flag2 = 0
        
        if len(loop_idx)>1:
            loops = [self[0:loop_idx[1]]]
            for i in range(1, len(loop_idx)-1):
                loops.append(self[loop_idx[i]: loop_idx[i+1]])
                print(self[loop_idx[i]: loop_idx[i+1]]._pos)
            return loops
        else:
            return [self]

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
