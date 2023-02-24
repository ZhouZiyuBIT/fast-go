import numpy as np
import yaml


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

