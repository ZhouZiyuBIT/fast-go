import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)*10
ps = np.stack((x,y), axis=1)
print(ps.shape)
segments = np.stack((ps[:-1], ps[1:]), axis=1)
print(segments.shape)
cmap = 'jet' # viridis, jet, hsv等也是常用的颜色映射方案
# colors = color_map(np.cos(x)[:-1], cmap)
# colors = color_map(y[:-1], cmap)
colors = plt.colormaps[cmap](y[:-1])
line_segments = LineCollection(segments, colors=colors, linewidths=3, linestyles='solid', cmap=cmap)

fig, ax = plt.subplots()
ax.set_xlim(np.min(x)-0.1, np.max(x)+0.1)
ax.set_ylim(np.min(y)-0.1, np.max(y)+0.1)
ax.add_collection(line_segments)
cb = fig.colorbar(line_segments, cmap='jet')

plt.show()