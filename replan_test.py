import numpy as np
from quadrotor import QuadrotorModel

from time_optimal_planner import WayPointOpt

def random_pos_move():
    return np.random.random(3)*np.array([2,2,0.5])-np.array([1, 1, 0.25])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plotting import GatesShape
    xinit = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0])
    gate = GatesShape("./gates.yaml")

    dts = np.array([0.2]*gate._N)
    
    quad = QuadrotorModel('quad.yaml')
    
    wp_opt = WayPointOpt(quad, gate._N, loop=True)
    wp_opt.define_opt()
    wp_opt.define_opt_t()
    
    g_pos_init = gate._pos
    res = wp_opt.solve_opt(xinit, g_pos_init.flatten(), dts)
    print("***************************************************************************************************************************")
    res_t = wp_opt.solve_opt_t(xinit, g_pos_init.flatten())
    for i in range(10):
        print(i,"##############################################################################")
        # g_pos_init[1] += random_pos_move()
        res_t = wp_opt.solve_opt_t(xinit, g_pos_init.flatten())
