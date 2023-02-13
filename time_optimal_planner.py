import numpy as np
import casadi as ca

from quadrotor import QuadrotorModel, QuadrotorSimpleModel, RPG_Quad

# Quaternion Multiplication
def quat_mult(q1,q2):
    ans = ca.vertcat(q2[0,:] * q1[0,:] - q2[1,:] * q1[1,:] - q2[2,:] * q1[2,:] - q2[3,:] * q1[3,:],
           q2[0,:] * q1[1,:] + q2[1,:] * q1[0,:] - q2[2,:] * q1[3,:] + q2[3,:] * q1[2,:],
           q2[0,:] * q1[2,:] + q2[2,:] * q1[0,:] + q2[1,:] * q1[3,:] - q2[3,:] * q1[1,:],
           q2[0,:] * q1[3,:] - q2[1,:] * q1[2,:] + q2[2,:] * q1[1,:] + q2[3,:] * q1[0,:])
    return ans

# Quaternion-Vector Rotation
def rotate_quat(q1,v1):
    ans = quat_mult(quat_mult(q1, ca.vertcat(0, v1)), ca.vertcat(q1[0,:],-q1[1,:], -q1[2,:], -q1[3,:]))
    return ca.vertcat(ans[1,:], ans[2,:], ans[3,:]) # to covert to 3x1 vec

class WayPointOpt():
    def __init__(self, quad:QuadrotorSimpleModel, wp_num:int, loop:bool):
        self._loop = loop
        
        self._quad = quad
        self._ddynamics = self._quad.ddynamics_dt()

        self._wp_num = wp_num
        self._N_per_wp = 20 # opt param
        self._Herizon = self._wp_num*self._N_per_wp

        self._X_dim = self._ddynamics.size1_in(0)
        self._U_dim = self._ddynamics.size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub

        self._DTs = ca.SX.sym('DTs', self._wp_num)
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        self._WPs_p = ca.SX.sym('WPs_p', 3, self._wp_num)
        # self._WPs_yaw = ca.SX.sym("WPs_yaw", 2, self._wp_num)
        if self._loop:
            self._X_init = self._Xs[:,-1]
        else:
            self._X_init = ca.SX.sym('X_init', self._X_dim)

        self._cost_Co = ca.diag([1,2,2,2]) # opt param
        self._cost_WP_p = ca.diag([1,1,1]) # opt param

        self._opt_option = {
            'verbose': False,
            # 'ipopt.tol': 1e-5,
            # 'ipopt.acceptable_tol': 1e-8,
            'ipopt.max_iter': 1000,
            'ipopt.warm_start_init_point': 'yes',
            # 'ipopt.print_level': 0,
        }
        self._opt_t_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 1000,
            # 'ipopt.warm_start_init_point': 'yes',
            # 'ipopt.print_level': 0
        }

        #################################################################
        self._nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []

        self._nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []

        self._nlp_x_t = []
        self._nlp_lbx_t = []
        self._nlp_ubx_t = []

        self._nlp_g_orientation = []
        self._nlp_lbg_orientation = []
        self._nlp_ubg_orientation = []

        self._nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []

        self._nlp_g_wp_p = []
        self._nlp_lbg_wp_p = []
        self._nlp_ubg_wp_p = []

        # self._nlp_g_wp_yaw = []
        # self._nlp_lbg_wp_yaw = []
        # self._nlp_ubg_wp_yaw = []

        self._nlp_g_quat = []
        self._nlp_lbg_quat = []
        self._nlp_ubg_quat = []

        if self._loop:
            self._nlp_p_xinit = []
        else:
            self._nlp_p_xinit = [ self._X_init ]
        self._nlp_p_dt = []
        self._nlp_p_wp_p = []
        
        self._nlp_obj_orientation = 0
        self._nlp_obj_minco = 0
        self._nlp_obj_time = 0
        self._nlp_obj_wp_p = 0
        self._nlp_obj_quat = 0
        self._nlp_obj_dyn = 0

        ###################################################################

        for i in range(self._wp_num):
            self._nlp_x_x += [ self._Xs[:, i*self._N_per_wp] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, i*self._N_per_wp] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            self._nlp_x_t += [ self._DTs[i] ]
            self._nlp_lbx_t += [0]
            self._nlp_ubx_t += [0.5]

            if i==0:
                dd_dyn = self._Xs[:,0]-self._ddynamics( self._X_init, self._Us[:,0], self._DTs[0])
                self._nlp_g_dyn += [ dd_dyn ]
                self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            else:
                dd_dyn = self._Xs[:,i*self._N_per_wp]-self._ddynamics( self._Xs[:,i*self._N_per_wp-1], self._Us[:,i*self._N_per_wp], self._DTs[i])
                self._nlp_g_dyn += [ dd_dyn ]
                self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]

            self._nlp_g_wp_p += [ (self._Xs[:3,(i+1)*self._N_per_wp-1]-self._WPs_p[:,i]).T@(self._Xs[:3,(i+1)*self._N_per_wp-1]-self._WPs_p[:,i]) ]
            self._nlp_lbg_wp_p += [0]
            self._nlp_ubg_wp_p += [ 0.001 ]

            self._nlp_p_dt += [ self._DTs[i] ]
            self._nlp_p_wp_p += [ self._WPs_p[:,i] ]

            q_orientation = rotate_quat(self._Xs[6:10, i*self._N_per_wp],[1,0,0])
            g_vector = self._WPs_p[:,i] - self._Xs[:3, i*self._N_per_wp]
            v_cross = ca.cross(q_orientation, g_vector)
            self._nlp_obj_orientation += v_cross.T@v_cross
            # self._nlp_g_orientation += [ v_cross.T@v_cross ]
            # self._nlp_lbg_orientation += [-0.01 ]
            # self._nlp_ubg_orientation += [ 0.01 ]
            
            # q1 = self._Xs[6:10, i*self._N_per_wp]
            # v1 = rotate_quat(ca.vertcat( q1[0], -q1[1], -q1[2], -q1[3] ), self._WPs_p[:,i]-self._Xs[:3, i*self._N_per_wp] )
            # self._nlp_g_orientation += [ v1[1]/v1[0], v1[2]/v1[0] ]
            # self._nlp_lbg_orientation += [-0.5,-1.0 ]
            # self._nlp_ubg_orientation += [ 0.5, 1.0 ]

            self._nlp_obj_minco += (self._Us[:,i*self._N_per_wp]-[-9.91,0,0,0]).T@self._cost_Co@(self._Us[:,i*self._N_per_wp]-[-9.91,0,0,0])
            self._nlp_obj_time += self._DTs[i]*self._N_per_wp
            self._nlp_obj_wp_p += (self._Xs[:3,(i+1)*self._N_per_wp-1]-self._WPs_p[:,i]).T@self._cost_WP_p@(self._Xs[:3,(i+1)*self._N_per_wp-1]-self._WPs_p[:,i])
            
            for j in range(1, self._N_per_wp):
                self._nlp_x_x += [ self._Xs[:, i*self._N_per_wp+j] ]
                self._nlp_lbx_x += self._X_lb
                self._nlp_ubx_x += self._X_ub
                self._nlp_x_u += [ self._Us[:, i*self._N_per_wp+j] ]
                self._nlp_lbx_u += self._U_lb
                self._nlp_ubx_u += self._U_ub

                q_orientation = rotate_quat(self._Xs[6:10, i*self._N_per_wp+j],[1,0,0])
                g_vector = self._WPs_p[:,i] - self._Xs[:3, i*self._N_per_wp+j]
                v_cross = ca.cross(q_orientation, g_vector)
                self._nlp_obj_orientation += v_cross.T@v_cross
                # self._nlp_g_orientation += [ v_cross.T@v_cross ]
                # self._nlp_lbg_orientation += [-0.01 ]
                # self._nlp_ubg_orientation += [ 0.01 ]
                
                if (j < self._N_per_wp*0.9) and (j > self._N_per_wp*(0.3)):
                    q1 = self._Xs[6:10, i*self._N_per_wp+j]
                    v1 = rotate_quat(ca.vertcat( q1[0], -q1[1], -q1[2], -q1[3]), self._WPs_p[:,i]-self._Xs[:3, i*self._N_per_wp+j] )
                    self._nlp_g_orientation += [ v1[1]/v1[0], v1[2]/v1[0] ]
                    self._nlp_lbg_orientation += [-0.5,-0.5 ]
                    self._nlp_ubg_orientation += [ 0.5, 0.5 ]
                
                dd_dyn = self._Xs[:,i*self._N_per_wp+j]-self._ddynamics( self._Xs[:,i*self._N_per_wp+j-1], self._Us[:,i*self._N_per_wp+j], self._DTs[i])
                self._nlp_g_dyn += [ dd_dyn ]
                self._nlp_obj_dyn += dd_dyn.T@dd_dyn
                self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
                self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]

                self._nlp_obj_minco += (self._Us[:,i*self._N_per_wp+j]-[-9.91,0,0,0]).T@self._cost_Co@(self._Us[:,i*self._N_per_wp+j]-[-9.91,0,0,0])

    def define_opt(self):
        nlp_dect = {
            'f': self._nlp_obj_dyn + self._nlp_obj_wp_p + self._nlp_obj_minco,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_wp_p+self._nlp_p_dt)),
            # 'g': ca.vertcat(*(self._nlp_g_dyn)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        self._xu0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon)
        for i in range(self._Herizon):
            self._xu0[i*self._X_dim+6] = 1
            self._xu0[self._Herizon*self._X_dim+i*self._U_dim] = -9.81

    def solve_opt(self, xinit, wp_p, dts):
        if self._loop:
            p = np.zeros(3*self._wp_num+self._wp_num)
            p[:3*self._wp_num] = wp_p
            p[3*self._wp_num:3*self._wp_num+self._wp_num] = dts
        else:
            p = np.zeros(self._X_dim+3*self._wp_num+self._wp_num)
            p[:self._X_dim] = xinit
            p[self._X_dim:self._X_dim+3*self._wp_num] = wp_p
            p[self._X_dim+3*self._wp_num:] = dts
        res = self._opt_solver(
            x0=self._xu0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u),
            # lbg=(),
            # ubg=(),
            p=p
        )
        self._xu0 = res['x'].full().flatten()
        self._dt0 = dts
        self._xut0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon+self._wp_num)
        self._xut0[:(self._X_dim+self._U_dim)*self._Herizon] = self._xu0
        self._xut0[(self._X_dim+self._U_dim)*self._Herizon:] = self._dt0
        return res

    def define_opt_t(self):
        nlp_dect = {
            'f': self._nlp_obj_time,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_t)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_wp_p)),
            'g': ca.vertcat(*(self._nlp_g_dyn+self._nlp_g_wp_p)),
        }
        self._opt_t_solver = ca.nlpsol('opt_t', 'ipopt', nlp_dect, self._opt_t_option)

    def solve_opt_t(self, xinit, wp_p):
        if self._loop:
            p = np.zeros(3*self._wp_num)
            p = wp_p
        else:
            p = np.zeros(self._X_dim+3*self._wp_num)
            p[:self._X_dim] = xinit
            p[self._X_dim:self._X_dim+3*self._wp_num] = wp_p
        res = self._opt_t_solver(
            x0=self._xut0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_t),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_t),
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_wp_p),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_wp_p),
            p=p
        )
        self._xut0 = res['x'].full().flatten()
        return res

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    xinit = np.array([-5,4.5,1.2, 0,0,0, 1,0,0,0, 0,0,0])
    wp = np.array([[-1.1, -1.6, 3.6],
        [9.2, 6.6, 1.0],
        [9.2, -4.0, 1.2],
        [-4.5, -6.0, 3.5],
        [-4.5, -6.0, 0.8],
        [4.75, -0.9, 1.2],
        [-2.8, 6.8, 1.2],
        [4.75, -0.9, 1.2]]
                  )
    dts = np.array([0.3]*8)
    
    # quad = QuadrotorModel('quad.yaml')
    quad = RPG_Quad('rpg_quad.yaml')
    
    wp_opt = WayPointOpt(quad, 8, loop=False)
    wp_opt.define_opt()
    wp_opt.define_opt_t()
    
    res = wp_opt.solve_opt(xinit, wp.flatten(), dts)
    res_t = wp_opt.solve_opt_t(xinit, wp.flatten())
    
    plot_pos = np.zeros((3, wp_opt._Herizon))
    plot_pos_t = np.zeros((3, wp_opt._Herizon))
    for i in range(wp_opt._Herizon):
        x = res['x'].full().flatten()
        plot_pos[:, i] = x[wp_opt._X_dim*i+0:wp_opt._X_dim*i+3]
        x_t = res_t['x'].full().flatten()
        plot_pos_t[:, i] = x_t[wp_opt._X_dim*i+0:wp_opt._X_dim*i+3]
    plt.scatter(wp[:,0], wp[:,1])
    plt.plot(plot_pos[0,:], plot_pos[1,:])
    plt.plot(plot_pos_t[0,:], plot_pos_t[1,:])
    plt.show()
