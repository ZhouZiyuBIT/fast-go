import casadi as ca
import numpy as np

from quadrotor import QuadrotorModel, QuadrotorSimpleModel

def linear_table(n, t, p0, p:ca.SX):
    x = []
    x += [ (p[:,0]-p0)*t+p0 ]
    for i in range(1,n):
        x += [ (p[:,i]-p[:,i-1])*(t-i) + p[:,i-1] ]
    
    y = ca.conditional(ca.floor(t), x, p[:,-1])
    
    return y

def linear(n, l, p:ca.SX, ls):
    y = 0
    for i in range(n-1):
        y += ca.logic_and(ls[i]<=l, l<ls[i+1])*( p[:,i]+(l-ls[i])/(ls[i+1]-ls[i])*(p[:,i+1]-p[:,i]) )
    y += (ls[n-1]<=l)*p[:,n-1]
    # y += (ls[n-1]<=l)*( p[:,n-2]+(l-ls[n-2])/(ls[n-1]-ls[n-2])*(p[:,n-1]-p[:,n-2]) )
    return y

def p_cost(v, Th):
    # 1
    c = v.T@v
    
    # 2
    # ll = v.T@v
    # c = (ll<=Th*Th)*ll
    # c += (ll>Th*Th)*(ll-Th*Th)*1000+Th*Th
    
    # 3
    # ll = v.T@v
    # c = (ll>Th*Th)*(ll-Th*Th)
    return c

class TrackerOpt():
    def __init__(self, quad:QuadrotorModel):
        self._quad = quad
        self._Herizon = 10
        self._ddynamics = []
        for n in range(self._Herizon):
            self._ddynamics += [self._quad.ddynamics(0.03+n*0.04)]
        
        self._X_dim = self._ddynamics[0].size1_in(0)
        self._U_dim = self._ddynamics[0].size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub
        
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        self._l = ca.SX.sym('l', self._Herizon)
        
        self._X_init = ca.SX.sym("X_init", self._X_dim)
        self._trj_N = 20
        self._Trj_p = ca.SX.sym("Trj_p", 3, self._trj_N)
        self._Trj_yaw = ca.SX.sym("Trj_yaw", 1, self._trj_N)
        self._Trj_p_ls = [0]
        self._Trj_yaw_ls = [0]
        for i in range(self._trj_N-1):
            self._Trj_p_ls.append(self._Trj_p_ls[i] + ca.sqrt( (self._Trj_p[:,i+1]-self._Trj_p[:,i]).T@(self._Trj_p[:,i+1]-self._Trj_p[:,i]) ))
            self._Trj_yaw_ls.append(self._Trj_yaw_ls[i] + ca.norm_2( self._Trj_yaw[i+1]-self._Trj_yaw[i] ))
        self._opt_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 20,
            # 'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0,
        }
        
        self._nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []
        
        self._nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []
        
        self._nlp_x_l = []
        self._nlp_lbx_l = []
        self._nlp_ubx_l = []

        self._nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []
        
        self._nlp_g_quat = []
        self._nlp_lbg_quat = []
        self._nlp_ubg_quat = []
        
        self._nlp_p_xinit = []
        self._nlp_p_Trj_p = []
        self._nlp_p_Trj_yaw = []
        
        self._nlp_obj_dyn = 0
        self._nlp_obj_trjp = 0
        self._nlp_obj_trjyaw = 0
        self._nlp_obj_l = 0
        
        self._nlp_x_x += [ self._Xs[:, 0] ]
        self._nlp_lbx_x += self._X_lb
        self._nlp_ubx_x += self._X_ub
        self._nlp_x_u += [ self._Us[:, 0] ]
        self._nlp_lbx_u += self._U_lb
        self._nlp_ubx_u += self._U_ub
        
        self._nlp_g_quat += [ self._Xs[6:10,0].T@self._Xs[6:10,0]-1 ]
        self._nlp_lbg_quat += [-0.1 ]
        self._nlp_ubg_quat += [ 0.1 ]
        
        dd_dyn = self._Xs[:,0]-self._ddynamics[0]( self._X_init, self._Us[:,0] )
        self._nlp_g_dyn += [ dd_dyn ]
        self._nlp_obj_dyn += dd_dyn.T@dd_dyn
        self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
        self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
        
        self._nlp_obj_l += self._l[0]
        trjp = linear(self._trj_N, self._l[0], self._Trj_p, self._Trj_p_ls)
        # self._nlp_obj_trjp += (self._Xs[:3,0]-trjp).T@(self._Xs[:3,0]-trjp)
        self._nlp_obj_trjp += p_cost(self._Xs[:3,0]-trjp, 0.5)
        trjyaw =  linear(self._trj_N, self._l[0], self._Trj_yaw, self._Trj_yaw_ls)
        c_trjyaw = ca.cos(trjyaw/2)
        s_trjyaw = ca.sin(trjyaw/2)
        _qw = self._Xs[6,0]
        _qz = self._Xs[9,0]
        _q_sqrt = ca.sqrt(_qw*_qw+_qz*_qz)
        self._nlp_obj_trjyaw += (_qw-c_trjyaw*_q_sqrt)*(_qw-c_trjyaw*_q_sqrt) + (_qz-s_trjyaw*_q_sqrt)*(_qz-s_trjyaw*_q_sqrt)
        
        self._nlp_x_l += [self._l[0]]
        self._nlp_lbx_l += [0]
        self._nlp_ubx_l += [50]
        
        for i in range(1,self._Herizon):
            self._nlp_x_x += [ self._Xs[:, i] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, i] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            
            self._nlp_g_quat += [ self._Xs[6:10,i].T@self._Xs[6:10,i]-1 ]
            self._nlp_lbg_quat += [-0.1 ]
            self._nlp_ubg_quat += [ 0.1 ]

            dd_dyn = self._Xs[:,i]-self._ddynamics[i]( self._Xs[:,i-1], self._Us[:,i] )
            self._nlp_g_dyn += [ dd_dyn ]
            self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
            
            self._nlp_obj_l += self._l[i]
            trjp = linear(self._trj_N, self._l[i], self._Trj_p, self._Trj_p_ls)
            # self._nlp_obj_trjp += (self._Xs[:3,i]-trjp).T@(self._Xs[:3,i]-trjp)
            self._nlp_obj_trjp += p_cost(self._Xs[:3,i]-trjp, 0.5)
            trjyaw =  linear(self._trj_N, self._l[i], self._Trj_yaw, self._Trj_yaw_ls)
            c_trjyaw = ca.cos(trjyaw/2)
            s_trjyaw = ca.sin(trjyaw/2)
            _qw = self._Xs[6,i]
            _qz = self._Xs[9,i]
            _q_sqrt = ca.sqrt(_qw*_qw+_qz*_qz)
            self._nlp_obj_trjyaw += (_qw-c_trjyaw*_q_sqrt)*(_qw-c_trjyaw*_q_sqrt) + (_qz-s_trjyaw*_q_sqrt)*(_qz-s_trjyaw*_q_sqrt)
            
            self._nlp_x_l += [self._l[i]]
            self._nlp_lbx_l += [0]
            self._nlp_ubx_l += [50]
        
        self._nlp_p_xinit += [self._X_init]
        
        for i in range(self._trj_N):
            self._nlp_p_Trj_p += [self._Trj_p[:,i]]
            self._nlp_p_Trj_yaw += [self._Trj_yaw[:,i]]
    
    def reset_xul(self):
        self._xul0 = np.zeros((self._X_dim+self._U_dim+1)*self._Herizon)
        self._xul0[-self._Herizon]=1.2
        for i in range(self._Herizon):
            self._xul0[i*self._X_dim+6] = 1
       
    def define_opt(self):
        print(self._nlp_obj_l)
        nlp_dect = {
            'f': -1*self._nlp_obj_l+30*(self._nlp_obj_trjp),#+30*self._nlp_obj_trjyaw,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_l)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_Trj_p+self._nlp_p_Trj_yaw)),
            'g': ca.vertcat(*(self._nlp_g_dyn)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        
        self.reset_xul()
        
    
    def solve(self, xinit, Trjp, Trjyaw):
        p = np.zeros(self._X_dim+4*self._trj_N)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim+3*self._trj_N] = Trjp
        p[self._X_dim+3*self._trj_N:self._X_dim+3*self._trj_N+self._trj_N] = Trjyaw
        res = self._opt_solver(
            x0=self._xul0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_l),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_l),
            lbg=(self._nlp_lbg_dyn),
            ubg=(self._nlp_ubg_dyn),
            p=p
        )
        
        self._xul0 = res['x'].full().flatten()
        
        return res

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    quad = QuadrotorModel('quad.yaml')
    tracker = TrackerOpt(quad)
    tracker.define_opt()
    tracker.reset_xul()
    
    xinit = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0])
    trjp = np.zeros(20*3)
    trjyaw = np.zeros(20)
    plot_x = np.zeros(20)
    plot_y = np.zeros(20)
    
    for i in range(20):
        trjp[3*i+0] = np.sin(i/5)*4
        trjp[3*i+1] = np.cos(i/3)*3
        plot_x[i] = trjp[3*i+0]
        plot_y[i] = trjp[3*i+1]
    
    res = tracker.solve(xinit, trjp, trjyaw)
    x = res['x'].full().flatten()
    print(x[-5:])
