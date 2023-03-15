import casadi as ca
import numpy as np

import os, sys
BASEPATH = os.path.abspath(__file__).split('script', 1)[0]+'script/fast-go/'
sys.path += [BASEPATH]

from quadrotor import QuadrotorModel, rotate_quat

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
    # y += (ls[n-1]<=l)*p[:,n-1]
    y += (ls[n-1]<=l)*( p[:,n-2]+(l-ls[n-2])/(ls[n-1]-ls[n-2])*(p[:,n-1]-p[:,n-2]) )
    return y

def linear2(n, l, p:ca.SX, ls):
    y = 0
    v = 0
    for i in range(n-1):
        y += ca.logic_and(ls[i]<=l, l<ls[i+1])*( p[:,i]+(l-ls[i])/(ls[i+1]-ls[i])*(p[:,i+1]-p[:,i]) )
        v += ca.logic_and(ls[i]<=l, l<ls[i+1])*( (p[:,i+1]-p[:,i])/(ls[i+1]-ls[i]) )
    # y += (ls[n-1]<=l)*p[:,n-1]
    y += (ls[n-1]<=l)*( p[:,n-2]+(l-ls[n-2])/(ls[n-1]-ls[n-2])*(p[:,n-1]-p[:,n-2]) )
    v += (ls[n-1]<=l)*( (p[:,n-1]-p[:,n-2])/(ls[n-1]-ls[n-2]) )
    return y, v

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
            self._ddynamics += [self._quad.ddynamics(0.06+n*0.01)]
        
        self._X_dim = self._ddynamics[0].size1_in(0)
        self._U_dim = self._ddynamics[0].size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub
        
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        self._l = ca.SX.sym('l', self._Herizon)
        self._max_l = ca.SX.sym("max_l", 1)
        
        self._X_init = ca.SX.sym("X_init", self._X_dim)
        self._trj_N = 30
        self._Trj_p = ca.SX.sym("Trj_p", 3, self._trj_N)
        self._Trj_p_ls = [0]
        for i in range(self._trj_N-1):
            self._Trj_p_ls.append(self._Trj_p_ls[i] + ca.sqrt( (self._Trj_p[:,i+1]-self._Trj_p[:,i]).T@(self._Trj_p[:,i+1]-self._Trj_p[:,i]) ))
        self._max_v = ca.SX.sym("max_v",1)
        self._punish_v_coef = ca.SX.sym("v_coef",1)

        self._opt_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 25,
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

        self._nlp_g_l = []
        self._nlp_lbg_l = []
        self._nlp_ubg_l = []

        self._nlp_g_v = []
        self._nlp_lbg_v = []
        self._nlp_ubg_v = []
        
        self._nlp_p_xinit = []
        self._nlp_p_Trj_p = []
        self._nlp_p_max_v = [self._max_v] # for safty
        self._nlp_p_punish_v_coef = [self._punish_v_coef]
        
        self._nlp_obj_dyn = 0
        self._nlp_obj_trjp = 0
        self._nlp_obj_l = 0
        self._nlp_obj_v_punish = 0      
        self._nlp_x_x += [ self._Xs[:, 0] ]
        self._nlp_lbx_x += self._X_lb
        self._nlp_ubx_x += self._X_ub
        self._nlp_x_u += [ self._Us[:, 0] ]
        self._nlp_lbx_u += self._U_lb
        self._nlp_ubx_u += self._U_ub
        
        dd_dyn = self._Xs[:,0]-self._ddynamics[0]( self._X_init, self._Us[:,0] )
        self._nlp_g_dyn += [ dd_dyn ]
        self._nlp_obj_dyn += dd_dyn.T@dd_dyn
        self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
        self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
        
        self._nlp_obj_l += self._l[0]
        trjp = linear(self._trj_N, self._l[0], self._Trj_p, self._Trj_p_ls)
        self._nlp_obj_trjp += p_cost(self._Xs[:3,0]-trjp, 0.5)
        
        self._nlp_x_l += [self._l[0]]
        self._nlp_lbx_l += [0]
        self._nlp_ubx_l += [ca.inf]
        
        for i in range(1,self._Herizon):
            self._nlp_x_x += [ self._Xs[:, i] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, i] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            
            dd_dyn = self._Xs[:,i]-self._ddynamics[i]( self._Xs[:,i-1], self._Us[:,i] )
            self._nlp_g_dyn += [ dd_dyn ]
            self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
            
            self._nlp_obj_l += self._l[i]
            trjp = linear(self._trj_N, self._l[i], self._Trj_p, self._Trj_p_ls)
            self._nlp_obj_trjp += p_cost(self._Xs[:3,i]-trjp, 0.5)
            
            self._nlp_x_l += [self._l[i]]
            self._nlp_lbx_l += [0]
            self._nlp_ubx_l += [ca.inf]
        
            self._nlp_g_v += [ self._Xs[3:6, i].T@self._Xs[3:6, i] - self._max_v*self._max_v ]
            self._nlp_lbg_v += [-ca.inf]
            self._nlp_ubg_v += [ 0 ]
            
            self._nlp_obj_v_punish += self._Xs[3:6, i].T@self._Xs[3:6, i]*self._punish_v_coef

        self._nlp_p_xinit += [self._X_init]
        

        for i in range(1, self._Herizon):
            self._nlp_g_l += [self._l[i]-self._l[i-1]]
            self._nlp_lbg_l += [0]
            self._nlp_ubg_l += [ ca.inf ]
        self._nlp_g_l += [ self._l[-1] ]
        self._nlp_lbg_l += [0]
        self._nlp_ubg_l += [20]

        for i in range(self._trj_N):
            self._nlp_p_Trj_p += [self._Trj_p[:,i]]
    
    def reset_xul(self):
        self._xul0 = np.zeros((self._X_dim+self._U_dim+1)*self._Herizon)
        self._xul0[-self._Herizon]=0
        for i in range(self._Herizon):
            self._xul0[i*self._X_dim+6] = 1

    def load_so(self, so_path):
        self._opt_solver = ca.nlpsol("opt", "ipopt", so_path, self._opt_option)
        self.reset_xul()
    
    def define_opt(self):
        nlp_dect = {
            'f': -1*self._nlp_obj_l+30*(self._nlp_obj_trjp),#+30*self._nlp_obj_trjyaw,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_l)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_Trj_p)),
            'g': ca.vertcat(*(self._nlp_g_dyn+self._nlp_g_l)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        
        self.reset_xul()
        
    
    def solve(self, xinit, Trjp, max_l:float):
        p = np.zeros(self._X_dim+3*self._trj_N)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim+3*self._trj_N] = Trjp
        res = self._opt_solver(
            x0=self._xul0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_l),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_l),
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_l),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_l),
            p=p
        )
        
        self._xul0 = res['x'].full().flatten()
        
        return res

class TrackerOpt2():
    def __init__(self, quad:QuadrotorModel):
        self._quad = quad
        self._Herizon = 10
        self._ddynamics = []
        for n in range(self._Herizon):
            self._ddynamics += [self._quad.ddynamics(0.06+n*0.01)]
        
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
        self._trj_N = 30
        self._Trj_p = ca.SX.sym("Trj_p", 3, self._trj_N)
        self._Trj_p_ls = [0]
        for i in range(self._trj_N-1):
            self._Trj_p_ls.append(self._Trj_p_ls[i] + ca.sqrt( (self._Trj_p[:,i+1]-self._Trj_p[:,i]).T@(self._Trj_p[:,i+1]-self._Trj_p[:,i]) ))

        self._opt_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 25,
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

        self._nlp_g_l = []
        self._nlp_lbg_l = []
        self._nlp_ubg_l = []
        
        self._nlp_p_xinit = []
        self._nlp_p_Trj_p = []
        
        self._nlp_obj_dyn = 0
        self._nlp_obj_trjp = 0
        self._nlp_obj_l = 0
        self._nlp_obj_v = 0      
        self._nlp_x_x += [ self._Xs[:, 0] ]
        self._nlp_lbx_x += self._X_lb
        self._nlp_ubx_x += self._X_ub
        self._nlp_x_u += [ self._Us[:, 0] ]
        self._nlp_lbx_u += self._U_lb
        self._nlp_ubx_u += self._U_ub
        
        dd_dyn = self._Xs[:,0]-self._ddynamics[0]( self._X_init, self._Us[:,0] )
        self._nlp_g_dyn += [ dd_dyn ]
        self._nlp_obj_dyn += dd_dyn.T@dd_dyn
        self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
        self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
        
        self._nlp_obj_l += self._l[0]
        trjp = linear(self._trj_N, self._l[0], self._Trj_p, self._Trj_p_ls)
        self._nlp_obj_trjp += p_cost(self._Xs[:3,0]-trjp, 0.5)

        self._nlp_obj_v += self._Xs[3:6, 0].T@self._Xs[3:6, 0]
        
        self._nlp_x_l += [self._l[0]]
        self._nlp_lbx_l += [0]
        self._nlp_ubx_l += [ca.inf]
        
        for i in range(1,self._Herizon):
            self._nlp_x_x += [ self._Xs[:, i] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, i] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            
            dd_dyn = self._Xs[:,i]-self._ddynamics[i]( self._Xs[:,i-1], self._Us[:,i] )
            self._nlp_g_dyn += [ dd_dyn ]
            self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
            
            self._nlp_obj_l += self._l[i]
            trjp = linear(self._trj_N, self._l[i], self._Trj_p, self._Trj_p_ls)
            self._nlp_obj_trjp += p_cost(self._Xs[:3,i]-trjp, 0.5)
            
            self._nlp_obj_v += self._Xs[3:6, i].T@self._Xs[3:6, i]

            self._nlp_x_l += [self._l[i]]
            self._nlp_lbx_l += [0]
            self._nlp_ubx_l += [ca.inf]

        self._nlp_p_xinit += [self._X_init]
        

        for i in range(1, self._Herizon):
            self._nlp_g_l += [self._l[i]-self._l[i-1]]
            self._nlp_lbg_l += [0]
            self._nlp_ubg_l += [ ca.inf ]
        self._nlp_g_l += [ self._l[-1] ]
        self._nlp_lbg_l += [0]
        self._nlp_ubg_l += [20]

        for i in range(self._trj_N):
            self._nlp_p_Trj_p += [self._Trj_p[:,i]]
    
    def reset_xul(self):
        self._xul0 = np.zeros((self._X_dim+self._U_dim+1)*self._Herizon)
        self._xul0[-self._Herizon]=0
        for i in range(self._Herizon):
            self._xul0[i*self._X_dim+6] = 1
       
    def define_opt(self):
        nlp_dect = {
            'f': self._nlp_obj_v+self._nlp_obj_trjp,#+30*self._nlp_obj_trjyaw,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_l)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_Trj_p)),
            'g': ca.vertcat(*(self._nlp_g_dyn+self._nlp_g_l)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        
        self.reset_xul()
        
    
    def solve(self, xinit, Trjp, max_l:float):
        p = np.zeros(self._X_dim+3*self._trj_N)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim+3*self._trj_N] = Trjp
        res = self._opt_solver(
            x0=self._xul0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_l),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_l),
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_l),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_l),
            p=p
        )
        
        self._xul0 = res['x'].full().flatten()
        
        return res

def traj_ploy(n, ploy_coef, ts, t):
    p=0
    v=0
    for i in range(n-1):
        a0=ploy_coef[:3,i]
        a1=ploy_coef[3:6,i]
        a2=ploy_coef[6:9,i]
        a3=ploy_coef[9:12,i]
        dt=t-ts[i]
        dtdt = dt*dt
        dtdtdt = dtdt*dt
        p += ca.logic_and(ts[i]<=t, t<ts[i+1])*( a0+a1*dt+a2*dtdt+a3*dtdtdt )
        v += ca.logic_and(ts[i]<=t, t<ts[i+1])*( a1+2*a2*dt+3*a3*dtdt )
    
    dt=t-ts[n-1]
    dtdt = dt*dt
    dtdtdt = dtdt*dt
    p += (ts[n-1]<=t)*( a0+a1*dt+a2*dtdt+a3*dtdtdt )
    v += (ts[n-1]<=t)*( a1+2*a2*dt+3*a3*dtdt )
    return p, v

def camera_project(quad_pos, quad_quat, point):
    v = point-quad_pos
    quad_quat_ = ca.vertcat(quad_quat[0], -quad_quat[1],-quad_quat[2],-quad_quat[3])
    v_b = rotate_quat(quad_quat_, v)
    return ca.vertcat( v_b[1]/v_b[0], v_b[2]/v_b[0] )

def aim_point(quad_pos, quad_quat, point):
    v = point-quad_pos
    siny = v[1]/ca.norm_2(v[:2])
    cosy = v[0]/ca.norm_2(v[:2])
    yaw = ca.acos(cosy)*ca.sign(siny)
    c_yaw = ca.cos(yaw/2)
    s_yaw = ca.sin(yaw/2)

    qw = quad_quat[0]
    qz = quad_quat[3]
    q_sqrt = ca.sqrt(qw*qw+qz*qz)
    delta_yaw = ca.vertcat(qw-c_yaw*q_sqrt, qz-s_yaw*q_sqrt)
    
    return delta_yaw.T@delta_yaw

class TrackerPosVel():
    def __init__(self, quad:QuadrotorModel):
        self._quad = quad
        self._Herizon = 10
        self._ddynamics = []
        for n in range(self._Herizon):
            self._ddynamics += [self._quad.ddynamics(0.05+n*0.0)]
        
        self._X_dim = self._ddynamics[0].size1_in(0)
        self._U_dim = self._ddynamics[0].size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub
        
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        self._t = ca.SX.sym('t', self._Herizon)
        
        self._X_init = ca.SX.sym("X_init", self._X_dim)
        self._trj_N = 15
        self._Trj_p = ca.SX.sym("Trj_p", 3, self._trj_N)
        self._Trj_v = ca.SX.sym("Trj_v", 3, self._trj_N)
        self._Trj_dt = ca.SX.sym("Trj_dt", 1, self._trj_N)
        self._Trj_t_ts =  [0]
        for i in range(self._trj_N-1):
            self._Trj_t_ts.append(self._Trj_t_ts[i] + self._Trj_dt[i])
        self._Aim_point = ca.SX.sym("Aim_point", 3)

        self._opt_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 25,
            # 'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0,
        }
        
        self._nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []
        
        self._nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []
        
        self._nlp_x_t = []
        self._nlp_lbx_t = []
        self._nlp_ubx_t = []

        self._nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []

        self._nlp_g_t = []
        self._nlp_lbg_t = []
        self._nlp_ubg_t = []
        
        self._nlp_p_xinit = []
        self._nlp_p_Trj_p = []
        self._nlp_p_Trj_v = []
        self._nlp_p_Trj_dt = []
        
        self._nlp_obj_dyn = 0
        self._nlp_obj_trjp = 0
        self._nlp_obj_trjv = 0
        self._nlp_obj_aim_point = 0
        self._nlp_x_x += [ self._Xs[:, 0] ]
        self._nlp_lbx_x += self._X_lb
        self._nlp_ubx_x += self._X_ub
        self._nlp_x_u += [ self._Us[:, 0] ]
        self._nlp_lbx_u += self._U_lb
        self._nlp_ubx_u += self._U_ub
        
        dd_dyn = self._Xs[:,0]-self._ddynamics[0]( self._X_init, self._Us[:,0] )
        self._nlp_g_dyn += [ dd_dyn ]
        self._nlp_obj_dyn += dd_dyn.T@dd_dyn
        self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
        self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
        
        trjp = linear(self._trj_N, self._t[0], self._Trj_p, self._Trj_t_ts)
        trjv = linear(self._trj_N, self._t[0], self._Trj_v, self._Trj_t_ts)
        self._nlp_obj_trjp += p_cost(self._Xs[:3,0]-trjp, 0.5)
        self._nlp_obj_trjv += (self._Xs[3:6,0]-trjv).T@(self._Xs[3:6,0]-trjv)
        # vv = (self._Xs[3:6,0].T@trjv/(trjv.T@trjv))*trjv
        # self._nlp_obj_trjv += (vv-trjv).T@(vv-trjv)

        # pv_norm = camera_project(self._Xs[:3,0], self._Xs[6:10,0], self._Aim_point)
        self._nlp_obj_aim_point += aim_point(self._Xs[:3,0], self._Xs[6:10,0], self._Aim_point)
        
        self._nlp_x_t += [self._t[0]]
        self._nlp_lbx_t += [0]
        self._nlp_ubx_t += [ca.inf]
        
        for i in range(1,self._Herizon):
            self._nlp_x_x += [ self._Xs[:, i] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, i] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            
            dd_dyn = self._Xs[:,i]-self._ddynamics[i]( self._Xs[:,i-1], self._Us[:,i] )
            self._nlp_g_dyn += [ dd_dyn ]
            self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
            
            trjp = linear(self._trj_N, self._t[i], self._Trj_p, self._Trj_t_ts)
            trjv = linear(self._trj_N, self._t[i], self._Trj_v, self._Trj_t_ts)
            self._nlp_obj_trjp += p_cost(self._Xs[:3,i]-trjp, 0.5)
            self._nlp_obj_trjv += (self._Xs[3:6,i]-trjv).T@(self._Xs[3:6,i]-trjv)
            # vv = (self._Xs[3:6,i].T@trjv/(trjv.T@trjv))*trjv
            # self._nlp_obj_trjv += (vv-trjv).T@(vv-trjv)

            # pv_norm = camera_project(self._Xs[:3,i], self._Xs[6:10,i], self._Aim_point)
            self._nlp_obj_aim_point += aim_point(self._Xs[:3,i], self._Xs[6:10,i], self._Aim_point)
            
            self._nlp_x_t += [self._t[i]]
            self._nlp_lbx_t += [0]
            self._nlp_ubx_t += [ca.inf]

        self._nlp_p_xinit += [self._X_init]
        

        for i in range(1, self._Herizon):
            self._nlp_g_t += [self._t[i]-self._t[i-1]]
            self._nlp_lbg_t += [0]
            self._nlp_ubg_t += [ ca.inf ]
        self._nlp_g_t += [ self._t[-1] ]
        self._nlp_lbg_t += [0]
        self._nlp_ubg_t += [20]

        for i in range(self._trj_N):
            self._nlp_p_Trj_p += [self._Trj_p[:,i]]
            self._nlp_p_Trj_v += [self._Trj_v[:,i]]
            self._nlp_p_Trj_dt += [self._Trj_dt[:,i]]
    
    def reset_xut(self):
        self._xut0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon+self._Herizon)
        self._xut0[-self._Herizon]=0
        for i in range(self._Herizon):
            self._xut0[i*self._X_dim+6] = 1

    def load_so(self, so_path):
        self._opt_solver = ca.nlpsol("opt", "ipopt", so_path, self._opt_option)
        self.reset_xut()
    
    def define_opt(self):
        nlp_dect = {
            'f': 10*self._nlp_obj_trjp+2*self._nlp_obj_trjv, #+0.0*self._nlp_obj_aim_point,#+30*self._nlp_obj_trjyaw,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_t)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_Trj_p+self._nlp_p_Trj_v+self._nlp_p_Trj_dt+[self._Aim_point])),
            'g': ca.vertcat(*(self._nlp_g_dyn+self._nlp_g_t)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        
        self.reset_xut()
        
    
    def solve(self, xinit, Trjp, Trjv, dt, aim, max_l:float):
        p = np.zeros(self._X_dim+3*self._trj_N+3*self._trj_N+self._trj_N+3)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim+3*self._trj_N] = Trjp
        p[self._X_dim+3*self._trj_N:self._X_dim+6*self._trj_N] = Trjv
        p[self._X_dim+6*self._trj_N:self._X_dim+7*self._trj_N] = dt
        p[self._X_dim+7*self._trj_N:self._X_dim+7*self._trj_N+3] = aim
        res = self._opt_solver(
            x0=self._xut0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_t),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_t),
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_t),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_t),
            p=p
        )
        
        self._xut0 = res['x'].full().flatten()
        
        return res

class TrackerPosVel2():
    def __init__(self, quad:QuadrotorModel):
        self._quad = quad
        self._Herizon = 10
        self._ddynamics = []
        for n in range(self._Herizon):
            self._ddynamics += [self._quad.ddynamics(0.05+n*0.0)]
        
        self._X_dim = self._ddynamics[0].size1_in(0)
        self._U_dim = self._ddynamics[0].size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub
        
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        self._t = ca.SX.sym('t', self._Herizon)
        
        self._X_init = ca.SX.sym("X_init", self._X_dim)
        self._trj_N = 30
        self._Trj_ploy = ca.SX.sym("Trj_ploy", 12, self._trj_N)
        self._Trj_dt = ca.SX.sym("Trj_dt", 1, self._trj_N)
        self._Trj_t_ts =  [0]
        for i in range(self._trj_N-1):
            self._Trj_t_ts.append(self._Trj_t_ts[i] + self._Trj_dt[i])

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
        
        self._nlp_x_t = []
        self._nlp_lbx_t = []
        self._nlp_ubx_t = []

        self._nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []

        self._nlp_g_t = []
        self._nlp_lbg_t = []
        self._nlp_ubg_t = []
        
        self._nlp_p_xinit = []
        self._nlp_p_Trj_ploy = []
        self._nlp_p_Trj_dt = []
        
        self._nlp_obj_dyn = 0
        self._nlp_obj_trjp = 0
        self._nlp_obj_trjv = 0
        self._nlp_obj_t = 0
        self._nlp_x_x += [ self._Xs[:, 0] ]
        self._nlp_lbx_x += self._X_lb
        self._nlp_ubx_x += self._X_ub
        self._nlp_x_u += [ self._Us[:, 0] ]
        self._nlp_lbx_u += self._U_lb
        self._nlp_ubx_u += self._U_ub
        
        dd_dyn = self._Xs[:,0]-self._ddynamics[0]( self._X_init, self._Us[:,0] )
        self._nlp_g_dyn += [ dd_dyn ]
        self._nlp_obj_dyn += dd_dyn.T@dd_dyn
        self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
        self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
        
        trjp, trjv = traj_ploy(self._trj_N, self._Trj_ploy, self._Trj_t_ts, self._t[0])
        self._nlp_obj_trjp += p_cost(self._Xs[:3,0]-trjp, 0.5)
        self._nlp_obj_trjv += (self._Xs[3:6,0]-trjv).T@(self._Xs[3:6,0]-trjv)
        # vv = (self._Xs[3:6,0].T@trjv/(trjv.T@trjv))*trjv
        # self._nlp_obj_trjv += (vv-trjv).T@(vv-trjv)
        self._nlp_obj_t += self._t[0]
        
        self._nlp_x_t += [self._t[0]]
        self._nlp_lbx_t += [0]
        self._nlp_ubx_t += [ca.inf]
        
        for i in range(1,self._Herizon):
            self._nlp_x_x += [ self._Xs[:, i] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, i] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            
            dd_dyn = self._Xs[:,i]-self._ddynamics[i]( self._Xs[:,i-1], self._Us[:,i] )
            self._nlp_g_dyn += [ dd_dyn ]
            self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
            
            trjp, trjv = traj_ploy(self._trj_N, self._Trj_ploy, self._Trj_t_ts, self._t[0]+i*0.05)
            self._nlp_obj_trjp += p_cost(self._Xs[:3,i]-trjp, 0.5)
            self._nlp_obj_trjv += (self._Xs[3:6,i]-trjv).T@(self._Xs[3:6,i]-trjv)
            # vv = (self._Xs[3:6,i].T@trjv/(trjv.T@trjv))*trjv
            # self._nlp_obj_trjv += (vv-trjv).T@(vv-trjv)
            
            # self._nlp_x_t += [self._t[i]]
            # self._nlp_lbx_t += [0]
            # self._nlp_ubx_t += [ca.inf]

        self._nlp_p_xinit += [self._X_init]
        

        # for i in range(1, self._Herizon):
        #     self._nlp_g_t += [self._t[i]-self._t[i-1]]
        #     self._nlp_lbg_t += [0]
        #     self._nlp_ubg_t += [ ca.inf ]
        # self._nlp_g_t += [ self._t[-1] ]
        # self._nlp_lbg_t += [0]
        # self._nlp_ubg_t += [20]

        for i in range(self._trj_N):
            self._nlp_p_Trj_ploy += [self._Trj_ploy[:,i]]
            self._nlp_p_Trj_dt += [self._Trj_dt[:,i]]
    
    def reset_xut(self):
        self._xut0 = np.zeros((self._X_dim+self._U_dim)*self._Herizon+1)
        # self._xut0[-self._Herizon]=0
        for i in range(self._Herizon):
            self._xut0[i*self._X_dim+6] = 1

    def load_so(self, so_path):
        self._opt_solver = ca.nlpsol("opt", "ipopt", so_path, self._opt_option)
        self.reset_xut()
    
    def define_opt(self):
        nlp_dect = {
            'f': 10*self._nlp_obj_trjp+1*self._nlp_obj_trjv,#+30*self._nlp_obj_trjyaw,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_t)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_Trj_ploy+self._nlp_p_Trj_dt)),
            'g': ca.vertcat(*(self._nlp_g_dyn+self._nlp_g_t)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        
        self.reset_xut()
        
    
    def solve(self, xinit, Trj, dt, max_l:float):
        p = np.zeros(self._X_dim+12*self._trj_N+self._trj_N)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim+12*self._trj_N] = Trj
        p[self._X_dim+12*self._trj_N:self._X_dim+13*self._trj_N] = dt
        res = self._opt_solver(
            x0=self._xut0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_t),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_t),
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_t),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_t),
            p=p
        )
        
        self._xut0 = res['x'].full().flatten()
        
        return res

class TrackerMPCC():
    def __init__(self, quad:QuadrotorModel):
        self._quad = quad
        self._Herizon = 10
        self._ddynamics = []
        for n in range(self._Herizon):
            self._ddynamics += [self._quad.ddynamics(0.06+n*0.01)]
        
        self._X_dim = self._ddynamics[0].size1_in(0)
        self._U_dim = self._ddynamics[0].size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub
        
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        self._l = ca.SX.sym('l', self._Herizon)
        self._max_l = ca.SX.sym("max_l", 1)
        
        self._X_init = ca.SX.sym("X_init", self._X_dim)
        self._trj_N = 30
        self._Trj_p = ca.SX.sym("Trj_p", 3, self._trj_N)
        self._Trj_p_ls = [0]
        for i in range(self._trj_N-1):
            self._Trj_p_ls.append(self._Trj_p_ls[i] + ca.sqrt( (self._Trj_p[:,i+1]-self._Trj_p[:,i]).T@(self._Trj_p[:,i+1]-self._Trj_p[:,i]) ))
        self._max_v = ca.SX.sym("max_v",1)
        self._punish_v_coef = ca.SX.sym("v_coef",1)

        self._opt_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 25,
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

        self._nlp_g_l = []
        self._nlp_lbg_l = []
        self._nlp_ubg_l = []

        self._nlp_g_v = []
        self._nlp_lbg_v = []
        self._nlp_ubg_v = []
        
        self._nlp_p_xinit = []
        self._nlp_p_Trj_p = []
        self._nlp_p_max_v = [self._max_v] # for safty
        self._nlp_p_punish_v_coef = [self._punish_v_coef]
        
        self._nlp_obj_dyn = 0
        self._nlp_obj_trjp = 0
        self._nlp_obj_trjv = 0
        self._nlp_obj_l = 0
        self._nlp_x_x += [ self._Xs[:, 0] ]
        self._nlp_lbx_x += self._X_lb
        self._nlp_ubx_x += self._X_ub
        self._nlp_x_u += [ self._Us[:, 0] ]
        self._nlp_lbx_u += self._U_lb
        self._nlp_ubx_u += self._U_ub
        
        dd_dyn = self._Xs[:,0]-self._ddynamics[0]( self._X_init, self._Us[:,0] )
        self._nlp_g_dyn += [ dd_dyn ]
        self._nlp_obj_dyn += dd_dyn.T@dd_dyn
        self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
        self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
        
        self._nlp_obj_l += self._l[0]
        # trjp = linear(self._trj_N, self._l[0], self._Trj_p, self._Trj_p_ls)
        # self._nlp_obj_trjp += p_cost(self._Xs[:3,0]-trjp, 0.5)
        trjp, trjv = linear2(self._trj_N, self._l[0], self._Trj_p, self._Trj_p_ls)
        self._nlp_obj_trjp += 1*ca.cross(trjv, self._Xs[:3,0]-trjp).T@ca.cross(trjv, self._Xs[:3,0]-trjp) + 1*(trjv.T@(self._Xs[:3,0]-trjp))*(trjv.T@(self._Xs[:3,0]-trjp))
        self._nlp_obj_trjv += trjv.T@(self._Xs[3:6,0])
        
        self._nlp_x_l += [self._l[0]]
        self._nlp_lbx_l += [0]
        self._nlp_ubx_l += [ca.inf]
        
        for i in range(1,self._Herizon):
            self._nlp_x_x += [ self._Xs[:, i] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, i] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            
            dd_dyn = self._Xs[:,i]-self._ddynamics[i]( self._Xs[:,i-1], self._Us[:,i] )
            self._nlp_g_dyn += [ dd_dyn ]
            self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
            
            self._nlp_obj_l += self._l[i]
            # trjp = linear(self._trj_N, self._l[i], self._Trj_p, self._Trj_p_ls)
            # self._nlp_obj_trjp += p_cost(self._Xs[:3,i]-trjp, 0.5)
            trjp, trjv = linear2(self._trj_N, self._l[i], self._Trj_p, self._Trj_p_ls)
            self._nlp_obj_trjp += 1*ca.cross(trjv, self._Xs[:3,i]-trjp).T@ca.cross(trjv, self._Xs[:3,i]-trjp) + 1*(trjv.T@(self._Xs[:3,i]-trjp))*(trjv.T@(self._Xs[:3,i]-trjp))
            self._nlp_obj_trjv += trjv.T@(self._Xs[3:6,i])
            
            self._nlp_x_l += [self._l[i]]
            self._nlp_lbx_l += [0]
            self._nlp_ubx_l += [ca.inf]
        
            self._nlp_g_v += [ self._Xs[3:6, i].T@self._Xs[3:6, i] - self._max_v*self._max_v ]
            self._nlp_lbg_v += [-ca.inf]
            self._nlp_ubg_v += [ 0 ]
            

        self._nlp_p_xinit += [self._X_init]
        

        for i in range(1, self._Herizon):
            self._nlp_g_l += [self._l[i]-self._l[i-1]]
            self._nlp_lbg_l += [0]
            self._nlp_ubg_l += [ ca.inf ]
        self._nlp_g_l += [ self._l[-1] ]
        self._nlp_lbg_l += [0]
        self._nlp_ubg_l += [20]

        for i in range(self._trj_N):
            self._nlp_p_Trj_p += [self._Trj_p[:,i]]
    
    def reset_xul(self):
        self._xul0 = np.zeros((self._X_dim+self._U_dim+1)*self._Herizon)
        self._xul0[-self._Herizon]=0
        for i in range(self._Herizon):
            self._xul0[i*self._X_dim+6] = 1

    def load_so(self, so_path):
        self._opt_solver = ca.nlpsol("opt", "ipopt", so_path, self._opt_option)
        self.reset_xul()
    
    def define_opt(self):
        nlp_dect = {
            'f': -1*self._nlp_obj_trjv+30*(self._nlp_obj_trjp),#+30*self._nlp_obj_trjyaw,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_l)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_Trj_p)),
            'g': ca.vertcat(*(self._nlp_g_dyn+self._nlp_g_l)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        
        self.reset_xul()
        
    
    def solve(self, xinit, Trjp, max_l:float):
        p = np.zeros(self._X_dim+3*self._trj_N)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim+3*self._trj_N] = Trjp
        res = self._opt_solver(
            x0=self._xul0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_l),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_l),
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_l),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_l),
            p=p
        )
        
        self._xul0 = res['x'].full().flatten()
        
        return res

class TrackerP():
    def __init__(self, quad:QuadrotorModel):
        self._quad = quad
        self._Herizon = 10
        self._ddynamics = self._quad.ddynamics_dt()
        # for n in range(self._Herizon):
        #     self._ddynamics += [self._quad.ddynamics_dt()]
        
        self._X_dim = self._ddynamics.size1_in(0)
        self._U_dim = self._ddynamics.size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub
        
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Herizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Herizon)
        self._dt = ca.SX.sym('dt', self._Herizon)
        
        self._X_init = ca.SX.sym("X_init", self._X_dim)
        self._Trj_p = ca.SX.sym("Trj_p", 3, self._Herizon)
        self._Trj_v = ca.SX.sym("Trj_v", 3, self._Herizon)
        
        self._opt_option = {
            'verbose': False,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 200,
            # 'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0,
        }
        
        self._nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []
        
        self._nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []
        
        self._nlp_x_dt = []
        self._nlp_lbx_dt = []
        self._nlp_ubx_dt = []

        self._nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []

        self._nlp_g_v = []
        self._nlp_lbg_v = []
        self._nlp_ubg_v = []
        
        self._nlp_p_xinit = []
        self._nlp_p_Trj_p = []
        self._nlp_p_Trj_v = []
        
        self._nlp_obj_dyn = 0
        self._nlp_obj_trjp = 0
        self._nlp_obj_trjv = 0
        

        self._nlp_x_x += [ self._Xs[:, 0] ]
        self._nlp_lbx_x += self._X_lb
        self._nlp_ubx_x += self._X_ub
        self._nlp_x_u += [ self._Us[:, 0] ]
        self._nlp_lbx_u += self._U_lb
        self._nlp_ubx_u += self._U_ub
        
        dd_dyn = self._Xs[:,0]-self._ddynamics( self._X_init, self._Us[:,0], self._dt[0] )
        self._nlp_g_dyn += [ dd_dyn ]
        self._nlp_obj_dyn += dd_dyn.T@dd_dyn
        self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
        self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
        
        self._nlp_x_dt += [self._dt[0]]
        self._nlp_lbx_dt += [0.02]
        self._nlp_ubx_dt += [0.2]

        self._nlp_obj_trjp += (self._Xs[:3,0]-self._Trj_p[:,0]).T@(self._Xs[:3,0]-self._Trj_p[:,0])
        vv=self._Trj_v[:,0]/ca.norm_2(self._Trj_v[:,0])
        self._nlp_obj_trjv += (vv.T@self._Xs[3:6,0])
        
        for i in range(1,self._Herizon):
            self._nlp_x_x += [ self._Xs[:, i] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [ self._Us[:, i] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            
            dd_dyn = self._Xs[:,i]-self._ddynamics( self._Xs[:,i-1], self._Us[:,i], self._dt[i] )
            self._nlp_g_dyn += [ dd_dyn ]
            self._nlp_obj_dyn += dd_dyn.T@dd_dyn
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
            
            self._nlp_x_dt += [self._dt[i]]
            self._nlp_lbx_dt += [0.02]
            self._nlp_ubx_dt += [0.2]

            self._nlp_obj_trjp += (self._Xs[:3,i]-self._Trj_p[:,i]).T@(self._Xs[:3,i]-self._Trj_p[:,i])
            vv=self._Trj_v[:,i]/ca.norm_2(self._Trj_v[:,i])
            self._nlp_obj_trjv += (vv.T@self._Xs[3:6,i])
        

        self._nlp_p_xinit += [self._X_init]

        for i in range(self._Herizon):
            self._nlp_p_Trj_p += [self._Trj_p[:,i]]
            self._nlp_p_Trj_v += [self._Trj_v[:,i]]
    
    def reset_xut(self):
        self._xut0 = np.zeros((self._X_dim+self._U_dim+1)*self._Herizon)
        for i in range(self._Herizon):
            self._xut0[-i-1] = 0.1
            self._xut0[i*self._X_dim+6] = 1

    def load_so(self, so_path):
        self._opt_solver = ca.nlpsol("opt", "ipopt", so_path, self._opt_option)
        self.reset_xut()
    
    def define_opt(self):
        nlp_dect = {
            'f': 100*self._nlp_obj_trjp-0.0001*self._nlp_obj_trjv,#+30*self._nlp_obj_trjyaw,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_dt)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_Trj_p+self._nlp_p_Trj_v)),
            'g': ca.vertcat(*(self._nlp_g_dyn)),
        }
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        
        self.reset_xut()
        
    
    def solve(self, xinit, Trjp, Trjv):
        p = np.zeros(self._X_dim+3*self._Herizon+3*self._Herizon)
        p[:self._X_dim] = xinit
        p[self._X_dim:self._X_dim+3*self._Herizon] = Trjp
        p[self._X_dim+3*self._Herizon:self._X_dim+6*self._Herizon] = Trjv

        res = self._opt_solver(
            x0=self._xut0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_dt),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_dt),
            lbg=(self._nlp_lbg_dyn),
            ubg=(self._nlp_ubg_dyn),
            p=p
        )
        
        self._xut0 = res['x'].full().flatten()
        
        return res


if __name__ == "__main__":
    
    quad = QuadrotorModel(BASEPATH+"quad/quad_real.yaml")
    tracker = TrackerOpt(quad)
    tracker.define_opt()
    tracker._opt_solver.generate_dependencies("tracker_opt.c")
    
