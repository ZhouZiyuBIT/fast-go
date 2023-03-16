import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

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

qw = ca.SX.sym("qw")
qx = ca.SX.sym("qx")
qy = ca.SX.sym("qy")
qz = ca.SX.sym("qz")

vx = ca.SX.sym("vx")
vy = ca.SX.sym("vy")
vz = ca.SX.sym("vz")

vv = rotate_quat(ca.vertcat(qw, qx, qy,qz), ca.vertcat(vx,vy,vz))
print(vv)
