

from torch import nn
import torch
from matplotlib import pyplot as plt
import numpy as np

#### Define system ####
# the f and h have the first dimention as batch dimention on x, u, and y 
class RK4_f(nn.Module):
    def forward(self, x, u):
        k1 = self.dt*self.deriv(x,u)
        k2 = self.dt*self.deriv(x+k1/2,u)
        k3 = self.dt*self.deriv(x+k2/2,u)
        k4 = self.dt*self.deriv(x+k3,u)
        return x + (k1+2*k2+2*k3+k4)/6

class pendulum_f(RK4_f):
    def __init__(self, m, l, g, dt, gamma=0.01):
        super().__init__()
        self.m = m
        self.l = l
        self.g = g
        self.dt = dt
        self.gamma = gamma
    
    def deriv(self, x, u):
        th, dth = x.T
        dthdt = dth
        ddthdt = -g/(l*m)*torch.sin(th) + u[:,0]*u[:,1] - dth*self.gamma
        return torch.stack([dthdt,ddthdt],dim=1)
    
class pendulum_h(nn.Module):
    def __init__(self, m, l, g):
        super().__init__()
        self.m = m
        self.l = l
        self.g = g
    
    def forward(self, x, u):
        th, dth = x.T
        sinth = torch.sin(th)
        costh = torch.cos(th)
        energy = 1/2*self.m*self.l**2*dth**2 - self.m*self.l*self.g*(1 + torch.cos(th))
        return torch.stack([sinth,costh+1,energy,th],dim=1)

### create system ###
m = 1
l = 1
g = 1
dt = 0.3
h = pendulum_h(m, l, g)
f = pendulum_f(m, l, g, dt)
nu = 2    # 2 inputs
ny = 4    # 4 outputs
nx = 2    # 2 states

### define target ###
y_target = [0, 0, 0, 0]  # target output
x_init = [0, 0]          # initial state

### MPC paramters ###
T = 50                                # MPC time horizon
u_inits = [[1]*nu]*T                  # initial estimated inputs
y_targets = [y_target]*T              # y_targets is the same for all time
y_weight = [0.0, 0.0, 1.0, 0]
u_bounds = [(-0.3,0.3), (-0.3, 0.3)]  # bounds on the input
u_weight = 0.01                       # weight on the input

from cool_MPC import MPC_solver
mpc = MPC_solver(f, h, nu, ny, nx=nx)

## prepare plotting ##
plt.figure(figsize=(12,4))
ax1 = plt.subplot(1,3,1)
plt.title('input (u)')
plt.xlabel('index time')
ax2 = plt.subplot(1,3,2)
plt.title('energy')
plt.xlabel('index time')
ax3 = plt.subplot(1,3,3)
plt.title('angle')
plt.xlabel('index time')

import time

t_start = time.time()
u_sol = mpc.solve(x_init=x_init, T=T, u_inits=u_inits, \
    y_targets=y_targets, y_weight=y_weight, u_weight=u_weight, \
        plot=False, u_bounds=u_bounds, verbose=1)
print('time', time.time() - t_start)

ax1.plot(u_sol)
ysol = mpc.integrate(x_init, u_sol)
ax2.plot(np.array(ysol)[:,2])
ax3.plot(np.array(ysol)[:,3])
plt.show()