
from torch import nn
import torch

from matplotlib import pyplot as plt



class RK4_f(nn.Module):
    def forward(self, x, u):
        k1 = self.dt*self.deriv(x,u)
        k2 = self.dt*self.deriv(x+k1/2,u)
        k3 = self.dt*self.deriv(x+k2/2,u)
        k4 = self.dt*self.deriv(x+k3,u)
        return x + (k1+2*k2+2*k3+k4)/6

class Silverbox_f(RK4_f):
    def __init__(self, delta=0.1, alpha=1, beta=0.05, gamma = 1, dt = 0.1):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
    
    def deriv(self, x, u):
        x, xd = x.T
        dxddt = self.gamma*u - self.delta*xd - self.alpha*x - self.beta*x**3
        dxdt = xd
        return torch.stack([dxdt,dxddt],dim=1)

class Silverbox_h(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, u):
        return x[:,0]

nx = 2
nu = None
ny = None
f = Silverbox_f()
h = Silverbox_h()

from cool_MPC import MPC_solver

import time
import numpy as np

time0 = time.time()
x_init = [1, 0]
betas =  [0, 0.1, 0.2, 0.5, 2, 5, 10, 20, 50, 100]
for beta in betas:
    f = Silverbox_f(beta=beta)
    h = Silverbox_h()
    T = 30
    print('beta', beta)
    mpc = MPC_solver(f, h, nu, ny, nx=nx)
    u_sol = mpc.solve(x_init=x_init, T=T, u_inits=[0.1]*T, y_targets=[0]*T, u_weight=0.05, verbose=1, line_search_start=1)
    # print(mpc.timer.percent())
    plt.plot(mpc.integrate(x_init, u_sol))
plt.legend([str(a) for a in betas])
print('total time:',time.time()-time0)
print()
plt.show()
