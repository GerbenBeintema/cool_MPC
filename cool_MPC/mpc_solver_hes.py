
from tabnanny import verbose
import torch
from torch.autograd.functional import jacobian, hessian
import numpy as np
from scipy.optimize import minimize

from cool_linear_solver import Variable, Constrained_least_squares

from matplotlib import pyplot as plt

from .tictoctimer import Tictoctimer

class MPC_solver_hes(object):
    def __init__(self, f_batched, h_batched, nu, ny, nx=None, eps_tol=0.015):
        self.f_batched = f_batched
        self.h_batched = h_batched
        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.eps_tol = eps_tol

    def f(self, x, u):
        x = torch.as_tensor(x,dtype=torch.float32)
        u = torch.as_tensor(u,dtype=torch.float32)
        return self.f_batched(x[None],u[None]).detach().numpy()[0]
    def h(self, x, u):
        x = torch.as_tensor(x,dtype=torch.float32)
        u = torch.as_tensor(u,dtype=torch.float32)
        y = self.h_batched(x[None],u[None]).detach().numpy()[0]
        return y
    def evaluate(self, x_init, u_seq, y_targets, y_weight, u_weight):
        Losses = []
        x = x_init
        for u, ytarg in zip(u_seq, y_targets):
            ynow = self.h_batched(x[None], u[None])[0]
            Losses.append(torch.sum(y_weight*(ynow-ytarg)**2) + torch.sum(u_weight*u**2))
            x = self.f_batched(x[None],u[None])[0]
        return torch.mean(torch.stack(Losses))

    def integrate(self, x_init, u_seq):
        x = x_init
        Y = []
        for u in u_seq:
            Y.append(self.h(x,u))
            x = self.f(x,u)
        return np.array(Y)

    def solve_it(self, x_init, T, u_inits, y_targets, y_weight=1.0, u_weight=1e-5, u_bounds=None, verbose=0):
        x_init = torch.as_tensor(x_init, dtype=torch.float32)
        u_inits = torch.as_tensor(u_inits, dtype=torch.float32)
        y_targets = torch.as_tensor(y_targets, dtype=torch.float32)
        y_weight = torch.as_tensor(y_weight, dtype=torch.float32)
        u_weight = torch.as_tensor(u_weight, dtype=torch.float32)

        eval_u = lambda u_seq: self.evaluate(x_init=x_init, u_seq=u_seq, y_targets=y_targets, y_weight=y_weight, u_weight=u_weight)
        fun_val = lambda u_seq: eval_u(torch.as_tensor(u_seq)).detach().numpy()
        fun_J = lambda u_seq: jacobian(eval_u, torch.as_tensor(u_seq)).numpy()
        fun_H = lambda u_seq: hessian(eval_u, torch.as_tensor(u_seq)).numpy()

        u_seq = np.array(u_inits)
        self.timer.tic('J cal')
        J = fun_J(u_seq)
        self.timer.toc('J cal')
        self.timer.tic('H cal')
        H = fun_H(u_seq) #update if bounds are given
        self.timer.toc('H cal')
        self.timer.tic('solve')
        du_seq = np.linalg.solve(H, -J)
        self.timer.toc('solve')


        self.timer.tic('minimize')
        g = lambda eps: fun_val(u_seq+eps[0]*du_seq)
        out = minimize(g, [1.], method='Nelder-Mead', tol=self.eps_tol)
        eps_best = out.x[0]
        if verbose:
            print('eps:', eps_best, 'val', out.fun)
        self.timer.toc('minimize')
        
        u_seq = u_seq + eps_best*du_seq
        return u_seq

    def solve(self, x_init, T, u_inits, y_targets, y_weight=1.0, u_weight=1e-5, u_bounds=None, u_diff_tol = 1e-4, plot=False, verbose=0):
        self.timer = Tictoctimer()
        if plot:
            ax1 = plt.subplot(1,2,1)
            plt.title('y')
            plt.grid()
            ax2 = plt.subplot(1,2,2)
            plt.title('u')
            plt.grid()
        
        u_old = u_inits
        k = 0
        while True:
            k += 1
            u_new = self.solve_it(x_init=x_init, T=T, u_inits=u_old, y_targets=y_targets, y_weight=y_weight, u_weight=u_weight, u_bounds=u_bounds, verbose=verbose)
            
            if plot:
                ax1.plot(self.integrate(x_init, u_new))
                ax2.plot(u_new)
            
            u_diff = np.mean((np.array(u_new)-np.array(u_old))**2)**0.5
            if verbose:
                print(f'Itteration {k} u_diff={u_diff}')
            if u_diff < u_diff_tol:
                return u_new
            else:
                u_old = u_new