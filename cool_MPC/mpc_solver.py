
import torch
from torch.autograd.functional import jacobian
import numpy as np
from scipy.optimize import minimize

from cool_linear_solver import Variable, Constrained_least_squares

from matplotlib import pyplot as plt

from .tictoctimer import Tictoctimer

class MPC_solver(object):
    def __init__(self, f_batched, h_batched, nu, ny, nx=None, eps_tol=0.015):
        self.f_batched = f_batched
        self.h_batched = h_batched
        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.timer = Tictoctimer()
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
    def dfdx(self, x, u):
        x = torch.tensor(x,dtype=torch.float32,requires_grad=True)
        u = torch.tensor(u,dtype=torch.float32)
        fnow = lambda x: self.f_batched(x[None],u[None])[0]
        return jacobian(fnow, x).numpy()
    def dfdu(self, x, u):
        x = torch.tensor(x,dtype=torch.float32)
        u = torch.tensor(u,dtype=torch.float32,requires_grad=True)
        fnow = lambda u: self.f_batched(x[None],u[None])[0]
        return jacobian(fnow, u).numpy()
    def dhdx(self, x, u):
        x = torch.tensor(x,dtype=torch.float32,requires_grad=True)
        u = torch.tensor(u,dtype=torch.float32)
        hnow = lambda x: self.h_batched(x[None],u[None])[0]
        return jacobian(hnow, x).numpy()
    def dhdu(self, x, u):
        x = torch.tensor(x,dtype=torch.float32)
        u = torch.tensor(u,dtype=torch.float32,requires_grad=True)
        hnow = lambda u: self.h_batched(x[None],u[None])[0]
        return jacobian(hnow, u).numpy()

    def integrate(self, x_init, u_seq):
        x = torch.as_tensor(x_init, dtype=torch.float32)[None]
        Y = []
        with torch.no_grad():
            for u in torch.as_tensor(u_seq, dtype=torch.float32):
                u = u[None]
                self.timer.tic('h_batched')
                y = self.h_batched(x,u)[0]
                self.timer.toc('h_batched')

                Y.append(y.numpy())
                self.timer.tic('f_batched')
                x = self.f_batched.forward(x,u)
                self.timer.toc('f_batched')
        return np.array(Y)

    def evaluate(self, x_init, u_seq, y_targets, y_weight, u_weight):
        
        Y_controlled = self.integrate(x_init, u_seq)
        # self.timer.toc('integrate')
        diff_y = (Y_controlled-np.array(y_targets))*np.array(y_weight)
        diff_u = np.array(u_seq)*u_weight
        MSE = 1/2*(np.sum(diff_y**2) + np.sum(diff_u**2))/(np.prod(diff_y.shape,dtype=int) + np.prod(diff_u.shape,dtype=int))
        return MSE
    
    def solve_it(self, x_init, T, u_inits, y_targets, y_weight=1.0, u_weight=1e-5, u_bounds=None, verbose=0):
        f, h, dfdx, dfdu, dhdx, dhdu = self.f, self.h, self.dfdx, self.dfdu, self.dhdx, self.dhdu
        x0t = np.copy(x_init)
        dxt = np.zeros_like(x0t)
        du = Variable('du')
        
        self.timer.start()
        self.timer.tic('lpv prop')
        errors = []
        for t,(u0t,y_target) in enumerate(zip(u_inits, y_targets)):
            yt = h(x0t,u0t) + dhdx(x0t,u0t)@dxt + np.dot(dhdu(x0t,u0t),du[t])
            errors.append(yt - y_target)
            
            dxt = dfdx(x0t,u0t)@dxt + dfdu(x0t,u0t)*du[t]
            x0t = f(x0t,u0t)
        self.timer.toc('lpv prop')

        self.timer.tic('eqs_make')
        sys = Constrained_least_squares()
        for eq in errors:
            if self.ny is not None:
                if isinstance(y_weight,(int,float)):
                    y_weight = [y_weight]*self.ny
                for eqi, y_weight_i in zip(eq,y_weight):
                    sys.add_objective(eqi*y_weight_i)
            else:
                sys.add_objective(eq)

        if u_weight is not None:
            for t in range(T):
                sys.add_objective((u_inits[t]+du[t])*u_weight)
        
        if u_bounds is not None:
            for t in range(T):
                umin, umax = u_bounds
                sys.add_inequality(u_inits[t]+du[t] <= umax)
                sys.add_inequality(umin <= u_inits[t]+du[t])
        self.timer.toc('eqs_make')
        self.timer.tic('solve')
        sys.solve()
        self.timer.toc('solve')

        dusol = [sys[du[i]] for i in range(T)]
        def unow(eps):
            u = [eps*dui + u0i for dui, u0i in zip(dusol, u_inits)]
            return u if u_bounds is None else np.clip(u, umin, umax)
        eval_now = lambda eps: self.evaluate(x_init, unow(eps[0]), y_target, y_weight, u_weight)
        out = minimize(fun=eval_now, x0=[0.5], method='Nelder-Mead', tol=self.eps_tol)

        eps_best = out.x[0]
        if verbose:
            print('eps:', eps_best, 'val', out.fun, 'nit', out.nit)
        self.timer.pause()

        usol = unow(eps_best)
        return usol

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