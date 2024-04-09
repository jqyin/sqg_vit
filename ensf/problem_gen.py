import torch
import numpy as np

# the lorenz drift
def lorenz96_drift(x, t):
    return (torch.roll(x, -1) - torch.roll(x, 2))*torch.roll(x, 1) - x + 8


class IntegratorODE:
    def __init__(self, drift_fun=None):
        self.drift_fun = drift_fun

    def integrate(self, xt, t_start, t_end, num_steps, method='fe', save_path=False):
        """
        Solving a ODE from t_start to t_end with a batch of inputs
        """
        if self.drift_fun is None:
            raise ValueError('Drift function not specified!')

        dt = (t_end - t_start) / num_steps
        t_current = t_start

        path_all = []
        t_vec = []

        # saving the path
        if save_path:
            path_all.append(xt)
            t_vec.append(t_current)

        for i in range(num_steps):
            xt = self.integrate_one_step(xt, t_current, dt, method)
            t_current += dt
            # saving the path
            if save_path:
                path_all.append(xt)
                t_vec.append(t_current)

        if save_path:
            return xt, [path_all, t_vec]
        else:
            return xt

    def integrate_one_step(self, xt, t_current, dt, method):
        """
        perform one-step update with specified method
        """
        if method == 'rk4':
            return self.rk4(xt=xt, f=self.drift_fun, t=t_current, dt=dt)
        elif method == 'fe':
            return self.forward_euler(xt=xt, f=self.drift_fun, t=t_current, dt=dt)
        else:
            raise NotImplementedError('Not implemented!')

    @staticmethod
    def forward_euler(xt, f, t, dt):
        xt = xt + f(xt, t) * dt
        return xt

    @staticmethod
    def rk4(xt, f, t, dt):
        k1 = f(xt, t)
        k2 = f(xt + dt / 2 * k1, t + dt / 2)
        k3 = f(xt + dt / 2 * k2, t + dt / 2)
        k4 = f(xt + dt * k3, t + dt)
        return xt + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def set_drift_fun(self, drift_fun):
        self.drift_fun = drift_fun


# filtering settings
# lorenz system
n_dim = 1000000
dt_true = 0.01
N_step = 100

# forward model solver
step_forward_true = 5
solver_type_true = 'rk4'

#
rep_total = 1

t_vec = torch.linspace(0, N_step*dt_true, N_step+1)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    # initial seed
    state_target = 10 * torch.rand(n_dim)

    ode_int = IntegratorODE(drift_fun=lorenz96_drift)

    path_all = []
    noise_all = []
    print('generating L96 path:')
    # generating path
    for rep_id in range(rep_total):
        print('\tpath', rep_id)
        # burn in stage
        N_burn_in = 500
        print(f'\t\tburn-in: {N_burn_in} steps')
        t_current = 0.
        for i in range(N_burn_in):
            state_target = ode_int.integrate(state_target, t_start=t_current, t_end=t_current+dt_true,
                                             num_steps=step_forward_true, method=solver_type_true)
            t_current += dt_true

        print(f'\t\tgenerating path: in {N_step} steps')
        path_0 = [state_target]
        t_current = 0.
        for i in range(N_step):
            state_target = ode_int.integrate(state_target, t_start=t_current, t_end=t_current+dt_true,
                                             num_steps=step_forward_true, method=solver_type_true)
            path_0.append(state_target)
            t_current += dt_true
        path_0 = torch.stack(path_0, dim=0)


        # generate fixed noise
        noise_0 = []
        for i in range(N_step):
            noise = torch.randn(n_dim)
            noise_0.append(noise)
        noise_0 = torch.stack(noise_0, dim=0)

        path_all.append(path_0)
        noise_all.append(noise_0)

    path_all = torch.stack(path_all, dim=0)
    noise_all = torch.stack(noise_all, dim=0)

    problem_all = {'t_vec': t_vec, 'path_all': path_all, 'noise_all': noise_all,
                   'rep_total': rep_total, 'n_dim': n_dim, 'dt_true': dt_true,
                   'N_step': N_step, 'step_forward_true': step_forward_true, 'solver_type_true':solver_type_true }
    torch.save(problem_all, 'problem_all.pt')

