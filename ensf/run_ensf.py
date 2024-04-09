import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import sys
import os
import pandas as pd
import logging
import os, argparse
from mpi4py import MPI


from problem_gen import *


# compact version
def cond_alpha(t):
    # conditional information
    # alpha_t(0) = 1
    # alpha_t(1) = esp_alpha \approx 0
    return 1 - (1 - eps_alpha) * t


def cond_sigma_sq(t):
    # conditional sigma^2
    # sigma2_t(0) = 0
    # sigma2_t(1) = 1
    # sigma(t) = t
    return eps_sigma_sq + (1 - eps_sigma_sq) * t


# drift function of forward SDE
def f(t):
    # f=d_(log_alpha)/dt
    alpha_t = cond_alpha(t)
    f_t = -(1 - eps_alpha) / alpha_t
    return f_t


def g_sq(t):
    # g = d(sigma_t^2)/dt -2f sigma_t^2
    d_sigma_sq_dt = (1 - eps_sigma_sq)
    g2 = d_sigma_sq_dt - 2 * f(t) * cond_sigma_sq(t)
    return g2


def g(t):
    return np.sqrt(g_sq(t))


# generate sample with reverse SDE
def reverse_SDE(x0, score_likelihood=None, time_steps=100,
                correction_step=0, correction_eps=0.001,
                drift_fun=f, diffuse_fun=g, alpha_fun=cond_alpha, sigma2_fun=cond_sigma_sq,
                save_path=False):
    # x_T: sample from standard Gaussian
    # x_0: target distribution to sample from

    # reverse SDE sampling process
    # N1 = x_T.shape[0]
    # N2 = x0.shape[0]
    # d = x_T.shape[1]

    # Generate the time mesh
    dt = 1.0 / time_steps

    # Initialization
    xt = torch.randn(ensemble_size, n_dim, device=device)
    t = 1.0

    # define storage
    if save_path:
        path_all = [xt]
        t_vec = [t]

    # forward Euler sampling
    for i in range(time_steps):
        # prior score evaluation
        alpha_t = alpha_fun(t)
        sigma2_t = sigma2_fun(t)

        # Evaluate the diffusion term
        diffuse = diffuse_fun(t)

        # Evaluate the drift term
        # drift = drift_fun(t)*xt - diffuse**2 * score_eval

        # print(i,xt.abs().max(),torch.sum(torch.isnan(xt)))

        # Update
        if score_likelihood is not None:
            xt += - dt * (
                        drift_fun(t) * xt - diffuse ** 2 * (-(xt - alpha_t * x0) / sigma2_t + score_likelihood(xt, t))) \
                  + np.sqrt(dt) * diffuse * torch.randn_like(xt)
        else:
            xt += - dt * (drift_fun(t) * xt + diffuse ** 2 * ((xt - alpha_t * x0) / sigma2_t)) + np.sqrt(
                dt) * diffuse * torch.randn_like(xt)

        # correction
        for j in range(correction_step):
            xt += (correction_eps / 2.0 * (-(xt - alpha_t * x0) / sigma2_t + score_likelihood(xt, t)) +
                   np.sqrt(correction_eps) * torch.randn_like(xt))

        # Store the state in the path
        if save_path:
            path_all.append(xt)
            t_vec.append(t)

        # update time
        t = t - dt

    if save_path:
        return path_all, t_vec
    else:
        return xt


def auto_score(xt, obs, forward_model=None):
    """
    Get score with automatic differentiation, if no forward model, then it is just likelihood
    :param xt: (ensemble_num, dim)
    :param obs: (obs_dim)
    :return: score (ensemble_num, dim)
    """
    xt.requires_grad_(True)
    xt.grad = None
    if forward_model is not None:
        y_hat = forward_model(xt)
    else:
        y_hat = xt

    # removing some obs
    y_hat_rm = y_hat[:, select_dim]
    obs_rm = obs[select_dim]

    # loss error
    loss = torch.sum((y_hat_rm - obs_rm) ** 2) * (-0.5) / obs_sigma ** 2

    loss.backward()
    grad = xt.grad
    xt.requires_grad_(False)
    return grad


def ana_score(xt, obs):
    # likelihood score only
    # obs: (d)
    # xt: (ensemble, d)

    score_x = -(torch.atan(xt) - obs)/obs_sigma**2 * (1./(1. + xt**2))
    # score_x = -(xt - obs) / obs_sigma ** 2

    ## delete score information
    score_x[:, ~select_idx] = 0
    return score_x


# define likelihood score
def likelihood_score_t(xt, t, score_fun):
    # obs: (d)
    # xt: (ensemble, d)

    # static score
    score_x = score_fun(xt)

    # damping function
    tau = g_tau(t)
    return tau * score_x

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='EnSF command line arguments', \
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    args = parser.parse_args()

    seed = args.seed
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()


    def print0(*msg):
        if rank == 0:
            msgs = " ".join(map(str,msg))   
            print(msgs)


    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    ####################################################################
    ####################################################################



    # computation setting
    # default_dtype = torch.float64
    default_dtype = torch.float32
    # default_dtype = torch.float16

    torch.set_default_dtype(default_dtype)

    device = 'cuda'
    # device = 'cpu'

    logging.basicConfig(level=logging.INFO, filename="log_atan.log", filemode="w+",
                        format="%(asctime)s - %(message)s",
                        datefmt='%m/%d/%Y %I:%M:%S %p')


    # load problem
    problem_all = torch.load('problem_all.pt')
    t_vec = problem_all['t_vec']
    path_all = problem_all['path_all']
    noise_all = problem_all['noise_all']

    path_all = path_all.to(device=device, dtype=default_dtype)
    noise_all = noise_all.to(device=device, dtype=default_dtype)

    print0('path_all: ', path_all.shape, path_all.device, path_all.dtype )
    print0('noise_all: ',noise_all.shape, noise_all.device, noise_all.dtype )
    logging.info(f'path_all: {path_all.shape} {path_all.device} {path_all.dtype}')
    logging.info(f'noise_all: {noise_all.shape} {noise_all.device} {noise_all.dtype}')

    # get problem info
    n_dim = problem_all['n_dim']
    dt_true = problem_all['dt_true']
    N_step = problem_all['N_step']
    step_forward_true = problem_all['step_forward_true']
    solver_type_true = problem_all['solver_type_true']
    rep_total = problem_all['rep_total']


    # ode integrator
    ode_int = IntegratorODE(drift_fun=lorenz96_drift)



    # load parameter
    ####################################################################

    case_name = 'my_case'
    # obs parameter
    obs_sigma = 0.1
    obs_ratio = 1  # full observation
    obs_gap = 10  # gap for each observation


    # EnSF setup
    method_name = 'my_method'
    # define the diffusion model
    eps_alpha = 0.01
    eps_sigma_sq = 0.05

    # ensemble size
    ensemble_size = 1

    # forward model solver
    pred_dt_ratio = 1  # forward solver discretization
    step_forward = obs_gap * pred_dt_ratio
    solver_type = 'rk4'

    # update step sampling
    update_cycle = 1
    update_sampling_step = 100
    update_correction_step = 0
    update_correction_eps = 0.01


    # pre-prediction sampling
    pred_cycle = 3
    pred_sampling_step = 100
    pred_correction_step = 0
    pred_correction_eps = 0.01


    g_tau_scale = 1
    # damping function(tau(0) = 1;  tau(1) = 0;)
    def g_tau(t):
        return (1-t)*g_tau_scale

    #######################

    # partial obs
    select_dim = torch.arange(n_dim//obs_ratio, device=device)*obs_ratio
    select_idx = torch.zeros(n_dim, device=device, dtype=torch.bool)
    select_idx[select_dim] = True


    total_filtering_cycle = int(N_step/obs_gap)  # total number of filterong cycles

    dt_obs = dt_true*obs_gap  # time gap for each observation
    ####################################################################


    # filtering
    ####################################################################
    full_name = f'{case_name}_{method_name}'
    print0(full_name)
    logging.info(full_name)

    rmse_all = []
    est_all = []
    true_state_all = []
    obs_all = []
    for rep_id in range(rep_total):
        print0(f'rep-{rep_id}:')
        logging.info(f'rep-{rep_id}:')
        path_true = path_all[rep_id]
        noise_true = noise_all[rep_id]

        # set seed
        torch.manual_seed(rep_id)

        # initial state ensemble (true)
        x_state = path_true[0] + torch.randn(ensemble_size, n_dim, device=device)*0.0

        # info containers
        rmse_save0 = []
        state_save0 = []
        est_save0 = []
        obs_save0 = []

        # filtering cycles
        t_current = 0
        t_vec_temp = []
        torch.cuda.empty_cache()
        t1 = time.time()
        for i in range(total_filtering_cycle):

            ##############################################################
            # get state/obs
            state_target = path_true[(i+1)*obs_gap]
            # obs = state_target + noise_true[i*obs_gap] * obs_sigma
            obs = torch.atan(state_target) + noise_true[i*obs_gap] * obs_sigma

            # define the forward model
            forward_fun = partial(ode_int.integrate, t_start=t_current, t_end=t_current+dt_obs, num_steps=step_forward, method=solver_type, save_path=False)
            ##############################################################



            ##############################################################
            # pre-prediction step

            if pred_cycle > 0:
                # define likelihood score
                score_pre = partial(auto_score, obs=obs, forward_model=forward_fun)
                score_pre_t = partial(likelihood_score_t, score_fun = score_pre)
                for _ in range(pred_cycle):
                    x_state = reverse_SDE(x0=x_state, score_likelihood=score_pre_t,
                                          time_steps=pred_sampling_step,
                                          correction_step=pred_correction_step, correction_eps=pred_correction_eps)
            ##############################################################



            ##############################################################
            # prediction step

            # state forward in time
            x_state = forward_fun(x_state)
            t_current += dt_obs
            ##############################################################



            ##############################################################
            # update step

            if update_cycle == -1:
                # simply replace state with obs
                x_state[:, select_idx] = obs[select_idx]
            #

            if update_cycle > 0:
                # define likelihood score
                # score_obs = partial(auto_score, obs=obs)
                score_obs = partial(ana_score, obs=obs)
                score_obs_t = partial(likelihood_score_t, score_fun=score_obs)
                # generate posterior sample
                for _ in range(update_cycle):
                    x_state = reverse_SDE(x0=x_state, score_likelihood=score_obs_t,
                                          time_steps=update_sampling_step,
                                          correction_step=update_correction_step, correction_eps=update_correction_eps)
            ##############################################################

            # get state estimates and rmse
            x_est = torch.mean(x_state, dim=0)
            rmse_temp = torch.sqrt(torch.mean((x_est - state_target) ** 2))

            # get time
            # if x_state.device.type == 'cuda':
            #     torch.cuda.current_stream().synchronize()
            # t2 = time.time()
            # print(f'\t RMSE = {rmse_temp.item():.4f}')
            # print(f'\t time = {t2 - t1:.4f} ')

            # save information
            rmse_save0.append(rmse_temp.item())
            state_save0.append(state_target)
            est_save0.append(x_est)
            obs_save0.append(obs)
            t_vec_temp.append(t_current)

            # break
            if rmse_temp.item() > 1000:
                print0('diverge!')
                logging.info('diverge!')
                break

            if torch.isnan(rmse_temp):
                print0('NaN!')
                logging.info('NaN!')
                break

        # sync time
        if x_state.device.type == 'cuda':
            torch.cuda.current_stream().synchronize()
        t2 = time.time()
        print0(f'\t final RMSE = {rmse_temp.item():.4f}')
        logging.info(f'\t final RMSE = {rmse_temp.item():.4f}')
        print0(f'\t time = {t2 - t1:.4f} ')
        logging.info(f'\t time = {t2 - t1:.4f} ')

        # save result
        state_save0 = torch.stack(state_save0, dim=0).cpu().numpy()
        est_save0 = torch.stack(est_save0, dim=0).cpu().numpy()
        obs_save0 = torch.stack(obs_save0, dim=0).cpu().numpy()
        rmse_save0 = np.array(rmse_save0)

        # save result
        rmse_all.append(rmse_save0)
        est_all.append(est_save0)
        obs_all.append(obs_save0)
        true_state_all.append(state_save0)

    # save all result
    rmse_all = np.stack(rmse_all, axis=0)
    est_all = np.stack(est_all, axis=0)
    obs_all = np.stack(obs_all, axis=0)
    true_state_all = np.stack(true_state_all, axis=0)
    t_vec_temp = np.array(t_vec_temp)

    # collect
    all_state = comm.reduce(true_state_all, root=0)
    all_obs = comm.reduce(obs_all, root=0)
    all_est = comm.reduce(est_all, root=0)
    all_rmse = comm.reduce(rmse_all, root=0)

    if rank == 0: 
        all_state = all_state / world_size
        all_obs = all_obs / world_size
        all_est = all_est / world_size
        all_rmse = all_rmse / world_size

        np.savez(f'result_{full_name}.npz',
             rmse_all=all_rmse,
             est_all=all_est,
             true_state_all=all_state,
             obs_all=all_obs,
             select_idx=select_idx.cpu().numpy(),
             t_vec_temp=t_vec_temp)

        # show results
        plt.figure(figsize=(6, 6))
        plt.plot(t_vec_temp, rmse_all[0], 'r', alpha=0.5)
        plt.title('RMSE')
        plt.grid()
        plt.savefig(f'preview_{full_name}.png', dpi=200)
        plt.close()
