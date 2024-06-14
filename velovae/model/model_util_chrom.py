import logging
import numpy as np
from anndata import AnnData
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import scvelo as scv
from .scvelo_util import mRNA, tau_inv, test_bimodality
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.stats import dirichlet, bernoulli, kstest
from tqdm.notebook import tqdm_notebook
from .model_util import assign_gene_mode_binary
from scipy.stats import median_abs_deviation
from scipy.sparse import issparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
logger = logging.getLogger(__name__)


def inv(x):
    x_inv = 1 / x * (x != 0)
    return x_inv


def chromatin(tau, c0, alpha_c):
    expac = np.exp(-alpha_c * tau)
    return c0 * expac + 1 * (1 - expac)


def unspliced(tau, u0, c0, alpha_c, alpha, beta):
    expac = np.exp(-alpha_c * tau)
    expb = np.exp(-beta * tau)
    const = (1 - c0) * alpha * inv(beta - alpha_c)
    return u0 * expb + alpha * inv(beta) * (1 - expb) + const * (expb - expac)


def spliced(tau, s0, u0, c0, alpha_c, alpha, beta, gamma):
    expac = np.exp(-alpha_c * tau)
    expb = np.exp(-beta * tau)
    expg = np.exp(-gamma * tau)
    const = (1 - c0) * alpha * inv(beta - alpha_c)
    out = s0 * expg + (alpha * inv(gamma)) * (1 - expg)
    out += (beta * inv(gamma - beta)) * ((alpha * inv(beta)) - u0 - const) * (expg - expb)
    out += (beta * inv(gamma - alpha_c)) * const * (expg - expac)
    return out


def compute_exp(tau, c0, u0, s0, kc, alpha_c, rho, alpha, beta, gamma):
    expac, expb, expg = np.exp(-alpha_c * tau), np.exp(-beta * tau), np.exp(-gamma * tau)
    const = (kc - c0) * rho * alpha * inv(beta - alpha_c)
    c = c0 * expac + kc * (1 - expac)
    u = u0 * expb + (rho * alpha * kc * inv(beta)) * (1 - expb) + const * (expb - expac)
    s = s0 * expg + (rho * alpha * kc * inv(gamma)) * (1 - expg)
    s += (beta * inv(gamma - beta)) * ((rho * alpha * kc * inv(beta)) - u0 - const) * (expg - expb)
    s += (beta * inv(gamma - alpha_c)) * const * (expg - expac)
    return c, u, s


def vectorize(t, t_, alpha_c, alpha, beta, gamma=None, c0=0, u0=0, s0=0, sorted=False):
    o = np.array(t < t_, dtype=int)
    tau = t * o + (t - t_) * (1 - o)

    c0_ = chromatin(t_, c0, alpha_c)
    u0_ = unspliced(t_, u0, c0, alpha_c, alpha, beta)
    s0_ = spliced(t_, s0, u0, c0, alpha_c, alpha, beta, gamma if gamma is not None else beta / 2)

    # vectorize u0, s0 and alpha
    c0 = c0 * o + c0_ * (1 - o)
    u0 = u0 * o + u0_ * (1 - o)
    s0 = s0 * o + s0_ * (1 - o)
    rho = 1 * o + 0 * (1 - o)
    kc = 1 * o + 0 * (1 - o)

    if sorted:
        idx = np.argsort(t)
        tau, rho, kc, c0, u0, s0 = tau[idx], rho[idx], kc[idx], c0[idx], u0[idx], s0[idx]
    return tau, rho, kc, c0, u0, s0


def pred_single(t, alpha_c, alpha, beta, gamma, ts, cinit=0, uinit=0, sinit=0):
    tau, rho, kc, c0, u0, s0 = vectorize(t, ts, alpha_c, alpha, beta, gamma, c0=cinit, u0=uinit, s0=sinit)
    tau = np.clip(tau, a_min=0, a_max=None)
    ct, ut, st = compute_exp(tau, c0, u0, s0, kc, alpha_c, rho, alpha, beta, gamma)
    return ct.squeeze(), ut.squeeze(), st.squeeze()


def pred_exp(tau, c0, u0, s0, kc, alpha_c, rho, alpha, beta, gamma):
    expac, expb, expg = torch.exp(-alpha_c*tau), torch.exp(-beta*tau), torch.exp(-gamma*tau)
    eps = 1e-6

    cpred = c0*expac + kc*(1-expac)
    upred = u0*expb + rho*alpha*kc/beta*(1-expb) + (kc-c0)*rho*alpha/(beta-alpha_c+eps)*(expb-expac)
    spred = s0*expg + rho*alpha*kc/gamma*(1-expg)
    spred += (rho*alpha*kc/beta-u0-(kc-c0)*rho*alpha/(beta-alpha_c+eps))*beta/(gamma-beta+eps)*(expg-expb)
    spred += (kc-c0)*rho*alpha*beta/(gamma-alpha_c+eps)/(beta-alpha_c+eps)*(expg-expac)
    return cpred, upred, spred


def pred_exp_numpy(tau, c0, u0, s0, kc, alpha_c, rho, alpha, beta, gamma):
    expac, expb, expg = np.exp(-alpha_c*tau), np.exp(-beta*tau), np.exp(-gamma*tau)
    eps = 1e-6

    cpred = c0*expac + kc*(1-expac)
    upred = u0*expb + rho*alpha*kc/beta*(1-expb) + (kc-c0)*rho*alpha/(beta-alpha_c+eps)*(expb-expac)
    spred = s0*expg + rho*alpha*kc/gamma*(1-expg)
    spred += (rho*alpha*kc/beta-u0-(kc-c0)*rho*alpha/(beta-alpha_c+eps))*beta/(gamma-beta+eps)*(expg-expb)
    spred += (kc-c0)*rho*alpha*beta/(gamma-alpha_c+eps)/(beta-alpha_c+eps)*(expg-expac)
    return np.clip(cpred, a_min=0, a_max=1), np.clip(upred, a_min=0, a_max=None), np.clip(spred, a_min=0, a_max=None)


def pred_exp_numpy_backward(tau, c, u, s, kc, alpha_c, rho, alpha, beta, gamma):
    expac, expb, expg = np.exp(alpha_c*tau), np.exp(beta*tau), np.exp(gamma*tau)
    eps = 1e-6

    c0pred = c*expac + kc*(1-expac)
    c0pred = np.clip(c0pred, a_min=0, a_max=1)
    u0pred = u*expb + rho*alpha*kc/beta*(1-expb) + (kc-c)*rho*alpha/(beta-alpha_c+eps)*(np.exp((beta-alpha_c)*tau)-1)
    u0pred = np.clip(u0pred, a_min=0, a_max=None)
    s0pred = s*expg + rho*alpha*kc/gamma*(1-expg)
    s0pred += (rho*alpha*kc/beta-u-(kc-c)*rho*alpha/(beta-alpha_c+eps))*beta/(gamma-beta+eps)*(np.exp((gamma-beta)*tau)-1)
    s0pred += (kc-c)*rho*alpha*beta/(gamma-alpha_c+eps)/(beta-alpha_c+eps)*(np.exp((gamma-alpha_c)*tau)-1)
    s0pred = np.clip(s0pred, a_min=0, a_max=None)
    return c0pred, u0pred, s0pred


def linreg(u, s):
    q = np.sum(s*s)
    r = np.sum(u*s)
    k = r/q
    if np.isinf(k) or np.isnan(k):
        k = 1.0+np.random.rand()
    return k


def assign_time_rna(u, s, u0_, s0_, alpha, beta, gamma, t_num=1000, t_max=20):
    tau = np.linspace(0, t_max, t_num)
    exp_mat = np.hstack((np.full((len(u), 1), 1), np.reshape(u, (-1, 1)), np.reshape(s, (-1, 1))))
    ct, ut, st = compute_exp(tau, 1, 0, 0, 1, 0, 1, alpha, beta, gamma)
    anchor_mat = np.hstack((np.reshape(ct, (-1, 1)), np.reshape(ut, (-1, 1)), np.reshape(st, (-1, 1))))
    tree = KDTree(anchor_mat)
    dd, ii = tree.query(exp_mat, k=1)
    dd = dd**2
    t_pred = tau[ii]

    exp_ss = np.array([[1, u0_, s0_]])
    ii_ss = tree.query(exp_ss, k=1)[1]
    t_ = tau[ii_ss][0]
    c0_pred, u0_pred, s0_pred = compute_exp(t_, 1, 0, 0, 1, 0, 1, alpha, beta, gamma)

    ct_, ut_, st_ = compute_exp(tau, c0_pred, u0_pred, s0_pred, 1, 0, 0, alpha, beta, gamma)
    anchor_mat_ = np.hstack((np.reshape(ct_, (-1, 1)), np.reshape(ut_, (-1, 1)), np.reshape(st_, (-1, 1))))
    tree_ = KDTree(anchor_mat_)
    dd_, ii_ = tree_.query(exp_mat, k=1)
    dd_ = dd_**2
    t_pred_ = tau[ii_]

    res = np.array([dd, dd_])
    # t = np.array([t_pred, t_pred_+np.ones((len(t_pred_)))*t_])
    t = np.array([t_pred, t_pred_+t_])
    o = np.argmin(res, axis=0)
    t_latent = np.array([t[o[i], i] for i in range(len(t_pred))])

    return t_latent, t_


def assign_time(c, u, s, c0_, u0_, s0_, alpha_c, alpha, beta, gamma, std_c_=None, std_s=None, weight_c=0.6, t_num=1000, t_max=20):
    tau = np.linspace(0, t_max, t_num)
    exp_mat = np.hstack((np.reshape(c, (-1, 1))/std_c_*std_s*weight_c, np.reshape(u, (-1, 1)), np.reshape(s, (-1, 1))))
    ct, ut, st = compute_exp(tau, 0, 0, 0, 1, alpha_c, 1, alpha, beta, gamma)
    anchor_mat = np.hstack((np.reshape(ct, (-1, 1))/std_c_*std_s*weight_c, np.reshape(ut, (-1, 1)), np.reshape(st, (-1, 1))))
    tree = KDTree(anchor_mat)
    dd, ii = tree.query(exp_mat, k=1)
    dd = dd**2
    t_pred = tau[ii]

    exp_ss = np.array([[c0_/std_c_*std_s*weight_c, u0_, s0_]])
    ii_ss = tree.query(exp_ss, k=1)[1]
    t_ = tau[ii_ss][0]
    c0_pred, u0_pred, s0_pred = compute_exp(t_, 0, 0, 0, 1, alpha_c, 1, alpha, beta, gamma)

    ct_, ut_, st_ = compute_exp(tau, c0_pred, u0_pred, s0_pred, 0, alpha_c, 0, alpha, beta, gamma)
    anchor_mat_ = np.hstack((np.reshape(ct_, (-1, 1))/std_c_*std_s*weight_c, np.reshape(ut_, (-1, 1)), np.reshape(st_, (-1, 1))))
    tree_ = KDTree(anchor_mat_)
    dd_, ii_ = tree_.query(exp_mat, k=1)
    dd_ = dd_**2
    t_pred_ = tau[ii_]

    res = np.array([dd, dd_])
    # t = np.array([t_pred, t_pred_+np.ones((len(t_pred_)))*t_])
    t = np.array([t_pred, t_pred_+t_])
    o = np.argmin(res, axis=0)
    t_latent = np.array([t[o[i], i] for i in range(len(t_pred))])

    return t_latent, t_


def init_gene_rna(u, s, percent, fit_scaling=True, tmax=1):
    std_u, std_s = np.std(u), np.std(s)
    scaling = np.clip(std_u / std_s, 1e-6, 1e6) if fit_scaling else 1.0
    if np.isnan(scaling):
        scaling = 1.0
    u = u/scaling

    # initialize beta and gamma from extreme quantiles of s
    mask_s = s >= np.percentile(s, percent)
    mask_u = u >= np.percentile(u, percent)
    mask = mask_s & mask_u
    if np.sum(mask) < 10:
        mask = mask_s

    # initialize alpha, beta and gamma
    beta = 1.0
    gamma = linreg(u[mask], s[mask]) + 1e-6
    if gamma < 0.05 / scaling:
        gamma *= 1.2
    elif gamma > 1.5 / scaling:
        gamma /= 1.2
    u_inf, s_inf = u[mask].mean(), s[mask].mean()
    u0_, s0_ = u_inf, s_inf
    alpha = u_inf*beta

    # initialize switching from u quantiles and alpha from s quantiles
    tstat_u, pval_u, means_u = test_bimodality(u, kde=True)
    tstat_s, pval_s, means_s = test_bimodality(s, kde=True)
    pval_steady = max(pval_u, pval_s)
    steady_u = means_u[1]
    if pval_steady < 1e-3:
        u_inf = np.mean([u_inf, steady_u])
        alpha = gamma * s_inf
        beta = alpha / u_inf
        u0_, s0_ = u_inf, s_inf
    t_ = tau_inv(u0_, s0_, 0, 0, alpha, beta, gamma)
    tau = tau_inv(u, s, 0, 0, alpha, beta, gamma)
    tau = np.clip(tau, 0, t_)
    tau_ = tau_inv(u, s, u0_, s0_, 0, beta, gamma)
    tau_ = np.clip(tau_, 0, np.max(tau_[s > 0]))
    ut, st = mRNA(tau, 0, 0, alpha, beta, gamma)
    ut_, st_ = mRNA(tau_, u0_, s0_, 0, beta, gamma)
    distu, distu_ = (u - ut), (u - ut_)
    dists, dists_ = (s - st), (s - st_)
    res = np.array([distu ** 2 + dists ** 2, distu_ ** 2 + dists_ ** 2])
    t = np.array([tau, tau_+np.ones((len(tau_)))*t_])
    o = np.argmin(res, axis=0)
    t_latent = np.array([t[o[i], i] for i in range(len(tau))])

    realign_ratio = tmax / np.max(t_latent)
    alpha, beta, gamma = alpha/realign_ratio, beta/realign_ratio, gamma/realign_ratio
    t_ *= realign_ratio
    t_latent *= realign_ratio
    return alpha, beta, gamma, t_latent, u0_, s0_, t_, scaling


def init_gene(c, u, s, percent, fit_scaling=True, tmax=1):
    std_u, std_s = np.std(u), np.std(s)
    scaling = np.clip(std_u / std_s, 1e-6, 1e6) if fit_scaling else 1.0
    if np.isnan(scaling):
        scaling = 1.0
    u = u/scaling
    scaling_c = np.clip(np.percentile(c, 99.5), 1e-3, None)
    c = c/scaling_c
    std_c_ = np.clip(np.std(c), 1e-6, None)

    # initialize beta and gamma from extreme quantiles of s
    thresh = np.mean(c) + np.std(c)
    mask_c = c >= thresh
    if np.sum(mask_c) < 5:
        mask_c = c >= np.mean(c)
    u_, s_ = u[mask_c], s[mask_c]
    mask_s = s >= np.percentile(s_, percent)
    mask_u = u >= np.percentile(u_, percent)
    mask = mask_s & mask_u & mask_c
    if np.sum(mask) < 10:
        mask = mask_s & mask_c
    if np.sum(mask) < 10:
        mask = mask_s

    # initialize alpha, beta and gamma
    beta = 1.0
    gamma = linreg(u[mask], s[mask]) + 1e-6
    if gamma < 0.05 / scaling:
        gamma *= 1.2
    elif gamma > 1.5 / scaling:
        gamma /= 1.2
    c_inf, u_inf, s_inf = c[mask].mean(), u[mask].mean(), s[mask].mean()
    c0_, u0_, s0_ = c_inf, u_inf, s_inf
    alpha = u_inf*beta/c_inf

    # initialize switching from u quantiles and alpha from s quantiles
    tstat_u, pval_u, means_u = test_bimodality(u, kde=True)
    tstat_s, pval_s, means_s = test_bimodality(s, kde=True)
    pval_steady = max(pval_u, pval_s)
    steady_u = means_u[1]
    if pval_steady < 1e-3:
        u_inf = np.mean([u_inf, steady_u])
        alpha = gamma * s_inf / c_inf
        beta = alpha * c_inf / u_inf
        u0_, s0_ = u_inf, s_inf
    t_ = tau_inv(u0_, s0_, 0, 0, alpha, beta, gamma)
    alpha_c = -np.log(1 - np.clip(np.median(c[mask]), 0.001, 0.999)) / t_
    if alpha_c == beta:
        alpha_c -= 1e-3
    if alpha_c == gamma:
        gamma += 1e-3
    t_latent, t_ = assign_time(c, u, s, c0_, u0_, s0_, alpha_c, alpha, beta, gamma, std_c_, std_s)

    realign_ratio = tmax / np.max(t_latent)
    alpha_c, alpha, beta, gamma = alpha_c/realign_ratio, alpha/realign_ratio, beta/realign_ratio, gamma/realign_ratio
    t_ *= realign_ratio
    t_latent *= realign_ratio
    return alpha_c, alpha, beta, gamma, t_latent, c0_, u0_, s0_, t_, scaling_c, scaling


def init_params(c, u, s, percent, fit_scaling=True, global_std=False, tmax=1, rna_only=False):
    ngene = u.shape[1]
    params = np.ones((ngene, 6))  # alpha_c, alpha, beta, gamma, scaling_c, scaling
    params[:, 0] = 0.1
    params[:, 1] = np.clip(np.max(u, 0), 0.001, None)
    params[:, 3] = np.clip(np.max(u, 0), 0.001, None)/np.clip(np.max(s, 0), 0.001, None)
    params[:, 4] = np.clip(np.max(c, 0), 0.001, None)
    t = np.zeros((ngene, len(s)))
    ts = np.zeros((ngene))
    c0, u0, s0 = np.zeros((ngene)), np.zeros((ngene)), np.zeros((ngene))
    cpred, upred, spred = np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)

    for i in tqdm_notebook(range(ngene)):
        ci, ui, si = c[:, i], u[:, i], s[:, i]
        cfilt = ci[(si > 0) & (ui > 0) & (ci > 0)]
        ufilt = ui[(si > 0) & (ui > 0) & (ci > 0)]
        sfilt = si[(si > 0) & (ui > 0) & (ci > 0)]
        if (len(sfilt) >= 5):
            if rna_only:
                out = init_gene_rna(ufilt, sfilt, percent, fit_scaling, tmax)
                alpha, beta, gamma, t_, u0_, s0_, ts_, scaling = out
                c0_ = 1
                params[i, :] = np.array([0, alpha, beta, gamma, 1, scaling])
            else:
                out = init_gene(cfilt, ufilt, sfilt, percent, fit_scaling, tmax)
                alpha_c, alpha, beta, gamma, t_, c0_, u0_, s0_, ts_, scaling_c, scaling = out
                params[i, :] = np.array([alpha_c, alpha, beta, gamma, scaling_c, scaling])
            t[i, (si > 0) & (ui > 0) & (ci > 0)] = t_
            c0[i] = c0_
            u0[i] = u0_
            s0[i] = s0_
            ts[i] = ts_
        else:
            c0[i] = 1.0
            u0[i] = np.max(ui)
            s0[i] = np.max(si)
        if rna_only:
            c0[i] = 1
            params[i, 0] = 0
            params[i, 4] = 1

        cpred_i, upred_i, spred_i = pred_single(t[i],
                                                params[i, 0],
                                                params[i, 1],
                                                params[i, 2],
                                                params[i, 3],
                                                ts[i],
                                                cinit=1 if rna_only else 0)
        cpred[:, i] = cpred_i * params[i, 4]
        upred[:, i] = upred_i * params[i, 5]
        spred[:, i] = spred_i

    if global_std:
        sigma_c = np.nanstd(c, 0)
        sigma_u = np.nanstd(u, 0)
        sigma_s = np.nanstd(s, 0)
    else:
        dist_c = c - cpred
        dist_u = u - upred
        dist_s = s - spred
        sigma_c = np.nanstd(dist_c, 0)
        sigma_u = np.nanstd(dist_u, 0)
        sigma_s = np.nanstd(dist_s, 0)
    mu_c = np.nanmean(c, 0)
    mu_u = np.nanmean(u, 0)
    mu_s = np.nanmean(s, 0)

    alpha_c, alpha, beta, gamma = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    scaling_c, scaling = params[:, 4], params[:, 5]

    return alpha_c, alpha, beta, gamma, scaling_c, scaling, ts, c0, u0, s0, sigma_c, sigma_u, sigma_s, mu_c, mu_u, mu_s, t.T, cpred, upred, spred


def get_ts_global(tgl, c, u, s, perc):
    tsgl = np.zeros((u.shape[1]))
    for i in range(u.shape[1]):
        ci, ui, si = c[:, i], u[:, i], s[:, i]
        zero_mask = (ci > 0) & (ui > 0) & (si > 0)
        mask_c = ci >= np.mean(ci)
        mask_u, mask_s = ui >= np.percentile(ui[mask_c], perc), si >= np.percentile(si[mask_c], perc)
        tsgl[i] = np.median(tgl[mask_c & mask_u & mask_s & zero_mask])
        if np.isnan(tsgl[i]):
            tsgl[i] = np.median(tgl[mask_c & (mask_u | mask_s) & zero_mask])
        if np.isnan(tsgl[i]):
            tsgl[i] = np.median(tgl)
    assert not np.any(np.isnan(tsgl))
    return tsgl


def reinit_gene(c, u, s, t, ts):
    mask_c = (c >= np.mean(c[c < 1])) & (c < 1)
    if np.sum(mask_c) < 5:
        mask_c = c >= np.mean(c)
    mask1_u = u >= np.quantile(u[mask_c], 0.95)
    mask1_s = s >= np.quantile(s[mask_c], 0.95)
    c1, u1, s1 = np.median(c[mask_c & (mask1_u | mask1_s)]), np.median(u[mask_c & (mask1_u | mask1_s)]), np.median(s[mask_c & (mask1_s | mask1_u)])

    if u1 == 0 or np.isnan(u1):
        u1 = np.max(u)
    if s1 == 0 or np.isnan(s1):
        s1 = np.max(s)

    t1 = np.median(t[mask_c & (mask1_u | mask1_s)])
    if t1 <= 0:
        tm = np.max(t[mask_c & (mask1_u | mask1_s)])
        t1 = tm if tm > 0 else 1.0

    mask2_u = (u[mask_c] >= u1*0.49) & (u[mask_c] <= u1*0.51) & (t[mask_c] <= ts)
    mask2_s = (s[mask_c] >= s1*0.49) & (s[mask_c] <= s1*0.51) & (t[mask_c] <= ts)
    if np.any(mask2_u) or np.any(mask2_s):
        t2 = np.median(t[mask_c][mask2_u | mask2_s])
        if abs(t1 - t2) < 1e-3:
            t1 = t2 + 1e-3
        u2, _ = np.median(u[mask_c][mask2_u]), np.median(s[mask_c][mask2_s])
        t0 = max(0, np.log((u1-u2) / (u1*np.exp(-t2) - u2*np.exp(-t1))))
    else:
        t0 = 0
    beta = 1.0
    alpha = u1/(1-np.exp(t0-t1)) if u1 > 0 else 0.1*np.random.rand()
    alpha_c = -np.log(1 - c1) / (t1-t0) if t1 > t0 else 0.1
    if alpha <= 0 or np.isnan(alpha) or np.isinf(alpha):
        alpha = u1
    gamma = alpha / np.quantile(s[mask_c], 0.95)
    if gamma <= 0 or np.isnan(gamma) or np.isinf(gamma):
        gamma = 2.0
    return alpha_c, alpha, beta, gamma, t0


def reinit_params(c, u, s, t, ts):
    G = u.shape[1]
    alpha_c, alpha, beta, gamma, ton = np.zeros((G)), np.zeros((G)), np.zeros((G)), np.zeros((G)), np.zeros((G))
    for i in range(G):
        alpha_c_g, alpha_g, beta_g, gamma_g, ton_g = reinit_gene(c[:, i], u[:, i], s[:, i], t, ts[i])
        alpha_c[i] = alpha_c_g
        alpha[i] = alpha_g
        beta[i] = beta_g
        gamma[i] = gamma_g
        ton[i] = ton_g
    return alpha_c, alpha, beta, gamma, ton


def pred_steady(tau_s, alpha_c, alpha, beta, gamma):
    eps = 1e-6
    expac, expb, expg = torch.exp(-alpha_c*tau_s), torch.exp(-beta*tau_s), torch.exp(-gamma*tau_s)
    expbd = torch.exp(-beta*tau_s)
    c0 = torch.tensor([1.0], device=alpha.device)-expac
    u0 = alpha/(beta+eps)*(torch.tensor([1.0], device=alpha.device)-expbd) + alpha/(beta-alpha_c+eps)*(expbd-expac)
    s0 = alpha/(gamma+eps)*(torch.tensor([1.0], device=alpha.device)-expg)
    s0 += (alpha/beta-alpha/(beta-alpha_c+eps))*beta/(gamma-beta+eps)*(expg-expb)
    s0 += alpha*beta/(gamma-alpha_c+eps)/(beta-alpha_c+eps)*(expg-expac)
    return c0, u0, s0


def ode(t, alpha_c, alpha, beta, gamma, to, ts, neg_slope=0.0):
    eps = 1e-6
    o = (t <= ts).int()

    tau_on = F.leaky_relu(t-to, negative_slope=neg_slope)
    expac, expb, expg = torch.exp(-alpha_c*tau_on), torch.exp(-beta*tau_on), torch.exp(-gamma*tau_on)
    expbd = torch.exp(-beta*tau_on)
    chat_on = torch.tensor([1.0], device=alpha.device)-expac
    uhat_on = alpha/(beta+eps)*(torch.tensor([1.0], device=alpha.device)-expbd) + alpha/(beta-alpha_c+eps)*(expbd-expac)
    shat_on = alpha/(gamma+eps)*(torch.tensor([1.0], device=alpha.device)-expg)
    shat_on += (alpha/beta-alpha/(beta-alpha_c+eps))*beta/(gamma-beta+eps)*(expg-expb)
    shat_on += alpha*beta/(gamma-alpha_c+eps)/(beta-alpha_c+eps)*(expg-expac)

    c0_, u0_, s0_ = pred_steady(F.relu(ts-to), alpha_c, alpha, beta, gamma)

    tau_off = F.leaky_relu(t-ts, negative_slope=neg_slope)
    expac, expb, expg = torch.exp(-alpha_c*tau_off), torch.exp(-beta*tau_off), torch.exp(-gamma*tau_off)
    expbd = torch.exp(-beta*tau_off)
    chat_off = c0_*expac
    uhat_off = u0_*expbd
    shat_off = s0_*expg-u0_*beta/(gamma-beta+eps)*(expg-expb)
    return (chat_on*o + chat_off*(1-o)), (uhat_on*o + uhat_off*(1-o)), (shat_on*o + shat_off*(1-o))


def pred_steady_numpy(ts, alpha_c, alpha, beta, gamma):
    alpha_c_ = np.clip(alpha_c, a_min=0, a_max=None)
    alpha_ = np.clip(alpha, a_min=0, a_max=None)
    beta_ = np.clip(beta, a_min=0, a_max=None)
    gamma_ = np.clip(gamma, a_min=0, a_max=None)
    eps = 1e-6
    ts_ = ts.squeeze()
    expac, expb, expg = np.exp(-alpha_c_*ts_), np.exp(-beta_*ts_), np.exp(-gamma_*ts_)
    c0 = 1.0-expac
    u0 = alpha_/(beta_+eps)*(1.0-expb)+alpha_/(beta_-alpha_c_+eps)*(expb-expac)
    s0 = alpha_/(gamma_+eps)*(1.0-expg)+(alpha_/beta_-alpha_/(beta_-alpha_c_+eps))*beta_/(gamma_-beta_+eps)*(expg-expb)
    s0 += alpha_*beta_/(gamma_-alpha_c_+eps)/(beta_-alpha_c_+eps)*(expg-expac)
    return c0, u0, s0


def ode_numpy(t, alpha_c, alpha, beta, gamma, to, ts, scaling_c=None, scaling_u=None, scaling_s=None, offset_c=None, offset_u=None, offset_s=None, k=10.0):
    eps = 1e-6
    o = (t <= ts).astype(int)

    tau_on = F.softplus(torch.tensor(t-to), beta=k).numpy()
    assert np.all(~np.isnan(tau_on))
    expac, expb, expg = np.exp(-alpha_c*tau_on), np.exp(-beta*tau_on), np.exp(-gamma*tau_on)
    expbd = np.exp(-beta*tau_on)
    chat_on = 1.0-expac
    uhat_on = alpha/(beta+eps)*(1.0-expbd) + alpha/(beta-alpha_c+eps)*(expbd-expac)
    shat_on = alpha/(gamma+eps)*(1.0-expg) + (alpha/beta-alpha/(beta-alpha_c+eps))*beta/(gamma-beta+eps)*(expg-expb)
    shat_on += alpha*beta/(gamma-alpha_c+eps)/(beta-alpha_c+eps)*(expg-expac)

    c0_, u0_, s0_ = pred_steady_numpy(np.clip(ts-to, 0, None), alpha_c, alpha, beta, gamma)
    if ts.ndim == 2 and to.ndim == 2:
        c0_ = c0_.reshape(-1, 1)
        u0_ = u0_.reshape(-1, 1)
        s0_ = s0_.reshape(-1, 1)
    tau_off = F.softplus(torch.tensor(t-ts), beta=k).numpy()
    assert np.all(~np.isnan(tau_off))
    expac, expb, expg = np.exp(-alpha_c*tau_off), np.exp(-beta*tau_off), np.exp(-gamma*tau_off)
    expbd = np.exp(-beta*tau_off)
    chat_off = c0_*expac
    uhat_off = u0_*expbd
    shat_off = s0_*expg-u0_*beta/(gamma-beta+eps)*(expg-expb)

    chat, uhat, shat = (chat_on*o + chat_off*(1-o)), (uhat_on*o + uhat_off*(1-o)), (shat_on*o + shat_off*(1-o))
    if scaling_c is not None:
        chat *= scaling_c
    if scaling_u is not None:
        uhat *= scaling_u
    if scaling_s is not None:
        shat *= scaling_s
    if offset_c is not None:
        chat += offset_c
    if offset_u is not None:
        uhat += offset_u
    if offset_s is not None:
        shat += offset_s
    return chat, uhat, shat


def kl_gaussian(mu1, std1, mu2, std2):
    kl = torch.sum(torch.log(std2/std1)+std1.pow(2)/(2*std2.pow(2))-0.5+(mu1-mu2).pow(2)/(2*std2.pow(2)), 1)
    return torch.mean(kl)


def reparameterize(mu, std):
    eps = torch.randn_like(mu)
    return std*eps + mu


def softplusinv(x, beta=1.0):
    return (1/beta * np.log(np.exp(beta * np.clip(x, 0, 20)) - 1)) * (x <= 20) + x * (x > 20)


def knn_approx(c, u, s, c0, u0, s0, k):
    x = np.concatenate((c, u, s), 1)
    x0 = np.concatenate((c0, u0, s0), 1)
    pca = PCA(n_components=30, svd_solver='arpack', random_state=2022)
    x0_pca = pca.fit_transform(x0)
    x_pca = pca.transform(x)
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(x_pca)
    knn = knn_model.kneighbors(x0_pca, return_distance=False)
    return knn.astype(int)


def _hist_equal(t, t_query, perc=0.95, n_bin=51):
    # Perform histogram equalization across all local times.
    tmax = t.max() - t.min()
    t_ub = np.quantile(t, perc)
    t_lb = t.min()
    delta_t = (t_ub - t_lb)/(n_bin-1)
    bins = [t_lb+i*delta_t for i in range(n_bin)]+[t.max()+0.01]
    pdf_t, edges = np.histogram(t, bins, density=True)
    pt, edges = np.histogram(t, bins, density=False)

    # Perform histogram equalization
    cdf_t = np.concatenate(([0], np.cumsum(pt)))
    cdf_t = cdf_t/cdf_t[-1]
    t_out = np.zeros((len(t)))
    t_out_query = np.zeros((len(t_query)))
    for i in range(n_bin):
        mask = (t >= bins[i]) & (t < bins[i+1])
        t_out[mask] = (cdf_t[i] + (t[mask]-bins[i])*pdf_t[i])*tmax
        mask_q = (t_query >= bins[i]) & (t_query < bins[i+1])
        t_out_query[mask_q] = (cdf_t[i] + (t_query[mask_q]-bins[i])*pdf_t[i])*tmax
    return t_out, t_out_query


def knnx0_index(t,
                z,
                t_query,
                z_query,
                dt,
                k,
                bins=20,
                adaptive=0.0,
                std_t=None,
                forward=False,
                hist_eq=False):
    ############################################################
    # Same functionality as knnx0, but returns the neighbor index
    ############################################################
    Nq = len(t_query)
    n1 = 0
    len_avg = 0
    if hist_eq:
        t, t_query = _hist_equal(t, t_query)
    k_global = np.clip(len(t)//bins, 10, 1000)
    print(f'Using {k_global} latent neighbors to select ancestors.')
    knn_z = NearestNeighbors(n_neighbors=k_global)
    knn_z.fit(z)
    neighbor_index = []
    for i in tqdm_notebook(range(Nq)):
        if adaptive > 0:
            dt_r, dt_l = adaptive*std_t[i], adaptive*std_t[i] + (dt[1]-dt[0])
        else:
            dt_r, dt_l = dt[0], dt[1]
        if forward:
            t_ub, t_lb = t_query[i] + dt_l, t_query[i] + dt_r
        else:
            t_ub, t_lb = t_query[i] - dt_r, t_query[i] - dt_l
        _, filt = knn_z.kneighbors(z_query[i:i+1])
        filt_bin = np.zeros_like(t).astype(bool)
        filt_bin[filt] = True
        indices = np.where((t >= t_lb) & (t < t_ub) & filt_bin)[0]
        k_ = len(indices)
        delta_t = dt[1] - dt[0]  # increment / decrement of the time window boundary
        while k_ < k and t_lb > t[filt].min() - 5*delta_t and t_ub < t[filt].max() + 5*delta_t:
            if forward:
                t_lb = np.clip(t_lb - delta_t/5, t_query[i], None)
                t_ub = t_ub + delta_t
            else:
                t_lb = t_lb - delta_t
                t_ub = np.clip(t_ub + delta_t/5, None, t_query[i])
            indices = np.where((t >= t_lb) & (t < t_ub) & filt_bin)[0]  # select cells in the bin
            k_ = len(indices)
        len_avg = len_avg + k_
        if k_ > 0:
            k_neighbor = k if k_ > k else max(1, k_//2)
            knn_model = NearestNeighbors(n_neighbors=k_neighbor)
            knn_model.fit(z[indices])
            _, ind = knn_model.kneighbors(z_query[i:i+1])
            if k_neighbor > 1:
                neighbor_index.append(indices[ind.squeeze()].astype(int))
            else:
                neighbor_index.append(np.array([indices[ind.squeeze()].astype(int)]))
        else:
            neighbor_index.append([])
            n1 = n1+1
    print(f"Percentage of Invalid Sets: {n1/Nq:.3f}")
    print(f"Average Set Size: {len_avg//Nq}")
    return neighbor_index


def get_x0(c,
           u,
           s,
           t,
           dt,
           neighbor_index,
           c0_init=None,
           u0_init=None,
           s0_init=None,
           forward=False):
    N = len(neighbor_index)  # training + validation
    c0 = np.zeros((N, c.shape[1])) if c0_init is None else np.tile(c0_init, (N, 1))
    u0 = np.zeros((N, u.shape[1])) if u0_init is None else np.tile(u0_init, (N, 1))
    s0 = np.zeros((N, s.shape[1])) if s0_init is None else np.tile(s0_init, (N, 1))
    t0 = np.ones((N))*(t.min() - dt[0])
    # Used as the default u/s counts at the final time point
    t_98 = np.quantile(t, 0.98)
    p = 0.98
    while not np.any(t >= t_98) and p > 0.01:
        p = p - 0.01
        t_98 = np.quantile(t, p)
    c_end, u_end, s_end = c[t >= t_98].mean(0), u[t >= t_98].mean(0), s[t >= t_98].mean(0)

    for i in range(N):
        if len(neighbor_index[i]) > 0:
            c0[i] = c[neighbor_index[i]].mean(0)
            u0[i] = u[neighbor_index[i]].mean(0)
            s0[i] = s[neighbor_index[i]].mean(0)
            t0[i] = t[neighbor_index[i]].mean()
        elif forward:
            c0[i] = c_end
            u0[i] = u_end
            s0[i] = s_end
            t0[i] = t_98 + (t_98-t.min()) * 0.01
    return c0, u0, s0, t0


# Modified from DeepVelo (Cui et al. 2024)
def pearson(x, y, mask=None):
    if mask is None:
        x = x - torch.mean(x, dim=0)
        y = y - torch.mean(y, dim=0)
        x = x / (torch.std(x, dim=0) + 1e-9)
        y = y / (torch.std(y, dim=0) + 1e-9)
        return torch.mean(x * y, dim=1)  # (N,)
    else:
        num_valid_data = torch.sum(mask, dim=0)  # (D,)

        y = y * mask
        x = x * mask
        x = x - torch.sum(x, dim=0) / (num_valid_data + 1e-9)
        y = y - torch.sum(y, dim=0) / (num_valid_data + 1e-9)
        y = y * mask
        x = x * mask
        x = x / torch.sqrt(torch.sum(torch.pow(x, 2), dim=0) + 1e-9)
        y = y / torch.sqrt(torch.sum(torch.pow(y, 2), dim=0) + 1e-9)
        return torch.sum(x * y, dim=1)  # (N,)


def loss_vel(c0, chat, vc, u0, uhat, vu, s0, shat, vs, rna_only=False, rna_only_idx=None, condition=None, w_hvg=None):
    if rna_only:
        delta_x = torch.cat([uhat-u0, shat-s0], 1)
        v = torch.cat([vu, vs], 1)
        if w_hvg is not None:
            w_hvg_ = w_hvg.repeat((1, 2))
            delta_x = w_hvg_ * delta_x
            v = w_hvg_ * v
    else:
        if len(rna_only_idx) > 0 and condition is not None:
            mask = torch.zeros_like(condition)
            mask[:, rna_only_idx] = 1
            mask_flip = (~mask.bool()).int()
            c0 = c0 * torch.sum(condition * mask_flip, 1, keepdim=True) + torch.ones_like(c0) * torch.sum(condition * mask, 1, keepdim=True)
            chat = chat * torch.sum(condition * mask_flip, 1, keepdim=True) + torch.ones_like(chat) * torch.sum(condition * mask, 1, keepdim=True)
        delta_x = torch.cat([chat-c0, uhat-u0, shat-s0], 1)
        v = torch.cat([vc, vu, vs], 1)
        if w_hvg is not None:
            w_hvg_ = w_hvg.repeat((1, 3))
            delta_x = w_hvg_ * delta_x
            v = w_hvg_ * v
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = cos_sim(delta_x, v)
    return loss_cos


def cosine_similarity(u, s, beta, gamma, s_knn, w_hvg=None):
    V = (beta * u - gamma * s)
    ds = torch.reshape(s, (s.size(0), 1, s.size(1))) - s_knn   # cell x knn x gene
    if w_hvg is not None:
        V = w_hvg * V
        ds = w_hvg * ds
    res = None
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    for i in range(ds.size(1)):
        if res is None:
            res = cos_sim(ds[:, i, :], V).mean()
        else:
            res += cos_sim(ds[:, i, :], V).mean()
    return res


def ellipse_axis(x, y, xc, yc, slope):
    return (y - yc) - (slope * (x - xc))


def compute_quantile_scores(adata, n_pcs=30, n_neighbors=30):
    if 'connectivities' not in adata.obsp:
        neighbors = sc.Neighbors(adata)
        neighbors.compute_neighbors(n_neighbors=n_neighbors, knn=True, n_pcs=n_pcs)
        conn = neighbors.connectivities
    else:
        conn = adata.obsp['connectivities'].copy()
    conn.setdiag(1)
    conn_norm = conn.multiply(1.0 / conn.sum(1)).tocsr()

    quantile_scores = np.zeros(adata.shape)
    quantile_scores_2bit = np.zeros((adata.shape[0], adata.shape[1], 2))
    quantile_gene = np.full(adata.n_vars, False)
    for idx, gene in enumerate(adata.var_names):
        u = np.array(adata[:, gene].layers['Mu'])
        s = np.array(adata[:, gene].layers['Ms'])
        non_zero = (u > 0) & (s > 0)
        if np.sum(non_zero) < 10:
            continue

        mean_u, mean_s = np.mean(u[non_zero]), np.mean(s[non_zero])
        std_u, std_s = np.std(u[non_zero]), np.std(s[non_zero])
        u_ = (u - mean_u)/std_u
        s_ = (s - mean_s)/std_s
        X = np.reshape(s_[non_zero], (-1, 1))
        Y = np.reshape(u_[non_zero], (-1, 1))

        # Ax^2 + Bxy + Cy^2 + Dx + Ey + 1 = 0
        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = -np.ones_like(X)
        x, _, _, _ = np.linalg.lstsq(A, b)
        x = x.squeeze()
        A, B, C, D, E = x
        good_fit = B**2 - 4*A*C < 0  # being ellipse
        theta = np.arctan(B/(A - C))/2 if x[0] > x[2] else np.pi/2 + np.arctan(B/(A - C))/2  # major axis points upper right
        good_fit = good_fit & (theta < np.pi/2) & (theta > 0)

        xc = (B*E - 2*C*D)/(4*A*C - B**2)
        yc = (B*D - 2*A*E)/(4*A*C - B**2)
        slope_major = np.tan(theta)
        theta2 = np.pi/2 + theta
        slope_minor = np.tan(theta2)

        major_bit = ellipse_axis(s_, u_, xc, yc, slope_major)
        minor_bit = ellipse_axis(s_, u_, xc, yc, slope_minor)
        quant1 = (major_bit > 0) & (minor_bit < 0)
        quant2 = (major_bit > 0) & (minor_bit > 0)
        quant3 = (major_bit < 0) & (minor_bit > 0)
        quant4 = (major_bit < 0) & (minor_bit < 0)

        if (np.sum(quant1 | quant4) < 10) or (np.sum(quant2 | quant3) < 10):
            good_fit = False

        quantile_scores[:, idx:idx+1] = (-3.) * quant1 + (-1.) * quant2 + 1. * quant3 + 3. * quant4
        quantile_scores_2bit[:, idx:idx+1, 0] = 1. * (quant1 | quant2)
        quantile_scores_2bit[:, idx:idx+1, 1] = 1. * (quant2 | quant3)
        quantile_gene[idx] = good_fit

    quantile_scores = csr_matrix.dot(conn_norm, quantile_scores)
    quantile_scores_2bit[:, :, 0] = csr_matrix.dot(conn_norm, quantile_scores_2bit[:, :, 0])
    quantile_scores_2bit[:, :, 1] = csr_matrix.dot(conn_norm, quantile_scores_2bit[:, :, 1])

    if np.any(np.isnan(quantile_scores)):
        logger.warn('nan found during ellipse fit')
    if np.any(np.isinf(quantile_scores)):
        logger.warn('inf found during ellipse fit')
    if np.any(np.isnan(quantile_scores_2bit)):
        logger.warn('nan found during ellipse fit')
    if np.any(np.isinf(quantile_scores_2bit)):
        logger.warn('inf found during ellipse fit')

    adata.layers['quantile_scores'] = quantile_scores
    adata.layers['quantile_scores_1st_bit'] = quantile_scores_2bit[:, :, 0]
    adata.layers['quantile_scores_2nd_bit'] = quantile_scores_2bit[:, :, 1]

    perc_good = np.sum(quantile_gene) / adata.n_vars * 100
    print(f'{np.sum(quantile_gene)} out of {adata.n_vars} = {perc_good:.3g}% genes have good ellipse fits.')

    adata.obs['quantile_score_sum'] = np.sum(adata.layers['quantile_scores'][:, quantile_gene], axis=1)
    adata.var['quantile_genes'] = quantile_gene


def cluster_by_quantile(adata,
                        n_clusters=7,
                        affinity='euclidean',
                        linkage='ward'):
    compute_quantile_scores(adata)

    n_clusters = int(n_clusters)
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    cluster = cluster.fit_predict(np.vstack((adata.layers['quantile_scores_1st_bit'],
                                             adata.layers['quantile_scores_2nd_bit'])).transpose())
    return cluster


def sample_dir_mix(w, yw, std_prior):
    # Sample from a mixture of dirichlet distributions
    mu_0, mu_1 = np.mean(w[yw == 0]), np.mean(w[yw == 1])
    alpha_w_0 = find_dirichlet_param(mu_0, std_prior)
    alpha_w_1 = find_dirichlet_param(mu_1, std_prior)
    np.random.seed(42)
    q1 = dirichlet.rvs(alpha_w_0, size=len(w))[:, 0]
    np.random.seed(42)
    q2 = dirichlet.rvs(alpha_w_1, size=len(w))[:, 0]
    wq = np.sum(yw == 1)/len(yw)
    np.random.seed(42)
    b = bernoulli.rvs(wq, size=len(w))
    q = (b == 0)*q1 + (b == 1)*q2
    return q


def assign_gene_mode_auto(adata,
                          w_noisy,
                          thred=0.05,
                          std_prior=0.1,
                          n_clusters=7):
    # Cluster by ellipse fit
    y = cluster_by_quantile(adata, n_clusters=n_clusters)
    adata.var['quantile_cluster'] = y

    # Sample weights from Dirichlet(mu=0.5, std=std_prior)
    alpha_neutral = find_dirichlet_param(0.5, std_prior)
    q_neutral = dirichlet.rvs(alpha_neutral, size=adata.n_vars)[:, 0]
    w = np.empty((adata.n_vars))
    pval_ind = []
    pval_rep = []
    cluster_type = np.zeros((y.max()+1))
    alpha_ind, alpha_rep = find_dirichlet_param(0.6, std_prior), find_dirichlet_param(0.4, std_prior)

    # Perform Komogorov-Smirnov Test
    for i in range(y.max()+1):
        n = np.sum(y == i)
        _, pval_1 = kstest(w_noisy[y == i], q_neutral, alternative='greater', method='asymp')
        _, pval_2 = kstest(w_noisy[y == i], q_neutral, alternative='less', method='asymp')
        pval_ind.append(pval_1)
        pval_rep.append(pval_2)
        if pval_1 < thred and pval_2 < thred:  # uni/bi-modal dirichlet
            cluster_type[i] = 0
            _, pval_3 = kstest(w_noisy[y == i], q_neutral)
            if pval_3 < thred:
                km = KMeans(2, n_init='auto', random_state=42)
                yw = km.fit_predict(w_noisy[y == i].reshape(-1, 1))
                w[y == i] = sample_dir_mix(w_noisy[y == i], yw, std_prior)
            else:
                np.random.seed(42)
                w[y == i] = dirichlet.rvs(alpha_neutral, size=n)[:, 0]

        elif pval_1 >= 0.05:  # induction
            cluster_type[i] = 1
            np.random.seed(42)
            w[y == i] = dirichlet.rvs(alpha_ind, size=n)[:, 0]
        elif pval_2 >= 0.05:  # repression
            cluster_type[i] = 2
            np.random.seed(42)
            w[y == i] = dirichlet.rvs(alpha_rep, size=n)[:, 0]

    pval_ind = np.array(pval_ind)
    pval_rep = np.array(pval_rep)
    print(f'KS-test result: {cluster_type}')
    # If no repressive cluster is found, pick the one with the highest p value
    if np.all(cluster_type == 1):
        ymax = np.argmax(pval_rep)
        print(f'Assigning cluster {ymax} to repressive.')
        np.random.seed(42)
        w[y == ymax] = dirichlet.rvs(alpha_rep, size=np.sum(y == ymax))[:, 0]

    # If no inductive cluster is found, pick the one with the highest p value
    if np.all(cluster_type == 2):
        ymax = np.argmax(pval_ind)
        print(f'Assigning cluster {ymax} to inductive.')
        np.random.seed(42)
        w[y == ymax] = dirichlet.rvs(alpha_ind, size=np.sum(y == ymax))[:, 0]
    return w


def find_dirichlet_param(mu, std, n_basis=2):
    alpha_i = ((mu/std)*((1-mu)/std) - 1) * mu
    params = [alpha_i for i in range(n_basis-1)]
    params.append((1-mu)/mu*alpha_i)
    return np.array(params)


def assign_gene_mode(adata,
                     w_noisy,
                     assign_type='binary',
                     thred=0.05,
                     std_prior=0.1,
                     n_cluster_thred=3):
    # Assign one of ('inductive', 'repressive', 'mixture') to gene clusters
    # `assign_type' specifies which strategy to use
    if assign_type == 'binary':
        return assign_gene_mode_binary(adata, w_noisy, thred)
    elif assign_type == 'auto':
        return assign_gene_mode_auto(adata, w_noisy, thred, std_prior, n_cluster_thred)
    elif assign_type == 'inductive':
        alpha_ind = find_dirichlet_param(0.8, std_prior)
        np.random.seed(42)
        return dirichlet.rvs(alpha_ind, size=adata.n_vars)[:, 0]
    elif assign_type == 'repressive':
        alpha_rep = find_dirichlet_param(0.2, std_prior)
        np.random.seed(42)
        return dirichlet.rvs(alpha_rep, size=adata.n_vars)[:, 0]


# Modified from MultiVelo
def aggregate_peaks_10x(adata_atac, peak_annot_file, linkage_file, peak_dist=10000, min_corr=0.5, gene_body=False, return_dict=False, verbose=False):
    """Peak to gene aggregation.

    This function aggregates promoter and enhancer peaks to genes based on the 10X linkage file.

    Parameters
    ----------
    adata_atac: :class:`~anndata.AnnData`
        ATAC anndata object which stores raw peak counts.
    peak_annot_file: `str`
        Peak annotation file from 10X CellRanger ARC.
    linkage_file: `str`
        Peak-gene linkage file from 10X CellRanger ARC. This file stores highly correlated peak-peak
        and peak-gene pair information.
    peak_dist: `int` (default: 10000)
        Maximum distance for peaks to be included for a gene.
    min_corr: `float` (default: 0.5)
        Minimum correlation for a peak to be considered as enhancer.
    gene_body: `bool` (default: `False`)
        Whether to add gene body peaks to the associated promoters.
    return_dict: `bool` (default: `False`)
        Whether to return promoter and enhancer dictionaries.
    verbose: `bool` (default: `False`)
        Whether to print number of genes with promoter peaks.

    Returns
    -------
    A new ATAC anndata object which stores gene aggreagted peak counts.
    Additionally, if `return_dict==True`:
        A dictionary which stores genes and promoter peaks.
        And a dictionary which stores genes and enhancer peaks.
    """
    promoter_dict = {}
    distal_dict = {}
    gene_body_dict = {}
    corr_dict = {}

    # read annotations
    with open(peak_annot_file) as f:
        header = next(f)
        tmp = header.split('\t')
        if len(tmp) == 4:
            cellranger_version = 1
        elif len(tmp) == 6:
            cellranger_version = 2
        else:
            raise ValueError('Peak annotation file should contain 4 columns (CellRanger ARC 1.0.0) or 5 columns (CellRanger ARC 2.0.0)')
        if verbose:
            print(f'CellRanger ARC identified as {cellranger_version}.0.0')
        if cellranger_version == 1:
            for line in f:
                tmp = line.rstrip().split('\t')
                tmp1 = tmp[0].split('_')
                peak = f'{tmp1[0]}:{tmp1[1]}-{tmp1[2]}'
                if tmp[1] != '':
                    genes = tmp[1].split(';')
                    dists = tmp[2].split(';')
                    types = tmp[3].split(';')
                    for i, gene in enumerate(genes):
                        dist = dists[i]
                        annot = types[i]
                        if annot == 'promoter':
                            if gene not in promoter_dict:
                                promoter_dict[gene] = [peak]
                            else:
                                promoter_dict[gene].append(peak)
                        elif annot == 'distal':
                            if dist == '0':
                                if gene not in gene_body_dict:
                                    gene_body_dict[gene] = [peak]
                                else:
                                    gene_body_dict[gene].append(peak)
                            else:
                                if gene not in distal_dict:
                                    distal_dict[gene] = [peak]
                                else:
                                    distal_dict[gene].append(peak)
        else:
            for line in f:
                tmp = line.rstrip().split('\t')
                peak = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                gene = tmp[3]
                dist = tmp[4]
                annot = tmp[5]
                if annot == 'promoter':
                    if gene not in promoter_dict:
                        promoter_dict[gene] = [peak]
                    else:
                        promoter_dict[gene].append(peak)
                elif annot == 'distal':
                    if dist == '0':
                        if gene not in gene_body_dict:
                            gene_body_dict[gene] = [peak]
                        else:
                            gene_body_dict[gene].append(peak)
                    else:
                        if gene not in distal_dict:
                            distal_dict[gene] = [peak]
                        else:
                            distal_dict[gene].append(peak)

    # read linkages
    with open(linkage_file) as f:
        for line in f:
            tmp = line.rstrip().split('\t')
            if tmp[12] == "peak-peak":
                peak1 = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                peak2 = f'{tmp[3]}:{tmp[4]}-{tmp[5]}'
                tmp2 = tmp[6].split('><')[0][1:].split(';')
                tmp3 = tmp[6].split('><')[1][:-1].split(';')
                corr = float(tmp[7])
                for t2 in tmp2:
                    gene1 = t2.split('_')
                    for t3 in tmp3:
                        gene2 = t3.split('_')
                        # one of the peaks is in promoter, peaks belong to the same gene or are close in distance
                        if ((gene1[1] == "promoter") != (gene2[1] == "promoter")) and ((gene1[0] == gene2[0]) or (float(tmp[11]) < peak_dist)):
                            if gene1[1] == "promoter":
                                gene = gene1[0]
                            else:
                                gene = gene2[0]
                            if gene in corr_dict:
                                # peak 1 is in promoter, peak 2 is not in gene body -> peak 2 is added to gene 1
                                if peak2 not in corr_dict[gene] and gene1[1] == "promoter" and (gene2[0] not in gene_body_dict or peak2 not in gene_body_dict[gene2[0]]):
                                    corr_dict[gene][0].append(peak2)
                                    corr_dict[gene][1].append(corr)
                                # peak 2 is in promoter, peak 1 is not in gene body -> peak 1 is added to gene 2
                                if peak1 not in corr_dict[gene] and gene2[1] == "promoter" and (gene1[0] not in gene_body_dict or peak1 not in gene_body_dict[gene1[0]]):
                                    corr_dict[gene][0].append(peak1)
                                    corr_dict[gene][1].append(corr)
                            else:
                                # peak 1 is in promoter, peak 2 is not in gene body -> peak 2 is added to gene 1
                                if gene1[1] == "promoter" and (gene2[0] not in gene_body_dict or peak2 not in gene_body_dict[gene2[0]]):
                                    corr_dict[gene] = [[peak2], [corr]]
                                # peak 2 is in promoter, peak 1 is not in gene body -> peak 1 is added to gene 2
                                if gene2[1] == "promoter" and (gene1[0] not in gene_body_dict or peak1 not in gene_body_dict[gene1[0]]):
                                    corr_dict[gene] = [[peak1], [corr]]
            elif tmp[12] == "peak-gene":
                peak1 = f'{tmp[0]}:{tmp[1]}-{tmp[2]}'
                tmp2 = tmp[6].split('><')[0][1:].split(';')
                gene2 = tmp[6].split('><')[1][:-1]
                corr = float(tmp[7])
                for t2 in tmp2:
                    gene1 = t2.split('_')
                    # peak 1 belongs to gene 2 or are close in distance -> peak 1 is added to gene 2
                    if ((gene1[0] == gene2) or (float(tmp[11]) < peak_dist)):
                        gene = gene1[0]
                        if gene in corr_dict:
                            if peak1 not in corr_dict[gene] and gene1[1] != "promoter" and (gene1[0] not in gene_body_dict or peak1 not in gene_body_dict[gene1[0]]):
                                corr_dict[gene][0].append(peak1)
                                corr_dict[gene][1].append(corr)
                        else:
                            if gene1[1] != "promoter" and (gene1[0] not in gene_body_dict or peak1 not in gene_body_dict[gene1[0]]):
                                corr_dict[gene] = [[peak1], [corr]]
            elif tmp[12] == "gene-peak":
                peak2 = f'{tmp[3]}:{tmp[4]}-{tmp[5]}'
                gene1 = tmp[6].split('><')[0][1:]
                tmp3 = tmp[6].split('><')[1][:-1].split(';')
                corr = float(tmp[7])
                for t3 in tmp3:
                    gene2 = t3.split('_')
                    # peak 2 belongs to gene 1 or are close in distance -> peak 2 is added to gene 1
                    if ((gene1 == gene2[0]) or (float(tmp[11]) < peak_dist)):
                        gene = gene1
                        if gene in corr_dict:
                            if peak2 not in corr_dict[gene] and gene2[1] != "promoter" and (gene2[0] not in gene_body_dict or peak2 not in gene_body_dict[gene2[0]]):
                                corr_dict[gene][0].append(peak2)
                                corr_dict[gene][1].append(corr)
                        else:
                            if gene2[1] != "promoter" and (gene2[0] not in gene_body_dict or peak2 not in gene_body_dict[gene2[0]]):
                                corr_dict[gene] = [[peak2], [corr]]

    gene_dict = promoter_dict
    enhancer_dict = {}
    promoter_genes = list(promoter_dict.keys())
    if gene_body:
        promoter_genes.extend(list(gene_body_dict.keys()))
        promoter_genes = list(set(promoter_genes))
    if verbose:
        print(f'Found {len(promoter_genes)} genes with promoter peaks')
    for gene in promoter_genes:
        if gene_body:  # add gene-body peaks
            if gene in gene_body_dict:
                if gene not in gene_dict:
                    gene_dict[gene] = gene_body_dict[gene]
                else:
                    for peak in gene_body_dict[gene]:
                        if peak not in gene_dict[gene]:
                            gene_dict[gene].append(peak)
        enhancer_dict[gene] = []
        if gene in corr_dict:  # add enhancer peaks
            for j, peak in enumerate(corr_dict[gene][0]):
                corr = corr_dict[gene][1][j]
                if corr > min_corr:
                    if peak not in gene_dict[gene]:
                        gene_dict[gene].append(peak)
                        enhancer_dict[gene].append(peak)

    # aggregate to genes
    adata_atac_X_copy = adata_atac.X.A
    gene_mat = np.zeros((adata_atac.shape[0], len(promoter_genes)))
    var_names_dict = {x: i for i, x in enumerate(adata_atac.var_names.to_numpy())}
    for i, gene in tqdm_notebook(enumerate(promoter_genes), total=len(promoter_genes)):
        peaks = gene_dict[gene]
        for peak in peaks:
            if peak in var_names_dict:
                gene_mat[:, i] += adata_atac_X_copy[:, var_names_dict[peak]]
    gene_mat[gene_mat < 0] = 0
    gene_mat = AnnData(X=csr_matrix(gene_mat))
    gene_mat.obs_names = pd.Index(list(adata_atac.obs_names))
    gene_mat.var_names = pd.Index(promoter_genes)
    gene_mat = gene_mat[:, gene_mat.X.sum(0) > 0]
    if return_dict:
        return gene_mat, promoter_dict, enhancer_dict
    else:
        return gene_mat


# Modified from MultiVelo
def velocity_graph(adata, key='vae', xkey=None, batch_corrected=False, velocity_offset=False, t_perc=1, **kwargs):
    vkey = f'{key}_velocity'
    if vkey+'_norm' not in adata.layers.keys():
        t = adata.obs[f"{key}_time"].to_numpy()
        v = adata.layers[vkey]
        if velocity_offset:
            v = v - np.mean(v[(t <= np.percentile(t, t_perc))], 0)
        adata.layers[vkey+'_norm'] = v / np.sum(np.abs(v), 0)
        adata.uns[vkey+'_norm_params'] = adata.uns[vkey+'_params']
    if vkey+'_norm_genes' not in adata.var.columns:
        adata.var[vkey+'_norm_genes'] = adata.var[vkey+'_genes']
    if xkey is None:
        if batch_corrected:
            xkey = 's_leveled'
        else:
            xkey = 'Ms'
    if xkey == 's_leveled' and 's_leveled' not in adata.layers.keys():
        logger.warn('Batch corrected s_leveled matrix not found in layers. Using Ms instead.')
        xkey = 'Ms'
    scv.tl.velocity_graph(adata, vkey=vkey+'_norm', xkey=xkey, **kwargs)


# Modified from https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html
def is_outlier(adata, metric, lower_nmads=20, upper_nmads=20):
    M = adata.obs[metric]
    lower_bound = max(np.min(M), np.median(M) - lower_nmads * median_abs_deviation(M))
    upper_bound = min(np.max(M), np.median(M) + upper_nmads * median_abs_deviation(M))
    if metric.startswith('log1p_'):
        print(f'{metric[6:]} lower_bound {np.expm1(lower_bound)}, upper_bound {np.expm1(upper_bound)}')
    else:
        print(f'{metric} lower_bound {lower_bound}, upper_bound {upper_bound}')
    outlier = (M < lower_bound) | (upper_bound < M)
    plt.figure()
    sns.distplot(M)
    sns.rugplot(M)
    sns.distplot(M[~outlier])
    sns.rugplot(M[~outlier])
    return outlier


# The following code was modified from https://github.com/scverse/scanpy/pull/2731 to add intercept back to residuals
def sanitize_anndata(adata):
    adata._sanitize()


def view_to_actual(adata):
    if adata.is_view:
        adata._init_as_actual(adata.copy())


def _get_obs_rep(adata, *, use_raw=False, layer=None, obsm=None, obsp=None):
    if not isinstance(use_raw, bool):
        raise TypeError(f"use_raw expected to be bool, was {type(use_raw)}.")

    is_layer = layer is not None
    is_raw = use_raw is not False
    is_obsm = obsm is not None
    is_obsp = obsp is not None
    choices_made = sum((is_layer, is_raw, is_obsm, is_obsp))
    assert choices_made <= 1
    if choices_made == 0:
        return adata.X
    elif is_layer:
        return adata.layers[layer]
    elif use_raw:
        return adata.raw.X
    elif is_obsm:
        return adata.obsm[obsm]
    elif is_obsp:
        return adata.obsp[obsp]
    else:
        assert False, (
            "That was unexpected. Please report this bug at:\n\n\t"
            " https://github.com/scverse/scanpy/issues"
        )


def _set_obs_rep(adata, val, *, use_raw=False, layer=None, obsm=None, obsp=None):
    is_layer = layer is not None
    is_raw = use_raw is not False
    is_obsm = obsm is not None
    is_obsp = obsp is not None
    choices_made = sum((is_layer, is_raw, is_obsm, is_obsp))
    assert choices_made <= 1
    if choices_made == 0:
        adata.X = val
    elif is_layer:
        adata.layers[layer] = val
    elif use_raw:
        adata.raw.X = val
    elif is_obsm:
        adata.obsm[obsm] = val
    elif is_obsp:
        adata.obsp[obsp] = val
    else:
        assert False, (
            "That was unexpected. Please report this bug at:\n\n\t"
            " https://github.com/scverse/scanpy/issues"
        )


def _regress_out_chunk(data):
    data_chunk = data[0]
    regressors = data[1]
    variable_is_categorical = data[2]
    add_intercept = data[3]

    responses_chunk_list = []
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    for col_index in range(data_chunk.shape[1]):
        if not (data_chunk[:, col_index] != data_chunk[0, col_index]).any():
            responses_chunk_list.append(data_chunk[:, col_index])
            continue

        if variable_is_categorical:
            regres = np.c_[np.ones(regressors.shape[0]), regressors[:, col_index]]
        else:
            regres = regressors
        try:
            if add_intercept:
                result = sm.GLM(
                    data_chunk[:, col_index], sm.add_constant(regres), family=sm.families.Gaussian()
                ).fit()
                new_column = result.resid_response + result.params.iloc[0]
            else:
                result = sm.GLM(
                    data_chunk[:, col_index], regres, family=sm.families.Gaussian()
                ).fit()
                new_column = result.resid_response
        except PerfectSeparationError:
            new_column = np.zeros(data_chunk.shape[0])

        responses_chunk_list.append(new_column)

    return np.vstack(responses_chunk_list)


def regress_out(adata, keys, layer=None, n_jobs=12, copy=False, add_intercept=False):
    """\
    Regress out (mostly) unwanted sources of variation.

    Uses simple linear regression. This is inspired by Seurat's `regressOut`
    function in R [Satija15]. Note that this function tends to overcorrect
    in certain circumstances as described in :issue:`526`.

    Parameters
    ----------
    adata
        The annotated data matrix.
    keys
        Keys for observation annotation on which to regress.
    layer
        If provided, which element of layers to use in regression.
    n_jobs
        Number of jobs for parallel computation.
        `None` means using :attr:`scanpy._settings.ScanpyConfig.n_jobs`.
    copy
        Determines whether a copy of `adata` is returned.
    add_intercept
        If True, regress_out will add intercept back to residuals in order to transform results back into gene-count space. Defaults to False

    Returns
    -------
    Depending on `copy` returns or updates `adata` with the corrected data matrix.
    """
    from pandas.api.types import CategoricalDtype
    adata = adata.copy() if copy else adata

    sanitize_anndata(adata)

    view_to_actual(adata)

    if isinstance(keys, str):
        keys = [keys]

    X = _get_obs_rep(adata, layer=layer)

    if issparse(X):
        X = X.toarray()

    variable_is_categorical = False
    if keys[0] in adata.obs_keys() and isinstance(
        adata.obs[keys[0]].dtype, CategoricalDtype
    ):
        if len(keys) > 1:
            raise ValueError(
                "If providing categorical variable, "
                "only a single one is allowed. For this one "
                "we regress on the mean for each category."
            )
        regressors = np.zeros(X.shape, dtype="float32")
        for category in adata.obs[keys[0]].cat.categories:
            mask = (category == adata.obs[keys[0]]).values
            for ix, x in enumerate(X.T):
                regressors[mask, ix] = x[mask].mean()
        variable_is_categorical = True
    else:
        if keys:
            regressors = adata.obs[keys]
        else:
            regressors = adata.obs.copy()

        regressors.insert(0, "ones", 1.0)

    len_chunk = np.ceil(min(1000, X.shape[1]) / n_jobs).astype(int)
    n_chunks = np.ceil(X.shape[1] / len_chunk).astype(int)

    tasks = []
    chunk_list = np.array_split(X, n_chunks, axis=1)
    if variable_is_categorical:
        regressors_chunk = np.array_split(regressors, n_chunks, axis=1)
    for idx, data_chunk in enumerate(chunk_list):
        if variable_is_categorical:
            regres = regressors_chunk[idx]
        else:
            regres = regressors
        tasks.append(tuple((data_chunk, regres, variable_is_categorical, add_intercept)))

    from joblib import Parallel, delayed

    res = Parallel(n_jobs=n_jobs)(delayed(_regress_out_chunk)(task) for task in tasks)

    _set_obs_rep(adata, np.vstack(res).T, layer=layer)
    return adata if copy else None


# The following code was modified from https://github.com/theislab/scvelo/blob/main/scvelo
def get_modality(adata, modality):
    if modality in ["X", None]:
        return adata.X
    elif modality in adata.layers.keys():
        return adata.layers[modality]
    elif modality in adata.obsm.keys():
        if isinstance(adata.obsm[modality], pd.DataFrame):
            return adata.obsm[modality].values
        else:
            return adata.obsm[modality]


def verify_dtypes(adata) -> None:
    try:
        _ = adata[:, 0]
    except Exception:
        uns = adata.uns
        adata.uns = {}
        try:
            _ = adata[:, 0]
            print(
                "Safely deleted unstructured annotations (adata.uns), \n"
                "as these do not comply with permissible anndata datatypes."
            )
        except Exception:
            print(
                "The data might be corrupted. Please verify all annotation datatypes."
            )
            adata.uns = uns


def get_size(adata, modality=None):
    X = get_modality(adata=adata, modality=modality)
    return sum(X, axis=1)


def set_initial_size(adata, layers=None):
    if layers is None:
        layers = ["unspliced", "spliced"]
    verify_dtypes(adata)
    layers = [
        layer
        for layer in layers
        if layer in adata.layers.keys()
        and f"initial_size_{layer}" not in adata.obs.keys()
    ]
    for layer in layers:
        adata.obs[f"initial_size_{layer}"] = get_size(adata, layer)
    if "initial_size" not in adata.obs.keys():
        adata.obs["initial_size"] = get_size(adata)


def materialize_as_ndarray(key):
    if isinstance(key, (list, tuple)):
        return tuple(np.asarray(arr) for arr in key)
    return np.asarray(key)


def get_mean_var(X, ignore_zeros=False, perc=None):
    data = X.data if issparse(X) else X
    mask_nans = np.isnan(data) | np.isinf(data) | np.isneginf(data)

    if issparse(X):
        n_nonzeros = X.getnnz(axis=0)
    else:
        n_nonzeros = (X != 0).sum(axis=0)

    if ignore_zeros:
        n_counts = n_nonzeros
    else:
        n_counts = X.shape[0]

    if mask_nans.sum() > 0:
        if issparse(X):
            data[mask_nans] = 0
            n_nans = (n_nonzeros - (X != 0).sum(0)).A1
        else:
            X[mask_nans] = 0
            n_nans = mask_nans.sum(0)
        n_counts -= n_nans

    if perc is not None:
        if np.size(perc) < 2:
            perc = [perc, 100] if perc < 50 else [0, perc]
        lb, ub = np.percentile(data, perc)
        if issparse(X):
            X.data = np.clip(data, lb, ub)
        else:
            X = np.clip(data, lb, ub)

    if issparse(X):
        mean = (X.sum(0) / n_counts).A1
        mean_sq = (X.multiply(X).sum(0) / n_counts).A1
    else:
        mean = X.sum(0) / n_counts
        mean_sq = np.multiply(X, X).sum(0) / n_counts

    n_counts = np.clip(n_counts, 2, None)  # to avoid division by zero
    var = (mean_sq - mean**2) * (n_counts / (n_counts - 1))

    mean = np.nan_to_num(mean)
    var = np.nan_to_num(var)

    return mean, var


def filter_genes_dispersion(
    data,
    flavor="seurat",
    min_disp=None,
    max_disp=None,
    min_mean=None,
    max_mean=None,
    n_bins=20,
    n_top_genes=None,
    retain_genes=None,
    log=True,
    subset=True,
    copy=False,
):
    """Extract highly variable genes.

    Expects non-logarithmized data.
    The normalized dispersion is obtained by scaling with the mean and standard
    deviation of the dispersions for genes falling into a given bin for mean
    expression of genes. This means that for each bin of mean expression, highly
    variable genes are selected.

    Parameters
    ----------
    data : :class:`~anndata.AnnData`, `np.ndarray`, `sp.sparse`
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    flavor : {'seurat', 'cell_ranger', 'svr'}, optional (default: 'seurat')
        Choose the flavor for computing normalized dispersion. If choosing
        'seurat', this expects non-logarithmized data - the logarithm of mean
        and dispersion is taken internally when `log` is at its default value
        `True`. For 'cell_ranger', this is usually called for logarithmized data
        - in this case you should set `log` to `False`. In their default
        workflows, Seurat passes the cutoffs whereas Cell Ranger passes
        `n_top_genes`.
    min_mean=0.0125, max_mean=3, min_disp=0.5, max_disp=`None` : `float`, optional
        If `n_top_genes` unequals `None`, these cutoffs for the means and the
        normalized dispersions are ignored.
    n_bins : `int` (default: 20)
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
    n_top_genes : `int` or `None` (default: `None`)
        Number of highly-variable genes to keep.
    retain_genes: `list`, optional (default: `None`)
        List of gene names to be retained independent of thresholds.
    log : `bool`, optional (default: `True`)
        Use the logarithm of the mean to variance ratio.
    subset : `bool`, optional (default: `True`)
        Keep highly-variable genes only (if True) else write a bool
        array for highly-variable genes while keeping all genes.
    copy : `bool`, optional (default: `False`)
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned.

    Returns
    -------
    If an AnnData `adata` is passed, returns or updates `adata` depending on \
    `copy`. It filters the `adata` and adds the annotations
    """

    import warnings
    adata = data.copy() if copy else data
    set_initial_size(adata)

    mean, var = materialize_as_ndarray(get_mean_var(adata.X))

    if n_top_genes is not None and adata.n_vars < n_top_genes:
        print(
            "Skip filtering by dispersion since number "
            "of variables are less than `n_top_genes`."
        )
    else:
        if flavor == "svr":
            from sklearn.svm import SVR

            log_mu = np.log2(mean)
            log_cv = np.log2(np.sqrt(var) / mean)
            clf = SVR(gamma=150.0 / len(mean))
            clf.fit(log_mu[:, None], log_cv)
            score = log_cv - clf.predict(log_mu[:, None])
            nth_score = np.sort(score)[::-1][n_top_genes - 1]
            adata.var["highly_variable"] = score >= nth_score

        else:
            cut_disp = [min_disp, max_disp, min_mean, max_mean]
            if n_top_genes is not None and not all(x is None for x in cut_disp):
                print("If you pass `n_top_genes`, all cutoffs are ignored.")
            if min_disp is None:
                min_disp = 0.5
            if max_disp is None:
                max_disp = np.inf
            if min_mean is None:
                min_mean = 0.0125
            if max_mean is None:
                max_mean = 3

            mean[mean == 0] = 1e-12
            dispersion = var / mean
            if log:
                dispersion[dispersion == 0] = np.nan
                dispersion = np.log(dispersion)
                mean = np.log1p(mean)
                mean[np.isnan(mean)] = 0  # deal with negative means

            df = pd.DataFrame()
            df["mean"], df["dispersion"] = mean, dispersion

            if flavor == "seurat":
                df["mean_bin"] = pd.cut(df["mean"], bins=n_bins)
                disp_grouped = df.groupby("mean_bin")["dispersion"]
                disp_mean_bin = disp_grouped.mean()
                disp_std_bin = disp_grouped.std(ddof=1)

                one_gene_per_bin = disp_std_bin.isnull()

                disp_std_bin[one_gene_per_bin] = disp_mean_bin[one_gene_per_bin].values
                disp_mean_bin[one_gene_per_bin] = 0

                mu = disp_mean_bin[df["mean_bin"].values].values
                std = disp_std_bin[df["mean_bin"].values].values
                df["dispersion_norm"] = ((df["dispersion"] - mu) / std).fillna(0)
            elif flavor == "cell_ranger":
                from statsmodels import robust

                cut = np.percentile(df["mean"], np.arange(10, 105, 5))
                df["mean_bin"] = pd.cut(df["mean"], np.r_[-np.inf, cut, np.inf])
                disp_grouped = df.groupby("mean_bin")["dispersion"]
                disp_median_bin = disp_grouped.median()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    disp_mad_bin = disp_grouped.apply(robust.mad)
                mu = disp_median_bin[df["mean_bin"].values].values
                std = disp_mad_bin[df["mean_bin"].values].values
                df["dispersion_norm"] = (np.abs(df["dispersion"] - mu) / std).fillna(0)
            else:
                raise ValueError('`flavor` needs to be "seurat" or "cell_ranger"')
            dispersion_norm = df["dispersion_norm"].values
            if n_top_genes is not None:
                cut_off = df["dispersion_norm"].nlargest(n_top_genes).values[-1]
                gene_subset = df["dispersion_norm"].values >= cut_off
            else:
                gene_subset = np.logical_and.reduce(
                    (
                        mean > min_mean,
                        mean < max_mean,
                        dispersion_norm > min_disp,
                        dispersion_norm < max_disp,
                    )
                )

            adata.var["means"] = df["mean"].values
            adata.var["dispersions"] = df["dispersion"].values
            adata.var["dispersions_norm"] = df["dispersion_norm"].values
            adata.var["highly_variable"] = gene_subset

        if subset:
            gene_subset = adata.var["highly_variable"]
            if retain_genes is not None:
                if isinstance(retain_genes, str):
                    retain_genes = [retain_genes]
                gene_subset = gene_subset | adata.var_names.isin(retain_genes)
            adata._inplace_subset_var(gene_subset)

        print(f"Extracted {np.sum(gene_subset)} highly variable genes.")
        adata.uns['hvg'] = {'flavor': flavor}
    return adata if copy else None
