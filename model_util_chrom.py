import numpy as np
from numpy import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from .scvelo_util import tau_inv, test_bimodality
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import pynndescent
from scipy.spatial import KDTree


def inv(x):
    x_inv = 1 / x * (x != 0)
    return x_inv


def chromatin(tau, c0, alpha_c):
    expc = exp(-alpha_c * tau)
    return c0 * expc + 1 * (1 - expc)


def unspliced(tau, u0, c0, alpha_c, alpha, beta):
    expc = exp(-alpha_c * tau)
    expu = exp(-beta * tau)
    const = (1 - c0) * alpha * inv(beta - alpha_c)
    return u0 * expu + alpha / beta * (1 - expu) + const * (expu - expc)


def spliced(tau, s0, u0, c0, alpha_c, alpha, beta, gamma):
    expc = exp(-alpha_c * tau)
    expu = exp(-beta * tau)
    exps = exp(-gamma * tau)
    const = (1 - c0) * alpha * inv(beta - alpha_c)
    return s0 * exps + (alpha / gamma) * (1 - exps) + (beta * inv(gamma - beta)) * ((alpha / beta) - u0 - const) * (exps - expu) + (beta * inv(gamma - alpha_c)) * const * (exps - expc)


def compute_exp(tau, c0, u0, s0, kc, alpha_c, rho, alpha, beta, gamma):
    expc, expu, exps = exp(-alpha_c * tau), exp(-beta * tau), exp(-gamma * tau)
    const = (kc - c0) * rho * alpha * inv(beta - alpha_c)
    c = kc - (kc - c0) * expc
    u = u0 * expu + (rho * alpha * kc / beta) * (1 - expu) + const * (expu - expc)
    s = s0 * exps + (rho * alpha * kc / gamma) * (1 - exps)
    s += (beta * inv(gamma - beta)) * ((rho * alpha * kc / beta) - u0 - const) * (exps - expu)
    s += (beta * inv(gamma - alpha_c)) * const * (exps - expc)
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


def pred_single(t, alpha_c, alpha, beta, gamma, ts, scaling_c=1.0, scaling=1.0, cinit=0, uinit=0, sinit=0):
    beta = beta*scaling
    tau, rho, kc, c0, u0, s0 = vectorize(t, ts, alpha_c, alpha, beta, gamma, c0=cinit, u0=uinit, s0=sinit)
    tau = np.clip(tau, a_min=0, a_max=None)
    ct, ut, st = compute_exp(tau, c0, u0, s0, kc, alpha_c, rho, alpha, beta, gamma)
    ct = ct*scaling_c
    ut = ut*scaling
    return ct.squeeze(), ut.squeeze(), st.squeeze()


def pred_exp_numpy(t, ton, c0, u0, s0, kc, alpha_c, rho, alpha, beta, gamma):
    tau = np.clip(t-ton, 0, None).reshape(-1, 1)
    expac, expb, expg = np.exp(-alpha_c*t), np.exp(-beta*tau), np.exp(-gamma*tau)
    eps = 1e-6

    cpred = c0*expac+kc*(1-expac)
    upred = u0*expb + rho*alpha*kc/beta*(1-expb) + (kc-c0)*rho*alpha/(beta-alpha_c+eps)*(expb-expac)
    spred = s0*expg + rho*alpha*kc/gamma*(1-expg)
    spred += (rho*alpha*kc/beta-u0-(kc-c0)*rho*alpha/(beta-alpha_c+eps))*beta/(gamma-beta+eps)*(expg-expb)
    spred += (kc-c0)*rho*alpha*beta/(gamma-alpha_c+eps)/(beta-alpha_c+eps)*(expg-expac)
    return np.clip(cpred, a_min=0, a_max=1), np.clip(upred, a_min=0, a_max=None), np.clip(spred, a_min=0, a_max=None)


def pred_exp(t, ton, neg_slope, c0, u0, s0, kc, alpha_c, rho, alpha, beta, gamma):
    tau = F.leaky_relu(t - ton, neg_slope)
    expac, expb, expg = torch.exp(-alpha_c*tau), torch.exp(-beta*tau), torch.exp(-gamma*tau)
    eps = 1e-6

    cpred = c0*expac + kc*(1-expac)
    upred = u0*expb + rho*alpha*kc/beta*(1-expb) + (kc-c0)*rho*alpha/(beta-alpha_c+eps)*(expb-expac)
    spred = s0*expg + rho*alpha*kc/gamma*(1-expg)
    spred += (rho*alpha*kc/beta-u0-(kc-c0)*rho*alpha/(beta-alpha_c+eps))*beta/(gamma-beta+eps)*(expg-expb)
    spred += (kc-c0)*rho*alpha*beta/(gamma-alpha_c+eps)/(beta-alpha_c+eps)*(expg-expac)
    return nn.functional.sigmoid(cpred), nn.functional.relu(upred), nn.functional.relu(spred)


def linreg(u, s):
    q = np.sum(s*s)
    r = np.sum(u*s)
    k = r/q
    if np.isinf(k) or np.isnan(k):
        k = 1.0+np.random.rand()
    return k


def assign_time(c, u, s, c0_, u0_, s0_, alpha_c, alpha, beta, gamma, std_c_=None, std_s=None, t_num=1000, t_max=20):
    tau = np.linspace(0, t_max, t_num)
    ct, ut, st = compute_exp(tau, 0, 0, 0, 1, alpha_c, alpha, beta, gamma)
    anchor_mat = np.hstack((np.reshape(ct, (-1, 1))/std_c_*std_s, np.reshape(ut, (-1, 1)), np.reshape(st, (-1, 1))))
    exp_mat = np.hstack((np.reshape(c, (-1, 1))/std_c_*std_s, np.reshape(u, (-1, 1)), np.reshape(s, (-1, 1))))
    tree = KDTree(anchor_mat)
    dd, ii = tree.query(exp_mat, k=1)
    dd = dd**2
    t_pred = tau[ii]

    exp_ss = np.array([[c0_/std_c_*std_s, u0_, s0_]])
    ii_ss = tree.query(exp_ss, k=1)[1]
    t_ = tau[ii_ss][0]
    c0_pred, u0_pred, s0_pred = compute_exp(t_, 0, 0, 0, 1, alpha_c, alpha, beta, gamma)

    ct_, ut_, st_ = compute_exp(tau, c0_pred, u0_pred, s0_pred, 0, alpha_c, 0, beta, gamma)
    anchor_mat_ = np.hstack((np.reshape(ct_, (-1, 1))/std_c_*std_s, np.reshape(ut_, (-1, 1)), np.reshape(st_, (-1, 1))))
    tree_ = KDTree(anchor_mat_)
    dd_, ii_ = tree_.query(exp_mat, k=1)
    dd_ = dd_**2
    t_pred_ = tau[ii_]

    res = np.array([dd, dd_])
    t = np.array([t_pred, t_pred_+np.ones((len(t_pred_)))*t_])
    o = np.argmin(res, axis=0)
    t_latent = np.array([t[o[i], i] for i in range(len(t_pred))])

    return t_latent, t_


def init_gene(s, u, c, percent, fit_scaling=False, tmax=1):
    std_u, std_s = np.std(u), np.std(s)
    scaling = std_u / std_s if fit_scaling else 1.0
    u = u/scaling
    scaling_c = np.max(c)
    c = c/scaling_c
    std_c_ = np.std(c)

    # initialize beta and gamma from extreme quantiles of s
    thresh = np.mean(c) + np.std(c)
    mask_c = c >= thresh
    u_, s_ = u[mask_c], s[mask_c]
    mask_s = s >= np.percentile(s_, percent, axis=0)
    mask_u = u >= np.percentile(u_, percent, axis=0)
    mask = mask_s & mask_u & mask_c
    if not np.any(mask):
        mask = mask_s & mask_c
    if not np.any(mask):
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
    alpha_c = -np.log(1 - np.clip(np.median(c[mask]), 0, 0.999)) / t_
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


def init_params(data, percent, fit_scaling=True, tmax=1):
    ngene = data.shape[1]//3
    c = data[:, :ngene]
    u = data[:, ngene:ngene*2]
    s = data[:, ngene*2:]

    params = np.ones((ngene, 6))
    params[:, 0] = 0.1
    params[:, 1] = np.random.rand((ngene))*np.clip(np.max(u, 0), 0.001, None)
    params[:, 3] = np.random.rand((ngene))*np.clip(np.max(u, 0), 0.001, None)/np.clip(np.max(s, 0), 0.001, None)
    params[:, 4] = np.clip(np.max(c, 0), 0.001, None)
    T = np.zeros((ngene, len(s)))
    ts = np.zeros((ngene))
    c0, u0, s0 = np.zeros((ngene)), np.zeros((ngene)), np.zeros((ngene))

    for i in range(ngene):
        si, ui, ci = s[:, i], u[:, i], c[:, i]
        sfilt, ufilt, cfilt = si[(si > 0) & (ui > 0) & (ci > 0)], ui[(si > 0) & (ui > 0) & (ci > 0)], ci[(si > 0) & (ui > 0) & (ci > 0)]
        if (len(sfilt) > 3) and (len(ufilt) > 3) and (len(cfilt) > 3):
            alpha_c, alpha, beta, gamma, t, c0_, u0_, s0_, ts_, scaling_c, scaling = init_gene(sfilt, ufilt, cfilt, percent, fit_scaling, tmax)
            params[i, :] = np.array([alpha_c, alpha, beta, gamma, scaling_c, scaling])
            T[i, (si > 0) & (ui > 0) & (ci > 0)] = t
            c0[i] = c0_
            u0[i] = u0_
            s0[i] = s0_
            ts[i] = ts_
        else:
            c0[i] = np.percentile(c, 95)
            u0[i] = np.max(u)
            s0[i] = np.max(s)

    dist_c, dist_u, dist_s = np.zeros(c.shape), np.zeros(u.shape), np.zeros(s.shape)
    for i in range(ngene):
        cpred, upred, spred = pred_single(T[i], params[i, 0], params[i, 1], params[i, 2], params[i, 3], ts[i], params[i, 4], params[i, 5])  # upred has the original scale
        dist_c[:, i] = c[:, i] - cpred
        dist_u[:, i] = u[:, i] - upred
        dist_s[:, i] = s[:, i] - spred

    sigma_c = np.clip(np.std(dist_c, 0), 0.1, None)
    sigma_u = np.clip(np.std(dist_u, 0), 0.1, None)
    sigma_s = np.clip(np.std(dist_s, 0), 0.1, None)
    sigma_c[np.isnan(sigma_c)] = 0.1
    sigma_u[np.isnan(sigma_u)] = 0.1
    sigma_s[np.isnan(sigma_s)] = 0.1

    return params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5], ts, c0, u0, s0, sigma_c, sigma_u, sigma_s, T.T


def get_ts_global(tgl, C, U, S, perc):
    tsgl = np.zeros((U.shape[1]))
    for i in range(U.shape[1]):
        c, u, s = C[:, i], U[:, i], S[:, i]
        zero_mask = (c > 0) & (u > 0) & (s > 0)
        mask_c = c >= np.mean(c)
        mask_u, mask_s = u >= np.percentile(u[mask_c], perc), s >= np.percentile(s[mask_c], perc)
        tsgl[i] = np.median(tgl[mask_c & mask_u & mask_s & zero_mask])
        if np.isnan(tsgl[i]):
            tsgl[i] = np.median(tgl[mask_c & (mask_u | mask_s) & zero_mask])
        if np.isnan(tsgl[i]):
            tsgl[i] = np.median(tgl)
    assert not np.any(np.isnan(tsgl))
    return tsgl


def reinit_gene(c, u, s, t, ts):
    mask_c = (c > np.mean(c[c < 1])) & (c < 1)
    mask1_u = u > np.quantile(u[mask_c], 0.95)
    mask1_s = s > np.quantile(s[mask_c], 0.95)
    u1, s1 = np.median(u[mask_c & (mask1_u | mask1_s)]), np.median(s[mask_c & (mask1_s | mask1_u)])
    c1 = np.median(c[mask_c & (mask1_u | mask1_s)])

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


def reinit_params(C, U, S, t, ts):
    G = U.shape[1]
    alpha_c, alpha, beta, gamma, ton = np.zeros((G)), np.zeros((G)), np.zeros((G)), np.zeros((G)), np.zeros((G))
    for i in range(G):
        alpha_c_g, alpha_g, beta_g, gamma_g, ton_g = reinit_gene(C[:, i], U[:, i], S[:, i], t, ts[i])
        alpha_c[i] = alpha_c_g
        alpha[i] = alpha_g
        beta[i] = beta_g
        gamma[i] = gamma_g
        ton[i] = ton_g
    return alpha_c, alpha, beta, gamma, ton


def pred_steady_numpy(ts, alpha_c, alpha, beta, gamma):
    alpha_c_, alpha_, beta_, gamma_ = np.clip(alpha_c, a_min=0, a_max=None), np.clip(alpha, a_min=0, a_max=None), np.clip(beta, a_min=0, a_max=None), np.clip(gamma, a_min=0, a_max=None)
    eps = 1e-6
    ts_ = ts.squeeze()
    expac, expb, expg = np.exp(-alpha_c_*ts_), np.exp(-beta_*ts_), np.exp(-gamma_*ts_)
    c0 = 1.0-expac
    u0 = alpha_/(beta_+eps)*(1.0-expb)+alpha_/(beta_-alpha_c_+eps)*(expb-expac)
    s0 = alpha_/(gamma_+eps)*(1.0-expg)+(alpha_/beta_-alpha_/(beta_-alpha_c_+eps))*beta_/(gamma_-beta_+eps)*(expg-expb)
    s0 += alpha_*beta_/(gamma_-alpha_c_+eps)/(beta_-alpha_c_+eps)*(expg-expac)
    return c0, u0, s0


def pred_steady(tau_s, alpha_c, alpha, beta, gamma):
    eps = 1e-6
    expac, expb, expg = torch.exp(-alpha_c*tau_s), torch.exp(-beta*tau_s), torch.exp(-gamma*tau_s)
    c0 = torch.tensor([1.0]).to(alpha.device)-expac
    u0 = alpha/(beta+eps)*(torch.tensor([1.0]).to(alpha.device)-expb) + alpha/(beta-alpha_c+eps)*(expb-expac)
    s0 = alpha/(gamma+eps)*(torch.tensor([1.0]).to(alpha.device)-expg) + (alpha/beta-alpha/(beta-alpha_c+eps))*beta/(gamma-beta+eps)*(expg-expb)
    s0 += alpha*beta/(gamma-alpha_c+eps)/(beta-alpha_c+eps)*(expg-expac)
    return c0, u0, s0


def ode_numpy(t, alpha_c, alpha, beta, gamma, to, ts, scaling_c=None, scaling=None, k=10.0):
    eps = 1e-6
    o = (t <= ts).astype(int)

    tau_on = F.softplus(torch.tensor(t-to), beta=k).numpy()
    assert np.all(~np.isnan(tau_on))
    expac, expb, expg = np.exp(-alpha_c*tau_on), np.exp(-beta*tau_on), np.exp(-gamma*tau_on)
    chat_on = 1.0-expac
    uhat_on = alpha/(beta+eps)*(1.0-expb) + alpha/(beta-alpha_c+eps)*(expb-expac)
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
    chat_off = c0_*expac
    uhat_off = u0_*expb
    shat_off = s0_*expg-u0_*beta/(gamma-beta+eps)*(expg-expb)

    chat, uhat, shat = (chat_on*o + chat_off*(1-o)), (uhat_on*o + uhat_off*(1-o)), (shat_on*o + shat_off*(1-o))
    if scaling_c is not None:
        chat *= scaling_c
    if scaling is not None:
        uhat *= scaling
    return chat, uhat, shat


def ode(t, alpha_c, alpha, beta, gamma, to, ts, neg_slope=0.0):
    eps = 1e-6
    o = (t <= ts).int()

    tau_on = F.leaky_relu(t-to, negative_slope=neg_slope)
    expac, expb, expg = torch.exp(-alpha_c*tau_on), torch.exp(-beta*tau_on), torch.exp(-gamma*tau_on)
    chat_on = torch.tensor([1.0]).to(alpha.device)-expac
    uhat_on = alpha/(beta+eps)*(torch.tensor([1.0]).to(alpha.device)-expb) + alpha/(beta-alpha_c+eps)*(expb-expac)
    shat_on = alpha/(gamma+eps)*(torch.tensor([1.0]).to(alpha.device)-expg) + (alpha/beta-alpha/(beta-alpha_c+eps))*beta/(gamma-beta+eps)*(expg-expb)
    shat_on += alpha*beta/(gamma-alpha_c+eps)/(beta-alpha_c+eps)*(expg-expac)

    c0_, u0_, s0_ = pred_steady(F.relu(ts-to), alpha_c, alpha, beta, gamma)

    tau_off = F.leaky_relu(t-ts, negative_slope=neg_slope)
    expac, expb, expg = torch.exp(-alpha_c*tau_off), torch.exp(-beta*tau_off), torch.exp(-gamma*tau_off)
    chat_off = c0_*expac
    uhat_off = u0_*expb
    shat_off = s0_*expg-u0_*beta/(gamma-beta+eps)*(expg-expb)
    return (chat_on*o + chat_off*(1-o)), (uhat_on*o + uhat_off*(1-o)), (shat_on*o + shat_off*(1-o))


def kl_uniform(mu_t, std_t, t_start, t_end, **kwargs):
    tail = kwargs["tail"] if "tail" in kwargs else 0.05
    t0 = mu_t - np.sqrt(3)*std_t
    dt = np.sqrt(12)*std_t
    C = 1/((t_end-t_start)*(1+tail))
    lamb = 2/(tail*(t_end-t_start))

    t1 = t0+dt
    dt1_til = nn.functional.relu(torch.minimum(t_start, t1) - t0)
    dt2_til = nn.functional.relu(t1 - torch.maximum(t_end, t0))

    term1 = -lamb*(dt1_til.pow(2)+dt2_til.pow(2))/(2*dt)
    term2 = lamb*((t_start-t0)*dt1_til+(t1-t_end)*dt2_til)/dt
    return torch.mean(term1 + term2 - torch.log(C*dt))


def kl_gaussian(mu1, std1, mu2, std2, **kwargs):
    return torch.mean(torch.sum(torch.log(std2/std1) + std1.pow(2)/(2*std2.pow(2)) - 0.5 + (mu1-mu2).pow(2)/(2*std2.pow(2)), 1))


def knn_approx(C, U, S, c0, u0, s0, k):
    X = np.concatenate((C, U, S), 1)
    x0 = np.concatenate((c0, u0, s0), 1)
    pca = PCA(n_components=30, svd_solver='arpack', random_state=2022)
    X0_pca = pca.fit_transform(x0)
    X_pca = pca.transform(X)
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(X_pca)
    knn = knn_model.kneighbors(X0_pca, return_distance=False)
    return knn.astype(int)


def knnx0(C, U, S, t, z, t_query, z_query, dt, k):
    Nq = len(t_query)
    c0 = np.zeros((Nq, C.shape[1]))
    u0 = np.zeros((Nq, U.shape[1]))
    s0 = np.zeros((Nq, S.shape[1]))
    t0 = np.ones((Nq))*(t.min() - dt[0])
    knn_ind = np.zeros((Nq, k))

    n1 = 0
    len_avg = 0
    for i in range(Nq):
        t_ub, t_lb = t_query[i] - dt[0], t_query[i] - dt[1]
        indices = np.where((t >= t_lb) & (t < t_ub))[0]
        k_ = len(indices)
        len_avg = len_avg+k_
        if k_ > 0:
            if k_ < k:
                c0[i] = C[indices].mean(0)
                u0[i] = U[indices].mean(0)
                s0[i] = S[indices].mean(0)
                t0[i] = t[indices].mean()
                n1 = n1 + 1
                knn_ind[i, :k_] = indices
                knn_ind[i, k_:] = i  # set as self transitions for invalid cells
            else:
                knn_model = NearestNeighbors(n_neighbors=k)
                knn_model.fit(z[indices])
                _, ind = knn_model.kneighbors(z_query[i:i+1])
                ind_ = indices[ind.squeeze()].astype(int)
                c0[i] = np.mean(C[ind_], 0)
                u0[i] = np.mean(U[ind_], 0)
                s0[i] = np.mean(S[ind_], 0)
                t0[i] = np.mean(t[ind_])
                knn_ind[i, :] = ind_
    print(f"Percentage of Invalid Sets: {n1/Nq:.3f}")
    print(f"Average Set Size: {len_avg//Nq}")
    return c0, u0, s0, t0, knn_ind.astype(int)


def knnx0_bin(C,
              U,
              S,
              t,
              z,
              t_query,
              z_query,
              dt,
              k=None,
              n_graph=10,
              pruning_degree_multiplier=1.5,
              diversify_prob=1.0,
              max_bin_size=10000):
    tmin = min(t.min(), t_query.min())
    Nq = len(t_query)
    c0 = np.zeros((Nq, C.shape[1]))
    u0 = np.zeros((Nq, U.shape[1]))
    s0 = np.zeros((Nq, S.shape[1]))
    t0 = np.ones((Nq))*(t.min() - dt[0])
    knn_ind = np.zeros((Nq, Nq))

    delta_t = (np.quantile(t, 0.99)-tmin+1e-6)/(n_graph+1)

    indices = np.where(t < (tmin+delta_t))[0]
    rng = np.random.default_rng(2022)
    if len(indices) > max_bin_size:
        indices = rng.choice(indices, max_bin_size, replace=False)
    mask_init = t < np.quantile(t[indices], 0.2)
    c_init = C[mask_init].mean(0)
    u_init = U[mask_init].mean(0)
    s_init = S[mask_init].mean(0)
    indices_query = np.where(t_query < (tmin+delta_t))[0]
    c0[indices_query] = c_init
    u0[indices_query] = u_init
    s0[indices_query] = s_init
    t0[indices_query] = tmin
    knn_ind[indices_query, indices_query] = 1

    for i in range(n_graph):
        t_ub, t_lb = tmin+(i+1)*delta_t, tmin+i*delta_t
        indices = np.where((t >= t_lb) & (t < t_ub))[0]
        if len(indices) > max_bin_size:
            indices = rng.choice(indices, max_bin_size, replace=False)
        k_ = len(indices)
        if k_ == 0:
            continue
        if k is None:
            k = max(1, len(indices)//20)
        knn_model = pynndescent.NNDescent(z[indices], n_neighbors=k+1, pruning_degree_multiplier=pruning_degree_multiplier, diversify_prob=diversify_prob)
        indices_query = np.where((t_query >= t_ub) & (t_query < t_ub+delta_t))[0] if (i < n_graph-1) else np.where(t_query >= t_ub)[0]
        if len(indices_query) == 0:
            continue
        try:
            ind, _ = knn_model.query(z_query[indices_query], k=k)
            ind = ind.astype(int)
        except ValueError:
            knn_model = NearestNeighbors(n_neighbors=min(k, len(indices)))
            knn_model.fit(z[indices])
            _, ind = knn_model.kneighbors(z_query[indices_query])

        for j in range(len(indices_query)):
            neighbor_idx = indices[ind[j]]
            c0[indices_query[j]] = np.mean(C[neighbor_idx], 0)
            u0[indices_query[j]] = np.mean(U[neighbor_idx], 0)
            s0[indices_query[j]] = np.mean(S[neighbor_idx], 0)
            t0[indices_query[j]] = np.mean(t[neighbor_idx])
            knn_ind[indices_query[j], neighbor_idx] = 1
    k_max = np.max(np.sum(knn_ind, 1))
    knn_ind_ = np.zeros((Nq, min(k_max, 20)))
    for i in range(Nq):
        if np.sum(knn_ind[i]) > knn_ind_.shape[1]:
            knn_ind_[i] = rng.choice(np.where(knn_ind[i] > 0)[0], knn_ind_.shape[1], replace=False)
        else:
            knn_ind_[i, :np.sum(knn_ind[i])] = np.where(knn_ind[i] > 0)[0]
            knn_ind_[i, np.sum(knn_ind[i]):] = i
    return c0, u0, s0, t0, knn_ind_.astype(int)


def cosine_similarity(U, S, beta, gamma, S_knn, onehot=None):
    if onehot is not None:
        V = (torch.mm(onehot, beta) * U - torch.mm(onehot, gamma) * S) + 1e-6
    else:
        V = (beta * U - gamma * S) + 1e-6
    dS = torch.reshape(S, (S.size(0), 1, S.size(1))) - S_knn + 1e-6  # cell x knn x gene
    ds_norm = torch.linalg.norm(dS, dim=2)
    dS = dS / ds_norm[:, :, None]
    v_norm = torch.linalg.norm(V, dim=1)
    V2 = V / v_norm[:, None]
    V2 = torch.reshape(V2, (V2.size(0), V2.size(1), 1))  # cell x gene x 1
    cos_sim = torch.matmul(dS, V2)  # cell x knn x 1
    cos_sim = torch.sum(torch.mean(torch.sum(cos_sim, 2), 1))
    return cos_sim
