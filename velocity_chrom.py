import numpy as np
from scipy.ndimage import gaussian_filter1d
from .model_util_chrom import pred_exp_numpy
from .transition_graph import encode_type


def rna_velocity_vae(adata,
                     adata_atac,
                     key,
                     batch_key=None,
                     use_raw=False,
                     sigma=None,
                     approx=False,
                     return_copy=False):
    n_batch = 0
    batch = None
    if batch_key is not None and batch_key in adata.obs:
        batch_raw = adata.obs[batch_key].to_numpy()
        batch_names_raw = np.unique(batch_raw)
        batch_dic, _ = encode_type(batch_names_raw)
        n_batch = len(batch_names_raw)
        batch = np.array([batch_dic[x] for x in batch_raw])
        onehot = np.zeros((batch.size, batch.max() + 1))
        onehot[np.arange(batch.size), batch] = 1

    kc = adata.layers[f"{key}_kc"]
    rho = adata.layers[f"{key}_rho"]
    if batch_key is not None:
        alpha_c = np.zeros((n_batch, adata.n_vars))
        alpha = np.zeros((n_batch, adata.n_vars))
        beta = np.zeros((n_batch, adata.n_vars))
        gamma = np.zeros((n_batch, adata.n_vars))
        scaling_c = np.zeros((n_batch, adata.n_vars))
        for i in range(n_batch):
            alpha_c[i, :] = adata.var[f"{key}_alpha_c_{i}"].to_numpy()
            alpha[i, :] = adata.var[f"{key}_alpha_{i}"].to_numpy()
            beta[i, :] = adata.var[f"{key}_beta_{i}"].to_numpy()
            gamma[i, :] = adata.var[f"{key}_gamma_{i}"].to_numpy()
            scaling_c[i, :] = adata.var[f"{key}_scaling_c_{i}"].to_numpy()
        alpha_c = np.dot(onehot, alpha_c)
        alpha = np.dot(onehot, alpha)
        beta = np.dot(onehot, beta)
        gamma = np.dot(onehot, gamma)
        scaling_c = np.dot(onehot, scaling_c)
    else:
        alpha_c = adata.var[f"{key}_alpha_c"].to_numpy()
        alpha = adata.var[f"{key}_alpha"].to_numpy()
        beta = adata.var[f"{key}_beta"].to_numpy()
        gamma = adata.var[f"{key}_gamma"].to_numpy()
        scaling_c = adata.var[f"{key}_scaling_c"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    t0 = adata.obs[f"{key}_t0"].to_numpy()
    c0 = adata.layers[f"{key}_c0"]
    u0 = adata.layers[f"{key}_u0"]
    s0 = adata.layers[f"{key}_s0"]

    if use_raw:
        c, u, s = adata_atac.layers['Mc'], adata.layers['Mu'], adata.layers['Ms']
    else:
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        if f"{key}_chat" in adata.layers and f"{key}_uhat" in adata.layers and f"{key}_shat" in adata.layers:
            c, u, s = adata.layers[f"{key}_chat"], adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
            c = c/scaling_c
            u = u/scaling
        else:
            tau = np.clip(t - t0, 0, None).reshape(-1, 1)
            c, u, s = pred_exp_numpy(tau, c0/scaling_c, u0/scaling, s0, kc, alpha_c, rho, alpha, beta, gamma)
            c, u, s = np.clip(c, 0, 1), np.clip(u, 0, None), np.clip(s, 0, None)
            adata.layers["chat"] = c * scaling_c
            adata.layers["uhat"] = u * scaling
            adata.layers["shat"] = s
    if approx:
        v = (s - s0)/((t - t0).reshape(-1, 1))
        vu = (u - u0)/((t - t0).reshape(-1, 1))
        vc = (c - c0)/((t - t0).reshape(-1, 1))
    else:
        v = beta * u - gamma * s
        vu = rho * alpha * c - beta * u
        vc = kc * alpha_c - alpha_c * c
    if sigma is not None:
        time_order = np.argsort(t)
        v[time_order] = gaussian_filter1d(v[time_order], sigma, axis=0, mode="nearest")
        vu[time_order] = gaussian_filter1d(vu[time_order], sigma, axis=0, mode="nearest")
        vc[time_order] = gaussian_filter1d(vc[time_order], sigma, axis=0, mode="nearest")
    adata.layers[f"{key}_velocity"] = v
    adata.layers[f"{key}_velocity_u"] = vu
    adata.layers[f"{key}_velocity_c"] = vc
    adata.var[f'{key}_velocity_genes'] = True
    if return_copy:
        return vc, vu, v, c, u, s
