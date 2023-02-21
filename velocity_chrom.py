import numpy as np
from scipy.ndimage import gaussian_filter1d
from .model_util_chrom import pred_exp_numpy
from .TransitionGraph import encode_type


def rna_velocity_vae(adata, adata_atac, key, batch_key=None, use_raw=False, use_scv_genes=False, sigma=None, approx=False, full_vb=False):
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
        alpha_c, alpha, beta, gamma = np.zeros((n_batch, adata.n_vars)), np.zeros((n_batch, adata.n_vars)), np.zeros((n_batch, adata.n_vars)), np.zeros((n_batch, adata.n_vars))
        for i in range(n_batch):
            alpha_c[i, :] = adata.var[f"{key}_alpha_c_{i}"].to_numpy()
            alpha[i, :] = adata.var[f"{key}_alpha_{i}"].to_numpy()
            beta[i, :] = adata.var[f"{key}_beta_{i}"].to_numpy()
            gamma[i, :] = adata.var[f"{key}_gamma_{i}"].to_numpy()
        alpha_c, alpha, beta, gamma = np.dot(onehot, alpha_c), np.dot(onehot, alpha), np.dot(onehot, beta), np.dot(onehot, gamma)
    else:
        alpha_c = adata.var[f"{key}_alpha_c"].to_numpy()
        alpha = adata.var[f"{key}_alpha"].to_numpy()
        beta = adata.var[f"{key}_beta"].to_numpy()
        gamma = adata.var[f"{key}_gamma"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    t0 = adata.obs[f"{key}_t0"].to_numpy()
    c0 = adata.layers[f"{key}_c0"]
    u0 = adata.layers[f"{key}_u0"]
    s0 = adata.layers[f"{key}_s0"]

    if use_raw:
        C, U, S = adata_atac.layers['Mc'], adata.layers['Mu'], adata.layers['Ms']
    else:
        scaling_c = adata.var[f"{key}_scaling_c"].to_numpy()
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        if f"{key}_chat" in adata.layers and f"{key}_uhat" in adata.layers and f"{key}_shat" in adata.layers:
            C, U, S = adata.layers[f"{key}_chat"], adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
            C = C/scaling_c
            U = U/scaling
        else:
            C, U, S = pred_exp_numpy(t, t0, c0/scaling_c, u0/scaling, s0, kc, alpha_c, rho, alpha, beta, gamma)
            C, U, S = np.clip(C, 0, None), np.clip(U, 0, None), np.clip(S, 0, None)
            adata.layers["chat"] = C * scaling_c
            adata.layers["uhat"] = U * scaling
            adata.layers["shat"] = S
    if approx:
        V = (S - s0)/((t - t0).reshape(-1, 1))
    else:
        V = (beta * U - gamma * S)
    if sigma is not None:
        time_order = np.argsort(t)
        V[time_order] = gaussian_filter1d(V[time_order], sigma, axis=0, mode="nearest")
    adata.layers[f"{key}_velocity"] = V
    if use_scv_genes:
        gene_mask = np.isnan(adata.var['fit_scaling'].to_numpy())
        V[:, gene_mask] = np.nan
    return V, C, U, S
