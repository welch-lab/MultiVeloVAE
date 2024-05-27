import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .model_util_chrom import pred_exp_numpy
from .transition_graph import encode_type
logger = logging.getLogger(__name__)


def rna_velocity_vae(adata,
                     adata_atac,
                     key,
                     batch_key=None,
                     ref_batch=None,
                     batch_hvg_key=None,
                     batch_correction=False,
                     use_raw=False,
                     rna_only=False,
                     sigma=None,
                     approx=False,
                     return_copy=False):
    n_batch = 0
    batch = None
    if batch_key is not None and batch_key in adata.obs:
        batch_raw = adata.obs[batch_key].to_numpy()
        batch_names_raw = np.unique(batch_raw)
        batch_dic, batch_dic_rev = encode_type(batch_names_raw)
        n_batch = len(batch_names_raw)
        batch = np.array([batch_dic[x] for x in batch_raw])
        onehot = np.zeros((batch.size, batch.max() + 1))
        onehot[np.arange(batch.size), batch] = 1

    kc = adata.layers[f"{key}_kc"]
    rho = adata.layers[f"{key}_rho"]
    if batch_key is not None and batch_key in adata.obs:
        if batch_correction:
            alpha_c = adata.var[f"{key}_alpha_c_{ref_batch}"].to_numpy()
            alpha = adata.var[f"{key}_alpha_{ref_batch}"].to_numpy()
            beta = adata.var[f"{key}_beta_{ref_batch}"].to_numpy()
            gamma = adata.var[f"{key}_gamma_{ref_batch}"].to_numpy()
            scaling_c = adata.var[f"{key}_scaling_c_{ref_batch}"].to_numpy()
            scaling_u = adata.var[f"{key}_scaling_u_{ref_batch}"].to_numpy()
            scaling_s = adata.var[f"{key}_scaling_s_{ref_batch}"].to_numpy()
            offset_c = adata.var[f"{key}_offset_c_{ref_batch}"].to_numpy()
            offset_u = adata.var[f"{key}_offset_u_{ref_batch}"].to_numpy()
            offset_s = adata.var[f"{key}_offset_s_{ref_batch}"].to_numpy()

            scaling_c_batch = np.zeros((n_batch, adata.n_vars))
            scaling_u_batch = np.zeros((n_batch, adata.n_vars))
            scaling_s_batch = np.zeros((n_batch, adata.n_vars))
            offset_c_batch = np.zeros((n_batch, adata.n_vars))
            offset_u_batch = np.zeros((n_batch, adata.n_vars))
            offset_s_batch = np.zeros((n_batch, adata.n_vars))
            for i in range(n_batch):
                scaling_c_batch[i, :] = adata.var[f"{key}_scaling_c_{i}"].to_numpy()
                scaling_u_batch[i, :] = adata.var[f"{key}_scaling_u_{i}"].to_numpy()
                scaling_s_batch[i, :] = adata.var[f"{key}_scaling_s_{i}"].to_numpy()
                offset_c_batch[i, :] = adata.var[f"{key}_offset_c_{i}"].to_numpy()
                offset_u_batch[i, :] = adata.var[f"{key}_offset_u_{i}"].to_numpy()
                offset_s_batch[i, :] = adata.var[f"{key}_offset_s_{i}"].to_numpy()
            scaling_c_batch = np.dot(onehot, scaling_c_batch)
            scaling_u_batch = np.dot(onehot, scaling_u_batch)
            scaling_s_batch = np.dot(onehot, scaling_s_batch)
            offset_c_batch = np.dot(onehot, offset_c_batch)
            offset_u_batch = np.dot(onehot, offset_u_batch)
            offset_s_batch = np.dot(onehot, offset_s_batch)
        else:
            alpha_c = np.zeros((n_batch, adata.n_vars))
            alpha = np.zeros((n_batch, adata.n_vars))
            beta = np.zeros((n_batch, adata.n_vars))
            gamma = np.zeros((n_batch, adata.n_vars))
            scaling_c = np.zeros((n_batch, adata.n_vars))
            scaling_u = np.zeros((n_batch, adata.n_vars))
            scaling_s = np.zeros((n_batch, adata.n_vars))
            offset_c = np.zeros((n_batch, adata.n_vars))
            offset_u = np.zeros((n_batch, adata.n_vars))
            offset_s = np.zeros((n_batch, adata.n_vars))
            for i in range(n_batch):
                alpha_c[i, :] = adata.var[f"{key}_alpha_c_{i}"].to_numpy()
                alpha[i, :] = adata.var[f"{key}_alpha_{i}"].to_numpy()
                beta[i, :] = adata.var[f"{key}_beta_{i}"].to_numpy()
                gamma[i, :] = adata.var[f"{key}_gamma_{i}"].to_numpy()
                scaling_c[i, :] = adata.var[f"{key}_scaling_c_{i}"].to_numpy()
                scaling_u[i, :] = adata.var[f"{key}_scaling_u_{i}"].to_numpy()
                scaling_s[i, :] = adata.var[f"{key}_scaling_s_{i}"].to_numpy()
                offset_c[i, :] = adata.var[f"{key}_offset_c_{i}"].to_numpy()
                offset_u[i, :] = adata.var[f"{key}_offset_u_{i}"].to_numpy()
                offset_s[i, :] = adata.var[f"{key}_offset_s_{i}"].to_numpy()
            alpha_c = np.dot(onehot, alpha_c)
            alpha = np.dot(onehot, alpha)
            beta = np.dot(onehot, beta)
            gamma = np.dot(onehot, gamma)
            scaling_c = np.dot(onehot, scaling_c)
            scaling_u = np.dot(onehot, scaling_u)
            scaling_s = np.dot(onehot, scaling_s)
            offset_c = np.dot(onehot, offset_c)
            offset_u = np.dot(onehot, offset_u)
            offset_s = np.dot(onehot, offset_s)
    else:
        alpha_c = adata.var[f"{key}_alpha_c"].to_numpy()
        alpha = adata.var[f"{key}_alpha"].to_numpy()
        beta = adata.var[f"{key}_beta"].to_numpy()
        gamma = adata.var[f"{key}_gamma"].to_numpy()
        scaling_c = adata.var[f"{key}_scaling_c"].to_numpy()
        scaling_u = adata.var[f"{key}_scaling_u"].to_numpy()
        scaling_s = adata.var[f"{key}_scaling_s"].to_numpy()
        offset_c = adata.var[f"{key}_offset_c"].to_numpy()
        offset_u = adata.var[f"{key}_offset_u"].to_numpy()
        offset_s = adata.var[f"{key}_offset_s"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    t0 = adata.obs[f"{key}_t0"].to_numpy()
    c0 = adata.layers[f"{key}_c0"]
    u0 = adata.layers[f"{key}_u0"]
    s0 = adata.layers[f"{key}_s0"]

    if use_raw:
        c, u, s = adata_atac.layers['Mc'], adata.layers['Mu'], adata.layers['Ms']
        if batch_correction:
            c = (c-offset_c_batch)/scaling_c_batch
            u = (u-offset_u_batch)/scaling_u_batch
            s = (s-offset_s_batch)/scaling_s_batch
        else:
            c = (c-offset_c)/scaling_c
            u = (u-offset_u)/scaling_u
            s = (s-offset_s)/scaling_s
    else:
        if f"{key}_chat" in adata.layers and f"{key}_uhat" in adata.layers and f"{key}_shat" in adata.layers:
            c = (adata.layers[f"{key}_chat"]-offset_c)/scaling_c
            u = (adata.layers[f"{key}_uhat"]-offset_u)/scaling_u
            s = (adata.layers[f"{key}_shat"]-offset_s)/scaling_s
            adata.layers["chat_scaled"] = c
            adata.layers["uhat_scaled"] = u
            adata.layers["shat_scaled"] = s
        else:
            tau = np.clip(t - t0, 0, None).reshape(-1, 1)
            c, u, s = pred_exp_numpy(tau, (c0-offset_c)/scaling_c, (u0-offset_u)/scaling_u, (s0-offset_s)/scaling_s, kc, alpha_c, rho, alpha, beta, gamma)
            c, u, s = np.clip(c, 0, 1), np.clip(u, 0, None), np.clip(s, 0, None)
            adata.layers["chat"] = c * scaling_c + offset_c
            adata.layers["uhat"] = u * scaling_u + offset_u
            adata.layers["shat"] = s * scaling_s + offset_s
            adata.layers["chat_scaled"] = c
            adata.layers["uhat_scaled"] = u
            adata.layers["shat_scaled"] = s
        if batch_correction:
            adata.layers["c_leveled"] = (adata_atac.layers['Mc']-offset_c_batch)/scaling_c_batch * scaling_c + offset_c
            adata.layers["u_leveled"] = (adata.layers['Mu']-offset_u_batch)/scaling_u_batch * scaling_u + offset_u
            adata.layers["s_leveled"] = (adata.layers['Ms']-offset_s_batch)/scaling_s_batch * scaling_s + offset_s
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
    adata.layers[f"{key}_velocity"] = v * scaling_s
    adata.layers[f"{key}_velocity_u"] = vu * scaling_u
    adata.layers[f"{key}_velocity_c"] = vc * scaling_c
    adata.var[f'{key}_velocity_genes'] = adata.var['quantile_genes'] & (adata.var[f"{key}_likelihood"] > (0.025 if not rna_only else 0.05))
    if ref_batch is not None and f"{batch_hvg_key}-{batch_dic_rev[ref_batch]}" in adata.var:
        adata.var[f'{key}_velocity_genes'] = adata.var[f'{key}_velocity_genes'] & adata.var[f"{batch_hvg_key}-{batch_dic_rev[ref_batch]}"]
    print(f"Selected {np.sum(adata.var[f'{key}_velocity_genes'])} velocity genes.")
    if np.sum(adata.var[f'{key}_velocity_genes']) < 0.2 * adata.n_vars:
        logger.warn('Less than 1/5 of genes assigned as velocity genes.')
    adata.uns[f"{key}_velocity_params"] = {'mode': 'dynamical'}
    if return_copy:
        return vc, vu, v, c, u, s
