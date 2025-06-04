import logging
import numpy as np
from scipy.sparse import issparse
from scipy.ndimage import gaussian_filter1d
from .model_util_chrom import pred_exp_numpy, encode_type
logger = logging.getLogger(__name__)


def velocity(adata,
             adata_atac,
             key,
             batch_key=None,
             ref_batch=None,
             batch_hvg_key=None,
             batch_correction=False,
             use_original=False,
             rna_only=False,
             likelihood_thred=None,
             use_only_ref=True,
             sigma=None,
             approx=False):
    """Compute multi-omic velocity for VAE.

    Args:
        adata (:class:`anndata.AnnData`):
            RNA AnnData object.
        adata_atac (:class:`anndata.AnnData`):
            ATAC AnnData object.
        key (str):
            Key of VAE variables.
        batch_key (str, optional):
            Field in adata.obs to find batch labels. Defaults to None.
        ref_batch (int, optional):
            Index to use as the reference batch. Defaults to None.
        batch_hvg_key (str, optional):
            Prefix of key for batch highly-variable genes in adata.var. Defaults to None.
        batch_correction (bool, optional):
            Whether the output was generated with batch correction. Defaults to False.
        use_original (bool, optional):
            whether to use the (noisy) input count to compute the velocity. Defaults to False.
        rna_only (bool, optional):
            Whether the model was trained with RNA only. Defaults to False.
        likelihood_thred (float, optional):
            Threshold to set high likelihood velocity genes. Defaults to None.
        use_only_ref (bool, optional):
            Select velocity genes only from highly variable gene list of the reference batch. Defaults to True.
        sigma (float, optional):
            Parameter used in Gaussian filtering of velocity values. Defaults to None.
        approx (bool, optional):
            Whether to use linear approximation to compute velocity. Defaults to False.
    """
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

        kc_batch = adata.layers[f"{key}_kc_batch"]
        rho_batch = adata.layers[f"{key}_rho_batch"]
        alpha_c_batch = np.zeros((n_batch, adata.n_vars))
        alpha_batch = np.zeros((n_batch, adata.n_vars))
        beta_batch = np.zeros((n_batch, adata.n_vars))
        gamma_batch = np.zeros((n_batch, adata.n_vars))
        scaling_c_batch = np.zeros((n_batch, adata.n_vars))
        scaling_u_batch = np.zeros((n_batch, adata.n_vars))
        scaling_s_batch = np.zeros((n_batch, adata.n_vars))
        offset_c_batch = np.zeros((n_batch, adata.n_vars))
        offset_u_batch = np.zeros((n_batch, adata.n_vars))
        offset_s_batch = np.zeros((n_batch, adata.n_vars))
        for i in range(n_batch):
            alpha_c_batch[i, :] = adata.var[f"{key}_alpha_c_{i}"].to_numpy()
            alpha_batch[i, :] = adata.var[f"{key}_alpha_{i}"].to_numpy()
            beta_batch[i, :] = adata.var[f"{key}_beta_{i}"].to_numpy()
            gamma_batch[i, :] = adata.var[f"{key}_gamma_{i}"].to_numpy()
            scaling_c_batch[i, :] = adata.var[f"{key}_scaling_c_{i}"].to_numpy()
            scaling_u_batch[i, :] = adata.var[f"{key}_scaling_u_{i}"].to_numpy()
            scaling_s_batch[i, :] = adata.var[f"{key}_scaling_s_{i}"].to_numpy()
            offset_c_batch[i, :] = adata.var[f"{key}_offset_c_{i}"].to_numpy()
            offset_u_batch[i, :] = adata.var[f"{key}_offset_u_{i}"].to_numpy()
            offset_s_batch[i, :] = adata.var[f"{key}_offset_s_{i}"].to_numpy()
        alpha_c_batch = np.dot(onehot, alpha_c_batch)
        alpha_batch = np.dot(onehot, alpha_batch)
        beta_batch = np.dot(onehot, beta_batch)
        gamma_batch = np.dot(onehot, gamma_batch)
        scaling_c_batch = np.dot(onehot, scaling_c_batch)
        scaling_u_batch = np.dot(onehot, scaling_u_batch)
        scaling_s_batch = np.dot(onehot, scaling_s_batch)
        offset_c_batch = np.dot(onehot, offset_c_batch)
        offset_u_batch = np.dot(onehot, offset_u_batch)
        offset_s_batch = np.dot(onehot, offset_s_batch)
    else:
        kc = adata.layers[f"{key}_kc"]
        rho = adata.layers[f"{key}_rho"]
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

    if use_original:
        c, u, s = adata_atac.layers['Mc'], adata.layers['Mu'], adata.layers['Ms']
        c = c.toarray() if issparse(c) else c
        u = u.toarray() if issparse(u) else u
        s = s.toarray() if issparse(s) else s
        if batch_key is not None and batch_key in adata.obs:
            c = (c-offset_c_batch)/scaling_c_batch
            u = (u-offset_u_batch)/scaling_u_batch
            s = (s-offset_s_batch)/scaling_s_batch
        else:
            c = (c-offset_c)/scaling_c
            u = (u-offset_u)/scaling_u
            s = (s-offset_s)/scaling_s
    else:
        if batch_correction or batch_key is None or batch_key not in adata.obs:
            c = (adata.layers[f"{key}_chat"]-offset_c)/scaling_c
            u = (adata.layers[f"{key}_uhat"]-offset_u)/scaling_u
            s = (adata.layers[f"{key}_shat"]-offset_s)/scaling_s
            adata.layers["chat_scaled"] = c
            adata.layers["uhat_scaled"] = u
            adata.layers["shat_scaled"] = s
        elif f"{key}_chat_batch" in adata.layers and f"{key}_uhat_batch" in adata.layers and f"{key}_shat_batch" in adata.layers:
            c = (adata.layers[f"{key}_chat_batch"]-offset_c_batch)/scaling_c_batch
            u = (adata.layers[f"{key}_uhat_batch"]-offset_u_batch)/scaling_u_batch
            s = (adata.layers[f"{key}_shat_batch"]-offset_s_batch)/scaling_s_batch
            adata.layers["chat_scaled"] = c
            adata.layers["uhat_scaled"] = u
            adata.layers["shat_scaled"] = s
        else:
            tau = np.clip(t - t0, 0, None).reshape(-1, 1)
            c, u, s = pred_exp_numpy(tau, c0, u0, s0, kc, alpha_c, rho, alpha, beta, gamma)
            c, u, s = np.clip(c, 0, 1), np.clip(u, 0, None), np.clip(s, 0, None)
            adata.layers["chat"] = c * scaling_c + offset_c
            adata.layers["uhat"] = u * scaling_u + offset_u
            adata.layers["shat"] = s * scaling_s + offset_s
        if batch_correction:
            c_mean_adjust = np.zeros((n_batch, adata.n_vars))
            u_mean_adjust = np.zeros((n_batch, adata.n_vars))
            s_mean_adjust = np.zeros((n_batch, adata.n_vars))
            c_std_adjust = np.zeros((n_batch, adata.n_vars))
            u_std_adjust = np.zeros((n_batch, adata.n_vars))
            s_std_adjust = np.zeros((n_batch, adata.n_vars))
            for i in range(n_batch):
                c_mean_adjust[i] = np.mean(adata.layers[f"{key}_chat_batch"][batch == i], 0) - np.mean(adata.layers[f"{key}_chat"][batch == i], 0)
                u_mean_adjust[i] = np.mean(adata.layers[f"{key}_uhat_batch"][batch == i], 0) - np.mean(adata.layers[f"{key}_uhat"][batch == i], 0)
                s_mean_adjust[i] = np.mean(adata.layers[f"{key}_shat_batch"][batch == i], 0) - np.mean(adata.layers[f"{key}_shat"][batch == i], 0)
                c_std_adjust[i] = np.std(adata.layers[f"{key}_chat_batch"][batch == i], 0) / np.std(adata.layers[f"{key}_chat"][batch == i], 0)
                u_std_adjust[i] = np.std(adata.layers[f"{key}_uhat_batch"][batch == i], 0) / np.std(adata.layers[f"{key}_uhat"][batch == i], 0)
                s_std_adjust[i] = np.std(adata.layers[f"{key}_shat_batch"][batch == i], 0) / np.std(adata.layers[f"{key}_shat"][batch == i], 0)
            c_mean_adjust = np.dot(onehot, c_mean_adjust)
            u_mean_adjust = np.dot(onehot, u_mean_adjust)
            s_mean_adjust = np.dot(onehot, s_mean_adjust)
            c_std_adjust = np.dot(onehot, c_std_adjust)
            u_std_adjust = np.dot(onehot, u_std_adjust)
            s_std_adjust = np.dot(onehot, s_std_adjust)
            adata.layers["c_leveled"] = (adata_atac.layers['Mc'] - c_mean_adjust) / c_std_adjust
            adata.layers["u_leveled"] = (adata.layers['Mu'] - u_mean_adjust) / u_std_adjust
            adata.layers["s_leveled"] = (adata.layers['Ms'] - s_mean_adjust) / s_std_adjust

    if approx:
        v = (s - (s0*scaling_s+offset_s)) / ((t - t0).reshape(-1, 1))
        vu = (u - (u0*scaling_u+offset_u)) / ((t - t0).reshape(-1, 1))
        vc = (c - (c0*scaling_c+offset_c)) / ((t - t0).reshape(-1, 1))
    elif batch_correction or batch_key is None or batch_key not in adata.obs:
        v = beta * u - gamma * s
        vu = rho * alpha * c - beta * u
        vc = kc * alpha_c - alpha_c * c
    else:
        v = beta_batch * u - gamma_batch * s
        vu = rho_batch * alpha_batch * c - beta_batch * u
        vc = kc_batch * alpha_c_batch - alpha_c_batch * c
    if sigma is not None:
        time_order = np.argsort(t)
        v[time_order] = gaussian_filter1d(v[time_order], sigma, axis=0, mode="nearest")
        vu[time_order] = gaussian_filter1d(vu[time_order], sigma, axis=0, mode="nearest")
        vc[time_order] = gaussian_filter1d(vc[time_order], sigma, axis=0, mode="nearest")
    if batch_correction or batch_key is None or batch_key not in adata.obs:
        v = v * scaling_s
        vu = vu * scaling_u
        vc = vc * scaling_c
    else:
        v = v * scaling_s_batch
        vu = vu * scaling_u_batch
        vc = vc * scaling_c_batch
    adata.layers[f"{key}_velocity"] = v
    adata.layers[f"{key}_velocity_u"] = vu
    adata.layers[f"{key}_velocity_c"] = vc

    if likelihood_thred is None:
        if rna_only:
            likelihood_thred = 0.05
        elif batch_correction:
            likelihood_thred = 0.01 + 0.01 * np.log2(n_batch)
        else:
            likelihood_thred = 0.025
    adata.var[f'{key}_velocity_genes'] = adata.var['quantile_genes'] & (adata.var[f"{key}_likelihood"] > likelihood_thred)
    if ref_batch is not None and use_only_ref and f"{batch_hvg_key}-{batch_dic_rev[ref_batch]}" in adata.var:
        adata.var[f'{key}_velocity_genes'] = adata.var[f'{key}_velocity_genes'] & adata.var[f"{batch_hvg_key}-{batch_dic_rev[ref_batch]}"]
    print(f"Selected {np.sum(adata.var[f'{key}_velocity_genes'])} velocity genes.")
    if np.sum(adata.var[f'{key}_velocity_genes']) < 0.2 * adata.n_vars:
        logger.warning('Less than 1/5 of genes assigned as velocity genes. Consider reduring the number of rounds in Stage 2: n_refine (default 6).')
    adata.uns[f"{key}_velocity_params"] = {'mode': 'dynamical'}
