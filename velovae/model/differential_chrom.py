import os
import logging
import numpy as np
from scipy.sparse import issparse
import pandas as pd
from .training_data_chrom import SCData, SCDataE
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel
from scipy.stats.distributions import chi2
from tqdm.auto import tqdm
from joblib import Parallel, delayed
logger = logging.getLogger(__name__)


def difference(v1, v2, norm=0.1, eps=1e-8):
    return (v1 - v2) / (norm + eps)


def fold_change(v1, v2, eps=1e-8):
    return v1 / (v2 + eps)


def log2_difference(v1, v2, norm=0.1):
    return np.log2(np.abs(v1 - v2) / norm + 1) * np.sign(v1 - v2)


def log2_fold_change(v1, v2, eps=1e-8):
    return np.log2(v1 + eps) - np.log2(v2 + eps)


# scVI False Discovery Proportion method [Boyeau2019]
def fdp(df, col, fdr):
    p_col = f'p_{col}_no_change'
    original_order = df.index
    df = df.sort_values(p_col, ascending=True)
    cum_fdr = np.cumsum(df[p_col].values) / np.arange(1, df.shape[0] + 1)
    kd = np.sum(cum_fdr <= fdr)
    df[f'fdr_{col}<{fdr}'] = False
    df.loc[df.index[:kd], f'fdr_{col}<{fdr}'] = True
    return df.loc[original_order]


# Inspired by [Lopez2018] and [Boyeau2019]
# https://docs.scvi-tools.org/en/stable/user_guide/background/differential_expression.html
def differential_dynamics(adata,
                          adata_atac,
                          model,
                          group1=None,
                          group2=None,
                          group_key=None,
                          idx1=None,
                          idx2=None,
                          batch_key=None,
                          batch_correction=True,
                          weight_batch_uniform=False,
                          mode='vanilla',
                          signed_velocity=True,
                          test_decoupling=False,
                          save_raw=False,
                          n_samples=5000,
                          norm=0.1,
                          delta=1,
                          fdr=0.05,
                          seed=0):
    """Calculate differential dynamics between two groups of cells.

    Args:
        adata (:class:`anndata.AnnData`):
            RNA AnnData object.
        adata_atac (:class:`anndata.AnnData`):
            ATAC AnnData object.
        model (:class:`velovae.VAEChrom`):
            A trained MultiVeloVAE model.
        group1 (str, optional):
            Name of group 1 in adata.obs[`group_key`]. Defaults to None.
        group2 (str, optional):
            Name of group 2 in adata.obs[`group_key`]. Defaults to None.
        group_key (str, optional):
            Field in adata.obs to find group labels. Defaults to None.
        idx1 ([list, :class:`numpy.ndarray`], optional):
            Indices of cells in group 1. Defaults to None.
        idx2 ([list, :class:`numpy.ndarray`], optional):
            Indices of cells in group 2. Defaults to None.
        batch_key (str, optional):
            Field in adata.obs to find batch labels. Defaults to None.
        batch_correction (bool, optional):
            Whether to perform batch correction before group comparisons. Defaults to True.
        weight_batch_uniform (bool, optional):
            Whether to draw cells equally or based on cell counts in each batch. Defaults to False.
        mode (str, optional):
            vanilla mode is one-sided and change mode is two-sided.
            change mode uses `delta` to specify null hypothesis and computes a P-value.
            Defaults to 'vanilla'.
        signed_velocity (bool, optional):
            Whether to use original velocity values or absolute values. Defaults to True.
        test_decoupling (bool, optional):
            Whether to test decoupling and coupling dynamics. Defaults to False.
        save_raw (bool, optional):
            Whether to save the predicted data in adata. Defaults to False.
        n_samples (int, optional):
            Number of data points in each group to generate. Defaults to 5000.
        norm (Float, optional):
            Normalization factor for log2 difference. Defaults to 0.1.
        delta (Float, optional):
            Interval of change used to define null hypothesis. Defaults to 1.
        fdr (Float, optional):
            False Discovery Rate threshold. Defaults to 0.05.
        seed (int, optional):
            Seed for random generator. Defaults to 0.

    Returns:
        :class:`pandas.DataFrame`: Output DataFrame with differential dynamics on various variables.
    """
    eps = 1e-8
    if idx1 is None and idx2 is None and group_key is None and batch_key is None:
        raise ValueError("Need to specify either the group_key that contains the groups or idx1 (and idx2) directly.")
    if group1 is not None:
        if group1 in adata.obs[group_key].unique():
            idx1 = adata.obs[group_key] == group1
        elif batch_key is not None and group1 in adata.obs[batch_key].unique():
            idx1 = adata.obs[batch_key] == group1
        else:
            raise ValueError("group1 not found in cell types or batch labels, try specifying idx1 directly.")
    else:
        if idx1 is None:
            raise ValueError("Need to specify either group1 or idx1.")
    if (len(idx1) == adata.n_obs) and np.array_equal(idx1, idx1.astype(bool)):
        idx1_bin = idx1
        idx1 = np.where(idx1)[0]
    else:
        idx1_bin = np.zeros(adata.n_obs, dtype=bool)
        idx1_bin[idx1] = True
    if group2 is not None:
        if group2 in adata.obs[group_key].unique():
            idx2 = adata.obs[group_key] == group2
        elif batch_key is not None and group2 in adata.obs[batch_key].unique():
            idx2 = adata.obs[batch_key] == group2
        else:
            raise ValueError("group2 not found in cell types or batch labels, try specifying idx2 directly.")
    else:
        if idx2 is None:
            print("Using the rest of cells as reference (control).")
            idx2 = np.setdiff1d(np.array(np.arange(adata.n_obs)), np.where(idx1)[0])
            if len(idx2) == 0:
                raise ValueError("Please specify idx2 explicitly.")
    if (len(idx2) == adata.n_obs) and np.array_equal(idx2, idx2.astype(bool)):
        idx2_bin = idx2
        idx2 = np.where(idx2)[0]
    else:
        idx2_bin = np.zeros(adata.n_obs, dtype=bool)
        idx2_bin[idx2] = True
    c = adata_atac.layers['Mc']
    c = c.toarray() if issparse(c) else c
    u = adata.layers['Mu']
    u = u.toarray() if issparse(u) else u
    s = adata.layers['Ms']
    s = s.toarray() if issparse(s) else s

    rng = np.random.default_rng(seed=seed)
    if batch_key is None:
        if model.enable_cvae:
            logger.warning("Batch correction was enabled during training. It's recommended to use the same batch_key to sample pairs.")
        g1_sample_idx = rng.choice(idx1, n_samples)
        g2_sample_idx = rng.choice(idx2, n_samples)
    else:
        batch_array = adata.obs[batch_key].values
        group1_batches = np.sort(batch_array[idx1].unique())
        group2_batches = np.sort(batch_array[idx2].unique())
        if np.array_equal(group1_batches, group2_batches):
            g1_sample_idx = []
            g2_sample_idx = []
            total_cells = np.sum([len(np.where(batch_array == batch)[0]) for batch in group1_batches])
            for batch in group1_batches:
                if weight_batch_uniform:
                    n_samples_cur_batch = n_samples // len(group1_batches)
                else:
                    n_samples_cur_batch = len(np.where(batch_array == batch)[0]) * n_samples // total_cells
                idx_batch1 = np.where((batch_array == batch) & idx1_bin)[0]
                idx_batch2 = np.where((batch_array == batch) & idx2_bin)[0]
                if len(idx_batch1) < 10:
                    logger.warning(f"Group1 in batch {batch} has less than 10 cells. Skipping this batch.")
                    continue
                if len(idx_batch2) < 10:
                    logger.warning(f"Group2 in batch {batch} has less than 10 cells. Skipping this batch.")
                    continue
                g1_sample_idx.append(rng.choice(idx_batch1, n_samples_cur_batch))
                g2_sample_idx.append(rng.choice(idx_batch2, n_samples_cur_batch))
            g1_sample_idx = np.concatenate(g1_sample_idx)
            g2_sample_idx = np.concatenate(g2_sample_idx)
        else:
            print("Different batches found in group1 and group2. Sampling pairs regardless of batch conditions.")
            g1_sample_idx = rng.choice(idx1, n_samples)
            g2_sample_idx = rng.choice(idx2, n_samples)

    if c.shape != u.shape or c.shape != s.shape:
        raise ValueError("c, u, and s must have the same shape.")
    if c.ndim == 1:
        c = c.reshape(1, -1)
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    x = np.concatenate((c, u, s), 1).astype(float)
    if not model.config['split_enhancer']:
        g1_dataset = SCData(x[g1_sample_idx], device=model.device)
        g2_dataset = SCData(x[g2_sample_idx], device=model.device)
    else:
        e = adata_atac.obsm['Me']
        e = e.toarray() if issparse(e) else e
        if e.ndim == 1:
            e = e.reshape(1, -1)
        g1_dataset = SCDataE(x[g1_sample_idx], e[[g1_sample_idx]], device=model.device)
        g2_dataset = SCDataE(x[g2_sample_idx], e[[g2_sample_idx]], device=model.device)

    g1_corrected = model.test(g1_dataset, batch=(None if batch_correction else batch_array[g1_sample_idx]), sample=True, seed=seed, out_of_sample=True)
    g2_corrected = model.test(g2_dataset, batch=(None if batch_correction else batch_array[g2_sample_idx]), sample=True, seed=seed, out_of_sample=True)

    var_info = [
        ('kc', 11, log2_difference, 'kc', norm),
        ('rho', 12, log2_difference, 'rho', norm),
        ('c', 0, log2_fold_change, 'c', norm),
        ('u', 1, log2_fold_change, 'u', norm),
        ('s', 2, log2_fold_change, 's', norm),
        ('vc', 3, log2_difference if signed_velocity else log2_fold_change, 'vc', None),
        ('vu', 4, log2_difference if signed_velocity else log2_fold_change, 'vu', None),
        ('vs', 5, log2_difference if signed_velocity else log2_fold_change, 'v', None)
    ]

    var_data = {}
    for name, idx, func, _, norm_factor in var_info:
        var_data[f"{name}1"] = g1_corrected[idx]
        var_data[f"{name}2"] = g2_corrected[idx]
        if name.startswith('v'):
            if not signed_velocity:
                var_data[f"{name}1"] = np.abs(var_data[f"{name}1"])
                var_data[f"{name}2"] = np.abs(var_data[f"{name}2"])
            if norm_factor is None:
                norm_factor = var_data[f"mean_{name[-1]}2"]
        var_data[f"mean_{name}1"] = np.mean(var_data[f"{name}1"], 0)
        var_data[f"mean_{name}2"] = np.mean(var_data[f"{name}2"], 0)

        var_data[f"{'ld' if func == log2_difference else 'lfc'}_{name}"] = func(var_data[f"{name}1"], var_data[f"{name}2"],
                                                                                norm_factor if func == log2_difference else eps)

    if test_decoupling:
        var_data["decoupling1"] = g1_corrected[11] - g1_corrected[12]
        var_data["decoupling2"] = g2_corrected[11] - g2_corrected[12]
        var_data["mean_decoupling1"] = np.mean(var_data["decoupling1"], 0)
        var_data["mean_decoupling2"] = np.mean(var_data["decoupling2"], 0)
        var_data["ld_decoupling"] = log2_difference(var_data["decoupling1"], var_data["decoupling2"], norm)
        var_data["coupling1"] = g1_corrected[11] + g1_corrected[12] - 1
        var_data["coupling2"] = g2_corrected[11] + g2_corrected[12] - 1
        var_data["mean_coupling1"] = np.mean(var_data["coupling1"], 0)
        var_data["mean_coupling2"] = np.mean(var_data["coupling2"], 0)
        var_data["ld_coupling"] = log2_difference(var_data["coupling1"], var_data["coupling2"], norm)

        var_info.append(('decoupling', None, log2_difference, 'decoupling', norm))
        var_info.append(('coupling', None, log2_difference, 'coupling', norm))

    if mode not in ['vanilla', 'change']:
        logging.warning(f"Mode {mode} not recognized. Using vanilla mode.")
        mode = 'vanilla'

    dfs = {}
    for name, _, func, disp_name, _ in var_info:
        transform_type = 'diff' if func == log2_difference else 'fc'
        transform_prefix = 'ld' if func == log2_difference else 'lfc'

        if name.startswith('v'):
            if signed_velocity:
                transform_type = 'diff'
                transform_prefix = 'ld'
            else:
                transform_type = 'fc'
                transform_prefix = 'lfc'

        if mode == 'vanilla':
            p1 = np.mean(var_data[f"{name}1"] > var_data[f"{name}2"], 0)
            p2 = 1.0 - p1
            bf = np.log(p1 + eps) - np.log(p2 + eps)

            df_data = {
                f'mean_{disp_name}1': var_data[f"mean_{name}1"],
                f'mean_{disp_name}2': var_data[f"mean_{name}2"],
                f'p1_{disp_name}': p1,
                f'p2_{disp_name}': p2,
                f'bayes_factor_{disp_name}': bf
            }

            df_data[f'log2_{transform_type}_{disp_name}'] = np.mean(var_data[f"{transform_prefix}_{name}"], 0)
            df = pd.DataFrame(df_data, index=adata.var_names)

        elif mode == 'change':
            p1 = np.mean(np.abs(var_data[f"{transform_prefix}_{name}"]) >= delta, 0)
            p2 = 1.0 - p1
            bf = np.log(p1 + eps) - np.log(p2 + eps)

            df_data = {
                f'mean_{disp_name}1': var_data[f"mean_{name}1"],
                f'mean_{disp_name}2': var_data[f"mean_{name}2"],
                f'p_{disp_name}_change': p1,
                f'p_{disp_name}_no_change': p2,
                f'bayes_factor_{disp_name}': bf
            }

            df_data[f'log2_{transform_type}_{disp_name}'] = np.mean(var_data[f"{transform_prefix}_{name}"], 0)
            df = pd.DataFrame(df_data, index=adata.var_names)
            df = fdp(df, disp_name, fdr)

        dfs[disp_name] = df

    df_dd = pd.concat(list(dfs.values()), axis=1)

    if group1 is None:
        group1 = '1'
    if group2 is None:
        group2 = '2'
    if save_raw:
        differential_dynamics_dict = {}
        for name, _, _, disp_name, _ in var_info:
            differential_dynamics_dict[f'{disp_name}_{group1}'] = var_data[f"{name}1"]
            differential_dynamics_dict[f'{disp_name}_{group2}'] = var_data[f"{name}2"]

        differential_dynamics_dict[f't_{group1}'] = g1_corrected[10]
        differential_dynamics_dict[f't_{group2}'] = g2_corrected[10]

        adata.uns['differential_dynamics'] = differential_dynamics_dict

    return df_dd


def dd_func(var1_g1_gene,
            var1_g2_gene,
            var2_g1_gene,
            var2_g2_gene,
            mean_norm_gene,
            t_both,
            t_bins,
            t1_bins,
            t2_bins,
            func,
            n_bins,
            n_samples,
            seed,
            kernel,
            eps):
    rng = np.random.default_rng(seed=seed)
    if func == 'ld':
        mean_norm_gene = mean_norm_gene if mean_norm_gene is not None else 1

    time_array, dd_array, count_gen = [], [], []
    for i in range(n_bins):
        time_bin = np.mean(t_both[t_bins == i])
        var1_g1_bin = var1_g1_gene[t1_bins == i]
        var1_g2_bin = var1_g2_gene[t2_bins == i]
        if len(var1_g1_bin) < 10 or len(var1_g2_bin) < 10:
            continue
        if var2_g1_gene is not None:
            var2_g1_bin = var2_g1_gene[t1_bins == i]
            var2_g2_bin = var2_g2_gene[t2_bins == i]
            if len(var2_g1_bin) < 10 or len(var2_g2_bin) < 10:
                continue
        time_array.append(time_bin)
        count_gen.append(len(var1_g1_bin))
        var1_g1_bin_perm = rng.choice(var1_g1_bin, n_samples)
        var1_g2_bin_perm = rng.choice(var1_g2_bin, n_samples)
        if func == 'lfc':
            fc_bin = fold_change(np.abs(var1_g1_bin_perm), np.abs(var1_g2_bin_perm), eps)
            dd_array.append(np.mean(fc_bin))
        else:
            diff_bin = difference(var1_g1_bin_perm, var1_g2_bin_perm, mean_norm_gene)
            dd_array.append(np.mean(diff_bin))

        if var2_g1_gene is not None:
            var2_g1_bin_perm = rng.choice(var2_g1_bin, n_samples)
            var2_g2_bin_perm = rng.choice(var2_g2_bin, n_samples)
            if func == 'lfc':
                fc_bin = fold_change(np.abs(var2_g1_bin_perm), np.abs(var2_g2_bin_perm), eps)
                dd_array[-1] /= np.mean(fc_bin + eps)
            else:
                diff_bin = difference(var2_g1_bin_perm, var2_g2_bin_perm, mean_norm_gene)
                dd_array[-1] -= np.mean(diff_bin)

    time_array = np.array(time_array)
    dd_array = np.array(dd_array)
    count_gen = np.array(count_gen)
    bounds = np.quantile(t_both, [0.005, 0.995])
    t_both_sorted = np.sort(t_both)
    t_both_sorted = t_both_sorted[(t_both_sorted >= bounds[0]) & (t_both_sorted <= bounds[1])]

    if kernel == 'RBF':
        kernel_ = 1.0 * RBF(1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
    elif kernel == 'ExpSineSquared':
        kernel_ = 1.0 * ExpSineSquared(1.0, 1.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
    elif kernel == 'RationalQuadratic':
        kernel_ = 1.0 * RationalQuadratic(1.0, 1.0, length_scale_bounds=(0.1, 10.0), alpha_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
    else:
        raise ValueError(f"Kernel {kernel} not supported. Must be one of ['RBF', 'ExpSineSquared', 'RationalQuadratic'].")
    gaussian_process = GaussianProcessRegressor(kernel=kernel_, random_state=seed, n_restarts_optimizer=10)
    gaussian_process.fit(time_array.reshape(-1, 1), dd_array.reshape(-1, 1))
    ll = gaussian_process.log_marginal_likelihood(gaussian_process.kernel_.theta)

    const = 0.0 if func == 'ld' else 1.0
    if var2_g1_gene is None:
        kernel_constant = ConstantKernel(const, constant_value_bounds='fixed') + WhiteKernel(0.1)
    else:
        kernel_constant = ConstantKernel(const) + WhiteKernel(0.1)
    gaussian_process_ = GaussianProcessRegressor(kernel=kernel_constant, random_state=seed, n_restarts_optimizer=10)
    gaussian_process_.fit(time_array.reshape(-1, 1), dd_array.reshape(-1, 1))
    ll_null = gaussian_process_.log_marginal_likelihood(gaussian_process_.kernel_.theta)
    lrt = -2 * (ll_null - ll)
    pval = chi2.sf(lrt, 1)
    if func == 'lfc':
        mean_out = np.mean(np.log2(dd_array + eps) * count_gen * n_samples / np.sum(count_gen))
    else:
        mean_out = np.mean(np.log2(np.abs(dd_array) + 1) * np.sign(dd_array) * count_gen * n_samples / np.sum(count_gen))
    return mean_out, pval


def differential_var_decoupling(adata,
                                genes=None,
                                group1=None,
                                group2=None,
                                var1='kc',
                                var2='rho',
                                signed_velocity=True,
                                n_bins=50,
                                n_samples=100,
                                seed=0,
                                kernel='RBF',
                                n_jobs=None):
    eps = 1e-8
    if isinstance(genes, str) or isinstance(genes, int):
        genes = [genes]
    elif genes is None:
        genes = adata.var_names
    gn = len(genes)
    mean_list = np.full(gn, np.nan)
    pval_list = np.full(gn, np.nan)
    var_names_dict = {k: v for v, k in enumerate(adata.var_names)}
    gene_idx = np.array([var_names_dict[gene] for gene in genes])
    if group1 is None:
        group1 = '1'
    if group2 is None:
        group2 = '2'
    if var1 not in ['kc', 'rho', 'c', 'u', 's', 'vc', 'vu', 'v']:
        raise ValueError(f"Variable {var1} not recognized. Must be one of ['kc', 'rho', 'c', 'u', 's', 'vc', 'vu', 'v'].")
    if var1 == 'v':
        var2 = None
        print('Testing velocity. Setting var2 to None.')
    if var2 is not None and var2 not in ['kc', 'rho', 'c', 'u', 's', 'vc', 'vu', 'v']:
        raise ValueError(f"Variable {var2} not recognized. Must be one of ['kc', 'rho', 'c', 'u', 's', 'vc', 'vu', 'v'].")
    default_func = {'kc': 'ld',
                    'rho': 'ld',
                    'c': 'lfc',
                    'u': 'lfc',
                    's': 'lfc',
                    'vc': 'ld' if signed_velocity else 'lfc',
                    'vu': 'ld' if signed_velocity else 'lfc',
                    'v': 'ld' if signed_velocity else 'lfc'}
    if var2 is not None and default_func[var1] != default_func[var2]:
        raise ValueError(f"{var1} and {var2} have different differential functions. Please select variables with simialr distributions, such as (kc, rho) or (c, u, s).")
    func = default_func[var1]
    if f'{var1}_{group1}' not in adata.uns['differential_dynamics'].keys():
        raise ValueError(f"{var1}_{group1} not found in adata.varm. Was differential_dynamics run with save_raw?")
    if f'{var1}_{group2}' not in adata.uns['differential_dynamics'].keys():
        raise ValueError(f"{var1}_{group2} not found in adata.varm. Was differential_dynamics run with save_raw?")
    var1_g1 = adata.uns['differential_dynamics'][f'{var1}_{group1}'][:, gene_idx]
    var1_g2 = adata.uns['differential_dynamics'][f'{var1}_{group2}'][:, gene_idx]
    if var2 is not None:
        if f'{var2}_{group1}' not in adata.uns['differential_dynamics'].keys():
            raise ValueError(f"{var2}_{group1} not found in adata.varm. Was differential_dynamics run with save_raw?")
        if f'{var2}_{group2}' not in adata.uns['differential_dynamics'].keys():
            raise ValueError(f"{var2}_{group2} not found in adata.varm. Was differential_dynamics run with save_raw?")
        var2_g1 = adata.uns['differential_dynamics'][f'{var2}_{group1}'][:, gene_idx]
        var2_g2 = adata.uns['differential_dynamics'][f'{var2}_{group2}'][:, gene_idx]
    t1 = adata.uns['differential_dynamics'][f't_{group1}']
    t2 = adata.uns['differential_dynamics'][f't_{group2}']
    t_both = np.concatenate([t1, t2])
    steps = np.quantile(t_both, np.linspace(0, 1, n_bins + 1))
    steps = steps[1:-1]
    t_bins = np.digitize(t_both, steps)
    t1_bins = np.digitize(t1, steps)
    t2_bins = np.digitize(t2, steps)
    if func == 'ld' and var1 == 'v':
        mean_norm = np.mean(adata.uns['differential_dynamics'][f's_{group2}'], 0)
    else:
        mean_norm = None

    if (n_jobs is None or not isinstance(n_jobs, int) or n_jobs < 0 or
            n_jobs > os.cpu_count()):
        n_jobs = os.cpu_count()
    if n_jobs > gn:
        n_jobs = gn
    batches = -(-gn // n_jobs)

    pbar = tqdm(total=gn)
    for group in range(batches):
        idx = range(group * n_jobs, np.min([gn, (group+1) * n_jobs]))
        res = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(dd_func)(
                var1_g1[:, i],
                var1_g2[:, i],
                var2_g1[:, i] if var2 is not None else None,
                var2_g2[:, i] if var2 is not None else None,
                mean_norm[i] if mean_norm is not None else None,
                t_both,
                t_bins,
                t1_bins,
                t2_bins,
                func,
                n_bins,
                n_samples,
                seed,
                kernel,
                eps)
            for i in idx)
        for i, r in zip(idx, res):
            mean_list[i], pval_list[i] = r[0], r[1]
        pbar.update(len(idx))
    return mean_list, pval_list
