import logging
import numpy as np
from scipy.sparse import issparse
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, ExpSineSquared, WhiteKernel
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def log2_difference(v1, v2, s2, eps=1e-8):
    return np.log2(np.abs(v1 - v2) / s2 + eps) * np.sign(v1 - v2)


def log2_fold_change(v1, v2, eps=1e-8):
    return np.log2(v1 + eps) - np.log2(v2 + eps)


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
                          signed_velocity=False,
                          output_raw=False,
                          n_samples=5000,
                          delta=1,
                          seed=0):
    # inspired by [Lopez2018] and [Boyeau2019]
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
            idx2 = np.setdiff1d(np.arange(adata.n_obs), np.where(idx1)[0])
    if (len(idx2) == adata.n_obs) and np.array_equal(idx2, idx2.astype(bool)):
        idx2_bin = idx2
        idx2 = np.where(idx2)[0]
    else:
        idx2_bin = np.zeros(adata.n_obs, dtype=bool)
        idx2_bin[idx2] = True
    c = adata_atac.layers['Mc']
    c = c.A if issparse(c) else c
    u = adata.layers['Mu']
    u = u.A if issparse(u) else u
    s = adata.layers['Ms']
    s = s.A if issparse(s) else s

    rng = np.random.default_rng(seed=seed)
    if batch_key is None:
        if model.enable_cvae:
            logger.warn("Batch correction was enabled during training. It's recommended to use the same batch_key to sample pairs.")
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
                    logger.warn(f"Group1 in batch {batch} has less than 10 cells. Skipping this batch.")
                    continue
                if len(idx_batch2) < 10:
                    logger.warn(f"Group2 in batch {batch} has less than 10 cells. Skipping this batch.")
                    continue
                g1_sample_idx.append(rng.choice(idx_batch1, n_samples_cur_batch))
                g2_sample_idx.append(rng.choice(idx_batch2, n_samples_cur_batch))
            g1_sample_idx = np.concatenate(g1_sample_idx)
            g2_sample_idx = np.concatenate(g2_sample_idx)
        else:
            print("Different batches found in group1 and group2. Sampling pairs regardless of batch conditions.")
            g1_sample_idx = rng.choice(idx1, n_samples)
            g2_sample_idx = rng.choice(idx2, n_samples)

    g1_corrected = model.test(c[g1_sample_idx], u[g1_sample_idx], s[g1_sample_idx], batch=(None if batch_correction else batch_array[g1_sample_idx]), sample=True, seed=seed)
    g2_corrected = model.test(c[g2_sample_idx], u[g2_sample_idx], s[g2_sample_idx], batch=(None if batch_correction else batch_array[g2_sample_idx]), sample=True, seed=seed)

    kc1 = g1_corrected[11]
    kc2 = g2_corrected[11]
    mean_kc1 = np.mean(kc1, 0)
    mean_kc2 = np.mean(kc2, 0)
    lfc_kc = log2_fold_change(kc1, kc2)

    rho1 = g1_corrected[12]
    rho2 = g2_corrected[12]
    mean_rho1 = np.mean(rho1, 0)
    mean_rho2 = np.mean(rho2, 0)
    lfc_rho = log2_fold_change(rho1, rho2)

    s1 = g1_corrected[2]
    s2 = g2_corrected[2]
    mean_s1 = np.mean(s1, 0)
    mean_s2 = np.mean(s2, 0)
    lfc_s = log2_fold_change(s1, s2)

    vs1 = g1_corrected[5]
    vs2 = g2_corrected[5]
    if signed_velocity:
        mean_vs1 = np.mean(vs1, 0)
        mean_vs2 = np.mean(vs2, 0)
        ld_vs = log2_difference(vs1, vs2, mean_s2)
    else:
        vs1 = np.abs(vs1)
        vs2 = np.abs(vs2)
        mean_vs1 = np.mean(vs1, 0)
        mean_vs2 = np.mean(vs2, 0)
        lfc_vs = log2_fold_change(vs1, vs2)

    if mode not in ['vanilla', 'change']:
        logging.warn(f"Mode {mode} not recognized. Using vanilla mode.")
        mode = 'vanilla'
    if mode == 'vanilla':
        p1_kc = np.mean(kc1 > kc2, 0)
        p2_kc = 1.0 - p1_kc
        bf_kc = np.log(p1_kc + eps) - np.log(p2_kc + eps)
        df_kc = pd.DataFrame({'mean_kc1': mean_kc1,
                              'mean_kc2': mean_kc2,
                              'p1_kc': p1_kc,
                              'p2_kc': p2_kc,
                              'bayes_factor_kc': bf_kc,
                              'log2_fc_kc': np.mean(lfc_kc, 0)},
                             index=adata.var_names)
        df_kc = df_kc.sort_values('bayes_factor_kc', ascending=False)

        p1_rho = np.mean(rho1 > rho2, 0)
        p2_rho = 1.0 - p1_rho
        bf_rho = np.log(p1_rho + eps) - np.log(p2_rho + eps)
        df_rho = pd.DataFrame({'mean_rho1': mean_rho1,
                               'mean_rho2': mean_rho2,
                               'p1_rho': p1_rho,
                               'p2_rho': p2_rho,
                               'bayes_factor_rho': bf_rho,
                               'log2_fc_rho': np.mean(lfc_rho, 0)},
                              index=adata.var_names)
        df_rho = df_rho.sort_values('bayes_factor_rho', ascending=False)

        p1_s = np.mean(s1 > s2, 0)
        p2_s = 1.0 - p1_s
        bf_s = np.log(p1_s + eps) - np.log(p2_s + eps)
        df_s = pd.DataFrame({'mean_s1': mean_s1,
                             'mean_s2': mean_s2,
                             'p1_s': p1_s,
                             'p2_s': p2_s,
                             'bayes_factor_s': bf_s,
                             'log2_fc_s': np.mean(lfc_s, 0)},
                            index=adata.var_names)
        df_s = df_s.sort_values('bayes_factor_s', ascending=False)

        p1_vs = np.mean(vs1 > vs2, 0)
        p2_vs = 1.0 - p1_vs
        bf_vs = np.log(p1_vs + eps) - np.log(p2_vs + eps)
        df_vs = pd.DataFrame({'mean_v1': mean_vs1,
                              'mean_v2': mean_vs2,
                              'p1_v': p1_vs,
                              'p2_v': p2_vs,
                              'bayes_factor_v': bf_vs},
                             index=adata.var_names)
        if signed_velocity:
            df_vs['log2_diff_v'] = np.mean(ld_vs, 0)
        else:
            df_vs['log2_fc_v'] = np.mean(lfc_vs, 0)
        df_vs = df_vs.sort_values('bayes_factor_v', ascending=False)

    elif mode == 'change':
        p1_kc = np.mean(np.abs(lfc_kc) >= delta, 0)
        p2_kc = 1.0 - p1_kc
        bf_kc = np.log(p1_kc + eps) - np.log(p2_kc + eps)
        df_kc = pd.DataFrame({'mean_kc1': mean_kc1,
                              'mean_kc2': mean_kc2,
                              'p_kc_change': p1_kc,
                              'p_kc_no_change': p2_kc,
                              'bayes_factor_kc': bf_kc,
                              'log2_fc_kc': np.mean(lfc_kc, 0)},
                             index=adata.var_names)
        df_kc = df_kc.sort_values('bayes_factor_kc', ascending=False)

        p1_rho = np.mean(np.abs(lfc_rho) >= delta, 0)
        p2_rho = 1.0 - p1_rho
        bf_rho = np.log(p1_rho + eps) - np.log(p2_rho + eps)
        df_rho = pd.DataFrame({'mean_rho1': mean_rho1,
                               'mean_rho2': mean_rho2,
                               'p_rho_change': p1_rho,
                               'p_rho_no_change': p2_rho,
                               'bayes_factor_rho': bf_rho,
                               'log2_fc_rho': np.mean(lfc_rho, 0)},
                              index=adata.var_names)
        df_rho = df_rho.sort_values('bayes_factor_rho', ascending=False)

        p1_s = np.mean(np.abs(lfc_s) >= delta, 0)
        p2_s = 1.0 - p1_s
        bf_s = np.log(p1_s + eps) - np.log(p2_s + eps)
        df_s = pd.DataFrame({'mean_s1': mean_s1,
                             'mean_s2': mean_s2,
                             'p_s_change': p1_s,
                             'p_s_no_change': p2_s,
                             'bayes_factor_s': bf_s,
                             'log2_fc_s': np.mean(lfc_s, 0)},
                            index=adata.var_names)
        df_s = df_s.sort_values('bayes_factor_s', ascending=False)

        p1_vs = np.mean(np.abs(ld_vs if signed_velocity else lfc_vs) >= delta, 0)
        p2_vs = 1.0 - p1_vs
        bf_vs = np.log(p1_vs + eps) - np.log(p2_vs + eps)
        df_vs = pd.DataFrame({'mean_v1': mean_vs1,
                              'mean_v2': mean_vs2,
                              'p_v_change': p1_vs,
                              'p_v_no_change': p2_vs,
                              'bayes_factor_v': bf_vs},
                             index=adata.var_names)
        if signed_velocity:
            df_vs['log2_diff_v'] = np.mean(ld_vs, 0)
        else:
            df_vs['log2_fc_v'] = np.mean(lfc_vs, 0)
        df_vs = df_vs.sort_values('bayes_factor_v', ascending=False)

    if output_raw:
        return df_kc, df_rho, df_s, df_vs, (kc1, kc2), (rho1, rho2), (s1, s2), (vs1, vs2), (g1_corrected[10], g2_corrected[10])
    else:
        return df_kc, df_rho, df_s, df_vs


def differential_dynamics_bin(adata,
                              var1,
                              var2,
                              gene,
                              t1,
                              t2,
                              s2=None,
                              func='lfc',
                              n_bins=50,
                              n_samples=100,
                              seed=0):
    if not isinstance(gene, str):
        raise ValueError("Please input only a single gene.")
    t_both = np.concatenate([t1, t2])
    steps = np.quantile(t_both, np.linspace(0, 1, n_bins + 1))
    steps[0] = steps[0] - 1
    steps[-1] = steps[-1] + 1
    t_bins = np.digitize(t_both, steps)
    t1_bins = np.digitize(t1, steps)
    t2_bins = np.digitize(t2, steps)
    gene_idx = adata.var_names == gene
    var1_gene = var1[:, gene_idx]
    var2_gene = var2[:, gene_idx]
    if func == 'ld':
        if s2 is None:
            raise ValueError("Need to provide spliced counts for computing velocity log difference.")
        mean_s2_gene = np.mean(s2[:, gene_idx])

    rng = np.random.default_rng(seed=seed)
    time_array, dd_array = [], []
    for i in range(n_bins):
        var1_bin = var1_gene[t1_bins == i]
        var2_bin = var2_gene[t2_bins == i]
        if len(var1_bin) < 10 or len(var2_bin) < 10:
            continue
        var1_bin_perm = rng.choice(var1_bin, n_samples)
        var2_bin_perm = rng.choice(var2_bin, n_samples)
        time_bin = np.mean(t_both[t_bins == i])
        time_array.append(time_bin)
        if func == 'lfc':
            lfc_bin = log2_fold_change(np.abs(var1_bin_perm), np.abs(var2_bin_perm))
            dd_array.append(np.mean(lfc_bin))
        elif func == 'ld':
            diff_bin = log2_difference(var1_bin_perm, var2_bin_perm, mean_s2_gene)
            dd_array.append(np.mean(diff_bin))
        else:
            raise ValueError(f"Mode {func} not recognized. Must be either 'lfc' or 'ld'.")
    time_array = np.array(time_array)
    dd_array = np.array(dd_array)
    bounds = np.quantile(t_both, [0.005, 0.995])
    t_both_sorted = np.sort(t_both)
    t_both_sorted = t_both_sorted[(t_both_sorted >= bounds[0]) & (t_both_sorted <= bounds[1])]

    kernel = 1.0 * ExpSineSquared(1.0, 1.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, random_state=seed, n_restarts_optimizer=10)
    gaussian_process.fit(time_array.reshape(-1, 1), dd_array.reshape(-1, 1))
    print(gaussian_process.kernel_)
    mean_prediction, std_prediction = gaussian_process.predict(t_both_sorted.reshape(-1, 1), return_std=True)

    plt.scatter(time_array, dd_array, label=f"Binned {'Log difference' if func == 'ld' else 'Log fold change'}")
    plt.plot(t_both_sorted, mean_prediction, label="Mean prediction")
    plt.plot(t_both_sorted, np.full_like(t_both_sorted, 0.0), label="Zero line", linestyle='--', color='black', alpha=0.5)
    plt.fill_between(
        t_both_sorted,
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"Credible interval",
    )
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel(f"{'Log difference' if func == 'ld' else 'Log fold change'}")
    plt.title("Gaussian process regression on differential dynamics")
