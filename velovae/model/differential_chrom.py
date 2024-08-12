import logging
import numpy as np
from scipy.sparse import issparse
import pandas as pd
logger = logging.getLogger(__name__)


def log2_difference(v1, v2, norm=0.1):
    return np.log2(np.abs(v1 - v2) / norm + 1) * np.sign(v1 - v2)


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
                          save_raw=False,
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
    ld_kc = log2_difference(kc1, kc2)

    rho1 = g1_corrected[12]
    rho2 = g2_corrected[12]
    mean_rho1 = np.mean(rho1, 0)
    mean_rho2 = np.mean(rho2, 0)
    ld_rho = log2_difference(rho1, rho2)

    c1 = g1_corrected[0]
    c2 = g2_corrected[0]
    mean_c1 = np.mean(c1, 0)
    mean_c2 = np.mean(c2, 0)
    ld_c = log2_difference(c1, c2, mean_c2)

    u1 = g1_corrected[1]
    u2 = g2_corrected[1]
    mean_u1 = np.mean(u1, 0)
    mean_u2 = np.mean(u2, 0)
    lfc_u = log2_fold_change(u1, u2)

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
                              'log2_diff_kc': np.mean(ld_kc, 0)},
                             index=adata.var_names)

        p1_rho = np.mean(rho1 > rho2, 0)
        p2_rho = 1.0 - p1_rho
        bf_rho = np.log(p1_rho + eps) - np.log(p2_rho + eps)
        df_rho = pd.DataFrame({'mean_rho1': mean_rho1,
                               'mean_rho2': mean_rho2,
                               'p1_rho': p1_rho,
                               'p2_rho': p2_rho,
                               'bayes_factor_rho': bf_rho,
                               'log2_diff_rho': np.mean(ld_rho, 0)},
                              index=adata.var_names)

        p1_c = np.mean(c1 > c2, 0)
        p2_c = 1.0 - p1_c
        bf_c = np.log(p1_c + eps) - np.log(p2_c + eps)
        df_c = pd.DataFrame({'mean_c1': mean_c1,
                             'mean_c2': mean_c2,
                             'p1_c': p1_c,
                             'p2_c': p2_c,
                             'bayes_factor_c': bf_c,
                             'log2_diff_c': np.mean(ld_c, 0)},
                             index=adata.var_names)

        p1_u = np.mean(u1 > u2, 0)
        p2_u = 1.0 - p1_u
        bf_u = np.log(p1_u + eps) - np.log(p2_u + eps)
        df_u = pd.DataFrame({'mean_u1': mean_u1,
                             'mean_u2': mean_u2,
                             'p1_u': p1_u,
                             'p2_u': p2_u,
                             'bayes_factor_u': bf_u,
                             'log2_fc_u': np.mean(lfc_u, 0)},
                            index=adata.var_names)

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

    elif mode == 'change':
        p1_kc = np.mean(np.abs(ld_kc) >= delta, 0)
        p2_kc = 1.0 - p1_kc
        bf_kc = np.log(p1_kc + eps) - np.log(p2_kc + eps)
        df_kc = pd.DataFrame({'mean_kc1': mean_kc1,
                              'mean_kc2': mean_kc2,
                              'p_kc_change': p1_kc,
                              'p_kc_no_change': p2_kc,
                              'bayes_factor_kc': bf_kc,
                              'log2_diff_kc': np.mean(ld_kc, 0)},
                             index=adata.var_names)

        p1_rho = np.mean(np.abs(ld_rho) >= delta, 0)
        p2_rho = 1.0 - p1_rho
        bf_rho = np.log(p1_rho + eps) - np.log(p2_rho + eps)
        df_rho = pd.DataFrame({'mean_rho1': mean_rho1,
                               'mean_rho2': mean_rho2,
                               'p_rho_change': p1_rho,
                               'p_rho_no_change': p2_rho,
                               'bayes_factor_rho': bf_rho,
                               'log2_diff_rho': np.mean(ld_rho, 0)},
                              index=adata.var_names)

        p1_c = np.mean(np.abs(ld_c) >= delta, 0)
        p2_c = 1.0 - p1_c
        bf_c = np.log(p1_c + eps) - np.log(p2_c + eps)
        df_c = pd.DataFrame({'mean_c1': mean_c1,
                             'mean_c2': mean_c2,
                             'p_c_change': p1_c,
                             'p_c_no_change': p2_c,
                             'bayes_factor_c': bf_c,
                             'log2_diff_c': np.mean(ld_c, 0)},
                            index=adata.var_names)

        p1_u = np.mean(np.abs(lfc_u) >= delta, 0)
        p2_u = 1.0 - p1_u
        bf_u = np.log(p1_u + eps) - np.log(p2_u + eps)
        df_u = pd.DataFrame({'mean_u1': mean_u1,
                             'mean_u2': mean_u2,
                             'p_u_change': p1_u,
                             'p_u_no_change': p2_u,
                             'bayes_factor_u': bf_u,
                             'log2_fc_u': np.mean(lfc_u, 0)},
                            index=adata.var_names)

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

    df_dd = pd.concat([df_kc, df_rho, df_c, df_u, df_s, df_vs], axis=1)

    if group1 is None:
        group1 = '1'
    if group2 is None:
        group2 = '2'
    if save_raw:
        adata.uns['differential_dynamics'] = {f'kc_{group1}': kc1,
                                              f'kc_{group2}': kc2,
                                              f'rho_{group1}': rho1,
                                              f'rho_{group2}': rho2,
                                              f'c_{group1}': c1,
                                              f'c_{group2}': c2,
                                              f'u_{group1}': u1,
                                              f'u_{group2}': u2,
                                              f's_{group1}': s1,
                                              f's_{group2}': s2,
                                              f'v_{group1}': vs1,
                                              f'v_{group2}': vs2,
                                              f't_{group1}': g1_corrected[10],
                                              f't_{group2}': g2_corrected[10]
                                             }
    return df_dd
