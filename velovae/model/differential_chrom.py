import logging
import numpy as np
from scipy.sparse import issparse
import pandas as pd
logger = logging.getLogger(__name__)


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
            raise ValueError("group1 not found in cell types or batch labels, try specifying idx1 directly")
    else:
        if idx1 is None:
            raise ValueError("Need to specify either group1 or idx1")
    if (len(idx1) == adata.n_obs) and np.array_equal(idx1, idx1.astype(bool)):
        idx1 = np.where(idx1)[0]
    if group2 is not None:
        if group2 in adata.obs[group_key].unique():
            idx2 = adata.obs[group_key] == group2
        elif batch_key is not None and group2 in adata.obs[batch_key].unique():
            idx2 = adata.obs[batch_key] == group2
        else:
            raise ValueError("group2 not found in cell types or batch labels, try specifying idx2 directly")
    else:
        if idx2 is None:
            print("Using the rest of cells as reference (control).")
            idx2 = np.setdiff1d(np.arange(adata.n_obs), np.where(idx1)[0])
    if (len(idx2) == adata.n_obs) and np.array_equal(idx2, idx2.astype(bool)):
        idx2 = np.where(idx2)[0]
    c = adata_atac.layers['Mc']
    c = c.A if issparse(c) else c
    u = adata.layers['Mu']
    u = u.A if issparse(u) else u
    s = adata.layers['Ms']
    s = s.A if issparse(s) else s
    g1_corrected = model.test(c[idx1], u[idx1], s[idx1], batch=(None if batch_correction else adata.obs[batch_key][idx1]), sample=True, seed=seed)
    g2_corrected = model.test(c[idx2], u[idx2], s[idx2], batch=(None if batch_correction else adata.obs[batch_key][idx2]), sample=True, seed=seed)
    rng = np.random.default_rng(seed=seed)
    if batch_key is None:
        if model.enable_cvae:
            logger.warn("Batch correction was enabled during training. It's recommended to use the same batch_key to sample pairs.")
        g1_sample_idx = rng.choice(np.arange(len(idx1)), n_samples)
        g2_sample_idx = rng.choice(np.arange(len(idx2)), n_samples)
    else:
        group1_batches = np.sort(adata.obs[batch_key][idx1].unique())
        group2_batches = np.sort(adata.obs[batch_key][idx2].unique())
        if np.array_equal(group1_batches, group2_batches):
            g1_sample_idx = []
            g2_sample_idx = []
            total_cells = np.sum([len(np.where(adata.obs[batch_key] == batch)[0]) for batch in group1_batches])
            for batch in group1_batches:
                if weight_batch_uniform:
                    n_samples_per_batch = n_samples // len(group1_batches)
                else:
                    n_samples_per_batch = len(np.where(adata.obs[batch_key] == batch)[0]) * n_samples // total_cells
                idx_batch1 = np.where(adata.obs[batch_key][idx1] == batch)[0]
                if len(idx_batch1) < 10:
                    logger.warn(f"Group1 in batch {batch} has less than 10 cells. Skipping this batch.")
                    continue
                idx_batch2 = np.where(adata.obs[batch_key][idx2] == batch)[0]
                if len(idx_batch2) < 10:
                    logger.warn(f"Group2 in batch {batch} has less than 10 cells. Skipping this batch.")
                    continue
                g1_sample_idx.append(rng.choice(idx_batch1, n_samples_per_batch))
                g2_sample_idx.append(rng.choice(idx_batch2, n_samples_per_batch))
            g1_sample_idx = np.concatenate(g1_sample_idx)
            g2_sample_idx = np.concatenate(g2_sample_idx)
        else:
            print("Different batches found in group1 and group2. Sampling pairs regardless of batch conditions.")
            g1_sample_idx = rng.choice(np.arange(len(idx1)), n_samples)
            g2_sample_idx = rng.choice(np.arange(len(idx2)), n_samples)

    kc1 = g1_corrected[11][g1_sample_idx]
    kc2 = g2_corrected[11][g2_sample_idx]
    mean_kc1 = np.mean(kc1, 0)
    mean_kc2 = np.mean(kc2, 0)
    lfc_kc = np.log2(kc1 + eps) - np.log2(kc2 + eps)

    rho1 = g1_corrected[12][g1_sample_idx]
    rho2 = g2_corrected[12][g2_sample_idx]
    mean_rho1 = np.mean(rho1, 0)
    mean_rho2 = np.mean(rho2, 0)
    lfc_rho = np.log2(rho1 + eps) - np.log2(rho2 + eps)

    s1 = g1_corrected[2][g1_sample_idx]
    s2 = g2_corrected[2][g2_sample_idx]
    mean_s1 = np.mean(s1, 0)
    mean_s2 = np.mean(s2, 0)
    lfc_s = np.log2(s1 + eps) - np.log2(s2 + eps)

    vs1 = g1_corrected[5][g1_sample_idx]
    vs2 = g2_corrected[5][g2_sample_idx]
    if signed_velocity:
        mean_vs1 = np.mean(vs1, 0)
        mean_vs2 = np.mean(vs2, 0)
        ld_vs = np.log2(np.abs(vs1 - vs2) / mean_s2 + eps) * np.sign(vs1 - vs2)
    else:
        vs1 = np.abs(vs1)
        vs2 = np.abs(vs2)
        mean_vs1 = np.mean(vs1, 0)
        mean_vs2 = np.mean(vs2, 0)
        lfc_vs = np.log2(vs1 + eps) - np.log2(vs2 + eps)

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

    return df_kc, df_rho, df_s, df_vs
