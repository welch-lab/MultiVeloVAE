import os
import time
import copy
import logging
import anndata as ad
from scipy.sparse import issparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..plotting_chrom import plot_sig_, plot_sig, plot_vel, plot_phase, plot_time
from ..plotting_chrom import plot_train_loss_log, plot_test_loss_log
from .model_util_chrom import hist_equal, encode_type, get_gene_index, convert_time
from .model_util_chrom import elbo_collapsed_categorical, find_dirichlet_param, assign_gene_mode_tprior, assign_gene_mode
from .model_util_chrom import pred_exp, pred_exp_numpy, pred_exp_numpy_backward, init_params, get_ts_global, reinit_params
from .model_util_chrom import kl_gaussian, reparameterize, softplusinv, knnx0_index, get_x0
from .model_util_chrom import pearson, loss_vel, cosine_similarity
from .training_data_chrom import SCData, SCDataE
from .velocity_chrom import velocity
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    Neural network encoder for the MultiVeloVAE model.

    This class encodes data (RNA counts and optionally chromatin accessibility) into
    latent representations. It maps input from the observation space to a distribution
    over latent variables z (cell state) and t (latent time).

    Attributes:
        t_network: A neural network to model pseudotime as a vector.
        split_enhancer: Whether to separately encode enhancer data (not tested).
        fc1, bn1, dpt1: First fully connected layer, batch norm, and dropout for main network.
        net: Sequential network combining first layer components.
        fc_mu_t, fc_std_t: Linear layers for latent time mean and standard deviation.
        fc_mu_z, fc_std_z: Linear layers for latent cell state mean and standard deviation.
        spt, sp1, sp2: SoftPlus activation functions for non-negative outputs.
        net_e, fc_mu_e, fc_std_e, sp3: Components for enhancer encoder (optional).

    Args:
        Cin (int):
            Input feature dimension (typically 3*number_of_genes for combined c/u/s data).
        dim_z (int):
            Latent variable dimensionality.
        dim_cond (int, optional):
            Dimension of condition vector for conditional VAE. Defaults to 0.
        dim_reg (int, optional):
            Dimension of regression covariates to regress out. Defaults to 0.
        hidden_size (int, optional):
            Size of hidden layer. Defaults to 256.
        t_network (bool, optional):
            If True, model pseudotime as a vector. Defaults to False.
        split_enhancer (bool, optional):
            If True, separately encode enhancer data. Defaults to False.
        Cin_e (int, optional):
            Enhancer input dimension. Required if split_enhancer is True. Defaults to None.
        checkpoint (str, optional):
            Path to checkpoint file to load weights. Defaults to None.
    """
    def __init__(self,
                 Cin,
                 dim_z,
                 dim_cond=0,
                 dim_reg=0,
                 hidden_size=256,
                 t_network=False,
                 split_enhancer=False,
                 Cin_e=None,
                 checkpoint=None):
        super(Encoder, self).__init__()
        self.t_network = t_network
        self.split_enhancer = split_enhancer

        self.fc1 = nn.Linear(Cin+dim_cond+dim_reg, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dpt1 = nn.Dropout(p=0.2)
        self.net = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1)

        self.fc_mu_t = nn.Linear(hidden_size, dim_z if t_network else 1)
        self.fc_std_t = nn.Linear(hidden_size, dim_z if t_network else 1)
        self.spt = nn.Softplus()
        self.sp1 = nn.Softplus()

        self.fc_mu_z = nn.Linear(hidden_size, dim_z)
        self.fc_std_z = nn.Linear(hidden_size, dim_z)
        self.sp2 = nn.Softplus()

        if split_enhancer:
            self.fc2 = nn.Linear(Cin_e, hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.dpt2 = nn.Dropout(p=0.2)
            self.net_e = nn.Sequential(self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)

            self.fc_mu_e = nn.Linear(hidden_size, dim_z)
            self.fc_std_e = nn.Linear(hidden_size, dim_z)
            self.sp3 = nn.Softplus()

        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint))
        else:
            self.init_weights()

    def init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in [self.fc_mu_t, self.fc_std_t, self.fc_mu_z, self.fc_std_z]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

        if self.split_enhancer:
            for m in self.net_e.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for m in [self.fc_mu_e, self.fc_std_e]:
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, data_in, data_in_e=None, condition=None, regressor=None):
        if condition is not None:
            data_in = torch.cat((data_in, condition), 1)
        if regressor is not None:
            data_in = torch.cat((data_in, regressor), 1)
        l1 = self.net(data_in)
        mu_tx = self.fc_mu_t(l1) if self.t_network else self.spt(self.fc_mu_t(l1))
        std_tx = self.sp1(self.fc_std_t(l1))
        mu_zx = self.fc_mu_z(l1)
        std_zx = self.sp2(self.fc_std_z(l1))
        if self.split_enhancer:
            l2 = self.net_e(data_in_e)
            mu_ex = self.fc_mu_e(l2)
            std_ex = self.sp3(self.fc_std_e(l2))
        else:
            mu_ex, std_ex = None, None
        return mu_tx, std_tx, mu_zx, std_zx, mu_ex, std_ex


class Decoder(nn.Module):
    """
    Neural network decoder for the MultiVeloVAE model.

    This decoder generates predictions of gene expression and chromatin accessibility
    based on latent cell state z and pseudotime t, using a mechanistic ODE model
    with neural networks to predict the regulatory parameters.

    The decoder models gene expression dynamics using the following system of ODEs:
    dc/dt = α_c × (k_c - c)
    du/dt = α × ρ × c - β × u
    ds/dt = β × u - γ × s

    where:
    - c = chromatin accessibility
    - u = unspliced RNA
    - s = spliced RNA
    - k_c, ρ are cell-gene specific chromatin and transcription states
    - α_c, α, β, γ are gene-specific and sample-specific velocity parameters

    Attributes:
        adata: AnnData object containing RNA data
        adata_atac: AnnData object containing chromatin accessibility data
        Other attributes for model configuration and parameters

    Args:
        adata (AnnData):
            AnnData object containing RNA data.
        adata_atac (AnnData):
            AnnData object containing chromatin accessibility data.
        train_idx (array-like):
            Indices of training data.
        dim_z (int):
            Latent variable dimensionality.
        dim_cond (int, optional):
            Dimension of condition vector for conditional VAE. Defaults to 0.
        dim_reg (int, optional):
            Dimension of regression covariates. Defaults to 0.
        batch_idx (array-like, optional):
            Batch indices for training data. Defaults to None.
        ref_batch (int, optional):
            Reference batch index. Defaults to None.
        hidden_size (int, optional):
            Size of hidden layers. Defaults to 256.
        split_enhancer (bool, optional):
            Whether to separately decode enhancer data. Defaults to False.
        parallel_arch (bool, optional):
            Whether to use parallel architecture for regulatory networks. Defaults to True.
        t_network (bool, optional):
            Whether pseudotime is encoded as a vector. Defaults to True.
        full_vb (bool, optional):
            Whether to use full variational Bayes for ODE parameters. Defaults to False.
        global_std (bool, optional):
            Whether to use global std for noise model. Defaults to True.
        log_params (bool, optional):
            Whether parameters are in log space. Defaults to False.
        rna_only (bool, optional):
            Whether to run in RNA-only mode. Defaults to False.
        rna_only_idx (list, optional):
            Indices of RNA-only batches. Defaults to [].
        perc (int, optional):
            Percentile for parameter initialization. Defaults to 98.
        tmax (float, optional):
            Maximum pseudotime. Defaults to 1.
        reinit (bool, optional):
            Whether to reinitialize parameters. Defaults to False.
        train_x0 (bool, optional):
            Whether to train initial conditions. Defaults to False.
        init_ton_zero (bool, optional):
            Whether to initialize activation time to zero. Defaults to True.
        init_method (str, optional):
            Method for initialization ('steady' or 'tprior'). Defaults to 'steady'.
        init_key (str, optional):
            Key in adata.obs for initialization. Defaults to None.
        checkpoint (str, optional):
            Path to checkpoint file to load weights. Defaults to None.
        vram_constrained (bool, optional):
            Whether to enable VRAM-constrained mode. Defaults to False.
    """
    def __init__(self,
                 adata,
                 adata_atac,
                 train_idx,
                 dim_z,
                 dim_cond=0,
                 dim_reg=0,
                 batch_idx=None,
                 ref_batch=None,
                 hidden_size=256,
                 split_enhancer=False,
                 parallel_arch=True,
                 t_network=True,
                 full_vb=False,
                 global_std=True,
                 log_params=False,
                 rna_only=False,
                 rna_only_idx=[],
                 perc=98,
                 tmax=1,
                 reinit=False,
                 train_x0=False,
                 init_ton_zero=True,
                 init_method='steady',
                 init_key=None,
                 checkpoint=None):
        super(Decoder, self).__init__()
        self.adata = adata
        self.adata_atac = adata_atac
        self.train_idx = train_idx
        self.dim_cond = dim_cond
        self.dim_reg = dim_reg
        if dim_cond == 1:
            dim_cond = 0
        self.cvae = True if dim_cond > 1 else False
        self.batch = batch_idx
        self.ref_batch = ref_batch
        self.split_enhancer = split_enhancer
        self.parallel_arch = parallel_arch
        self.t_network = t_network
        self.is_full_vb = full_vb
        self.global_std = global_std
        self.log_params = log_params
        self.rna_only = rna_only
        self.rna_only_idx = np.array(rna_only_idx)
        self.reinit = reinit
        self.train_x0 = train_x0
        self.init_ton_zero = init_ton_zero
        self.init_method = init_method
        self.init_key = init_key
        self.checkpoint = checkpoint
        self.tmax = tmax
        self.construct_nn(dim_z, dim_cond, dim_reg, hidden_size, perc)

    def construct_nn(self, dim_z, dim_cond, dim_reg, hidden_size, perc):
        G = self.adata.n_vars
        self.set_shape(G, dim_cond)

        if not self.split_enhancer:
            self.fc1 = nn.Linear(dim_z+dim_cond+dim_reg, hidden_size)
        else:
            self.fc1 = nn.Bilinear(dim_z, dim_z, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dpt1 = nn.Dropout(p=0.2)
        self.fc_out1 = nn.Linear(hidden_size, G)
        self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                     self.fc_out1, nn.Sigmoid())

        if not self.split_enhancer:
            self.fc2 = nn.Linear(dim_z+dim_cond+dim_reg, hidden_size)
        else:
            self.fc2 = nn.Bilinear(dim_z, dim_z, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dpt2 = nn.Dropout(p=0.2)
        self.fc_out2 = nn.Linear(hidden_size, G)
        if self.parallel_arch:
            self.net_kc = nn.Sequential(self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2,
                                        self.fc_out2, nn.Sigmoid())
        else:
            self.net_kc = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                        self.fc_out2, nn.Sigmoid())

        if self.split_enhancer:
            self.fc3 = nn.Linear(dim_z, hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)
            self.dpt3 = nn.Dropout(p=0.2)
            self.fc_out3 = nn.Linear(hidden_size, self.adata_atac.obsm['Me'].shape[1])
            self.net_e = nn.Sequential(self.fc3, self.bn3, nn.LeakyReLU(), self.dpt3,
                                       self.fc_out3)

        if self.t_network:
            self.net_t = nn.Linear(dim_z, 1)

        if self.checkpoint is not None:
            self.alpha_c = nn.Parameter(torch.empty(self.params_shape))
            self.alpha = nn.Parameter(torch.empty(self.params_shape))
            self.beta = nn.Parameter(torch.empty(self.params_shape))
            self.gamma = nn.Parameter(torch.empty(self.params_shape))
            self.register_buffer('sigma_c', torch.empty(G))
            self.register_buffer('sigma_u', torch.empty(G))
            self.register_buffer('sigma_s', torch.empty(G))
            self.register_buffer('zero_vec', torch.zeros(G))
            self.register_buffer('one_vec', torch.ones(G))

            self.ton = nn.Parameter(torch.empty(G))
            self.toff = nn.Parameter(torch.empty(G))
            self.c0_ = nn.Parameter(torch.empty(G))
            self.u0_ = nn.Parameter(torch.empty(G))
            self.s0_ = nn.Parameter(torch.empty(G))
            self.c0 = nn.Parameter(torch.empty(G))
            self.u0 = nn.Parameter(torch.empty(G))
            self.s0 = nn.Parameter(torch.empty(G))

            if self.cvae:
                self.scaling_c = nn.Parameter(torch.empty((self.dim_cond, G)))
                self.scaling_u = nn.Parameter(torch.empty((self.dim_cond, G)))
                self.scaling_s = nn.Parameter(torch.empty((self.dim_cond, G)))
                self.offset_c = nn.Parameter(torch.empty((self.dim_cond, G)))
                self.offset_u = nn.Parameter(torch.empty((self.dim_cond, G)))
                self.offset_s = nn.Parameter(torch.empty((self.dim_cond, G)))
                self.register_buffer('one_mat', torch.ones((self.dim_cond, G)))
                self.register_buffer('zero_mat', torch.zeros((self.dim_cond, G)))
            else:
                self.scaling_c = nn.Parameter(torch.empty(G))
                self.scaling_u = nn.Parameter(torch.empty(G))
                self.scaling_s = nn.Parameter(torch.empty(G))
                self.offset_c = nn.Parameter(torch.empty(G))
                self.offset_u = nn.Parameter(torch.empty(G))
                self.offset_s = nn.Parameter(torch.empty(G))

            self.load_state_dict(torch.load(self.checkpoint))
        else:
            c = self.adata_atac.layers['Mc' if not self.split_enhancer else 'Mp'][self.train_idx]
            u = self.adata.layers['Mu'][self.train_idx]
            s = self.adata.layers['Ms'][self.train_idx]
            self.init_weights()
            self.init_ode(c, u, s, perc)

    def set_shape(self, G, dim_cond):
        if self.is_full_vb:
            if self.cvae:
                self.params_shape = (dim_cond, 2, G)
            else:
                self.params_shape = (2, G)
        else:
            if self.cvae:
                self.params_shape = (dim_cond, G)
            else:
                self.params_shape = G

    def to_param(self, x, sigma=0.1):
        if self.is_full_vb:
            if self.cvae:
                y = np.tile(np.stack([x, sigma*np.ones(self.params_shape[2])]), (self.params_shape[0], 1, 1))
            else:
                y = np.stack([x, sigma*np.ones(self.params_shape[1])])
            return nn.Parameter(torch.tensor(np.log(y) if self.log_params else softplusinv(y)))
        else:
            if self.cvae:
                return nn.Parameter(torch.tensor(np.tile(np.log(x) if self.log_params else softplusinv(x), (self.params_shape[0], 1))))
            else:
                return nn.Parameter(torch.tensor(np.log(x) if self.log_params else softplusinv(x)))

    def init_weights(self, reinit_t=True):
        for m in self.net_rho.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        for m in self.net_kc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        if self.t_network and reinit_t:
            for m in self.net_t.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def init_ode(self, c, u, s, perc):
        G = self.adata.n_vars
        print("Initializing using the steady-state and dynamical models.")
        if len(self.rna_only_idx) > 0:
            rna_only_idx = np.isin(self.batch[self.train_idx], np.array(self.rna_only_idx))
        else:
            rna_only_idx = None
        out = init_params(c, u, s, perc, fit_scaling=True, global_std=self.global_std, tmax=self.tmax, rna_only=self.rna_only, rna_only_idx=rna_only_idx)
        alpha_c, alpha, beta, gamma, scaling_c, scaling_u, toff, c0, u0, s0, sigma_c, sigma_u, sigma_s, mu_c, mu_u, mu_s, t, cpred, upred, spred = out
        scaling_s = np.ones_like(scaling_u)
        offset_c, offset_u, offset_s = np.zeros_like(scaling_u), np.zeros_like(scaling_u), np.zeros_like(scaling_u)

        if self.init_method == 'tprior':
            w = assign_gene_mode_tprior(self.adata, self.init_key, self.train_idx)
            perc_good = 1
        else:
            dyn_mask = (t > self.tmax*0.01) & (np.abs(t-toff) > self.tmax*0.01)
            w = np.sum(((t < toff) & dyn_mask), 0) / (np.sum(dyn_mask, 0) + 1e-10)
            w, perc_good = assign_gene_mode(self.adata, w, 'auto', 0.05, 0.1, 7)
            sigma_c = np.clip(sigma_c, 1e-3, None)
            sigma_u = np.clip(sigma_u, 1e-3, None)
            sigma_s = np.clip(sigma_s, 1e-3, None)
            sigma_c[np.isnan(sigma_c)] = 1
            sigma_u[np.isnan(sigma_u)] = 1
            sigma_s[np.isnan(sigma_s)] = 1
            sigma_c = np.clip(sigma_c, np.min(sigma_c[self.adata.var['quantile_genes']]), None)
            sigma_u = np.clip(sigma_u, np.min(sigma_u[self.adata.var['quantile_genes']]), None)
            sigma_s = np.clip(sigma_s, np.min(sigma_s[self.adata.var['quantile_genes']]), None)
            mu_c[np.isnan(mu_c)] = 0
            mu_u[np.isnan(mu_u)] = 0
            mu_s[np.isnan(mu_s)] = 0
            mu_c = np.clip(mu_c, np.min(mu_c[self.adata.var['quantile_genes']]), np.max(mu_c[self.adata.var['quantile_genes']]))
            mu_u = np.clip(mu_u, np.min(mu_u[self.adata.var['quantile_genes']]), np.max(mu_u[self.adata.var['quantile_genes']]))
            mu_s = np.clip(mu_s, np.min(mu_s[self.adata.var['quantile_genes']]), np.max(mu_s[self.adata.var['quantile_genes']]))
        print(f"Initial induction: {np.sum(w >= 0.5)}, repression: {np.sum(w < 0.5)} out of {G}.")
        self.adata.var["w_init"] = w
        self.perc_good = perc_good
        logit_pw = 0.5*(np.log(w+1e-10) - np.log(1-w-1e-10))
        if not self.rna_only:
            logit_pw = np.stack([logit_pw, -0.5*logit_pw, 0.5*logit_pw, -logit_pw], 1)
        else:
            logit_pw = np.stack([logit_pw, np.zeros_like(logit_pw), np.zeros_like(logit_pw), -logit_pw], 1)
        self.logit_pw = nn.Parameter(torch.tensor(logit_pw))

        if self.init_method == "tprior":
            print("Reinitialization using prior time.")
            self.alpha_c_ = alpha_c
            self.alpha_ = alpha
            self.beta_ = beta
            self.gamma_ = gamma
            self.toff_ = toff
            self.c0_ = c0
            self.u0_ = u0
            self.s0_ = s0
            self.t_ = t
            self.cpred_ = cpred
            self.upred_ = upred
            self.spred_ = spred

            t_prior = self.adata.obs[self.init_key].to_numpy()
            t_prior = t_prior[self.train_idx]
            std_t = (np.std(t_prior)+1e-3)*0.05
            self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
            self.t_init -= self.t_init.min()
            self.t_init = self.t_init
            self.t_init = self.t_init/self.t_init.max()*self.tmax
            if self.reinit:
                toff = get_ts_global(self.t_init, c/scaling_c, u/scaling_u, s, 95)
                alpha_c, alpha, beta, gamma, ton = reinit_params(c/scaling_c, u/scaling_u, s, self.t_init, toff, rna_only=self.rna_only, rna_only_idx=rna_only_idx)

        else:
            self.alpha_c_ = alpha_c
            self.alpha_ = alpha
            self.beta_ = beta
            self.gamma_ = gamma
            self.toff_ = toff
            self.c0_ = c0
            self.u0_ = u0
            self.s0_ = s0
            self.t_ = t
            self.cpred_ = cpred
            self.upred_ = upred
            self.spred_ = spred

            if self.reinit:
                if self.init_key is not None:
                    self.t_init = self.adata.obs[self.init_key].to_numpy()[self.train_idx]
                else:
                    t = t+np.random.rand(t.shape[0], t.shape[1]) * 1e-3
                    t_eq = np.zeros(t.shape)
                    n_bin = t.shape[0]//50+1
                    for i in range(t.shape[1]):
                        t_eq[:, i] = hist_equal(t[:, i], self.tmax, 0.9, n_bin)
                    self.t_init = np.quantile(t_eq, 0.5, 1)
                toff = get_ts_global(self.t_init, c/scaling_c, u/scaling_u, s, 95)
                alpha_c, alpha, beta, gamma, ton = reinit_params(c/scaling_c, u/scaling_u, s, self.t_init, toff, rna_only=self.rna_only, rna_only_idx=rna_only_idx)

        if self.cvae:
            print("Computing scaling factors for each batch class.")
            scaling_c = np.ones((self.dim_cond, G))
            scaling_u = np.ones((self.dim_cond, G))
            scaling_s = np.ones((self.dim_cond, G))
            if self.ref_batch is None:
                self.ref_batch = 0
            ci = c[self.batch[self.train_idx] == self.ref_batch]
            ui = u[self.batch[self.train_idx] == self.ref_batch]
            si = s[self.batch[self.train_idx] == self.ref_batch]
            filt = (si > 0) * (ui > 0) * (ci > 0)
            ci[~filt] = np.nan
            ui[~filt] = np.nan
            si[~filt] = np.nan
            std_u_ref, std_s_ref = np.nanstd(ui, axis=0), np.nanstd(si, axis=0)
            if not self.rna_only:
                scaling_c[self.ref_batch] = np.clip(np.nanpercentile(ci, 99.5, axis=0), 1e-3, None)
            scaling_u[self.ref_batch] = np.clip(std_u_ref / std_s_ref, 1e-6, 1e6)
            scaling_s[self.ref_batch] = 1.0
            for i in range(self.dim_cond):
                if i != self.ref_batch:
                    ci = c[self.batch[self.train_idx] == i]
                    ui = u[self.batch[self.train_idx] == i]
                    si = s[self.batch[self.train_idx] == i]
                    filt = (si > 0) * (ui > 0) * (ci > 0)
                    if np.any(np.sum(filt, axis=0) == 0):
                        j = np.where(np.sum(filt, axis=0) == 0)[0]
                        logger.warning(f'Batch class {i} gene {j} has no valid data.')
                    ci[~filt] = np.nan
                    ui[~filt] = np.nan
                    si[~filt] = np.nan
                    std_u, std_s = np.nanstd(ui, axis=0), np.nanstd(si, axis=0)
                    if not self.rna_only and i not in self.rna_only_idx:
                        scaling_c[i] = np.clip(np.nanpercentile(ci, 99.5, axis=0), 1e-3, None)
                    scaling_u[i] = np.clip(std_u / (std_s_ref*(~np.isnan(std_s_ref)) + std_s*np.isnan(std_s_ref)), 1e-6, 1e6)
                    scaling_s[i] = np.clip(std_s / (std_s_ref*(~np.isnan(std_s_ref)) + std_s*np.isnan(std_s_ref)), 1e-6, 1e6)
            offset_c = np.zeros((self.dim_cond, G))
            offset_u = np.zeros((self.dim_cond, G))
            offset_s = np.zeros((self.dim_cond, G))
        if not self.cvae:
            if np.any(np.isnan(scaling_c)):
                logger.warning('Warning: scaling_c invalid (nan).')
            if np.any(np.isinf(scaling_c)):
                logger.warning('Warning: scaling_c invalid (inf).')
            if np.any(np.isnan(scaling_u)):
                logger.warning('Warning: scaling_u invalid (nan).')
            if np.any(np.isinf(scaling_u)):
                logger.warning('Warning: scaling_u invalid (inf).')
            if np.any(np.isnan(scaling_s)):
                logger.warning('Warning: scaling_s invalid (nan).')
            if np.any(np.isinf(scaling_s)):
                logger.warning('Warning: scaling_s invalid (inf).')
        scaling_c[np.isnan(scaling_c)] = 1.0
        scaling_u[np.isnan(scaling_u)] = 1.0
        scaling_s[np.isnan(scaling_s)] = 1.0

        self.alpha_c = self.to_param(alpha_c)
        self.alpha = self.to_param(alpha)
        self.beta = self.to_param(beta)
        self.gamma = self.to_param(gamma)
        self.scaling_c = nn.Parameter(torch.tensor(np.log(scaling_c) if self.log_params else softplusinv(scaling_c)))
        self.scaling_u = nn.Parameter(torch.tensor(np.log(scaling_u) if self.log_params else softplusinv(scaling_u)))
        self.scaling_s = nn.Parameter(torch.tensor(np.log(scaling_s) if self.log_params else softplusinv(scaling_s)))
        self.offset_c = nn.Parameter(torch.tensor(offset_c))
        self.offset_u = nn.Parameter(torch.tensor(offset_u))
        self.offset_s = nn.Parameter(torch.tensor(offset_s))

        self.c0_ = nn.Parameter(torch.tensor(np.log(c0+1e-10) if self.log_params else softplusinv(c0+1e-6)))
        self.u0_ = nn.Parameter(torch.tensor(np.log(u0+1e-10) if self.log_params else softplusinv(u0+1e-6)))
        self.s0_ = nn.Parameter(torch.tensor(np.log(s0+1e-10) if self.log_params else softplusinv(s0+1e-6)))
        if self.train_x0:
            self.c0 = nn.Parameter(torch.tensor(np.log(np.full_like(c0, 1e-6)) if self.log_params else softplusinv(np.full_like(c0, 1e-3))))
            self.u0 = nn.Parameter(torch.tensor(np.log(np.full_like(u0, 1e-6)) if self.log_params else softplusinv(np.full_like(u0, 1e-3))))
            self.s0 = nn.Parameter(torch.tensor(np.log(np.full_like(s0, 1e-6)) if self.log_params else softplusinv(np.full_like(s0, 1e-3))))
        else:
            self.c0 = nn.Parameter(torch.tensor(np.zeros_like(c0)))
            self.u0 = nn.Parameter(torch.tensor(np.zeros_like(u0)))
            self.s0 = nn.Parameter(torch.tensor(np.zeros_like(s0)))

        if self.init_ton_zero or (not self.reinit):
            self.ton = nn.Parameter(torch.zeros(G))
        else:
            self.ton = nn.Parameter(torch.tensor(ton+1e-10))
        if self.rna_only:
            sigma_c = np.full_like(sigma_c, 1.0)
        self.register_buffer('sigma_c', torch.tensor(sigma_c))
        self.register_buffer('sigma_u', torch.tensor(sigma_u))
        self.register_buffer('sigma_s', torch.tensor(sigma_s))
        self.register_buffer('mu_c', torch.tensor(mu_c))
        self.register_buffer('mu_u', torch.tensor(mu_u))
        self.register_buffer('mu_s', torch.tensor(mu_s))
        self.register_buffer('zero_vec', torch.zeros_like(self.u0))
        self.register_buffer('one_vec', torch.ones_like(self.u0))
        if self.cvae:
            self.register_buffer('one_mat', torch.ones_like(self.scaling_u))
            self.register_buffer('zero_mat', torch.zeros_like(self.scaling_u))
        if self.rna_only:
            self.alpha_c.requires_grad = False
            self.scaling_c.requires_grad = False
            self.offset_c.requires_grad = False
            self.c0.requires_grad = False

    def get_param(self, x, enforce_positive=True, idx=None):
        if x == 'ton':
            out = self.ton
        elif x == 'c0':
            out = self.c0
        elif x == 'u0':
            out = self.u0
        elif x == 's0':
            out = self.s0
        elif x == 'alpha_c':
            out = self.alpha_c
        elif x == 'alpha':
            out = self.alpha
        elif x == 'beta':
            out = self.beta
        elif x == 'gamma':
            out = self.gamma
        elif x == 'scaling_c':
            out = self.scaling_c
        elif x == 'scaling_u':
            out = self.scaling_u
        elif x == 'scaling_s':
            out = self.scaling_s
        elif x == 'offset_c':
            out = self.offset_c
        elif x == 'offset_u':
            out = self.offset_u
        elif x == 'offset_s':
            out = self.offset_s

        if idx is not None:
            out = out[idx]

        if enforce_positive:
            if self.log_params:
                out = out.exp()
            else:
                out = F.softplus(out)
        return out

    def _get_param(self, x, condition=None, four_basis=False, is_full_vb=False, sample=True, mask_idx=None, mask_to=1, enforce_positive=True, numpy=False):
        param = self.get_param(x, enforce_positive=enforce_positive)
        if numpy:
            param = param.detach()
        if is_full_vb:
            if sample:
                G = self.adata.n_vars
                eps = torch.randn(G, device=self.alpha_c.device)
                if condition is not None:
                    y = torch.exp(param[:, 0].log() + eps*(param[:, 1]))
                else:
                    y = torch.exp(param[0].log() + eps*(param[1]))
            else:
                if condition is not None:
                    y = param[:, 0]
                else:
                    y = param[0]
        else:
            y = param

        if condition is not None:
            if isinstance(mask_idx, int) or (isinstance(mask_idx, np.ndarray) and len(mask_idx) > 0):
                if (isinstance(mask_idx, np.ndarray) and len(mask_idx) > 1):
                    mask_idx = np.unique(mask_idx)
                mask = torch.zeros_like(condition)
                mask[:, mask_idx] = 1
                mask_flip = (~mask.bool()).int()
                y = torch.mm(condition * mask_flip, y)
                if isinstance(mask_to, int) and mask_to == 0:
                    y += torch.mm(condition * mask, self.zero_mat)
                elif isinstance(mask_to, int) and mask_to == 1:
                    y += torch.mm(condition * mask, self.one_mat)
                elif isinstance(mask_to, np.ndarray):
                    mask_to = mask_to.reshape((1, -1)).tile((self.dim_cond, 1))
                    y += torch.mm(condition * mask, mask_to)
            else:
                y = torch.mm(condition, y)

        if condition is not None and four_basis:
            y = y.unsqueeze(1)
        if numpy:
            y = y.cpu().numpy()
        return y

    def reparameterize(self, condition=None, sample=True, four_basis=False, numpy=False):
        alpha_c = self._get_param('alpha_c', condition, four_basis, self.is_full_vb, sample, mask_idx=self.rna_only_idx, mask_to=0, numpy=numpy)
        alpha = self._get_param('alpha', condition, four_basis, self.is_full_vb, sample, numpy=numpy)
        beta = self._get_param('beta', condition, four_basis, self.is_full_vb, sample, numpy=numpy)
        gamma = self._get_param('gamma', condition, four_basis, self.is_full_vb, sample, numpy=numpy)
        scaling_c = self._get_param('scaling_c', condition, four_basis, mask_idx=self.rna_only_idx, mask_to=1, numpy=numpy)
        scaling_u = self._get_param('scaling_u', condition, four_basis, numpy=numpy)
        scaling_s = self._get_param('scaling_s', condition, four_basis, mask_idx=self.ref_batch, numpy=numpy)
        offset_c = self._get_param('offset_c', condition, four_basis, mask_idx=np.append(self.rna_only_idx, self.ref_batch), mask_to=0, enforce_positive=False, numpy=numpy)
        offset_u = self._get_param('offset_u', condition, four_basis, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False, numpy=numpy)
        offset_s = self._get_param('offset_s', condition, four_basis, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False, numpy=numpy)
        return alpha_c, alpha, beta, gamma, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s

    def _get_param_special(self, kc, rho, t_, condition=None, four_basis=False):
        if four_basis:
            zero_mat = torch.zeros_like(rho)
            kc_ = torch.stack([kc, kc, zero_mat, zero_mat], 1) if not self.rna_only else torch.stack([kc for i in range(4)], 1)
            rho_ = torch.stack([rho, zero_mat, rho, zero_mat], 1)
            c0 = torch.stack([self.zero_vec, self.zero_vec, self.c0_.exp() if self.log_params else F.softplus(self.c0_), self.c0_.exp() if self.log_params else F.softplus(self.c0_)])
            u0 = torch.stack([self.zero_vec, self.u0_.exp() if self.log_params else F.softplus(self.u0_), self.zero_vec, self.u0_.exp() if self.log_params else F.softplus(self.u0_)])
            s0 = torch.stack([self.zero_vec, self.s0_.exp() if self.log_params else F.softplus(self.s0_), self.zero_vec, self.s0_.exp() if self.log_params else F.softplus(self.s0_)])
            tau = torch.stack([(t_ - self.ton) for i in range(4)], 1)
        else:
            kc_ = kc
            rho_ = rho
            if self.train_x0:
                c0 = self.c0.exp() if self.log_params else F.softplus(self.c0)
                u0 = self.u0.exp() if self.log_params else F.softplus(self.u0)
                s0 = self.s0.exp() if self.log_params else F.softplus(self.s0)
            else:
                c0 = self.c0
                u0 = self.u0
                s0 = self.s0
            tau = t_ - self.ton

        if self.rna_only:
            c0 = torch.ones_like(c0)
        if condition is not None and len(self.rna_only_idx) > 0:
            mask = torch.zeros_like(condition)
            mask[:, self.rna_only_idx] = 1
            mask_flip = (~mask.bool()).int()
            kc_copy = kc_.detach().clone()
            if four_basis:
                kc_ = torch.stack([torch.mm(condition * mask_flip, self.one_mat) for i in range(4)], 1) * kc_ + torch.stack([torch.mm(condition * mask, self.one_mat) for i in range(4)], 1) * kc_copy
                c0 = c0.reshape((1, 4, -1)).tile((self.dim_cond, 1, 1))
                c0 = torch.einsum('bi,ijm->bjm', condition * mask_flip, c0) + torch.einsum('bi,ijm->bjm', condition * mask, torch.ones_like(c0))
            else:
                kc_ = torch.mm(condition * mask_flip, self.one_mat) * kc_ + torch.mm(condition * mask, self.one_mat) * kc_copy
                c0 = c0.reshape((1, -1)).tile((self.dim_cond, 1))
                c0 = torch.mm(condition * mask_flip, c0) + torch.mm(condition * mask, self.one_mat)
        return kc_, rho_, c0, u0, s0, tau

    def forward(self, t, z, ze=None, c0=None, u0=None, s0=None, t0=None, condition=None, regressor=None, neg_slope=0.0, sample=True, four_basis=False, return_velocity=False):
        alpha_c, alpha, beta, gamma, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s = self.reparameterize(condition, sample, four_basis)

        z_in = z
        if condition is not None:
            z_in = torch.cat((z_in, condition), 1)
        if regressor is not None:
            z_in = torch.cat((z_in, regressor), 1)
        if not self.split_enhancer:
            rho = self.net_rho(z_in)
            ehat = None
        else:
            rho = self.net_rho(z_in, ze)
            ehat = self.net_e(ze)
        if self.rna_only:
            kc = torch.ones_like(rho)
        else:
            if not self.split_enhancer:
                kc = self.net_kc(z_in)
            else:
                kc = self.net_kc(z_in, ze)

        t_in = t
        t_ = self.net_t(t_in) * self.tmax if self.t_network else t
        if (c0 is None) or (u0 is None) or (s0 is None) or (t0 is None):
            kc, rho, c0, u0, s0, tau = self._get_param_special(kc, rho, t_, condition, four_basis)
            tau = F.leaky_relu(tau, neg_slope)
            chat, uhat, shat = pred_exp(tau,
                                        c0,
                                        u0,
                                        s0,
                                        kc,
                                        alpha_c,
                                        rho,
                                        alpha,
                                        beta,
                                        gamma)
        else:
            tau = F.leaky_relu(t_ - t0, neg_slope)
            chat, uhat, shat = pred_exp(tau,
                                        c0,
                                        u0,
                                        s0,
                                        kc,
                                        alpha_c,
                                        rho,
                                        alpha,
                                        beta,
                                        gamma)
            if len(self.rna_only_idx) > 0 and condition is not None:
                mask = np.in1d(np.arange(self.dim_cond), self.rna_only_idx)
                mask = torch.tensor(mask, device=chat.device).int()
                chat = chat * ((condition * (~mask.bool()).int()).sum(1, keepdim=True) > 0) + torch.ones_like(chat) * ((condition * mask).sum(1, keepdim=True) > 0)

        if return_velocity:
            vc = (kc * alpha_c - alpha_c * chat) * scaling_c
            vu = (rho * alpha * chat - beta * uhat) * scaling_u
            vs = (beta * uhat - gamma * shat) * scaling_s

            chat = chat * scaling_c + offset_c
            uhat = uhat * scaling_u + offset_u
            shat = shat * scaling_s + offset_s
            return chat, uhat, shat, ehat, t_, kc, rho, vc, vu, vs
        else:
            chat = chat * scaling_c + offset_c
            uhat = uhat * scaling_u + offset_u
            shat = shat * scaling_s + offset_s
            return chat, uhat, shat, ehat, t_, kc, rho


class VAEChrom():
    """
    MultiVeloVAE model for joint multi-omics velocity inference.

    This is the main class implementing the MultiVeloVAE model, which integrates
    chromatin accessibility and RNA sequencing data to infer cellular dynamics
    through a variational autoencoder framework with a mechanistic ODE model.

    The model learns a low-dimensional latent representation of cell state,
    a latent time for each cell, and the parameters of velocity ODE equations
    that explain the observed data. It can handle batch effects through a
    conditional VAE design. It can handle both multi-omic and RNA-only samples.

    The model consists of:
    1. An encoder that maps data to a distribution over latent variables
    2. A decoder that generates predictions using a mechanistic ODE model
    3. Training procedures including two-stage refinement

    Attributes:
        adata: AnnData object containing RNA data
        adata_atac: AnnData object containing chromatin accessibility data
        encoder: Neural network encoder
        decoder: Neural network decoder with ODE model
        config: Dictionary containing model configuration and hyperparameters
        Other attributes for training and inference

    Args:
        adata (AnnData):
            Input AnnData object for RNA data.
        adata_atac (AnnData, optional):
            Input AnnData object for chromatin data. Defaults to None for RNA-only.
        dim_z (int, optional):
            Latent cell state dimension. Defaults to None.
        batch_key (str, optional):
            Key of batch labels in adata.obs. Defaults to None.
        ref_batch (int, optional):
            Index to use as the reference batch. Defaults to None.
        batch_hvg_key (str, optional):
            Prefix of key for batch highly-variable genes in adata.var. Defaults to None.
        var_to_regress (str or list, optional):
            Continuous variable(s) to be regressed out. Defaults to None.
        device (torch.device, optional):
            Device used for training. Defaults to 'cpu'.
        hidden_size (int, optional):
            The width of the hidden layers of the encoder and decoder. Defaults to 256.
        full_vb (bool, optional):
            Whether to use full variational Bayes for rate parameters. Defaults to False.
        parallel_arch (bool, optional):
            Whether to use parallel architecture for the indicator functions. Defaults to True.
        t_network (bool, optional):
            Whether to use a neural network to estimate the time distribution. Defaults to True.
        four_basis (bool, optional):
            Whether to enable BasisVAE to model genes as induction and repression. Defaults to False.
        run_2nd_stage (bool, optional):
            Whether to run the second stage of training. Defaults to True.
        tmax (float):
            Maximum time, specifies the time range for initialization.
        init_method (str, optional):
            {'steady', 'tprior'}, initialization method. Defaults to 'steady'.
        init_tprior (str, optional):
            Key in adata.obs for initialization. Defaults to None.
        tprior (str, optional):
            Key in adata.obs containing the informative time prior. Defaults to None.
        deming_std (bool, optional):
            Whether to use Deming residual for the loss function std. Defaults to False.
        rna_only (bool, optional):
            Whether to run in RNA-only mode. Defaults to False.
        rna_only_idx (list, optional):
            List of indices of RNA-only samples. Defaults to [].
        learning_rate (float, optional):
            Learning rate for training. Defaults to None.
        early_stop_thred (float, optional):
            Early stopping threshold for training. Defaults to None.
        checkpoints (list, optional):
            List of two .pt files with pretrained parameters. Defaults to [None, None].
        plot_init (bool, optional):
            Whether to plot initialization results. Defaults to False.
        gene_plot (list, optional):
            List of gene names to plot. Defaults to [].
        cluster_key (str, optional):
            Key in adata.obs for plot colors. Defaults to 'clusters'.
        figure_path (str, optional):
            Path to save figures. Defaults to 'figures'.
        embed (str, optional):
            Key in adata.obsm of 2D embedding (tsne, umap, etc.). Defaults to None.
        vram_constrained (bool, optional):
            Whether to enable VRAM-constrained mode. Defaults to False.
    """
    def __init__(self,
                 adata,
                 adata_atac=None,
                 dim_z=None,
                 batch_key=None,
                 ref_batch=None,
                 batch_hvg_key=None,
                 var_to_regress=None,
                 device='cpu',
                 hidden_size=256,
                 full_vb=False,
                 parallel_arch=True,
                 t_network=True,
                 four_basis=False,
                 run_2nd_stage=True,
                 tmax=1,
                 init_method='steady',
                 init_tprior=None,
                 tprior=None,
                 deming_std=False,
                 rna_only=False,
                 rna_only_idx=[],
                 learning_rate=None,
                 early_stop_thred=None,
                 checkpoints=[None, None],
                 plot_init=False,
                 gene_plot=[],
                 cluster_key='clusters',
                 figure_path='figures',
                 embed=None,
                 vram_constrained=False):
        """MultiVeloVAE Model

        Args:
            adata ((:class:`anndata.AnnData`)):
                Input AnnData object for RNA data.
            adata_atac ((:class:`anndata.AnnData`)):
                Input AnnData object for chromatin data. Defaults to None for RNA-only.
            dim_z (int, optional):
                Latent cell state dimension. Defaults to None.
            batch_key (str, optional):
                Key of batch labels in adata.obs. Defaults to None.
            ref_batch (int, optional):
                Index to use as the reference batch. Defaults to None.
            batch_hvg_key (str, optional):
                Prefix of key for batch highly-variable genes in adata.var. Defaults to None.
            var_to_regress (str or list, optional):
                Continuous variable(s) to be regressed out. Defaults to None.
            device (torch.device, optional):
                Device used for training. Defaults to 'cpu'.
            hidden_size (int, optional):
                The width of the hidden layers of the encoder and decoder. Defaults to 256.
            full_vb (bool, optional):
                Whether to use the full variational Bayes feature to estimate rate parameter uncertainty.
                Defaults to False.
            parallel_arch (bool, optional):
                Whether to use parallel architecture for the indicator functions. Defaults to True.
            t_network (bool, optional):
                Whether to use a neural network to estimate the time distribution. Defaults to True.
            four_basis (bool, optional):
                Whether to enable BasisVAE to model genes as induction and repression. Defaults to False.
            run_2nd_stage (bool, optional):
                Whether to run the second stage of training. Defaults to True.
            tmax (float):
                Maximum time, specifies the time range for initialization.
            init_method (str, optional):
                {'steady', 'tprior'}, initialization method. Defaults to 'steady'.
            init_tprior (str, optional):
                Key in adata.obs storing the capture time or any prior time information.
                This is used in initialization. Defaults to None.
            tprior (str, optional):
                Key in adata.obs containing the informative time prior.
                This is used in model training. Defaults to None.
            deming_std (bool, optional):
                Whether to use Deming residual for the standard deviation of the loss function. Defaults to False.
            rna_only (bool, optional):
                Whether to run in RNA-only mode. Defaults to False.
            rna_only_idx (list, optional):
                List of indices of RNA-only samples. Defaults to [].
            learning_rate (float, optional):
                Learning rate for training. Defaults to None.
            early_stop_thred (float, optional):
                Early stopping threshold for training. Defaults to None.
            checkpoints (list, optional):
                Contains a list of two .pt files containing pretrained or saved model parameters.
                Defaults to [None, None].
            plot_init (bool, optional):
                Whether to plot the initialization results. Defaults to False.
            gene_plot (list, optional):
                List of gene names to plot. Defaults to [].
            cluster_key (str, optional):
                Key in adata.obs containing the cluster labels for plot colors. Defaults to 'clusters'.
            figure_path (str, optional):
                Path to save the figures. Defaults to 'figures'.
            vram_constrained (bool, optional):
                Whether to enable VRAM-constrained mode. Defaults to False.
        """
        if adata_atac is None or rna_only:
            rna_only = True
            adata_atac = ad.AnnData(X=np.ones((adata.n_obs, adata.n_vars)))
            adata_atac.layers['Mc'] = adata_atac.X
            print('Running in RNA-only mode.')
        if rna_only_idx is None:
            rna_only_idx = []
        elif isinstance(rna_only_idx, int):
            rna_only_idx = [rna_only_idx]
        elif len(rna_only_idx) == 0:
            rna_only_idx = []
        else:
            if not isinstance(rna_only_idx, (list, np.ndarray)):
                raise ValueError('rna_only_idx is invalid.')
        if batch_key is not None and len(rna_only_idx) > 0 and not rna_only:
            print('Running in mixed RNA-only mode.')
            if ref_batch is not None and ref_batch in rna_only_idx:
                raise ValueError('Error: reference batch cannot be RNA only when inferring chromatin. Exiting the program...')
        if ('Mc' not in adata_atac.layers) or ('Mu' not in adata.layers) or ('Ms' not in adata.layers):
            raise ValueError('Chromatin/Unspliced/Spliced count matrices not found in the layers! Exiting the program...')
        if issparse(adata_atac.layers['Mc']):
            adata_atac.layers['Mc'] = adata_atac.layers['Mc'].toarray()
        if issparse(adata.layers['Mu']):
            adata.layers['Mu'] = adata.layers['Mu'].toarray()
        if issparse(adata.layers['Ms']):
            adata.layers['Ms'] = adata.layers['Ms'].toarray()
        if np.any(adata_atac.layers['Mc'] < 0):
            logger.warning('Negative expression values detected in layers["Mc"]. Please make sure all values are non-negative.')
        if np.any(adata.layers['Mu'] < 0):
            logger.warning('Negative expression values detected in layers["Mu"]. Please make sure all values are non-negative.')
        if np.any(adata.layers['Ms'] < 0):
            logger.warning('Negative expression values detected in layers["Ms"]. Please make sure all values are non-negative.')

        self.config = {
            # model parameters
            "dim_z": dim_z,
            "hidden_size": hidden_size,
            "key": 'vae',
            "is_full_vb": full_vb,
            "split_enhancer": False,
            "indicator_arch": 'parallel' if parallel_arch else 'series',
            "t_network": t_network if tprior is None else False,
            "velocity_continuity": False,
            "velocity_correlation": False,
            "four_basis": four_basis,
            "batch_key": batch_key,
            "ref_batch": ref_batch,
            "batch_hvg_key": batch_hvg_key,
            "var_to_regress": var_to_regress,
            "reinit_params": tprior is not None,
            "init_ton_zero": True,
            "unit_scale": True,
            'loss_std_type': 'deming' if deming_std else 'global',
            "log_params": False,
            "rna_only": rna_only,
            "rna_only_idx": rna_only_idx,
            "tmax": tmax,
            "init_method": init_method,
            "init_key": init_tprior,
            "tprior": tprior,
            "std_z_prior": 0.05,
            "tail": 0.01,
            "std_t_scaling": 0.05,
            "n_neighbors": 4,
            "n_lineages": 20,
            "dt": (0.04, 0.08),

            # training parameters
            "n_epochs": 2000,
            "n_epochs_post": 500,
            "n_refine": 6,
            "n_refine_min": 3,
            "batch_size": 256,
            "learning_rate": None,
            "learning_rate_ode": None,
            "learning_rate_post": None,
            "lambda": 1e-2,
            "lambda_post": 1e-2,
            "kl_t": 1.0,
            "kl_z": 1.0,
            "kl_w": 1e-2,
            "kl_param": 0.1,
            "reg_param_batch": 1e-2,
            "reg_forward": 1.0,
            "reg_corr": 1.0,
            "reg_cos": 0.0,
            "2_norm": 1e-4,
            "test_iter": None,
            "n_warmup": 6,
            "n_warmup2": 16,
            "n_warmup_post": 4,
            "cosineannealing_iter": 20,
            "cosineannealing_warmup": 10,
            "weight_c": 0.6 if not rna_only else 0.0,
            "weight_soft_batch": 0.0,
            "early_stop": 6,
            "early_stop_thred": early_stop_thred,
            "x0_change_thred": 0.1,
            "train_test_split": 0.7,
            "neg_slope": 0.0,
            "neg_slope2": 1e-2,
            "k_alt": 0,
            "train_x0": True,
            "train_ton": False,
            "train_scaling": True,
            "train_offset": True,
            "knn_use_pred": True,
            "run_2nd_stage": run_2nd_stage,
            "save_epoch": 50,
            "vram_constrained": vram_constrained,

            # plotting
            "sparsify": 1}

        self.set_device(device)
        self.split_train_test(adata.n_obs)
        self.encode_batch(adata, adata_atac, batch_key, ref_batch, batch_hvg_key)
        if self.enable_cvae and self.config['rna_only']:
            self.config['rna_only_idx'] = np.arange(self.n_batch)
        self.rna_only_idx = np.array(self.config['rna_only_idx'])
        if self.enable_cvae and np.array_equal(np.array(set(self.rna_only_idx)), np.arange(self.n_batch)):
            self.config['rna_only'] = True
        if not self.enable_cvae and len(self.rna_only_idx) == 1 and self.rna_only_idx[0] == 0:
            self.config['rna_only'] = True
        if self.enable_cvae and np.any(np.array(self.rna_only_idx) > self.n_batch):
            raise ValueError('RNA-only index out of range.')
        self.init_regressor(adata, var_to_regress)
        self.set_lr(adata, adata_atac, learning_rate)
        self.get_prior(adata)

        self.cell_type_colors = None
        if cluster_key is not None and cluster_key in adata.obs.keys():
            self.cell_labels_raw = adata.obs[cluster_key].to_numpy()
            if adata.obs[cluster_key].dtype.name == 'category':
                self.cell_types_raw = adata.obs[cluster_key].cat.categories.to_numpy()
            else:
                self.cell_types_raw = np.unique(self.cell_labels_raw)
            self.label_dic, self.label_dic_rev = encode_type(self.cell_types_raw)
            n_type = len(self.cell_types_raw)
            self.cell_labels = np.array([self.label_dic[x] for x in self.cell_labels_raw])
            self.cell_types = np.array([self.label_dic[self.cell_types_raw[i]] for i in range(n_type)])

            if f'{cluster_key}_colors' in adata.uns.keys():
                cell_type_colors = adata.uns[f'{cluster_key}_colors']
                self.cell_type_colors = {self.cell_types_raw[i]: cell_type_colors[i] for i in range(n_type)}
        else:
            self.cell_labels_raw = np.full(adata.n_obs, '')
            self.cell_labels = np.full(adata.n_obs, '')

        self.adata = adata
        self.adata_atac = adata_atac

        self.encoder = Encoder(3*self.adata.n_vars,
                               self.config['dim_z'],
                               dim_cond=self.n_batch,
                               dim_reg=(0 if self.var_to_regress is None else self.var_to_regress.shape[1]),
                               hidden_size=hidden_size,
                               t_network=self.config['t_network'],
                               split_enhancer=self.config['split_enhancer'],
                               Cin_e=self.adata_atac.obsm['Me'].shape[1] if self.config['split_enhancer'] else None,
                               checkpoint=checkpoints[0]).float().to(self.device)

        self.decoder = Decoder(self.adata,
                               self.adata_atac,
                               self.train_idx,
                               self.config['dim_z'],
                               dim_cond=self.n_batch,
                               dim_reg=(0 if self.var_to_regress is None else self.var_to_regress.shape[1]),
                               batch_idx=self.batch_,
                               ref_batch=self.ref_batch,
                               hidden_size=hidden_size,
                               split_enhancer=self.config['split_enhancer'],
                               parallel_arch=parallel_arch,
                               t_network=self.config['t_network'],
                               full_vb=full_vb,
                               global_std=(not deming_std),
                               log_params=self.config["log_params"],
                               rna_only=rna_only,
                               rna_only_idx=rna_only_idx,
                               perc=98,
                               tmax=tmax,
                               reinit=self.config["reinit_params"],
                               train_x0=self.config["train_x0"],
                               init_ton_zero=self.config["init_ton_zero"],
                               init_method=init_method,
                               init_key=init_tprior,
                               checkpoint=checkpoints[1]).float().to(self.device)

        self.alpha_w = torch.tensor(find_dirichlet_param(0.5, 0.05, 4), dtype=torch.float, device=self.device)

        self.use_knn = False
        self.reg_velocity = False
        self.c0, self.u0, self.s0 = None, None, None
        self.c1, self.u1, self.s1 = None, None, None
        self.t0, self.t1 = None, None
        self.x0_index, self.x1_index = None, None
        self.z = None

        self.loss_train, self.loss_test = [], []
        self.loss_test_iter, self.loss_test_color = [], []
        self.rec_train, self.klt_train, self.klz_train = [], [], []
        self.rec_test, self.klt_test, self.klz_test = [], [], []
        if self.config['split_enhancer']:
            self.rec_e_train, self.kle_train = [], []
            self.rec_e_test, self.kle_test = [], []
        self.counter = 0
        self.n_drop = 0
        self.best_model_state = None
        self.best_counter = 0
        self.loss_idx_start = 0
        self.r = -1

        self.clip_fn = nn.Hardtanh(-1e16, 1e16)

        if full_vb:
            self.p_log_alpha_c = torch.tensor([[2], [0.5]], dtype=torch.float, device=self.device)
            self.p_log_alpha = torch.tensor([[2], [1]], dtype=torch.float, device=self.device)
            self.p_log_beta = torch.tensor([[2], [0.5]], dtype=torch.float, device=self.device)
            self.p_log_gamma = torch.tensor([[2], [0.5]], dtype=torch.float, device=self.device)
            self.p_params = [self.p_log_alpha_c, self.p_log_alpha, self.p_log_beta, self.p_log_gamma]

        if plot_init:
            self.plot_initial(gene_plot, figure_path, embed)

    def set_device(self, device):
        if 'cuda' in device:
            if torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                logger.warning('GPU not detected. Using CPU as the device.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

    def set_mode(self, mode, net='both'):
        if mode == 'train':
            if net == 'both':
                self.encoder.train()
                self.decoder.train()
            elif net == 'encoder':
                self.encoder.train()
            elif net == 'decoder':
                self.decoder.train()
        elif mode == 'eval':
            if net == 'both':
                self.encoder.eval()
                self.decoder.eval()
            elif net == 'encoder':
                self.encoder.eval()
            elif net == 'decoder':
                self.decoder.eval()
        else:
            logger.warning("Mode not recognized. Must be 'train' or 'test'!")

    def split_train_test(self, N):
        rand_perm = np.random.permutation(N)
        n_train = int(N*self.config["train_test_split"])
        self.train_idx = rand_perm[:n_train]
        self.test_idx = rand_perm[n_train:]

    def encode_batch(self, adata, adata_atac, batch_key, ref_batch, batch_hvg_key=None):
        self.n_batch = 0
        self.batch = None
        self.batch_ = None
        batch_count = None
        self.ref_batch = ref_batch
        if batch_key is not None and batch_key in adata.obs.keys():
            print('CVAE enabled. Performing batch effect correction.')
            self.batch_labels_raw = adata.obs[batch_key].to_numpy()
            if adata.obs[batch_key].dtype.name == 'category':
                self.batch_names_raw = adata.obs[batch_key].cat.categories.to_numpy()
                batch_count = adata.obs[batch_key].value_counts()[self.batch_names_raw].to_numpy()
            else:
                self.batch_names_raw, batch_count = np.unique(self.batch_labels_raw, return_counts=True)
            self.batch_dic, self.batch_dic_rev = encode_type(self.batch_names_raw)
            self.n_batch = len(self.batch_names_raw)
            self.batch_ = np.array([self.batch_dic[x] for x in self.batch_labels_raw])
            self.batch = torch.tensor(self.batch_, dtype=int, device=self.device)
        if isinstance(self.ref_batch, int):
            if self.ref_batch >= self.n_batch:
                self.ref_batch = self.n_batch - 1
            elif self.ref_batch < -self.n_batch:
                self.ref_batch = 0
            print(f'Reference batch set to {self.ref_batch} ({self.batch_names_raw[self.ref_batch]}).')
            self.config['ref_batch'] = self.batch_names_raw[self.ref_batch]
            if np.issubdtype(self.batch_names_raw.dtype, np.number) and 0 not in self.batch_names_raw:
                logger.warning('Integer batch names do not start from 0. Reference batch index may not match the actual batch name!')
        elif isinstance(self.ref_batch, str):
            if self.config['ref_batch'] in self.batch_names_raw:
                self.ref_batch = self.batch_dic[self.config['ref_batch']]
                print(f'Reference batch set to {self.ref_batch} ({self.batch_names_raw[self.ref_batch]}).')
                self.config['ref_batch'] = self.batch_names_raw[self.ref_batch]
            else:
                raise ValueError('Reference batch not found in the provided batch field!')
        elif batch_count is not None:
            self.ref_batch = 0
            print(f'Reference batch set to {self.ref_batch} ({self.batch_names_raw[self.ref_batch]}).')
            self.config['ref_batch'] = self.batch_names_raw[self.ref_batch]
        self.enable_cvae = self.n_batch > 0
        if self.config['dim_z'] is not None:
            if self.enable_cvae and 2*self.n_batch > self.config['dim_z']:
                logger.warning('Number of batch classes is larger than half of dim_z. Consider increasing dim_z.')
            if self.enable_cvae and 10*self.n_batch < self.config['dim_z']:
                logger.warning('Number of batch classes is smaller than 1/10 of dim_z. Consider decreasing dim_z.')
        else:
            if self.config['rna_only']:
                dim_z = 3 + adata.n_vars // 512
            else:
                dim_z = 4 + adata.n_vars // 256
            if self.enable_cvae:
                dim_z += 1
            self.config['dim_z'] = dim_z
            print(f'Latent dimension set to {dim_z}.')
        self.batch_hvg_genes_ = np.ones((self.n_batch, adata.n_vars), dtype=bool)
        if batch_hvg_key is not None:
            for batch in self.batch_names_raw:
                if f"{batch_hvg_key}-{batch}" in adata.var.keys():
                    self.batch_hvg_genes_[self.batch_dic[batch]] = adata.var[f"{batch_hvg_key}-{batch}"].to_numpy()
                else:
                    logger.warning(f'Highly variable genes for batch {batch} not found in var: {batch_hvg_key}-batch. All genes will be used for batch {batch}.')
                    self.batch_hvg_genes_[self.batch_dic[batch]] = True
        elif self.enable_cvae:
            for i in range(self.n_batch):
                ci = adata_atac.layers['Mc' if not self.config['split_enhancer'] else 'Mp'][self.batch_ == i]
                ui = adata.layers['Mu'][self.batch_ == i]
                si = adata.layers['Ms'][self.batch_ == i]
                filt = (si > 0) * (ui > 0) * (ci > 0)
                self.batch_hvg_genes_[i] = np.sum(filt, axis=0) > ci.shape[0] * 0.01
        self.batch_hvg_genes = torch.tensor(self.batch_hvg_genes_, dtype=torch.float, device=self.device)
        self.batch_count_balance = None
        if batch_count is not None:
            self.batch_count_balance = torch.tensor(batch_count / batch_count[self.ref_batch], dtype=torch.float, device=self.device)
        if self.enable_cvae:
            self.config['kl_t'] = self.config['kl_t'] * 2 * (self.n_batch-1)
            self.config['kl_z'] = self.config['kl_z'] * 2 * (self.n_batch-1)
            print(f"KL weights set to {self.config['kl_t']:.2f} {self.config['kl_z']:.2f}.")
        self.onehot = F.one_hot(self.batch, self.n_batch).float() if self.enable_cvae else None

    def init_regressor(self, adata, var_to_regress):
        self.var_to_regress = None
        if isinstance(var_to_regress, str):
            if var_to_regress in adata.obs.keys():
                self.var_to_regress = adata.obs[var_to_regress].to_numpy()[:, None]
                print(f'Found {var_to_regress} in obs. Regressing it out.')
        elif isinstance(var_to_regress, (list, tuple, np.ndarray)):
            var_found = []
            var_not_found = []
            for x in var_to_regress:
                if isinstance(x, str) and x in adata.obs.keys():
                    var = adata.obs[x].to_numpy()[:, None]
                    var = (var - var.mean()) / var.std()
                    if self.var_to_regress is None:
                        self.var_to_regress = var
                    else:
                        self.var_to_regress = np.concatenate((self.var_to_regress, var), axis=1)
                    var_found.append(x)
                else:
                    var_not_found.append(x)
            print(f'Found {var_found} in obs. Regressing them out.')
            if len(var_not_found) > 0:
                raise ValueError(f'Warning: Variables {var_not_found} not found. Please check the obs keys and try again.')
        if self.var_to_regress is not None:
            self.var_to_regress = torch.tensor(self.var_to_regress, dtype=torch.float, device=self.device)

    def set_lr(self, adata, adata_atac, learning_rate):
        if learning_rate is None:
            if self.enable_cvae:
                p = 0
                for i in range(self.n_batch):
                    idx = self.batch_ == i
                    batch_gene = self.batch_hvg_genes_[i]
                    pi = np.sum(adata.layers['Mu'][np.ix_(idx, batch_gene)] > 0) + np.sum(adata.layers['Ms'][np.ix_(idx, batch_gene)] > 0)
                    if i not in self.rna_only_idx and not self.config['rna_only']:
                        pi += np.sum(adata_atac.layers['Mc' if not self.config['split_enhancer'] else 'Mp'][np.ix_(idx, batch_gene)] > 0)
                    pi = pi / (np.sum(idx) * np.sum(batch_gene) * (2 if i in self.rna_only_idx or self.config['rna_only'] else 3))
                    print(f'Batch {i} sparsity: {1-pi:.3f}')
                    p += pi
                p = p / self.n_batch
            else:
                p = np.sum(adata.layers['Mu'] > 0) + (np.sum(adata.layers['Ms'] > 0))
                if not self.config['rna_only']:
                    p += np.sum(adata_atac.layers['Mc' if not self.config['split_enhancer'] else 'Mp'] > 0)
                p = p / (adata.n_obs * adata.n_vars * (2 if self.config['rna_only'] else 3))
            self.config["learning_rate"] = 10**(p-4)
            print(f'Learning rate set to {self.config["learning_rate"]*10000:.1f}e-4 based on data sparsity.')
        else:
            self.config["learning_rate"] = learning_rate
        self.config["learning_rate_post"] = self.config["learning_rate"]
        self.config["learning_rate_ode"] = self.config["learning_rate"] * 8 * (self.config["k_alt"]+1)
        if self.config["train_x0"] and not self.config["four_basis"]:
            self.config["learning_rate_ode"] = self.config["learning_rate_ode"] / 2
        if self.config["early_stop_thred"] is None:
            if self.enable_cvae:
                self.config["early_stop_thred"] = np.sum(self.batch_hvg_genes_) * 1e-3
            else:
                self.config["early_stop_thred"] = adata.n_vars / 2 * 1e-3
            if len(self.rna_only_idx) > 0:
                scaling_factor = max(-self.n_batch / len(self.rna_only_idx) / 3 + 2, 2/3)
                self.config["early_stop_thred"] = self.config["early_stop_thred"] * scaling_factor
            print(f"Early stop threshold set to {self.config['early_stop_thred']:.1f}.")

    def get_prior(self, adata):
        if self.config['tprior'] is None:
            print("Using Gaussian Prior.")
            if self.config['t_network']:
                self.p_t = torch.stack([torch.zeros(adata.n_obs, self.config['dim_z']),
                                        torch.ones(adata.n_obs, self.config['dim_z'])*self.config["std_t_scaling"]]).float().to(self.device)
            else:
                self.p_t = torch.stack([torch.ones(adata.n_obs, 1)*self.config['tmax']*0.5,
                                        torch.ones(adata.n_obs, 1)*self.config['tmax']*self.config["std_t_scaling"]]).float().to(self.device)
        else:
            print('Using informative time prior.')
            t = adata.obs[self.config['tprior']].to_numpy()
            t = t/t.max()*self.config['tmax']
            t_cap = np.sort(np.unique(t))
            std_t = np.zeros((len(t)))
            std_t[t == t_cap[0]] = (t_cap[1] - t_cap[0])*(0.5+0.5*self.config["std_t_scaling"])
            for i in range(1, len(t_cap)-1):
                std_t[t == t_cap[i]] = 0.5*(t_cap[i] - t_cap[i-1])*(0.5+0.5*self.config["std_t_scaling"]) + 0.5*(t_cap[i+1] - t_cap[i])*(0.5+0.5*self.config["std_t_scaling"])
            std_t[t == t_cap[-1]] = (t_cap[-1] - t_cap[-2])*(0.5+0.5*self.config["std_t_scaling"])
            self.p_t = torch.stack([torch.tensor(t).view(-1, 1), torch.tensor(std_t).view(-1, 1)]).float().to(self.device)

        if not self.config['t_network']:
            self.config['kl_t'] = self.config['kl_t'] * self.config['dim_z']

        self.p_z = torch.stack([torch.zeros(adata.n_obs, self.config['dim_z']),
                                torch.ones(adata.n_obs, self.config['dim_z'])*self.config["std_z_prior"]]).float().to(self.device)

    def update_config(self, config):
        for key in config:
            if key in self.config:
                self.config[key] = config[key]
            else:
                self.config[key] = config[key]
                logger.warning(f"Unknown hyperparameter: {key}")

    def plot_initial(self, gene_plot, figure_path="figures", embed=None):
        gind, gene_plot = get_gene_index(self.adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        condition = self.onehot[self.train_idx] if self.enable_cvae else None
        _, _, _, _, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s = self.decoder.reparameterize(condition, False, False, numpy=True)

        t = self.decoder.t_[:, gind]
        c = self.adata_atac.layers['Mc' if not self.config['split_enhancer'] else 'Mp'][self.train_idx, :][:, gind]
        u = self.adata.layers['Mu'][self.train_idx, :][:, gind]
        s = self.adata.layers['Ms'][self.train_idx, :][:, gind]
        chat = self.decoder.cpred_[:, gind]
        uhat = self.decoder.upred_[:, gind]
        shat = self.decoder.spred_[:, gind]

        if embed is not None and f'X_{embed}' in self.adata.obsm:
            Xembed = self.adata.obsm[f"X_{embed}"]
            Xembed_train = Xembed[self.train_idx]
            plot_time(np.quantile(self.decoder.t_, 0.5, 1), Xembed_train, save=f"{figure_path}/time-init.png")
        for i in range(len(gind)):
            idx = gind[i]
            scaling_c_ = scaling_c[:, idx] if self.enable_cvae else scaling_c[idx]
            scaling_u_ = scaling_u[:, idx] if self.enable_cvae else scaling_u[idx]
            scaling_s_ = scaling_s[:, idx] if self.enable_cvae else scaling_s[idx]
            offset_c_ = offset_c[:, idx] if self.enable_cvae else offset_c[idx]
            offset_u_ = offset_u[:, idx] if self.enable_cvae else offset_u[idx]
            offset_s_ = offset_s[:, idx] if self.enable_cvae else offset_s[idx]

            plot_sig(t[:, i].squeeze(),
                     c[:, i],
                     u[:, i],
                     s[:, i],
                     chat[:, i],
                     uhat[:, i],
                     shat[:, i],
                     self.cell_labels_raw[self.train_idx],
                     self.cell_type_colors,
                     gene_plot[i],
                     save=f"{figure_path}/sig-{gene_plot[i].replace('.', '-')}-init.png",
                     sparsify=self.config['sparsify'])

            plot_sig(t[:, i].squeeze(),
                     (c[:, i] - offset_c_) / scaling_c_,
                     (u[:, i] - offset_u_) / scaling_u_,
                     (s[:, i] - offset_s_) / scaling_s_,
                     (chat[:, i] - offset_c_) / scaling_c_,
                     (uhat[:, i] - offset_u_) / scaling_u_,
                     (shat[:, i] - offset_s_) / scaling_s_,
                     self.cell_labels_raw[self.train_idx],
                     self.cell_type_colors,
                     gene_plot[i],
                     save=f"{figure_path}/sigscaled-{gene_plot[i].replace('.', '-')}-init.png",
                     sparsify=self.config['sparsify'])

            plot_phase(c[:, i],
                       u[:, i],
                       s[:, i],
                       chat[:, i],
                       uhat[:, i],
                       shat[:, i],
                       gene_plot[i],
                       'cu',
                       None,
                       None,
                       self.cell_labels_raw[self.train_idx],
                       self.cell_type_colors,
                       save=f"{figure_path}/phase-{gene_plot[i].replace('.', '-')}-init-cu.png")

            plot_phase(c[:, i],
                       u[:, i],
                       s[:, i],
                       chat[:, i],
                       uhat[:, i],
                       shat[:, i],
                       gene_plot[i],
                       'us',
                       None,
                       None,
                       self.cell_labels_raw[self.train_idx],
                       self.cell_type_colors,
                       save=f"{figure_path}/phase-{gene_plot[i].replace('.', '-')}-init-us.png")

            plot_phase((c[:, i] - offset_c_) / scaling_c_,
                       (u[:, i] - offset_u_) / scaling_u_,
                       (s[:, i] - offset_s_) / scaling_s_,
                       (chat[:, i] - offset_c_) / scaling_c_,
                       (uhat[:, i] - offset_u_) / scaling_u_,
                       (shat[:, i] - offset_s_) / scaling_s_,
                       gene_plot[i],
                       'cu',
                       None,
                       None,
                       self.cell_labels_raw[self.train_idx],
                       self.cell_type_colors,
                       save=f"{figure_path}/phasescaled-{gene_plot[i].replace('.', '-')}-init-cu.png")

            plot_phase((c[:, i] - offset_c_) / scaling_c_,
                       (u[:, i] - offset_u_) / scaling_u_,
                       (s[:, i] / scaling_s_) - offset_s_,
                       (chat[:, i] - offset_c_) / scaling_c_,
                       (uhat[:, i] - offset_u_) / scaling_u_,
                       (shat[:, i] - offset_s_) / scaling_s_,
                       gene_plot[i],
                       'us',
                       None,
                       None,
                       self.cell_labels_raw[self.train_idx],
                       self.cell_type_colors,
                       save=f"{figure_path}/phasescaled-{gene_plot[i].replace('.', '-')}-init-us.png")

    def prepare_dataset(self, c=None, u=None, s=None, only_full=False):
        """Prepare dataset for training and testing

        Args:
            c (:class:`numpy.ndarray`, optional):
                Chromatin accessibility values (nxg). Defaults to None.
            u (:class:`numpy.ndarray`, optional):
                Unspliced RNA values (nxg). Defaults to None.
            s (:class:`numpy.ndarray`, optional):
                Spliced RNA values (nxg). Defaults to None.
            only_full (bool, optional):
                Whether to return only the full dataset, or datasets for training set and test set as well. Defaults to False.

        Returns:
            Tuple[:class:`SCData`, :class:`SCData`, :class:`SCData`]:
                if not only_full:
                    - Full dataset containing all cells
                    - Training dataset
                    - Test dataset
            :class:`SCData`:
                if only_full: Full dataset
        """
        chrom_key = 'Mc' if not self.config['split_enhancer'] else 'Mp'
        if c is None:
            c = self.adata_atac.layers[chrom_key].toarray() if issparse(self.adata_atac.layers[chrom_key]) else self.adata_atac.layers[chrom_key]
        if u is None:
            u = self.adata.layers['Mu'].toarray() if issparse(self.adata.layers['Mu']) else self.adata.layers['Mu']
        if s is None:
            s = self.adata.layers['Ms'].toarray() if issparse(self.adata.layers['Ms']) else self.adata.layers['Ms']
        X = np.concatenate((c, u, s), 1).astype(float)

        device = 'cpu' if self.config['vram_constrained'] else self.device
        if self.config['split_enhancer']:
            E = self.adata_atac.obsm['Me'].toarray() if issparse(self.adata_atac.obsm['Me']) else self.adata_atac.obsm['Me']
            E = E.astype(float)
        if not self.config['split_enhancer']:
            full_set = SCData(X, device=device)
            if not only_full:
                train_set = SCData(X[self.train_idx], device=device)
                test_set = None
                if len(self.test_idx) > 0:
                    test_set = SCData(X[self.test_idx], device=device)
        else:
            if not only_full:
                full_set = SCDataE(X, E, device=device)
                train_set = SCDataE(X[self.train_idx], E[self.train_idx], device=device)
                test_set = None
                if len(self.test_idx) > 0:
                    test_set = SCDataE(X[self.test_idx], E[self.test_idx], device=device)
        if not only_full:
            return full_set, train_set, test_set
        else:
            return full_set

    def forward(self, data_in, data_in_e=None, c0=None, u0=None, s0=None, t0=None, t1=None, condition=None, regressor=None, t=None, z=None, sample=True, return_velocity=False):
        if self.config['unit_scale']:
            sigma_c = self.decoder.sigma_c
            sigma_u = self.decoder.sigma_u
            sigma_s = self.decoder.sigma_s
            data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//3]/sigma_c,
                                       data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]/sigma_u,
                                       data_in[:, data_in.shape[1]//3*2:]/sigma_s), 1)
        else:
            _, _, _, _, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s = self.decoder.reparameterize(condition, False, False)
            data_in_scale = torch.cat(((data_in[:, :data_in.shape[1]//3]-offset_c)/scaling_c,
                                       (data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]-offset_u)/scaling_u,
                                       (data_in[:, data_in.shape[1]//3*2:]-offset_s)/scaling_s), 1)
        mu_t, std_t, mu_z, std_z, mu_e, std_e = self.encoder.forward(data_in_scale, data_in_e, condition, regressor)

        if t is None or z is None:
            if sample and not self.use_knn:
                t = reparameterize(mu_t, std_t)
                z = reparameterize(mu_z, std_z)
            else:
                t = mu_t
                z = mu_z
        if sample and not self.use_knn:
            if self.config['split_enhancer']:
                ze = reparameterize(mu_e, std_e)
            else:
                ze = None
        else:
            ze = mu_e

        if t1 is not None:
            chat, uhat, shat, ehat, t_, kc, rho, vc, vu, vs = self.decoder.forward(t,
                                                                                   z,
                                                                                   ze,
                                                                                   c0,
                                                                                   u0,
                                                                                   s0,
                                                                                   t0,
                                                                                   condition,
                                                                                   regressor,
                                                                                   neg_slope=self.config["neg_slope"],
                                                                                   sample=sample,
                                                                                   return_velocity=True)
            chat_fw, uhat_fw, shat_fw, _, _, kc, rho, vc_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                                                 z,
                                                                                                 ze,
                                                                                                 chat,
                                                                                                 uhat,
                                                                                                 shat,
                                                                                                 t_,
                                                                                                 condition,
                                                                                                 regressor,
                                                                                                 neg_slope=self.config["neg_slope"],
                                                                                                 sample=sample,
                                                                                                 return_velocity=True)
        else:
            out = self.decoder.forward(t,
                                       z,
                                       ze,
                                       c0,
                                       u0,
                                       s0,
                                       t0,
                                       condition,
                                       regressor,
                                       neg_slope=self.config["neg_slope"],
                                       sample=sample,
                                       four_basis=self.config["four_basis"] if not self.use_knn else False,
                                       return_velocity=return_velocity)
            if return_velocity:
                chat, uhat, shat, ehat, t_, kc, rho, vc, vu, vs = out
                chat_fw, uhat_fw, shat_fw, vc_fw, vu_fw, vs_fw = None, None, None, None, None, None
            else:
                chat, uhat, shat, ehat, t_, kc, rho = out
                chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = None, None, None, None, None, None, None, None, None
        return mu_t, std_t, mu_z, std_z, mu_e, std_e, chat, uhat, shat, ehat, t_, kc, rho, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw

    def vae_risk(self,
                 q_tx,
                 p_t,
                 q_zx,
                 p_z,
                 q_ex,
                 c,
                 u,
                 s,
                 e,
                 chat,
                 uhat,
                 shat,
                 ehat,
                 vc=None,
                 vu=None,
                 vs=None,
                 c0=None,
                 u0=None,
                 s0=None,
                 chat_fw=None,
                 uhat_fw=None,
                 shat_fw=None,
                 vc_fw=None,
                 vu_fw=None,
                 vs_fw=None,
                 c1=None,
                 u1=None,
                 s1=None,
                 s_knn=None,
                 condition=None,
                 sample=True):
        kldt = kl_gaussian(q_tx[0], q_tx[1], p_t[0], p_t[1])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])

        sigma_c = self.decoder.sigma_c
        sigma_u = self.decoder.sigma_u
        sigma_s = self.decoder.sigma_s

        if uhat.ndim == 3:
            logp = 0.5*((c.unsqueeze(1) - chat)/sigma_c*self.config['weight_c']).pow(2)
            logp += 0.5*((u.unsqueeze(1) - uhat)/sigma_u).pow(2)
            logp += 0.5*((s.unsqueeze(1) - shat)/sigma_s).pow(2)
            logp += torch.log(sigma_c)
            logp += torch.log(sigma_u)
            logp += torch.log(sigma_s*2*np.pi)
            logp = self.clip_fn(logp)
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
            logp = torch.sum(pw*logp, 1)
        else:
            logp = 0.5*((c - chat)/sigma_c*self.config['weight_c']).pow(2)
            logp += 0.5*((u - uhat)/sigma_u).pow(2)
            logp += 0.5*((s - shat)/sigma_s).pow(2)
            logp += torch.log(sigma_c)
            logp += torch.log(sigma_u)
            logp += torch.log(sigma_s*2*np.pi)
            logp = self.clip_fn(logp)

        if chat_fw is not None and uhat_fw is not None and shat_fw is not None:
            logp += 0.5*((c1 - chat_fw)/sigma_c*self.config['weight_c']).pow(2)
            logp += 0.5*((u1 - uhat_fw)/sigma_u).pow(2)
            logp += 0.5*((s1 - shat_fw)/sigma_s).pow(2)

        w_hvg = None
        if condition is not None:
            w_hvg = torch.mm(condition, self.batch_hvg_genes)
            logp = w_hvg * logp + (~w_hvg.bool()).int() * logp * self.config['weight_soft_batch']

        err_rec_n = logp.sum(1)
        if condition is not None:
            err_rec_n = err_rec_n / (condition * self.batch_count_balance).sum(1)
        err_rec = err_rec_n.mean()

        loss = err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz

        if self.config['split_enhancer']:
            err_rec_e = ((e - ehat)*self.config['weight_c']).pow(2)
            klde = kl_gaussian(q_ex[0], q_ex[1], p_z[0], p_z[1])
            loss += err_rec_e.sum(1).mean() + self.config["kl_z"]*klde
        else:
            err_rec_e, klde = 0, 0

        self.global_counter += 1

        if self.config['log_params']:
            params_eval = ['alpha', 'beta', 'gamma'] if self.config['rna_only'] else ['alpha_c', 'alpha', 'beta', 'gamma']
            for x in params_eval:
                loss += self.config['2_norm'] * torch.linalg.norm(self.decoder.get_param(x))

        if condition is not None:
            params_eval = ['alpha', 'beta', 'gamma'] if self.config['rna_only'] else ['alpha_c', 'alpha', 'beta', 'gamma']
            for x in params_eval:
                for i in range(self.n_batch):
                    if i != self.ref_batch:
                        loss += self.config['reg_param_batch'] * (self.decoder.get_param(x, idx=((i, 0) if self.config['is_full_vb'] else i)) -
                                                                  self.decoder.get_param(x, idx=((self.ref_batch, 0) if self.config['is_full_vb'] else self.ref_batch))).pow(2).mean()

        if not self.use_knn and self.config["four_basis"]:
            kldw = elbo_collapsed_categorical(self.decoder.logit_pw, self.alpha_w, 4, self.decoder.scaling_u.shape[0])
            loss += self.config["kl_w"]*kldw

        if self.config["is_full_vb"]:
            kld_params = None
            params_eval = ['alpha', 'beta', 'gamma'] if self.config['rna_only'] else ['alpha_c', 'alpha', 'beta', 'gamma']
            for i, x in enumerate(params_eval):
                if self.enable_cvae:
                    for j in range(self.n_batch):
                        if j in self.rna_only_idx and x == 'alpha_c':
                            continue
                        if kld_params is None:
                            kld_params = kl_gaussian(self.decoder.get_param(x, idx=(j, 0)).log().view(1, -1), self.decoder.get_param(x, idx=(j, 1)).view(1, -1), self.p_params[i][0], self.p_params[i][1])
                        else:
                            kld_params += kl_gaussian(self.decoder.get_param(x, idx=(j, 0)).log().view(1, -1), self.decoder.get_param(x, idx=(j, 1)).view(1, -1), self.p_params[i][0], self.p_params[i][1])

                else:
                    if kld_params is None:
                        kld_params = kl_gaussian(self.decoder.get_param(x, idx=0).log().view(1, -1), self.decoder.get_param(x, idx=1).view(1, -1), self.p_params[i][0], self.p_params[i][1])
                    else:
                        kld_params += kl_gaussian(self.decoder.get_param(x, idx=0).log().view(1, -1), self.decoder.get_param(x, idx=1).view(1, -1), self.p_params[i][0], self.p_params[i][1])
            kld_params /= u.shape[0]
            loss = loss + self.config["kl_param"]*kld_params

        if self.use_knn and (self.config["velocity_continuity"] or self.config["velocity_correlation"]):
            _, _, _, _, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s = self.decoder.reparameterize(condition, False, False, numpy=True)
            if self.config["velocity_continuity"]:
                loss_v1 = loss_vel((c0-offset_c)/scaling_c, (chat-offset_c)/scaling_c, vc,
                                   (u0-offset_u)/scaling_u, (uhat-offset_u)/scaling_u, vu,
                                   (s0-offset_s)/scaling_s, (shat-offset_s)/scaling_s, vs,
                                   self.config['rna_only'], self.rna_only_idx, condition, w_hvg=w_hvg)
                loss_v2 = loss_vel((chat-offset_c)/scaling_c, (chat_fw-offset_c)/scaling_c, vc_fw,
                                   (uhat-offset_u)/scaling_u, (uhat_fw-offset_u)/scaling_u, vu_fw,
                                   (shat-offset_s)/scaling_s, (shat_fw-offset_s)/scaling_s, vs_fw,
                                   self.config['rna_only'], self.rna_only_idx, condition, w_hvg=w_hvg)
                forward_loss = loss_v1 + loss_v2
                if condition is not None:
                    forward_loss = forward_loss / (condition * self.batch_count_balance).sum(1)
                loss = loss - self.config["reg_forward"] * forward_loss.mean()
            if self.config["velocity_correlation"]:
                correlation_loss = pearson(vs, (uhat-offset_u)/scaling_u, mask=w_hvg) - pearson(vs, (shat-offset_s)/scaling_s, mask=w_hvg)
                if condition is not None:
                    correlation_loss = correlation_loss / (condition * self.batch_count_balance).sum(1)
                loss = loss - self.config["reg_corr"] * correlation_loss.mean()

        if self.reg_velocity and s_knn is not None:
            _, _, _, _, _, scaling_u, scaling_s, _, offset_u, offset_s = self.decoder.reparameterize(condition, False, False, numpy=True)
            _, _, beta, gamma, _, _, _ = self.decoder.reparameterize(condition, sample)
            cos_sim = cosine_similarity((uhat-offset_u)/scaling_u, (shat-offset_s)/scaling_s, beta, gamma, s_knn, w_hvg=w_hvg)
            loss = loss - self.config["reg_cos"]*cos_sim

        return loss, err_rec, self.config["kl_t"]*kldt, self.config["kl_z"]*kldz, err_rec_e, self.config["kl_z"]*klde

    def pred_all(self, dataset, mode='test', output=None, gene_idx=None, batch=None, var_to_regress=None, encoder_only=False):
        if output is None:
            output = ["chat", "uhat", "shat", "t", "z", "kc", "rho"]
            if self.config['split_enhancer']:
                output.append('e')
        N, G = dataset.data.shape[0], dataset.data.shape[1]//3
        if gene_idx is None:
            gene_idx = np.array(range(G))
        elbo = 0
        rec = 0
        klt = 0
        klz = 0
        rec_e = 0
        kle = 0
        save_chat_fw = "chat_fw" in output and self.use_knn and self.config["velocity_continuity"]
        save_uhat_fw = "uhat_fw" in output and self.use_knn and self.config["velocity_continuity"]
        save_shat_fw = "shat_fw" in output and self.use_knn and self.config["velocity_continuity"]
        if var_to_regress is None:
            var_to_regress = self.var_to_regress
        if "chat" in output:
            chat_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
        if save_chat_fw:
            chat_fw_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
        if "uhat" in output:
            uhat_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
        if save_uhat_fw:
            uhat_fw_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
        if "shat" in output:
            shat_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
        if save_shat_fw:
            shat_fw_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
        if "t" in output:
            if self.config['t_network']:
                mu_t_out = np.zeros((N, self.config['dim_z']), dtype=np.float32)
                std_t_out = np.zeros((N, self.config['dim_z']), dtype=np.float32)
            else:
                mu_t_out = np.zeros((N, 1), dtype=np.float32)
                std_t_out = np.zeros((N, 1), dtype=np.float32)
            time_out = np.zeros((N))
        if "z" in output:
            mu_z_out = np.zeros((N, self.config['dim_z']), dtype=np.float32)
            std_z_out = np.zeros((N, self.config['dim_z']), dtype=np.float32)
        if "e" in output:
            mu_e_out = np.zeros((N, self.config['dim_z']), dtype=np.float32)
            std_e_out = np.zeros((N, self.config['dim_z']), dtype=np.float32)
        if "kc" in output:
            kc_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
        if "rho" in output:
            rho_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
        if "v" in output:
            vc_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
            vu_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
            vs_res = np.zeros((N, len(gene_idx)), dtype=np.float32)
        with torch.no_grad():
            B = min(N//5, 5000)
            Nb = N // B
            if Nb*B < N:
                Nb += 1

            w_hard = F.one_hot(torch.argmax(self.decoder.logit_pw, 1), num_classes=4).T
            for n in range(Nb):
                i = n*B
                j = min([(n+1)*B, N])
                data_in = dataset.data[i:j] if not self.config['vram_constrained'] else dataset.data[i:j].to(self.device)
                if self.config['split_enhancer']:
                    data_in_e = dataset.data_e[i:j] if not self.config['vram_constrained'] else dataset.data_e[i:j].to(self.device)
                else:
                    data_in_e = None
                if mode == "test":
                    idx = self.test_idx[i:j]
                elif mode == "train":
                    idx = self.train_idx[i:j]
                else:
                    idx = torch.arange(i, j, device=self.device)

                if encoder_only:
                    c0 = torch.zeros(len(idx), G, dtype=torch.float, device=self.device) if self.use_knn else None
                    u0 = torch.zeros(len(idx), G, dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.zeros(len(idx), G, dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.zeros(len(idx), 1, dtype=torch.float, device=self.device) if self.use_knn else None
                    t1 = torch.zeros(len(idx), 1, dtype=torch.float, device=self.device) if self.use_knn and self.config['velocity_continuity'] else None
                else:
                    c0 = self.c0[idx] if self.use_knn else None
                    u0 = self.u0[idx] if self.use_knn else None
                    s0 = self.s0[idx] if self.use_knn else None
                    t0 = self.t0[idx] if self.use_knn else None
                    t1 = self.t1[idx] if self.use_knn and self.config['velocity_continuity'] else None

                if batch is None:
                    condition = self.onehot[idx] if self.enable_cvae else None
                else:
                    condition = F.one_hot(batch[idx], self.n_batch).float() if self.enable_cvae else None
                regressor = var_to_regress[idx] if var_to_regress is not None else None
                out = self.forward(data_in, data_in_e, c0, u0, s0, t0, t1, condition, regressor, sample=False, return_velocity=("v" in output))
                mu_tx, std_tx, mu_zx, std_zx, mu_ex, std_ex, chat, uhat, shat, ehat, t_, kc, rho, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = out

                if not encoder_only:
                    p_t = self.p_t[:, idx, :]
                    p_z = self.p_z[:, idx, :]
                    loss = self.vae_risk((mu_tx, std_tx),
                                         p_t,
                                         (mu_zx, std_zx),
                                         p_z,
                                         (mu_ex, std_ex),
                                         data_in[:, :G],
                                         data_in[:, G:G*2],
                                         data_in[:, G*2:],
                                         data_in_e,
                                         chat,
                                         uhat,
                                         shat,
                                         ehat,
                                         vc,
                                         vu,
                                         vs,
                                         c0,
                                         u0,
                                         s0,
                                         chat_fw,
                                         uhat_fw,
                                         shat_fw,
                                         vc_fw,
                                         vu_fw,
                                         vs_fw,
                                         self.c1[idx] if self.use_knn and self.config['velocity_continuity'] else None,
                                         self.u1[idx] if self.use_knn and self.config['velocity_continuity'] else None,
                                         self.s1[idx] if self.use_knn and self.config['velocity_continuity'] else None,
                                         self.s_knn[idx] if self.reg_velocity else None,
                                         condition,
                                         sample=False)

                    elbo = elbo - ((j-i)/N)*loss[0].detach().cpu().item()
                    rec = rec - ((j-i)/N)*loss[1].detach().cpu().item()
                    klt = klt - ((j-i)/N)*loss[2].detach().cpu().item()
                    klz = klz - ((j-i)/N)*loss[3].detach().cpu().item()
                    if self.config['split_enhancer']:
                        rec_e = rec_e - ((j-i)/N)*loss[4].detach().cpu().item()
                        kle = kle - ((j-i)/N)*loss[5].detach().cpu().item()
                if "chat" in output:
                    if chat.ndim == 3:
                        chat = torch.sum(chat*w_hard, 1)
                    chat_res[i:j] = chat[:, gene_idx].detach().cpu().numpy()
                if "uhat" in output:
                    if uhat.ndim == 3:
                        uhat = torch.sum(uhat*w_hard, 1)
                    uhat_res[i:j] = uhat[:, gene_idx].detach().cpu().numpy()
                if "shat" in output:
                    if shat.ndim == 3:
                        shat = torch.sum(shat*w_hard, 1)
                    shat_res[i:j] = shat[:, gene_idx].detach().cpu().numpy()
                if save_chat_fw:
                    chat_fw_res[i:j] = chat_fw[:, gene_idx].detach().cpu().numpy()
                if save_uhat_fw:
                    uhat_fw_res[i:j] = uhat_fw[:, gene_idx].detach().cpu().numpy()
                if save_shat_fw:
                    shat_fw_res[i:j] = shat_fw[:, gene_idx].detach().cpu().numpy()
                if "t" in output:
                    if self.config['t_network']:
                        mu_t_out[i:j] = mu_tx.detach().cpu().numpy()
                        std_t_out[i:j] = std_tx.detach().cpu().numpy()
                    else:
                        mu_t_out[i:j] = mu_tx.detach().cpu().numpy()
                        std_t_out[i:j] = std_tx.detach().cpu().numpy()
                    time_out[i:j] = t_.detach().cpu().squeeze().numpy()
                if "z" in output:
                    mu_z_out[i:j] = mu_zx.detach().cpu().numpy()
                    std_z_out[i:j] = std_zx.detach().cpu().numpy()
                if "e" in output:
                    mu_e_out[i:j] = mu_ex.detach().cpu().numpy()
                    std_e_out[i:j] = std_ex.detach().cpu().numpy()
                if "kc" in output:
                    if kc.ndim == 3:
                        kc = torch.sum(kc*w_hard, 1)
                    kc_res[i:j] = kc[:, gene_idx].detach().cpu().numpy()
                if "rho" in output:
                    if rho.ndim == 3:
                        rho = torch.sum(rho*w_hard, 1)
                    rho_res[i:j] = rho[:, gene_idx].detach().cpu().numpy()
                if "v" in output:
                    if vc.ndim == 3:
                        vc = torch.sum(vc*w_hard, 1)
                    vc_res[i:j] = vc[:, gene_idx].detach().cpu().numpy()
                    if vu.ndim == 3:
                        vu = torch.sum(vu*w_hard, 1)
                    vu_res[i:j] = vu[:, gene_idx].detach().cpu().numpy()
                    if vs.ndim == 3:
                        vs = torch.sum(vs*w_hard, 1)
                    vs_res[i:j] = vs[:, gene_idx].detach().cpu().numpy()

        out = {}
        if "chat" in output:
            out['chat'] = chat_res
        if "uhat" in output:
            out['uhat'] = uhat_res
        if "shat" in output:
            out['shat'] = shat_res
        if save_chat_fw:
            out['chat_fw'] = chat_fw_res
        if save_uhat_fw:
            out['uhat_fw'] = uhat_fw_res
        if save_shat_fw:
            out['shat_fw'] = shat_fw_res
        if "t" in output:
            out['mu_t'] = mu_t_out
            out['std_t'] = std_t_out
            out['t'] = time_out
        if "z" in output:
            out['mu_z'] = mu_z_out
            out['std_z'] = std_z_out
        if "e" in output:
            out['mu_e'] = mu_e_out
            out['std_e'] = std_e_out
        if "kc" in output:
            out['kc'] = kc_res
        if "rho" in output:
            out['rho'] = rho_res
        if "v" in output:
            out["vc"] = vc_res
            out["vu"] = vu_res
            out["vs"] = vs_res

        return out, (elbo, rec, klt, klz, rec_e, kle)

    def eval(self,
             dataset,
             Xembed,
             testid=0,
             test_mode=True,
             gind=None,
             gene_plot=None,
             plot=False,
             path='figures'):
        self.set_mode('eval')
        mode = "test" if test_mode else "train"
        out_type = ["chat", "uhat", "shat", "t", "v"]
        if self.use_knn and self.config["velocity_continuity"]:
            out_type.extend(["chat_fw", "uhat_fw", "shat_fw"])
        out, loss = self.pred_all(dataset, mode=mode, output=out_type, gene_idx=gind)
        chat, uhat, shat, t_ = out["chat"], out["uhat"], out["shat"], out["t"]

        G = dataset.data.shape[1]//3

        testid_str = str(testid)
        if testid_str.startswith('train'):
            id_num = testid_str[6:]
            testid_str = 'train-' + '0'*(10-len(testid_str)) + id_num
        elif testid_str.startswith('test'):
            id_num = testid_str[5:]
            testid_str = 'test-' + '0'*(9-len(testid_str)) + id_num

        if plot:
            cell_idx = self.test_idx if test_mode else self.train_idx
            condition = self.onehot[cell_idx] if self.enable_cvae else None
            _, _, _, _, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s = self.decoder.reparameterize(condition, False, False, numpy=True)

            if not self.use_knn:
                plot_time(t_, Xembed, save=f"{path}/time-{testid_str}.png")

            for i in range(len(gind)):
                idx = gind[i]
                scaling_c_ = scaling_c[:, idx] if self.enable_cvae else scaling_c[idx]
                scaling_u_ = scaling_u[:, idx] if self.enable_cvae else scaling_u[idx]
                scaling_s_ = scaling_s[:, idx] if self.enable_cvae else scaling_s[idx]
                offset_c_ = offset_c[:, idx] if self.enable_cvae else offset_c[idx]
                offset_u_ = offset_u[:, idx] if self.enable_cvae else offset_u[idx]
                offset_s_ = offset_s[:, idx] if self.enable_cvae else offset_s[idx]

                plot_sig(t_.squeeze(),
                         dataset.data[:, idx].cpu().numpy(),
                         dataset.data[:, idx+G].cpu().numpy(),
                         dataset.data[:, idx+G*2].cpu().numpy(),
                         chat[:, i],
                         uhat[:, i],
                         shat[:, i],
                         self.cell_labels_raw[cell_idx.detach().cpu().numpy()],
                         self.cell_type_colors,
                         gene_plot[i],
                         save=f"{path}/sig-{gene_plot[i].replace('.', '-')}-{testid_str}.png",
                         sparsify=self.config['sparsify'])

                plot_sig(t_.squeeze(),
                         (dataset.data[:, idx].cpu().numpy() - offset_c_) / scaling_c_,
                         (dataset.data[:, idx+G].cpu().numpy() - offset_u_) / scaling_u_,
                         (dataset.data[:, idx+G*2].cpu().numpy() - offset_s_) / scaling_s_,
                         (chat[:, i] - offset_c_) / scaling_c_,
                         (uhat[:, i] - offset_u_) / scaling_u_,
                         (shat[:, i] - offset_s_) / scaling_s_,
                         self.cell_labels_raw[cell_idx.detach().cpu().numpy()],
                         self.cell_type_colors,
                         gene_plot[i],
                         save=f"{path}/sigscaled-{gene_plot[i].replace('.', '-')}-{testid_str}.png",
                         sparsify=self.config['sparsify'])

                plot_vel(t_.squeeze(),
                         (chat[:, i] - offset_c_) / scaling_c_,
                         (uhat[:, i] - offset_u_) / scaling_u_,
                         (shat[:, i] - offset_s_) / scaling_s_,
                         out["vc"][:, i], out["vu"][:, i], out["vs"][:, i],
                         self.t0[cell_idx].squeeze().detach().cpu().numpy() if self.use_knn else None,
                         ((self.c0[cell_idx, idx].detach().cpu().numpy() - offset_c_) / scaling_c_) if self.use_knn else None,
                         ((self.u0[cell_idx, idx].detach().cpu().numpy() - offset_u_) / scaling_u_) if self.use_knn else None,
                         ((self.s0[cell_idx, idx].detach().cpu().numpy() - offset_s_) / scaling_s_) if self.use_knn else None,
                         cell_labels=self.cell_labels_raw[cell_idx.detach().cpu().numpy()],
                         cell_type_colors=self.cell_type_colors,
                         title=gene_plot[i],
                         save=f"{path}/vel-{gene_plot[i].replace('.', '-')}-{testid_str}.png")

                if self.use_knn and self.config['velocity_continuity']:
                    plot_sig(t_.squeeze(),
                             dataset.data[:, idx].cpu().numpy(),
                             dataset.data[:, idx+G].cpu().numpy(),
                             dataset.data[:, idx+G*2].cpu().numpy(),
                             out["chat_fw"][:, i],
                             out["uhat_fw"][:, i],
                             out["shat_fw"][:, i],
                             self.cell_labels_raw[cell_idx.detach().cpu().numpy()],
                             self.cell_type_colors,
                             gene_plot[i],
                             save=f"{path}/sig-{gene_plot[i].replace('.', '-')}-{testid_str}-bw.png",
                             sparsify=self.config['sparsify'])

                    plot_sig(t_.squeeze(),
                             (dataset.data[:, idx].cpu().numpy() - offset_c_) / scaling_c_,
                             (dataset.data[:, idx+G].cpu().numpy() - offset_u_) / scaling_u_,
                             (dataset.data[:, idx+G*2].cpu().numpy() - offset_s_) / scaling_s_,
                             (out["chat_fw"][:, i] - offset_c_) / scaling_c_,
                             (out["uhat_fw"][:, i] - offset_u_) / scaling_u_,
                             (out["shat_fw"][:, i] - offset_s_) / scaling_s_,
                             self.cell_labels_raw[cell_idx.detach().cpu().numpy()],
                             self.cell_type_colors,
                             gene_plot[i],
                             save=f"{path}/sigscaled-{gene_plot[i].replace('.', '-')}-{testid_str}-bw.png",
                             sparsify=self.config['sparsify'])
        return loss

    def save_state_dict(self, loss_test=0, reset=False):
        if reset:
            self.best_model_state = None
            self.best_counter = 0
        elif len(self.loss_test) >= self.loss_idx_start + 1:
            if loss_test <= np.min(self.loss_test[self.loss_idx_start:]):
                self.best_counter = self.counter
                self.best_model_state = (copy.deepcopy(self.encoder.state_dict()), copy.deepcopy(self.decoder.state_dict()))
                self.loss_test_color.append('blue')
            else:
                self.loss_test_color.append('red')
            self.loss_test.append(loss_test)
            self.loss_test_iter.append(self.counter)
        else:
            self.loss_test_color.append('blue')
            self.loss_test.append(loss_test)
            self.loss_test_iter.append(self.counter)

    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, optimizer3=None, scheduler=None, scheduler2=None, scheduler3=None, k=0, net='both'):
        B = len(train_loader)
        self.set_mode('train', net)
        stop_training = False

        for i, data in enumerate(train_loader):
            if self.counter % self.config["test_iter"] == (2 if self.use_knn else 1):
                elbo_test, rec_test, klt_test, klz_test, rec_e_test, kle_test = self.eval(test_set,
                                                                                          None,
                                                                                          testid=self.counter,
                                                                                          test_mode=True)

                if len(self.loss_test) > 0:
                    if self.config["early_stop_thred"] is not None and (self.loss_test[-1] + elbo_test <= self.config["early_stop_thred"]):
                        self.n_drop = self.n_drop + 1
                    else:
                        self.n_drop = 0
                self.save_state_dict(-elbo_test)
                self.rec_test.append(-rec_test)
                self.klt_test.append(-klt_test)
                self.klz_test.append(-klz_test)
                if self.config['split_enhancer']:
                    self.rec_e_test.append(-rec_e_test)
                    self.kle_test.append(-kle_test)
                self.set_mode('train', net)

                if (self.n_drop >= self.config["early_stop"]) and (self.config["early_stop"] > 0):
                    stop_training = True
                    self.counter = self.counter + 1
                    break

            optimizer.zero_grad()
            if optimizer2 is not None:
                optimizer2.zero_grad()
            if optimizer3 is not None:
                optimizer3.zero_grad()

            data_x, idx = data[0], data[1]
            if self.config['split_enhancer']:
                data_e = data[2]
            else:
                data_e = None
            cell_idx = self.train_idx[idx]

            c0 = self.c0[cell_idx] if self.use_knn else None
            u0 = self.u0[cell_idx] if self.use_knn else None
            s0 = self.s0[cell_idx] if self.use_knn else None
            t0 = self.t0[cell_idx] if self.use_knn else None
            t1 = self.t1[cell_idx] if self.use_knn and self.config['velocity_continuity'] else None
            p_t = self.p_t[:, cell_idx, :]
            p_z = self.p_z[:, cell_idx, :]
            condition = self.onehot[cell_idx] if self.enable_cvae else None
            regressor = self.var_to_regress[cell_idx] if self.var_to_regress is not None else None
            out = self.forward(data_x, data_e, c0, u0, s0, t0, t1, condition, regressor)
            mu_tx, std_tx, mu_zx, std_zx, mu_ex, std_ex, chat, uhat, shat, ehat, _, _, _, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = out

            loss = self.vae_risk((mu_tx, std_tx),
                                 p_t,
                                 (mu_zx, std_zx),
                                 p_z,
                                 (mu_ex, std_ex),
                                 data_x[:, :data_x.shape[1]//3],
                                 data_x[:, data_x.shape[1]//3:data_x.shape[1]//3*2],
                                 data_x[:, data_x.shape[1]//3*2:],
                                 data_e,
                                 chat,
                                 uhat,
                                 shat,
                                 ehat,
                                 vc,
                                 vu,
                                 vs,
                                 c0,
                                 u0,
                                 s0,
                                 chat_fw,
                                 uhat_fw,
                                 shat_fw,
                                 vc_fw,
                                 vu_fw,
                                 vs_fw,
                                 self.c1[cell_idx] if self.use_knn and self.config["velocity_continuity"] else None,
                                 self.u1[cell_idx] if self.use_knn and self.config["velocity_continuity"] else None,
                                 self.s1[cell_idx] if self.use_knn and self.config["velocity_continuity"] else None,
                                 self.s_knn[cell_idx] if self.reg_velocity else None,
                                 condition)

            loss[0].backward()
            if not self.use_knn:
                torch.nn.utils.clip_grad_value_(self.encoder.parameters(), 1e7)
            torch.nn.utils.clip_grad_value_(self.decoder.parameters(), 1e7)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if k == 0:
                if optimizer2 is not None:
                    optimizer2.step()
                    if scheduler2 is not None:
                        scheduler2.step()
                if optimizer3 is not None:
                    optimizer3.step()
                    if scheduler3 is not None:
                        scheduler3.step()
            else:
                if optimizer2 is not None and ((i+1) % (k+1) == 0 or i == B-1):
                    optimizer2.step()
                    if scheduler2 is not None:
                        scheduler2.step()
                if optimizer3 is not None and ((i) % (k+1) == 0 or i == B-1):
                    optimizer3.step()
                    if scheduler3 is not None:
                        scheduler3.step()

            self.loss_train.append(loss[0].detach().cpu().item())
            self.rec_train.append(loss[1].detach().cpu().item())
            self.klt_train.append(loss[2].detach().cpu().item())
            self.klz_train.append(loss[3].detach().cpu().item())
            if self.config['split_enhancer']:
                self.rec_e_train.append(loss[4].detach().cpu().item())
                self.kle_train.append(loss[5].detach().cpu().item())
            self.counter = self.counter + 1

        return stop_training

    def update_x0(self, dataset, save=True):
        start = time.time()
        self.set_mode('eval')
        out, _ = self.pred_all(dataset, "both")
        chat, uhat, shat, t, mu_t, z = out["chat"], out["uhat"], out["shat"], out["t"], out["mu_t"], out["mu_z"]
        kc, rho = out["kc"], out["rho"]
        if self.enable_cvae:
            batch_ = torch.full((self.adata.n_obs,), self.ref_batch, dtype=int, device=self.device)
            out, _ = self.pred_all(dataset, mode="both", batch=batch_)
            t_, mu_t_, z_ = out["t"], out["mu_t"], out["mu_z"]
        if self.x0_index is None:
            self.z = torch.tensor(z, dtype=torch.float, device=self.device)
            if save:
                key = self.config['key']
                self.adata.obs[f"{key}_time_step1"] = t
                self.adata.obsm[f"{key}_t_step1"] = mu_t
                self.adata.obsm[f"{key}_z_step1"] = z
            if self.enable_cvae and save:
                self.adata.obs[f"{key}_time_step1_ref"] = t_
                self.adata.obsm[f"{key}_t_step1_ref"] = mu_t_
                self.adata.obsm[f"{key}_z_step1_ref"] = z_

        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        train_idx = self.train_idx.detach().cpu().numpy()
        if self.enable_cvae and not self.config['rna_only'] and len(self.rna_only_idx) > 0:
            use_index = train_idx[~np.isin(self.batch_[train_idx], self.rna_only_idx)]
        else:
            use_index = train_idx

        G = dataset.data.shape[1]//3
        c, u, s = dataset.data[:, :G].cpu().numpy(), dataset.data[:, G:G*2].cpu().numpy(), dataset.data[:, G*2:].cpu().numpy()
        alpha_c, alpha, beta, gamma, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s = self.decoder.reparameterize(self.onehot, False, False, numpy=True)

        init_mask = (t <= np.quantile(t, 0.02))
        if self.enable_cvae:
            c0_init, u0_init, s0_init = pred_exp_numpy_backward((t[init_mask]-(t.min()-dt[0]))[:, None],
                                                                (c[init_mask]-offset_c[init_mask])/scaling_c[init_mask],
                                                                (u[init_mask]-offset_u[init_mask])/scaling_u[init_mask],
                                                                (s[init_mask]-offset_s[init_mask])/scaling_s[init_mask],
                                                                kc[init_mask],
                                                                alpha_c[init_mask],
                                                                rho[init_mask],
                                                                alpha[init_mask],
                                                                beta[init_mask],
                                                                gamma[init_mask])
        else:
            c0_init, u0_init, s0_init = pred_exp_numpy_backward((t[init_mask]-(t.min()-dt[0]))[:, None],
                                                                (c[init_mask]-offset_c)/scaling_c,
                                                                (u[init_mask]-offset_u)/scaling_u,
                                                                (s[init_mask]-offset_s)/scaling_s,
                                                                kc[init_mask],
                                                                alpha_c,
                                                                rho[init_mask],
                                                                alpha,
                                                                beta,
                                                                gamma)
        for x in [c0_init, u0_init, s0_init]:
            x[np.isnan(x)] = 0
            x[np.isinf(x)] = 0
        c0_init = np.mean(c0_init, 0)
        u0_init = np.mean(u0_init, 0)
        s0_init = np.mean(s0_init, 0)

        print("Cell-wise KNN estimation.")
        if self.x0_index is None:
            self.x0_index = knnx0_index(t[use_index],
                                        z[use_index],
                                        t,
                                        z,
                                        dt,
                                        self.config["n_neighbors"],
                                        bins=self.config["n_lineages"],
                                        hist_eq=False)
        if self.enable_cvae:
            c0, u0, s0, t0 = get_x0((chat[use_index]-offset_c[use_index])/scaling_c[use_index],
                                    (uhat[use_index]-offset_u[use_index])/scaling_u[use_index],
                                    (shat[use_index]-offset_s[use_index])/scaling_s[use_index],
                                    t[use_index],
                                    dt,
                                    self.x0_index,
                                    c0_init,
                                    u0_init,
                                    s0_init,
                                    t.min() - dt[0])
        else:
            c0, u0, s0, t0 = get_x0((chat[use_index]-offset_c)/scaling_c,
                                    (uhat[use_index]-offset_u)/scaling_u,
                                    (shat[use_index]-offset_s)/scaling_s,
                                    t[use_index],
                                    dt,
                                    self.x0_index,
                                    c0_init,
                                    u0_init,
                                    s0_init,
                                    t.min() - dt[0])
        self.c0 = c0
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        self.c0_ = c0 * scaling_c + offset_c
        self.u0_ = u0 * scaling_u + offset_u
        self.s0_ = s0 * scaling_s + offset_s
        self.t0_ = t0

        if np.sum(np.isnan(self.t0)) > 0:
            raise ValueError('t0 contains nan. Consider raising early stopping threshold or reducing n_epochs.')
        if np.sum(np.isnan(self.c0)) > 0 or np.sum(np.isnan(self.u0)) > 0 or np.sum(np.isnan(self.s0)) > 0:
            raise ValueError('c0, u0, or s0 contains nan. Consider raising early stopping threshold or reducing n_epochs.')

        if self.config["velocity_continuity"]:
            end_mask = (t >= np.quantile(t, 0.98))
            if self.enable_cvae:
                c0_end, u0_end, s0_end = pred_exp_numpy(((t.max()+dt[0])-t[end_mask])[:, None],
                                                        (c[end_mask]-offset_c[end_mask])/scaling_c[end_mask],
                                                        (u[end_mask]-offset_u[end_mask])/scaling_u[end_mask],
                                                        (s[end_mask]-offset_s[end_mask])/scaling_s[end_mask],
                                                        kc[end_mask],
                                                        alpha_c[end_mask],
                                                        rho[end_mask],
                                                        alpha[end_mask],
                                                        beta[end_mask],
                                                        gamma[end_mask])
            else:
                c0_end, u0_end, s0_end = pred_exp_numpy(((t.max()+dt[0])-t[end_mask])[:, None],
                                                        (c[end_mask]-offset_c)/scaling_c,
                                                        (u[end_mask]-offset_u)/scaling_u,
                                                        (s[end_mask]-offset_s)/scaling_s,
                                                        kc[end_mask],
                                                        alpha_c,
                                                        rho[end_mask],
                                                        alpha,
                                                        beta,
                                                        gamma)
            for x in [c0_end, u0_end, s0_end]:
                x[np.isnan(x)] = 0
                x[np.isinf(x)] = 0
            c0_end = np.mean(c0_end, 0)
            u0_end = np.mean(u0_end, 0)
            s0_end = np.mean(s0_end, 0)
            if self.x1_index is None:
                self.x1_index = knnx0_index(t[use_index],
                                            z[use_index],
                                            t,
                                            z,
                                            dt,
                                            self.config["n_neighbors"],
                                            bins=self.config["n_lineages"],
                                            forward=True,
                                            hist_eq=False)
            if self.enable_cvae:
                c1, u1, s1, t1 = get_x0((chat[use_index]-offset_c[use_index])/scaling_c[use_index],
                                        (uhat[use_index]-offset_u[use_index])/scaling_u[use_index],
                                        (shat[use_index]-offset_s[use_index])/scaling_s[use_index],
                                        t[use_index],
                                        dt,
                                        self.x1_index,
                                        c0_end,
                                        u0_end,
                                        s0_end,
                                        t.max() + dt[0])
            else:
                c1, u1, s1, t1 = get_x0((chat[use_index]-offset_c)/scaling_c,
                                        (uhat[use_index]-offset_u)/scaling_u,
                                        (shat[use_index]-offset_s)/scaling_s,
                                        t[use_index],
                                        dt,
                                        self.x1_index,
                                        c0_end,
                                        u0_end,
                                        s0_end,
                                        t.max() + dt[0])
            self.c1 = c1
            self.u1 = u1
            self.s1 = s1
            self.t1 = t1

        self.c0 = torch.tensor(self.c0, dtype=torch.float, device=self.device)
        self.u0 = torch.tensor(self.u0, dtype=torch.float, device=self.device)
        self.s0 = torch.tensor(self.s0, dtype=torch.float, device=self.device)
        self.t0 = torch.tensor(self.t0, dtype=torch.float, device=self.device)
        if self.config["velocity_continuity"]:
            self.c1 = torch.tensor(self.c1, dtype=torch.float, device=self.device)
            self.u1 = torch.tensor(self.u1, dtype=torch.float, device=self.device)
            self.s1 = torch.tensor(self.s1, dtype=torch.float, device=self.device)
            self.t1 = torch.tensor(self.t1, dtype=torch.float, device=self.device)

        print(f"Finished. Actual Time: {convert_time(time.time()-start)}")
        return t

    def update_std_noise(self, dataset):
        G = dataset.data.shape[1]//3
        out, _ = self.pred_all(dataset,
                               mode='train',
                               output=["chat", "uhat", "shat"])
        std_c = np.clip((out["chat"] - dataset.data[:, :G].cpu().numpy()).std(0), 0.01, None)
        std_u = np.clip((out["uhat"] - dataset.data[:, G:G*2].cpu().numpy()).std(0), 0.01, None)
        std_s = np.clip((out["shat"] - dataset.data[:, G*2:].cpu().numpy()).std(0), 0.01, None)
        std_c[np.isnan(std_c)] = 1
        std_u[np.isnan(std_u)] = 1
        std_s[np.isnan(std_s)] = 1
        self.decoder.register_buffer('sigma_c', torch.tensor(std_c, dtype=torch.float, device=self.device))
        self.decoder.register_buffer('sigma_u', torch.tensor(std_u, dtype=torch.float, device=self.device))
        self.decoder.register_buffer('sigma_s', torch.tensor(std_s, dtype=torch.float, device=self.device))

    def train(self, config={}, plot=False, gene_plot=[], figure_path="figures", embed="umap", n_epochs=None):
        """The high-level API for training

        Args:
            config (dict, optional):
                Contains the hyper-parameters users want to modify.
                Users can change the default using this argument. Defaults to {}.
            plot (bool, optional):
                Whether to plot intermediate results. Used for debugging. Defaults to False.
            gene_plot (list, optional):
                Genes to plot. Effective only when plot is True. Defaults to [].
            figure_path (str, optional):
                Path to the folder for saving plots. Defaults to "figures".
            embed (str, optional):
                Low dimensional embedding in adata.obsm. Used for plotting.
                The actual key storing the embedding in adata.obsm is f'X_{embed}'. Defaults to "umap".
        """
        self.update_config(config)
        if n_epochs is not None:
            self.config["n_epochs"] = n_epochs
        start = time.time()
        self.global_counter = 0

        print("-------------------------- Train a MultiVeloVAE -------------------------")
        try:
            Xembed = self.adata.obsm[f"X_{embed}"]
            Xembed_train = Xembed[self.train_idx]
            Xembed_test = Xembed[self.test_idx]
        except KeyError:
            logger.warning(f"Embedding X_{embed} not found! Set to None.")
            Xembed = np.nan*np.ones((self.adata.n_obs, 2))
            Xembed_train = Xembed[self.train_idx]
            Xembed_test = Xembed[self.test_idx]
            plot = False

        print("*********      Creating Training and Validation Datasets      *********")
        full_set, train_set, test_set = self.prepare_dataset()

        data_loader = DataLoader(train_set,
                                 batch_size=self.config["batch_size"],
                                 shuffle=True,
                                 pin_memory=self.config["vram_constrained"])

        if self.config["test_iter"] is None:
            self.config["test_iter"] = len(self.train_idx)//self.config["batch_size"]*2

        self.train_idx = torch.tensor(self.train_idx, dtype=int, device=self.device)
        self.test_idx = torch.tensor(self.test_idx, dtype=int, device=self.device)
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")
        print("*********                      Finished.                      *********")

        gind, gene_plot = get_gene_index(self.adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        print("*********                      Stage  1                       *********")
        param_nn = list(self.encoder.parameters())
        param_nn += list(self.decoder.net_rho.parameters())
        if not self.config['rna_only']:
            param_nn += list(self.decoder.net_kc.parameters())
        if self.config['t_network']:
            param_nn += list(self.decoder.net_t.parameters())
        if self.config['split_enhancer']:
            param_nn += list(self.decoder.net_e.parameters())
        param_ode = [self.decoder.alpha_c,
                     self.decoder.alpha,
                     self.decoder.beta,
                     self.decoder.gamma]
        if self.config['train_x0'] and not self.config['four_basis']:
            param_ode.extend([self.decoder.c0,
                              self.decoder.u0,
                              self.decoder.s0])
            param_ode_init = [self.decoder.c0,
                              self.decoder.u0,
                              self.decoder.s0,
                              self.decoder.alpha,
                              self.decoder.gamma]
        if self.config['train_scaling']:
            param_ode.extend([self.decoder.scaling_c,
                              self.decoder.scaling_u])
            if self.enable_cvae:
                param_ode.append(self.decoder.scaling_s)
        if self.config['train_offset'] and self.enable_cvae:
            param_ode.extend([self.decoder.offset_c,
                              self.decoder.offset_u,
                              self.decoder.offset_s])
        if self.config['four_basis']:
            param_ode.extend([self.decoder.c0_,
                              self.decoder.u0_,
                              self.decoder.s0_,
                              self.decoder.logit_pw])
        if self.config['train_ton']:
            param_ode.append(self.decoder.ton)

        optimizer = torch.optim.AdamW(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.AdamW(param_ode, lr=self.config["learning_rate_ode"], weight_decay=self.config["lambda"])
        if self.config['train_x0'] and not self.config['four_basis']:
            optimizer_ode_init = torch.optim.AdamW(param_ode_init, lr=self.config["learning_rate_ode"], weight_decay=self.config["lambda"])

        for epoch in range(self.config["n_epochs"]):
            if epoch < self.config["n_warmup"]:
                stop_training = self.train_epoch(data_loader,
                                                 test_set,
                                                 optimizer=optimizer,
                                                 k=self.config["k_alt"])
            elif self.config["train_x0"] and not self.config['four_basis'] and epoch < self.config["n_warmup2"]:
                stop_training = self.train_epoch(data_loader,
                                                 test_set,
                                                 optimizer=optimizer,
                                                 optimizer2=optimizer_ode_init,
                                                 k=self.config["k_alt"])
            else:
                train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["cosineannealing_iter"], eta_min=0)
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: float((step+1) / self.config["cosineannealing_warmup"]))
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [self.config["cosineannealing_warmup"]])
                train_scheduler_ode = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ode, T_max=self.config["cosineannealing_iter"], eta_min=0)
                warmup_scheduler_ode = torch.optim.lr_scheduler.LambdaLR(optimizer_ode, lr_lambda=lambda step: float((step+1) / self.config["cosineannealing_warmup"]))
                scheduler_ode = torch.optim.lr_scheduler.SequentialLR(optimizer_ode, [warmup_scheduler_ode, train_scheduler_ode], [self.config["cosineannealing_warmup"]])
                stop_training = self.train_epoch(data_loader,
                                                 test_set,
                                                 optimizer,
                                                 optimizer_ode,
                                                 scheduler=scheduler,
                                                 scheduler2=scheduler_ode,
                                                 k=self.config["k_alt"])

            if epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0:
                elbo_train, _, _, _, _, _ = self.eval(train_set,
                                                      Xembed_train,
                                                      testid=f"train-{epoch+1}",
                                                      test_mode=False,
                                                      gind=gind,
                                                      gene_plot=gene_plot,
                                                      plot=plot,
                                                      path=figure_path)
                self.set_mode('train')
                elbo_test = -self.loss_test[-1] if len(self.loss_test) > 0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}\t\tTotal Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********     Stage 1: Early Stop Triggered at epoch {epoch+1}.     *********")
                if self.best_model_state is not None:
                    print(f"*********     Retrieving best model from iteration {self.best_counter}.      *********")
                    self.encoder.load_state_dict(self.best_model_state[0])
                    self.decoder.load_state_dict(self.best_model_state[1])
                break

        if self.config["run_2nd_stage"]:
            print("*********                      Stage  2                       *********")
            n_stage1 = epoch+1
            count_epoch = n_stage1
            n_test1 = len(self.loss_test)
            optimizer.zero_grad()
            optimizer_ode.zero_grad()
            self.set_mode('eval', 'encoder')
            self.save_state_dict(reset=True)
            self.config['test_iter'] = self.config['test_iter'] // (3 if self.enable_cvae else 2)
            for m in self.encoder.net.modules():
                for param in m.parameters():
                    param.requires_grad = False
                    param.grad = None
            for m in [self.encoder.fc_mu_t, self.encoder.fc_std_t, self.encoder.fc_mu_z, self.encoder.fc_std_z]:
                for param in m.parameters():
                    param.requires_grad = False
                    param.grad = None
            if self.config['split_enhancer']:
                for m in self.encoder.net_e.modules():
                    for param in m.parameters():
                        param.requires_grad = False
                        param.grad = None
                for m in [self.encoder.fc_mu_e, self.encoder.fc_std_e]:
                    for param in m.parameters():
                        param.requires_grad = False
                        param.grad = None
            if self.config['t_network']:
                for m in self.decoder.net_t.modules():
                    for param in m.parameters():
                        param.requires_grad = False
                        param.grad = None
            param_post = list(self.decoder.net_rho.parameters())
            if not self.config['rna_only']:
                param_post += list(self.decoder.net_kc.parameters())
            param_ode = [self.decoder.alpha_c,
                         self.decoder.alpha,
                         self.decoder.beta,
                         self.decoder.gamma]
            optimizer_post = torch.optim.AdamW(param_post, lr=self.config["learning_rate_post"], weight_decay=self.config["lambda_post"])
            optimizer_ode = torch.optim.AdamW(param_ode, lr=self.config["learning_rate_ode"], weight_decay=self.config["lambda"])

            sigma_c_prev = self.decoder.sigma_c.detach().cpu().numpy()
            sigma_u_prev = self.decoder.sigma_u.detach().cpu().numpy()
            sigma_s_prev = self.decoder.sigma_s.detach().cpu().numpy()
            c0_prev, u0_prev, s0_prev = None, None, None
            noise_change = np.inf
            x0_change = np.inf

            train_idx = self.train_idx.detach().cpu().numpy()
            G = full_set.data.shape[1]//3
            if self.enable_cvae:
                HVG = np.sum(self.batch_hvg_genes_) / self.n_batch

            if not self.enable_cvae:
                if self.config['rna_only']:
                    var_x = train_set.data[:, G:G*2].var(0) + train_set.data[:, G*2:].var(0)
                else:
                    var_x = train_set.data[:, :G].var(0) + train_set.data[:, G:G*2].var(0) + train_set.data[:, G*2:].var(0)
            else:
                var_x = torch.ones((self.n_batch, G), dtype=torch.float, device=self.device)
                for i in range(self.n_batch):
                    idx = self.batch_[train_idx] == i
                    batch_gene = self.batch_hvg_genes_[i]
                    ix = np.ix_(idx, batch_gene)
                    if i in self.rna_only_idx:
                        var_x[i, batch_gene] = train_set.data[:, G:G*2][ix].var(0) + train_set.data[:, G*2:][ix].var(0)
                    else:
                        var_x[i, batch_gene] = train_set.data[:, :G][ix].var(0) + train_set.data[:, G:G*2][ix].var(0) + train_set.data[:, G*2:][ix].var(0)
            var_x = var_x.detach().cpu().numpy()

            for r in range(self.config['n_refine']):
                self.r = r
                self.config['early_stop_thred'] *= 0.95
                stop_training = ((x0_change < self.config['x0_change_thred']) and r >= self.config['n_refine_min']) or (x0_change < 0.01)
                if self.config['loss_std_type'] == 'deming' and noise_change > 0.001 and r < self.config['n_refine']-1:
                    self.update_std_noise(train_set)
                    stop_training = False
                if stop_training:
                    print(f"*********      Stage 2: Early Stop Triggered at round {r}.      *********")
                    break

                t = self.update_x0(full_set)
                if plot:
                    if r == 0:
                        plot_time(self.t0_.squeeze(), Xembed, save=f"{figure_path}/timet0-train-test.png")
                        plot_time(t, Xembed, save=f"{figure_path}/time-train-test.png")
                    t0_plot = self.t0_[train_idx].squeeze()
                    for i in range(len(gind)):
                        idx = gind[i]
                        c0_plot = self.c0_[train_idx, idx]
                        u0_plot = self.u0_[train_idx, idx]
                        s0_plot = self.s0_[train_idx, idx]
                        c_plot = train_set.data[:, :G].cpu().numpy()[:, idx]
                        u_plot = train_set.data[:, G:G*2].cpu().numpy()[:, idx]
                        s_plot = train_set.data[:, G*2:].cpu().numpy()[:, idx]
                        plot_sig_(t[train_idx],
                                  c_plot,
                                  u_plot,
                                  s_plot,
                                  cell_labels=self.cell_labels_raw[train_idx],
                                  cell_type_colors=self.cell_type_colors,
                                  tpred=t0_plot,
                                  cpred=c0_plot,
                                  upred=u0_plot,
                                  spred=s0_plot,
                                  type_specific=False,
                                  title=gene_plot[i],
                                  save=f"{figure_path}/sigx0-{gene_plot[i].replace('.', '-')}-round{r+1}.png")

                if r == 0:
                    self.use_knn = True
                    self.decoder.logit_pw.requires_grad = False
                    self.decoder.init_weights(reinit_t=False)

                self.n_drop = 0
                self.loss_idx_start = len(self.loss_test)
                print(f"*********             Velocity Refinement Round {r+1}             *********")

                for epoch in range(self.config["n_epochs_post"]):
                    if epoch == 0:
                        elbo_train, _, _, _, _, _ = self.eval(train_set,
                                                              Xembed_train,
                                                              testid=f"train-round{r+1}-first",
                                                              test_mode=False,
                                                              gind=gind,
                                                              gene_plot=gene_plot,
                                                              plot=plot,
                                                              path=figure_path)
                        self.set_mode('train', 'decoder')
                    if epoch < self.config["n_warmup_post"]:
                        stop_training = self.train_epoch(data_loader,
                                                         test_set,
                                                         optimizer=optimizer_post,
                                                         k=self.config["k_alt"],
                                                         net='decoder')
                    else:
                        stop_training = self.train_epoch(data_loader,
                                                         test_set,
                                                         optimizer=optimizer_post,
                                                         optimizer2=optimizer_ode,
                                                         k=self.config["k_alt"],
                                                         net='decoder')

                    if stop_training or epoch == self.config["n_epochs_post"]:
                        elbo_train, _, _, _, _, _ = self.eval(train_set,
                                                              Xembed_train,
                                                              testid=f"train-round{r+1}-last",
                                                              test_mode=False,
                                                              gind=gind,
                                                              gene_plot=gene_plot,
                                                              plot=plot,
                                                              path=figure_path)
                        self.set_mode('train', 'decoder')
                        elbo_test = -self.loss_test[-1] if len(self.loss_test) > n_test1 else -np.inf
                        print(f"Epoch {epoch+count_epoch+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}\t\tTotal Time = {convert_time(time.time()-start)}")

                    if stop_training:
                        print(f"*********     Round {r+1}: Early Stop Triggered at epoch {epoch+count_epoch+1}.     *********")
                    if stop_training or epoch == self.config["n_epochs_post"]:
                        if self.best_model_state is not None:
                            print(f"*********     Retrieving best model from iteration {self.best_counter}.      *********")
                            self.encoder.load_state_dict(self.best_model_state[0])
                            self.decoder.load_state_dict(self.best_model_state[1])
                            self.save_state_dict(reset=True)
                        break

                count_epoch += (epoch+1)
                if self.config['loss_std_type'] == 'deming':
                    sigma_c = self.decoder.sigma_c.detach().cpu().numpy()
                    sigma_u = self.decoder.sigma_u.detach().cpu().numpy()
                    sigma_s = self.decoder.sigma_s.detach().cpu().numpy()
                    norm_delta_sigma = np.sum((sigma_c-sigma_c_prev)**2 + (sigma_u-sigma_u_prev)**2 + (sigma_s-sigma_s_prev)**2)
                    norm_sigma = np.sum(sigma_c_prev**2 + sigma_u_prev**2 + sigma_s_prev**2)
                    sigma_c_prev = self.decoder.sigma_c.detach().cpu().numpy()
                    sigma_u_prev = self.decoder.sigma_u.detach().cpu().numpy()
                    sigma_s_prev = self.decoder.sigma_s.detach().cpu().numpy()
                    noise_change = norm_delta_sigma/norm_sigma
                    print(f"Change in noise variance: {noise_change:.4f}")

                if r > 0:
                    c0_cur = self.c0_[train_idx]
                    u0_cur = self.u0_[train_idx]
                    s0_cur = self.s0_[train_idx]
                    if not self.enable_cvae:
                        if self.config['rna_only']:
                            norm_delta_x0 = ((u0_cur - u0_prev)**2 + (s0_cur - s0_prev)**2).mean(0)
                        else:
                            norm_delta_x0 = ((c0_cur - c0_prev)**2 + (u0_cur - u0_prev)**2 + (s0_cur - s0_prev)**2).mean(0)
                        x0_change = np.sum(np.sqrt(norm_delta_x0 / var_x)) / G
                    else:
                        norm_delta_x0 = np.zeros((self.n_batch, G), dtype=np.float32)
                        for i in range(self.n_batch):
                            idx = self.batch_[train_idx] == i
                            batch_gene = self.batch_hvg_genes_[i]
                            ix = np.ix_(idx, batch_gene)
                            if i in self.rna_only_idx:
                                norm_delta_x0[i, batch_gene] = ((u0_cur[ix]-u0_prev[ix])**2 + (s0_cur[ix]-s0_prev[ix])**2).mean(0)
                            else:
                                norm_delta_x0[i, batch_gene] = ((c0_cur[ix]-c0_prev[ix])**2 + (u0_cur[ix]-u0_prev[ix])**2 + (s0_cur[ix]-s0_prev[ix])**2).mean(0)
                        x0_change = np.sum(np.sqrt(norm_delta_x0 / var_x)) / self.n_batch / HVG
                    print(f"Change in x0: {x0_change:.3f}")
                c0_prev = self.c0_[train_idx]
                u0_prev = self.u0_[train_idx]
                s0_prev = self.s0_[train_idx]

            optimizer_post.zero_grad()
            optimizer_ode.zero_grad()

            if plot and self.config['n_refine'] > 1:
                t0_plot = self.t0_[train_idx].squeeze()
                for i in range(len(gind)):
                    idx = gind[i]
                    c0_plot = self.c0_[train_idx, idx]
                    u0_plot = self.u0_[train_idx, idx]
                    s0_plot = self.s0_[train_idx, idx]
                    plot_sig_(t0_plot,
                              c0_plot,
                              u0_plot,
                              s0_plot,
                              cell_labels=self.cell_labels_raw[train_idx],
                              cell_type_colors=self.cell_type_colors,
                              title=gene_plot[i],
                              save=f"{figure_path}/sigx0-{gene_plot[i].replace('.', '-')}-updated.png")

        elbo_train, rec_train, klt_train, klz_train, rec_e_train, kle_train = self.eval(train_set,
                                                                                        Xembed_train,
                                                                                        testid="train-final",
                                                                                        test_mode=False,
                                                                                        gind=gind,
                                                                                        gene_plot=gene_plot,
                                                                                        plot=plot,
                                                                                        path=figure_path)
        elbo_test, rec_test, klt_test, klz_test, rec_e_test, kle_test = self.eval(test_set,
                                                                                  Xembed_test,
                                                                                  testid="test-final",
                                                                                  test_mode=True,
                                                                                  gind=gind,
                                                                                  gene_plot=gene_plot,
                                                                                  plot=plot,
                                                                                  path=figure_path)
        self.loss_train.append(-elbo_train)
        self.rec_train.append(-rec_train)
        self.klt_train.append(-klt_train)
        self.klz_train.append(-klz_train)
        if self.config['split_enhancer']:
            self.rec_e_train.append(-rec_e_train)
            self.kle_train.append(-kle_train)

        self.save_state_dict(-elbo_test)
        self.rec_test.append(-rec_test)
        self.klt_test.append(-klt_test)
        self.klz_test.append(-klz_test)
        if self.config['split_enhancer']:
            self.rec_e_test.append(-rec_e_test)
            self.kle_test.append(-kle_test)
        if plot:
            plot_train_loss_log(self.loss_train, range(1, len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            plot_train_loss_log(self.rec_train, range(1, len(self.rec_train)+1), save=f'{figure_path}/train_loss_rec_velovae.png')
            plot_train_loss_log(self.klt_train, range(1, len(self.klt_train)+1), save=f'{figure_path}/train_loss_klt_velovae.png')
            plot_train_loss_log(self.klz_train, range(1, len(self.klz_train)+1), save=f'{figure_path}/train_loss_klz_velovae.png')
            if self.config['split_enhancer']:
                plot_train_loss_log(self.rec_e_train, range(1, len(self.rec_e_train)+1), save=f'{figure_path}/train_loss_rec_e_velovae.png')
                plot_train_loss_log(self.kle_train, range(1, len(self.kle_train)+1), save=f'{figure_path}/train_loss_kle_velovae.png')
            if self.config["test_iter"] > 0:
                plot_test_loss_log(self.loss_test, self.loss_test_iter, color=self.loss_test_color, save=f'{figure_path}/test_loss_velovae.png')
                plot_test_loss_log(self.rec_test, self.loss_test_iter, color=self.loss_test_color, save=f'{figure_path}/test_loss_rec_velovae.png')
                plot_test_loss_log(self.klt_test, self.loss_test_iter, color=self.loss_test_color, save=f'{figure_path}/test_loss_klt_velovae.png')
                plot_test_loss_log(self.klz_test, self.loss_test_iter, color=self.loss_test_color, save=f'{figure_path}/test_loss_klz_velovae.png')
                if self.config['split_enhancer']:
                    plot_test_loss_log(self.rec_e_test, self.loss_test_iter, color=self.loss_test_color, save=f'{figure_path}/test_loss_rec_e_velovae.png')
                    plot_test_loss_log(self.kle_test, self.loss_test_iter, color=self.loss_test_color, save=f'{figure_path}/test_loss_kle_velovae.png')
        print(f"Final: Train ELBO = {elbo_train:.3f},\tTest ELBO = {elbo_test:.3f}")
        print(f"*********      Finished. Total Time = {convert_time(time.time()-start)}     *********")

    def test(self, dataset, batch=None, covar=None, k=1, sample=False, seed=0, out_of_sample=False):
        """Predict latent variables and modality outputs for new data

        Args:
            dataset (:class:`anndata.AnnData`):
                Input AnnData object
            config (dict, optional):
                Contains the hyper-parameters users want to modify.
                Users can change the default using this argument. Defaults to {}.
            plot (bool, optional):
                Whether to plot intermediate results. Used for debugging. Defaults to False.
            gene_plot (list, optional):
                Genes to plot. Effective only when plot is True. Defaults to [].
            cluster_key (str, optional):
                Key in adata.obs storing the cell type annotation.. Defaults to "clusters".
            figure_path (str, optional):
                Path to the folder for saving plots. Defaults to "figures".
            embed (str, optional):
                Low dimensional embedding in adata.obsm. Used for plotting.
                The actual key storing the embedding should be f'X_{embed}'. Defaults to "umap".
        """
        self.set_mode('eval')

        if not self.enable_cvae:
            out, _ = self.pred_all(dataset, mode="both", encoder_only=out_of_sample)
        else:
            if batch is not None:
                print('Using supplied batch list for latent variable computation.')
                batch = np.array([self.batch_dic[x] for x in np.array(batch)])
                batch = torch.tensor(batch, dtype=int, device=self.device)
                out, _ = self.pred_all(dataset, mode="both", batch=batch, encoder_only=out_of_sample)
            else:
                print('Using reference batch for latent variable computation.')
                batch = torch.full((dataset.data.shape[0],), self.ref_batch, dtype=int, device=self.device)
                out, _ = self.pred_all(dataset, mode="both", batch=batch, encoder_only=out_of_sample)

        c0_test = np.zeros((dataset.data.shape[0], dataset.data.shape[1]//3), dtype=np.float32)
        u0_test = np.zeros((dataset.data.shape[0], dataset.data.shape[1]//3), dtype=np.float32)
        s0_test = np.zeros((dataset.data.shape[0], dataset.data.shape[1]//3), dtype=np.float32)
        t0_test = np.zeros(dataset.data.shape[0], dtype=np.float32)
        var_to_regress = None
        if self.var_to_regress is not None:
            if covar is not None:
                var_to_regress = covar
                if var_to_regress.ndim == 1:
                    var_to_regress = var_to_regress[:, None]
                if var_to_regress.shape[1] != self.var_to_regress.shape[1]:
                    raise ValueError(f"Number of covariates must be the same as the training data ({self.var_to_regress.shape[1]}).")
            else:
                var_to_regress_self = self.var_to_regress.detach().cpu().numpy()
                var_to_regress = np.zeros((dataset.data.shape[0], self.var_to_regress.shape[1]), dtype=np.float32)

        z_mu, z_std, t_mu, t_std = out["mu_z"], out["std_z"], out["mu_t"], out["std_t"]
        if sample:
            torch.manual_seed(seed)
            z_test = reparameterize(torch.tensor(z_mu, dtype=torch.float, device=self.device), torch.tensor(z_std, dtype=torch.float, device=self.device))
            t_test = reparameterize(torch.tensor(t_mu, dtype=torch.float, device=self.device), torch.tensor(t_std, dtype=torch.float, device=self.device))
        else:
            z_test, t_test = z_mu, t_mu
            z_test, t_test = torch.tensor(z_test, dtype=torch.float, device=self.device), torch.tensor(t_test, dtype=torch.float, device=self.device)

        knn_model = NearestNeighbors(n_neighbors=k)
        knn_model.fit(self.z.detach().cpu().numpy())
        ind = knn_model.kneighbors(z_test.detach().cpu().numpy(), return_distance=False)
        c0_self = self.c0.detach().cpu().numpy()
        u0_self = self.u0.detach().cpu().numpy()
        s0_self = self.s0.detach().cpu().numpy()
        t0_self = self.t0.detach().cpu().numpy()

        for i in range(z_test.shape[0]):
            if k == 1:
                c0_test[i] = c0_self[ind[i][0]]
                u0_test[i] = u0_self[ind[i][0]]
                s0_test[i] = s0_self[ind[i][0]]
                t0_test[i] = t0_self[ind[i][0], 0]
                if var_to_regress is not None and covar is None:
                    var_to_regress[i] = var_to_regress_self[ind[i][0]]
            elif k > 1:
                c0_test[i] = np.mean(c0_self[ind[i]], 0)
                u0_test[i] = np.mean(u0_self[ind[i]], 0)
                s0_test[i] = np.mean(s0_self[ind[i]], 0)
                t0_test[i] = np.mean(t0_self[ind[i]], 0)
                if var_to_regress is not None and covar is None:
                    var_to_regress[i] = np.mean(var_to_regress_self[ind[i]], 0)
        c0_test = torch.tensor(c0_test, dtype=torch.float, device=self.device)
        u0_test = torch.tensor(u0_test, dtype=torch.float, device=self.device)
        s0_test = torch.tensor(s0_test, dtype=torch.float, device=self.device)
        t0_test = torch.tensor(t0_test[:, None], dtype=torch.float, device=self.device)
        if var_to_regress is not None:
            var_to_regress = torch.tensor(var_to_regress, dtype=torch.float, device=self.device)

        N, G = dataset.data.shape[0], dataset.data.shape[1]//3
        chat_res = np.zeros((N, G), dtype=np.float32)
        uhat_res = np.zeros((N, G), dtype=np.float32)
        shat_res = np.zeros((N, G), dtype=np.float32)
        vc_res = np.zeros((N, G), dtype=np.float32)
        vu_res = np.zeros((N, G), dtype=np.float32)
        vs_res = np.zeros((N, G), dtype=np.float32)
        time_out = np.zeros((N), dtype=np.float32)
        kc_res = np.zeros((N, G), dtype=np.float32)
        rho_res = np.zeros((N, G), dtype=np.float32)
        with torch.no_grad():
            B = min(N//5, 5000)
            Nb = N // B
            if Nb*B < N:
                Nb += 1
            for n in range(Nb):
                i = n*B
                j = min([(n+1)*B, N])
                data_in = dataset.data[i:j]
                if self.config['split_enhancer']:
                    data_in_e = dataset.data_e[i:j]
                else:
                    data_in_e = None
                idx = torch.arange(i, j, device=self.device)

                c0 = c0_test[idx]
                u0 = u0_test[idx]
                s0 = s0_test[idx]
                t0 = t0_test[idx]
                t = t_test[idx]
                z = z_test[idx]
                condition = F.one_hot(batch[idx], self.n_batch).float() if self.enable_cvae else None
                regressor = var_to_regress[idx] if var_to_regress is not None else None
                out = self.forward(data_in, data_in_e, c0, u0, s0, t0, None, condition, regressor, t=t, z=z, sample=False, return_velocity=True)
                _, _, _, _, _, _, chat, uhat, shat, _, t_, kc, rho, _, _, _, vc, vu, vs, _, _, _ = out
                chat_res[i:j] = chat.detach().cpu().numpy()
                uhat_res[i:j] = uhat.detach().cpu().numpy()
                shat_res[i:j] = shat.detach().cpu().numpy()
                vc_res[i:j] = vc.detach().cpu().numpy()
                vu_res[i:j] = vu.detach().cpu().numpy()
                vs_res[i:j] = vs.detach().cpu().numpy()
                time_out[i:j] = t_.detach().cpu().squeeze().numpy()
                kc_res[i:j] = kc.detach().cpu().numpy()
                rho_res[i:j] = rho.detach().cpu().numpy()

        return chat_res, uhat_res, shat_res, vc_res, vu_res, vs_res, z_mu, z_std, t_mu, t_std, time_out, kc_res, rho_res

    def save_model(self, file_path=None, enc_name='encoder', dec_name='decoder'):
        """Function to save trained models

        Args:
            file_path (str, optional):
                Path to save the models
            enc_name (str, optional):
                Encoder file name. Defaults to 'encoder'.
            dec_name (str, optional):
                Decoder file name. Defaults to 'decoder'.
        """
        if file_path is not None:
            os.makedirs(file_path, exist_ok=True)
        else:
            file_path = '.'
        torch.save(self.encoder.state_dict(), f"{file_path}/{enc_name}.pt")
        torch.save(self.decoder.state_dict(), f"{file_path}/{dec_name}.pt")

    def save_anndata(self, file_path=None, file_name=None):
        """Function to save variables to Anndata object

        Args:
            file_path (str, optional):
                Path to save the anndata. Defaults to None.
            file_name (str, optional):
                Name of the h5ad file to be saved. Defaults to None.
        """
        self.set_mode('eval')
        if file_path is not None:
            os.makedirs(file_path, exist_ok=True)
        else:
            file_path = '.'

        key = self.config['key']
        if self.config['is_full_vb']:
            if self.enable_cvae:
                for i in range(self.n_batch):
                    if self.config['rna_only'] or i in self.rna_only_idx:
                        self.adata.var[f"{key}_alpha_c_{i}"] = 0
                    else:
                        self.adata.var[f"{key}_alpha_c_{i}"] = self.decoder.get_param('alpha_c', idx=(i, 0)).detach().cpu().numpy()
                    self.adata.var[f"{key}_alpha_{i}"] = self.decoder.get_param('alpha', idx=(i, 0)).detach().cpu().numpy()
                    self.adata.var[f"{key}_beta_{i}"] = self.decoder.get_param('beta', idx=(i, 0)).detach().cpu().numpy()
                    self.adata.var[f"{key}_gamma_{i}"] = self.decoder.get_param('gamma', idx=(i, 0)).detach().cpu().numpy()
                    if self.config['rna_only'] or i in self.rna_only_idx:
                        self.adata.var[f"{key}_std_log_alpha_c_{i}"] = 0
                    else:
                        self.adata.var[f"{key}_std_log_alpha_c_{i}"] = self.decoder.get_param('alpha_c', idx=(i, 1)).detach().cpu().numpy()
                    self.adata.var[f"{key}_std_log_alpha_{i}"] = self.decoder.get_param('alpha', idx=(i, 1)).detach().cpu().numpy()
                    self.adata.var[f"{key}_std_log_beta_{i}"] = self.decoder.get_param('beta', idx=(i, 1)).detach().cpu().numpy()
                    self.adata.var[f"{key}_std_log_gamma_{i}"] = self.decoder.get_param('gamma', idx=(i, 1)).detach().cpu().numpy()
            else:
                if self.config['rna_only']:
                    self.adata.var[f"{key}_alpha_c"] = 0
                else:
                    self.adata.var[f"{key}_alpha_c"] = self.decoder.get_param('alpha_c', idx=0).detach().cpu().numpy()
                self.adata.var[f"{key}_alpha"] = self.decoder.get_param('alpha', idx=0).detach().cpu().numpy()
                self.adata.var[f"{key}_beta"] = self.decoder.get_param('beta', idx=0).detach().cpu().numpy()
                self.adata.var[f"{key}_gamma"] = self.decoder.get_param('gamma', idx=0).detach().cpu().numpy()
                if self.config['rna_only']:
                    self.adata.var[f"{key}_std_log_alpha_c"] = 0
                else:
                    self.adata.var[f"{key}_std_log_alpha_c"] = self.decoder.get_param('alpha_c', idx=1).detach().cpu().numpy()
                self.adata.var[f"{key}_std_log_alpha"] = self.decoder.get_param('alpha', idx=1).detach().cpu().numpy()
                self.adata.var[f"{key}_std_log_beta"] = self.decoder.get_param('beta', idx=1).detach().cpu().numpy()
                self.adata.var[f"{key}_std_log_gamma"] = self.decoder.get_param('gamma', idx=1).detach().cpu().numpy()
        else:
            if self.enable_cvae:
                for i in range(self.n_batch):
                    if self.config['rna_only'] or i in self.rna_only_idx:
                        self.adata.var[f"{key}_alpha_c_{i}"] = 0
                    else:
                        self.adata.var[f"{key}_alpha_c_{i}"] = self.decoder.get_param('alpha_c', idx=i).detach().cpu().numpy()
                    self.adata.var[f"{key}_alpha_{i}"] = self.decoder.get_param('alpha', idx=i).detach().cpu().numpy()
                    self.adata.var[f"{key}_beta_{i}"] = self.decoder.get_param('beta', idx=i).detach().cpu().numpy()
                    self.adata.var[f"{key}_gamma_{i}"] = self.decoder.get_param('gamma', idx=i).detach().cpu().numpy()
            else:
                if self.config['rna_only']:
                    self.adata.var[f"{key}_alpha_c"] = 0
                else:
                    self.adata.var[f"{key}_alpha_c"] = self.decoder.get_param('alpha_c').detach().cpu().numpy()
                self.adata.var[f"{key}_alpha"] = self.decoder.get_param('alpha').detach().cpu().numpy()
                self.adata.var[f"{key}_beta"] = self.decoder.get_param('beta').detach().cpu().numpy()
                self.adata.var[f"{key}_gamma"] = self.decoder.get_param('gamma').detach().cpu().numpy()
        if self.enable_cvae:
            for i in range(self.n_batch):
                if self.config['rna_only'] or i in self.rna_only_idx:
                    self.adata.var[f"{key}_scaling_c_{i}"] = 1
                else:
                    self.adata.var[f"{key}_scaling_c_{i}"] = self.decoder.get_param('scaling_c', idx=i).detach().cpu().numpy()
                self.adata.var[f"{key}_scaling_u_{i}"] = self.decoder.get_param('scaling_u', idx=i).detach().cpu().numpy()
                if i == self.ref_batch:
                    self.adata.var[f"{key}_scaling_s_{i}"] = 1
                else:
                    self.adata.var[f"{key}_scaling_s_{i}"] = self.decoder.get_param('scaling_s', idx=i).detach().cpu().numpy()
                if self.config['rna_only'] or i in self.rna_only_idx or i == self.ref_batch:
                    self.adata.var[f"{key}_offset_c_{i}"] = 0
                else:
                    self.adata.var[f"{key}_offset_c_{i}"] = self.decoder.get_param('offset_c', enforce_positive=False, idx=i).detach().cpu().numpy()
                if i == self.ref_batch:
                    self.adata.var[f"{key}_offset_u_{i}"] = 0
                else:
                    self.adata.var[f"{key}_offset_u_{i}"] = self.decoder.get_param('offset_u', enforce_positive=False, idx=i).detach().cpu().numpy()
                if i == self.ref_batch:
                    self.adata.var[f"{key}_offset_s_{i}"] = 0
                else:
                    self.adata.var[f"{key}_offset_s_{i}"] = self.decoder.get_param('offset_s', enforce_positive=False, idx=i).detach().cpu().numpy()
        else:
            if self.config['rna_only']:
                self.adata.var[f"{key}_scaling_c"] = 1
            else:
                self.adata.var[f"{key}_scaling_c"] = self.decoder.get_param('scaling_c').detach().cpu().numpy()
            self.adata.var[f"{key}_scaling_u"] = self.decoder.get_param('scaling_u').detach().cpu().numpy()
            self.adata.var[f"{key}_scaling_s"] = self.decoder.get_param('scaling_s').detach().cpu().numpy()
            if self.config['rna_only']:
                self.adata.var[f"{key}_offset_c"] = 0
            else:
                self.adata.var[f"{key}_offset_c"] = self.decoder.get_param('offset_c', enforce_positive=False).detach().cpu().numpy()
            self.adata.var[f"{key}_offset_u"] = self.decoder.get_param('offset_u', enforce_positive=False).detach().cpu().numpy()
            self.adata.var[f"{key}_offset_s"] = self.decoder.get_param('offset_s', enforce_positive=False).detach().cpu().numpy()
        self.adata.var[f"{key}_sigma_c"] = self.decoder.sigma_c.detach().cpu().numpy()
        self.adata.var[f"{key}_sigma_u"] = self.decoder.sigma_u.detach().cpu().numpy()
        self.adata.var[f"{key}_sigma_s"] = self.decoder.sigma_s.detach().cpu().numpy()
        self.adata.var[f"{key}_mu_c"] = self.decoder.mu_c.detach().cpu().numpy()
        self.adata.var[f"{key}_mu_u"] = self.decoder.mu_u.detach().cpu().numpy()
        self.adata.var[f"{key}_mu_s"] = self.decoder.mu_s.detach().cpu().numpy()
        self.adata.var[f"{key}_ton"] = self.decoder.ton.detach().cpu().numpy()
        self.adata.varm[f"{key}_basis"] = F.softmax(self.decoder.logit_pw, 1).detach().cpu().numpy()

        full_set = self.prepare_dataset(only_full=True)
        G = full_set.data.shape[1]//3
        out, _ = self.pred_all(full_set, mode="both")
        chat, uhat, shat, t, std_t, t_, z, std_z, kc, rho = out["chat"], out["uhat"], out["shat"], out["mu_t"], out["std_t"], out["t"], out["mu_z"], out["std_z"], out["kc"], out["rho"]
        std_c = np.clip(np.nanstd(full_set.data[:, :G].cpu().numpy(), axis=0), 0.01, None)
        std_u = np.clip(np.nanstd(full_set.data[:, G:G*2].cpu().numpy(), axis=0), 0.01, None)
        std_s = np.clip(np.nanstd(full_set.data[:, G*2:].cpu().numpy(), axis=0), 0.01, None)
        diff_c = (full_set.data[:, :G].cpu().numpy() - chat) / std_c
        diff_u = (full_set.data[:, G:G*2].cpu().numpy() - uhat) / std_u
        diff_s = (full_set.data[:, G*2:].cpu().numpy() - shat) / std_s
        sigma_c = np.clip(np.nanstd(diff_c, axis=0), 1e-3, None)
        sigma_u = np.clip(np.nanstd(diff_u, axis=0), 1e-3, None)
        sigma_s = np.clip(np.nanstd(diff_s, axis=0), 1e-3, None)
        nll_c = 0.5 * np.log(2 * np.pi) + np.log(sigma_c) + 0.5 * np.nanmean(diff_c**2, axis=0) / (sigma_c**2)
        nll_u = 0.5 * np.log(2 * np.pi) + np.log(sigma_u) + 0.5 * np.nanmean(diff_u**2, axis=0) / (sigma_u**2)
        nll_s = 0.5 * np.log(2 * np.pi) + np.log(sigma_s) + 0.5 * np.nanmean(diff_s**2, axis=0) / (sigma_s**2)
        if self.config['rna_only']:
            nll = nll_u + nll_s
        else:
            nll = nll_c + nll_u + nll_s
        self.adata.var[f"{key}_likelihood"] = np.exp(-nll)
        if self.config['split_enhancer']:
            ehat, e, std_e = out["ehat"], out["mu_e"], out["std_e"]

        if self.enable_cvae:
            self.adata.layers[f"{key}_chat_batch"] = chat
            self.adata.layers[f"{key}_uhat_batch"] = uhat
            self.adata.layers[f"{key}_shat_batch"] = shat
            self.adata.obs[f"{key}_time_batch"] = t_
            self.adata.obsm[f"{key}_t_batch"] = t
            self.adata.obsm[f"{key}_std_t_batch"] = std_t
            self.adata.obsm[f"{key}_z_batch"] = z
            self.adata.obsm[f"{key}_std_z_batch"] = std_z
            self.adata.layers[f"{key}_kc_batch"] = kc
            self.adata.layers[f"{key}_rho_batch"] = rho
            if self.use_knn:
                out = self.test(full_set, covar=self.var_to_regress.detach().cpu().numpy() if self.var_to_regress is not None else None)
                chat, uhat, shat, _, _, _, z, std_z, t, std_t, t_, kc, rho = out
            else:
                batch_ = torch.full((self.adata.n_obs,), self.ref_batch, dtype=int, device=self.device)
                out, _ = self.pred_all(full_set, mode="both", batch=batch_)
                chat, uhat, shat, t, std_t, t_, z, std_z, kc, rho = out["chat"], out["uhat"], out["shat"], out["mu_t"], out["std_t"], out["t"], out["mu_z"], out["std_z"], out["kc"], out["rho"]

        self.adata.layers[f"{key}_chat"] = chat
        self.adata.layers[f"{key}_uhat"] = uhat
        self.adata.layers[f"{key}_shat"] = shat
        self.adata.obs[f"{key}_time"] = t_
        self.adata.layers[f"{key}_time"] = np.tile(self.adata.obs[f"{key}_time"].to_numpy()[:, None], self.adata.n_vars)
        self.adata.obsm[f"{key}_t"] = t
        self.adata.obsm[f"{key}_std_t"] = std_t
        self.adata.obsm[f"{key}_z"] = z
        self.adata.obsm[f"{key}_std_z"] = std_z
        self.adata.layers[f"{key}_kc"] = kc
        self.adata.layers[f"{key}_rho"] = rho
        if self.config['split_enhancer']:
            self.adata.layers[f"{key}_ehat"] = ehat
            self.adata.obsm[f"{key}_ze"] = e
            self.adata.obsm[f"{key}_std_ze"] = std_e

        self.adata.obs[f"{key}_t0"] = self.t0.detach().cpu().numpy().squeeze() if self.t0 is not None else 0
        self.adata.layers[f"{key}_c0"] = self.c0.detach().cpu().numpy() if self.c0 is not None else np.zeros_like(self.adata.layers['Mu'])
        self.adata.layers[f"{key}_u0"] = self.u0.detach().cpu().numpy() if self.u0 is not None else np.zeros_like(self.adata.layers['Mu'])
        self.adata.layers[f"{key}_s0"] = self.s0.detach().cpu().numpy() if self.s0 is not None else np.zeros_like(self.adata.layers['Mu'])

        self.adata.uns[f"{key}_train_idx"] = self.train_idx.detach().cpu().numpy()
        self.adata.uns[f"{key}_test_idx"] = self.test_idx.detach().cpu().numpy()

        print("Computing velocity.")
        velocity(self.adata, self.adata_atac, key,
                 batch_key=self.config['batch_key'],
                 ref_batch=self.ref_batch,
                 batch_hvg_key=self.config['batch_hvg_key'],
                 batch_correction=self.enable_cvae,
                 rna_only=self.config['rna_only'])

        if np.nanmax(t_) - np.nanmin(t_) < 0.1:
            logger.warning('Time shrinked to very narrow range.\nConsider reduring the number of epochs in Stage 1: n_epochs (default 2000)\n or increasing early_stop_thred, and re-train.')

        if file_name is not None:
            print("Writing anndata output to file.")
            self.adata.write_h5ad(f"{file_path}/{file_name}")
