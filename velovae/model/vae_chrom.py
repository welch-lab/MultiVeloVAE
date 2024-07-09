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
from ..plotting_chrom import plot_sig_, plot_sig, plot_vel, plot_phase, plot_time
from ..plotting_chrom import plot_train_loss_log, plot_test_loss_log
from .model_util import hist_equal, convert_time, get_gene_index
from .model_util import elbo_collapsed_categorical, assign_gene_mode_tprior
from .model_util_chrom import pred_exp, pred_exp_numpy_backward, init_params, get_ts_global, reinit_params
from .model_util_chrom import kl_gaussian, reparameterize, softplusinv, knnx0_index, get_x0
from .model_util_chrom import pearson, loss_vel, cosine_similarity, find_dirichlet_param, assign_gene_mode
from .transition_graph import encode_type
from .training_data_chrom import SCData
from .velocity_chrom import rna_velocity_vae
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self,
                 Cin,
                 dim_z,
                 dim_cond=0,
                 dim_reg=0,
                 N1=256,
                 t_network=False,
                 checkpoint=None):
        super(Encoder, self).__init__()
        self.t_network = t_network

        self.fc1 = nn.Linear(Cin+dim_cond+dim_reg, N1)
        self.bn1 = nn.BatchNorm1d(N1)
        self.dpt1 = nn.Dropout(p=0.2)
        self.net = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1)

        self.fc_mu_t = nn.Linear(N1, dim_z if t_network else 1)
        self.fc_std_t = nn.Linear(N1, dim_z if t_network else 1)
        self.spt = nn.Softplus()
        self.spt1 = nn.Softplus()

        self.fc_mu_z = nn.Linear(N1, dim_z)
        self.fc_std_z = nn.Linear(N1, dim_z)
        self.spt2 = nn.Softplus()

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

    def forward(self, data_in, condition=None, regressor=None):
        if condition is not None:
            data_in = torch.cat((data_in, condition), 1)
        if regressor is not None:
            data_in = torch.cat((data_in, regressor), 1)
        h = self.net(data_in)
        mu_tx = self.fc_mu_t(h) if self.t_network else self.spt(self.fc_mu_t(h))
        std_tx = self.spt1(self.fc_std_t(h))
        mu_zx = self.fc_mu_z(h)
        std_zx = self.spt2(self.fc_std_z(h))
        return mu_tx, std_tx, mu_zx, std_zx


class Decoder(nn.Module):
    def __init__(self,
                 adata,
                 adata_atac,
                 train_idx,
                 dim_z,
                 dim_cond=0,
                 dim_reg=0,
                 batch_idx=None,
                 ref_batch=None,
                 N1=256,
                 parallel_arch=True,
                 t_network=False,
                 full_vb=False,
                 global_std=False,
                 log_params=False,
                 rna_only=False,
                 rna_only_idx=[],
                 p=98,
                 tmax=1,
                 reinit=False,
                 init_ton_zero=False,
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
        self.parallel_arch = parallel_arch
        self.t_network = t_network
        self.is_full_vb = full_vb
        self.global_std = global_std
        self.log_params = log_params
        self.rna_only = rna_only
        self.rna_only_idx = np.array(rna_only_idx)
        self.reinit = reinit
        self.init_ton_zero = init_ton_zero
        self.init_method = init_method
        self.init_key = init_key
        self.checkpoint = checkpoint
        self.tmax = tmax
        self.construct_nn(dim_z, dim_cond, dim_reg, N1, p)

    def construct_nn(self, dim_z, dim_cond, dim_reg, N1, p):
        G = self.adata.n_vars
        self.set_shape(G, dim_cond)

        self.fc1 = nn.Linear(dim_z+dim_cond+dim_reg, N1)
        self.bn1 = nn.BatchNorm1d(N1)
        self.dpt1 = nn.Dropout(p=0.2)
        self.fc_out1 = nn.Linear(N1, G)
        self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                     self.fc_out1, nn.Sigmoid())

        self.fc2 = nn.Linear(dim_z+dim_cond+dim_reg if self.parallel_arch else G, N1)
        self.bn2 = nn.BatchNorm1d(N1)
        self.dpt2 = nn.Dropout(p=0.2)
        self.fc_out2 = nn.Linear(N1, G)
        self.net_kc = nn.Sequential(self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2,
                                    self.fc_out2, nn.Sigmoid())

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
            self.register_buffer('zero_vec', torch.empty(G))
            self.register_buffer('one_vec', torch.empty(G))

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
                self.register_buffer('one_mat', torch.empty((self.dim_cond, G)))
                self.register_buffer('zero_mat', torch.empty((self.dim_cond, G)))
            else:
                self.scaling_c = nn.Parameter(torch.empty(G))
                self.scaling_u = nn.Parameter(torch.empty(G))
                self.scaling_s = nn.Parameter(torch.empty(G))
                self.offset_c = nn.Parameter(torch.empty(G))
                self.offset_u = nn.Parameter(torch.empty(G))
                self.offset_s = nn.Parameter(torch.empty(G))

            self.load_state_dict(torch.load(self.checkpoint))
        else:
            c = self.adata_atac.layers['Mc'][self.train_idx]
            u = self.adata.layers['Mu'][self.train_idx]
            s = self.adata.layers['Ms'][self.train_idx]
            self.init_weights()
            self.init_ode(c, u, s, p)

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

    def init_ode(self, c, u, s, p):
        G = self.adata.n_vars
        print("Initializing using the steady-state and dynamical models.")
        out = init_params(c, u, s, p, fit_scaling=True, global_std=self.global_std, tmax=self.tmax, rna_only=self.rna_only)
        alpha_c, alpha, beta, gamma, scaling_c, scaling_u, toff, c0, u0, s0, sigma_c, sigma_u, sigma_s, mu_c, mu_u, mu_s, t, cpred, upred, spred = out
        scaling_s = np.ones_like(scaling_u)
        offset_c, offset_u, offset_s = np.zeros_like(scaling_u), np.zeros_like(scaling_u), np.zeros_like(scaling_u)

        if self.init_method == 'tprior':
            w = assign_gene_mode_tprior(self.adata, self.init_key, self.train_idx)
        else:
            dyn_mask = (t > self.tmax*0.01) & (np.abs(t-toff) > self.tmax*0.01)
            w = np.sum(((t < toff) & dyn_mask), 0) / (np.sum(dyn_mask, 0) + 1e-10)
            w = assign_gene_mode(self.adata, w, 'auto', 0.05, 0.1, 7)
            sigma_c = np.clip(sigma_c, 1e-3, None)
            sigma_u = np.clip(sigma_u, 1e-3, None)
            sigma_s = np.clip(sigma_s, 1e-3, None)
            sigma_c[np.isnan(sigma_c)] = 1
            sigma_u[np.isnan(sigma_u)] = 1
            sigma_s[np.isnan(sigma_s)] = 1
            sigma_c = np.clip(sigma_c, np.min(sigma_c[self.adata.var['quantile_genes']]), np.max(sigma_c[self.adata.var['quantile_genes']]))
            sigma_u = np.clip(sigma_u, np.min(sigma_u[self.adata.var['quantile_genes']]), np.max(sigma_u[self.adata.var['quantile_genes']]))
            sigma_s = np.clip(sigma_s, np.min(sigma_s[self.adata.var['quantile_genes']]), np.max(sigma_s[self.adata.var['quantile_genes']]))
            mu_c[np.isnan(mu_c)] = 0
            mu_u[np.isnan(mu_u)] = 0
            mu_s[np.isnan(mu_s)] = 0
            mu_c = np.clip(mu_c, np.min(mu_c[self.adata.var['quantile_genes']]), np.max(mu_c[self.adata.var['quantile_genes']]))
            mu_u = np.clip(mu_u, np.min(mu_u[self.adata.var['quantile_genes']]), np.max(mu_u[self.adata.var['quantile_genes']]))
            mu_s = np.clip(mu_s, np.min(mu_s[self.adata.var['quantile_genes']]), np.max(mu_s[self.adata.var['quantile_genes']]))
        print(f"Initial induction: {np.sum(w >= 0.5)}, repression: {np.sum(w < 0.5)} out of {G}.")
        self.adata.var["w_init"] = w
        logit_pw = 0.5*(np.log(w+1e-10) - np.log(1-w-1e-10))
        logit_pw = np.stack([logit_pw, -0.5*logit_pw, 0.5*logit_pw, -logit_pw], 1)
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
                toff = get_ts_global(self.t_init, u/scaling_u, s, 95)
                alpha_c, alpha, beta, gamma, ton = reinit_params(c/scaling_c, u/scaling_u, s, self.t_init, toff)

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
                alpha_c, alpha, beta, gamma, ton = reinit_params(c/scaling_c, u/scaling_u, s, self.t_init, toff)

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
                        logger.warn(f'Batch class {i} gene {j} has no valid data.')
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
                logger.warn('Warning: scaling_c invalid (nan).')
            if np.any(np.isinf(scaling_c)):
                logger.warn('Warning: scaling_c invalid (inf).')
            if np.any(np.isnan(scaling_u)):
                logger.warn('Warning: scaling_u invalid (nan).')
            if np.any(np.isinf(scaling_u)):
                logger.warn('Warning: scaling_u invalid (inf).')
            if np.any(np.isnan(scaling_s)):
                logger.warn('Warning: scaling_s invalid (nan).')
            if np.any(np.isinf(scaling_s)):
                logger.warn('Warning: scaling_s invalid (inf).')
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

        self.c0_ = nn.Parameter(torch.tensor(np.log(c0+1e-10) if self.log_params else softplusinv(c0)))
        self.u0_ = nn.Parameter(torch.tensor(np.log(u0+1e-10) if self.log_params else softplusinv(u0)))
        self.s0_ = nn.Parameter(torch.tensor(np.log(s0+1e-10) if self.log_params else softplusinv(s0)))
        self.c0 = nn.Parameter(torch.zeros_like(self.u0_))
        self.u0 = nn.Parameter(torch.zeros_like(self.u0_))
        self.s0 = nn.Parameter(torch.zeros_like(self.u0_))

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

    def _get_param(self, x, condition=None, four_basis=False, is_full_vb=False, sample=True, mask_idx=None, mask_to=1, enforce_positive=True, detach=False):
        param = self.get_param(x, enforce_positive=enforce_positive)
        if detach:
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
        return y

    def reparameterize(self, condition=None, sample=True, four_basis=False):
        alpha_c = self._get_param('alpha_c', condition, four_basis, self.is_full_vb, sample, mask_idx=self.rna_only_idx, mask_to=0)
        alpha = self._get_param('alpha', condition, four_basis, self.is_full_vb, sample)
        beta = self._get_param('beta', condition, four_basis, self.is_full_vb, sample)
        gamma = self._get_param('gamma', condition, four_basis, self.is_full_vb, sample)
        scaling_c = self._get_param('scaling_c', condition, four_basis, mask_idx=self.rna_only_idx, mask_to=1)
        scaling_u = self._get_param('scaling_u', condition, four_basis)
        scaling_s = self._get_param('scaling_s', condition, four_basis, mask_idx=self.ref_batch)
        offset_c = self._get_param('offset_c', condition, four_basis, mask_idx=np.append(self.rna_only_idx, self.ref_batch), mask_to=0, enforce_positive=False)
        offset_u = self._get_param('offset_u', condition, four_basis, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
        offset_s = self._get_param('offset_s', condition, four_basis, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)

        return alpha_c, alpha, beta, gamma, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s

    def forward(self, t, z, c0=None, u0=None, s0=None, t0=None, condition=None, regressor=None, neg_slope=0.0, sample=True, four_basis=False, return_velocity=False, use_input_time=False):
        alpha_c, alpha, beta, gamma, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s = self.reparameterize(condition, sample, four_basis)

        z_in = z
        if condition is not None:
            z_in = torch.cat((z_in, condition), 1)
        if regressor is not None:
            z_in = torch.cat((z_in, regressor), 1)
        rho = self.net_rho(z_in)
        if self.rna_only:
            kc = torch.ones_like(rho)
        else:
            kc = self.net_kc(z_in if self.parallel_arch else rho)

        t_in = t
        t_ = self.net_t(t_in) * self.tmax if self.t_network and (not use_input_time) else t
        if (c0 is None) or (u0 is None) or (s0 is None) or (t0 is None):
            if four_basis:
                zero_mat = torch.zeros_like(rho)
                kc_ = torch.stack([kc, kc, zero_mat, zero_mat], 1) if not self.rna_only else torch.stack([kc for i in range(4)], 1)
                rho_ = torch.stack([rho, zero_mat, rho, zero_mat], 1)
                c0 = torch.stack([self.zero_vec, self.zero_vec, self.c0_.exp() if self.log_params else F.softplus(self.c0_), self.c0_.exp() if self.log_params else F.softplus(self.c0_)])
                u0 = torch.stack([self.zero_vec, self.u0_.exp() if self.log_params else F.softplus(self.u0_), self.zero_vec, self.u0_.exp() if self.log_params else F.softplus(self.u0_)])
                s0 = torch.stack([self.zero_vec, self.s0_.exp() if self.log_params else F.softplus(self.s0_), self.zero_vec, self.s0_.exp() if self.log_params else F.softplus(self.s0_)])
                tau = torch.stack([F.leaky_relu(t_ - self.ton, neg_slope) for i in range(4)], 1)
            else:
                kc_ = kc
                rho_ = rho
                c0 = self.c0
                u0 = self.u0
                s0 = self.s0
                tau = F.leaky_relu(t_ - self.ton, neg_slope)

            if self.rna_only:
                c0 = torch.ones_like(c0)
            if condition is not None and len(self.rna_only_idx) > 0:
                mask = torch.zeros_like(condition)
                mask[:, self.rna_only_idx] = 1
                mask_flip = (~mask.bool()).int()
                if four_basis:
                    kc_ = torch.stack([torch.mm(condition * mask_flip, self.one_mat) for i in range(4)], 1) * kc_ + torch.stack([torch.mm(condition * mask, self.one_mat) for i in range(4)], 1)
                    c0 = c0.reshape((1, 4, -1)).tile((self.dim_cond, 1, 1))
                    c0 = torch.einsum('bi,ijm->bjm', condition * mask_flip, c0) + torch.einsum('bi,ijm->bjm', condition * mask, torch.ones_like(c0))
                else:
                    kc_ = torch.mm(condition * mask_flip, self.one_mat) * kc_ + torch.mm(condition * mask, self.one_mat)
                    c0 = c0.reshape((1, -1)).tile((self.dim_cond, 1))
                    c0 = torch.mm(condition * mask_flip, c0) + torch.mm(condition * mask, self.one_mat)

            chat, uhat, shat = pred_exp(tau,
                                        c0,
                                        u0,
                                        s0,
                                        kc_,
                                        alpha_c,
                                        rho_,
                                        alpha,
                                        beta,
                                        gamma)

        else:
            tau = F.leaky_relu(t_ - t0, neg_slope)
            chat, uhat, shat = pred_exp(tau,
                                        (c0-offset_c)/scaling_c,
                                        (u0-offset_u)/scaling_u,
                                        (s0-offset_s)/scaling_s,
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
            if ((c0 is None) or (u0 is None) or (s0 is None) or (t0 is None)) and four_basis:
                zero_mat = torch.zeros_like(rho)
                kc_ = torch.stack([kc, kc, zero_mat, zero_mat], 1)
                rho_ = torch.stack([rho, zero_mat, rho, zero_mat], 1)
            else:
                kc_ = kc
                rho_ = rho
            vc = (kc_ * alpha_c - alpha_c * chat) * scaling_c
            vu = (rho_ * alpha * chat - beta * uhat) * scaling_u
            vs = (beta * uhat - gamma * shat) * scaling_s

            chat = chat * scaling_c + offset_c
            uhat = uhat * scaling_u + offset_u
            shat = shat * scaling_s + offset_s
            return chat, uhat, shat, t_, kc, rho, vc, vu, vs
        else:
            chat = chat * scaling_c + offset_c
            uhat = uhat * scaling_u + offset_u
            shat = shat * scaling_s + offset_s
            return chat, uhat, shat, t_, kc, rho


class VAEChrom():
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
                 batch_size=256,
                 full_vb=False,
                 parallel_arch=True,
                 t_network=True,
                 velocity_continuity=False,
                 velocity_correlation=False,
                 four_basis=False,
                 run_2nd_stage=True,
                 refine_velocity=True,
                 tmax=1,
                 reinit_params=False,
                 init_method='steady',
                 init_key=None,
                 tprior=None,
                 init_ton_zero=True,
                 unit_scale=True,
                 deming_std=False,
                 log_params=False,
                 rna_only=False,
                 rna_only_idx=[],
                 learning_rate=None,
                 early_stop_thred=None,
                 checkpoints=[None, None],
                 plot_init=False,
                 gene_plot=[],
                 cluster_key='clusters',
                 figure_path='figures',
                 embed=None):
        if adata_atac is None:
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
            raise ValueError('rna_only_idx is invalid.')
        if len(rna_only_idx) > 0:
            print('Running in mixed RNA-only mode.')
            if ref_batch is not None and ref_batch in rna_only_idx:
                raise ValueError('Error: reference batch cannot be RNA only when inferring chromatin. Exiting the program...')
        if ('Mc' not in adata_atac.layers) or ('Mu' not in adata.layers) or ('Ms' not in adata.layers):
            raise ValueError('Chromatin/Unspliced/Spliced count matrices not found in the layers! Exiting the program...')
        if issparse(adata_atac.layers['Mc']):
            adata_atac.layers['Mc'] = adata_atac.layers['Mc'].A
        if issparse(adata.layers['Mu']):
            adata.layers['Mu'] = adata.layers['Mu'].A
        if issparse(adata.layers['Ms']):
            adata.layers['Ms'] = adata.layers['Ms'].A
        if np.any(adata_atac.layers['Mc'] < 0):
            logger.warn('Negative expression values detected in layers["Mc"]. Please make sure all values are non-negative.')
        if np.any(adata.layers['Mu'] < 0):
            logger.warn('Negative expression values detected in layers["Mu"]. Please make sure all values are non-negative.')
        if np.any(adata.layers['Ms'] < 0):
            logger.warn('Negative expression values detected in layers["Ms"]. Please make sure all values are non-negative.')

        self.config = {
            # model parameters
            "dim_z": dim_z,
            "hidden_size": hidden_size,
            "key": 'vae',
            "is_full_vb": full_vb,
            "indicator_arch": 'parallel' if parallel_arch else 'series',
            "t_network": t_network if tprior is None else False,
            "velocity_continuity": velocity_continuity,
            "velocity_correlation": velocity_correlation,
            "four_basis": four_basis,
            "batch_key": batch_key,
            "ref_batch": ref_batch,
            "batch_hvg_key": batch_hvg_key,
            "var_to_regress": var_to_regress,
            "reinit_params": reinit_params,
            "init_ton_zero": init_ton_zero,
            "unit_scale": unit_scale,
            'loss_std_type': 'deming' if deming_std else 'global',
            "log_params": log_params,
            "rna_only": rna_only,
            "rna_only_idx": rna_only_idx,
            "tmax": tmax,
            "init_method": init_method,
            "init_key": init_key,
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
            "n_refine": 8 if refine_velocity else 1,
            "batch_size": batch_size,
            "learning_rate": None,
            "learning_rate_x0": None,
            "learning_rate_ode": None,
            "learning_rate_post": None,
            "lambda": 1e-3,
            "lambda_post": 1e-3,
            "kl_t": 1.0,
            "kl_z": 1.0,
            "kl_w": 0.01,
            "kl_param": 0.1,
            "reg_param_batch": 1.0,
            "reg_forward": 1.0,
            "reg_corr": 1.0,
            "reg_cos": 0.0,
            "2_norm": 1e-3,
            "test_iter": None,
            "save_epoch": 50,
            "n_warmup": 6,
            "n_warmup2": 15,
            "n_warmup_post": 4,
            "weight_c": 0.6 if not rna_only else 0.0,
            "weight_soft_batch": 0.0,
            "early_stop": 6,
            "early_stop_thred": early_stop_thred,
            "train_test_split": 0.7,
            "neg_slope": 0.0,
            "neg_slope2": 0.01,
            "k_alt": 0,
            "train_ton": False,
            "train_scaling": True,
            "train_offset": True,
            "knn_use_pred": True,
            "run_2nd_stage": run_2nd_stage,

            # plotting
            "sparsify": 1}

        self.set_device(device)
        self.split_train_test(adata.n_obs)
        self.encode_batch(adata, batch_key, ref_batch, batch_hvg_key)
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

        self.rna_only_idx = np.array(self.config['rna_only_idx'])
        if not self.enable_cvae and len(self.rna_only_idx) == 1 and self.rna_only_idx[0] == 0:
            self.config['rna_only'] = True
        if self.enable_cvae and np.any(np.array(self.rna_only_idx) > self.n_batch):
            raise ValueError('RNA-only index out of range.')

        self.adata = adata
        self.adata_atac = adata_atac

        self.encoder = Encoder(3*self.adata.n_vars,
                               self.config['dim_z'],
                               dim_cond=self.n_batch,
                               dim_reg=(0 if self.var_to_regress is None else self.var_to_regress.shape[1]),
                               N1=hidden_size,
                               t_network=t_network,
                               checkpoint=checkpoints[0]).float().to(self.device)

        self.decoder = Decoder(self.adata,
                               self.adata_atac,
                               self.train_idx,
                               self.config['dim_z'],
                               dim_cond=self.n_batch,
                               dim_reg=(0 if self.var_to_regress is None else self.var_to_regress.shape[1]),
                               batch_idx=self.batch_,
                               ref_batch=self.ref_batch,
                               N1=hidden_size,
                               parallel_arch=parallel_arch,
                               t_network=t_network,
                               full_vb=full_vb,
                               global_std=(not deming_std),
                               log_params=log_params,
                               rna_only=rna_only,
                               rna_only_idx=rna_only_idx,
                               p=98,
                               tmax=tmax,
                               reinit=reinit_params,
                               init_ton_zero=init_ton_zero,
                               init_method=init_method,
                               init_key=init_key,
                               checkpoint=checkpoints[1]).float().to(self.device)

        self.alpha_w = torch.tensor(find_dirichlet_param(0.5, 0.05, 4), dtype=torch.float, device=self.device)

        self.use_knn = False
        self.reg_velocity = False
        self.c0, self.u0, self.s0 = None, None, None
        self.c1, self.u1, self.s1 = None, None, None
        self.t0, self.t1 = None, None
        self.x0_index, self.x1_index = None, None
        self.t, self.z = None, None
        self.t_, self.z_ = None, None

        self.loss_train, self.loss_test = [], []
        self.loss_test_iter, self.loss_test_color = [], []
        self.rec_train, self.klt_train, self.klz_train = [], [], []
        self.rec_test, self.klt_test, self.klz_test = [], [], []
        self.counter = 0
        self.n_drop = 0
        self.best_model_state = None
        self.best_counter = 0
        self.loss_idx_start = 0
        self.r = -1

        self.clip_fn = nn.Hardtanh(-1e30, 1e30)

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
                logger.warn('GPU not detected. Using CPU as the device.')
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
            logger.warn("Mode not recognized. Must be 'train' or 'test'!")

    def split_train_test(self, N):
        rand_perm = np.random.permutation(N)
        n_train = int(N*self.config["train_test_split"])
        self.train_idx = rand_perm[:n_train]
        self.test_idx = rand_perm[n_train:]

    def encode_batch(self, adata, batch_key, ref_batch, batch_hvg_key=None):
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
                logger.warn('Integer batch names do not start from 0. Reference batch index may not match the actual batch name!')
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
                logger.warn('Number of batch classes is larger than half of dim_z. Consider increasing dim_z.')
            if self.enable_cvae and 10*self.n_batch < self.config['dim_z']:
                logger.warn('Number of batch classes is smaller than 1/10 of dim_z. Consider decreasing dim_z.')
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
                    logger.warn(f'Highly variable genes for batch {batch} not found in var:batch_{batch_hvg_key}. All genes will be used for batch {batch}.')
                    self.batch_hvg_genes_[self.batch_dic[batch]] = True
        self.batch_hvg_genes = torch.tensor(self.batch_hvg_genes_, dtype=torch.float, device=self.device)
        self.batch_count_balance = None
        if batch_count is not None:
            self.batch_count_balance = torch.tensor(batch_count / batch_count[self.ref_batch], dtype=torch.float, device=self.device)
        if self.enable_cvae:
            self.config['kl_t'] = self.config['kl_t'] * 2
            self.config['kl_z'] = self.config['kl_z'] * (2**(self.n_batch-1))
            print(f"KL weights set to {self.config['kl_t']} {self.config['kl_z']}.")

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
            print(f'Found {var_found}. Regressing them out.')
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
                    if i not in self.config['rna_only_idx']:
                        p += np.sum(adata_atac.layers['Mc'][np.ix_(idx, batch_gene)] > 0)
                    p += np.sum(adata.layers['Mu'][np.ix_(idx, batch_gene)] > 0)
                    p += np.sum(adata.layers['Ms'][np.ix_(idx, batch_gene)] > 0)
            else:
                if self.config['rna_only']:
                    p = np.sum(adata.layers['Mu'] > 0) + (np.sum(adata.layers['Ms'] > 0))
                else:
                    p = np.sum(adata_atac.layers['Mc'] > 0) + np.sum(adata.layers['Mu'] > 0) + (np.sum(adata.layers['Ms'] > 0))
            if self.config['rna_only']:
                p = p / (adata.n_obs * adata.n_vars * 2)
            else:
                p = p / (adata.n_obs * adata.n_vars * 3)
            self.config["learning_rate"] = 10**(p-4)
            print(f'Learning rate set to {self.config["learning_rate"]*10000:.1f}e-4 based on data sparsity.')
        else:
            self.config["learning_rate"] = learning_rate
        self.config["learning_rate_post"] = self.config["learning_rate"]
        self.config["learning_rate_ode"] = 10*self.config["learning_rate"]
        self.config["learning_rate_t0"] = self.config["learning_rate"]
        if self.config['early_stop_thred'] is None:
            if self.enable_cvae:
                if self.config['batch_hvg_key'] is None:
                    self.config['early_stop_thred'] = 0.4 * self.n_batch
                else:
                    self.config['early_stop_thred'] = 0.8 * self.n_batch
            else:
                self.config['early_stop_thred'] = 0.4
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

        self.p_z = torch.stack([torch.zeros(adata.n_obs, self.config['dim_z']),
                                torch.ones(adata.n_obs, self.config['dim_z'])*self.config["std_z_prior"]]).float().to(self.device)

    def update_config(self, config):
        for key in config:
            if key in self.config:
                self.config[key] = config[key]
            else:
                self.config[key] = config[key]
                logger.warn(f"Unknown hyperparameter: {key}")

    def plot_initial(self, gene_plot, figure_path="figures", embed=None):
        gind, gene_plot = get_gene_index(self.adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        onehot = F.one_hot(self.batch[self.train_idx], self.n_batch).float() if self.enable_cvae else None
        scaling_c = self.decoder._get_param('scaling_c', onehot, False, False, mask_idx=self.rna_only_idx, mask_to=1, detach=True).cpu().numpy()
        scaling_u = self.decoder._get_param('scaling_u', onehot, False, False, detach=True).cpu().numpy()
        scaling_s = self.decoder._get_param('scaling_s', onehot, False, False, mask_idx=self.ref_batch, mask_to=1, detach=True).cpu().numpy()
        offset_c = self.decoder._get_param('offset_c', onehot, False, False, mask_idx=np.append(self.rna_only_idx, self.ref_batch), mask_to=0, enforce_positive=False, detach=True).cpu().numpy()
        offset_u = self.decoder._get_param('offset_u', onehot, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False, detach=True).cpu().numpy()
        offset_s = self.decoder._get_param('offset_s', onehot, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False, detach=True).cpu().numpy()

        t = self.decoder.t_[:, gind]
        c = self.adata_atac.layers['Mc'][self.train_idx, :][:, gind]
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

    def forward(self, data_in, t=None, z=None, c0=None, u0=None, s0=None, t0=None, t1=None, condition=None, regressor=None, sample=True, return_velocity=False, use_input_time=None):
        if self.config['unit_scale']:
            sigma_c = self.decoder.sigma_c
            sigma_u = self.decoder.sigma_u
            sigma_s = self.decoder.sigma_s
            data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//3]/sigma_c,
                                       data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]/sigma_u,
                                       data_in[:, data_in.shape[1]//3*2:]/sigma_s), 1)
        else:
            scaling_c = self.decoder._get_param('scaling_c', condition, False, False, mask_idx=self.rna_only_idx, mask_to=1)
            scaling_u = self.decoder._get_param('scaling_u', condition, False, False)
            scaling_s = self.decoder._get_param('scaling_s', condition, False, False, mask_idx=self.ref_batch, mask_to=1)
            offset_c = self.decoder._get_param('offset_c', condition, False, False, mask_idx=np.append(self.rna_only_idx, self.ref_batch), mask_to=0, enforce_positive=False)
            offset_u = self.decoder._get_param('offset_u', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            offset_s = self.decoder._get_param('offset_s', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            data_in_scale = torch.cat(((data_in[:, :data_in.shape[1]//3]-offset_c)/scaling_c,
                                       (data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]-offset_u)/scaling_u,
                                       (data_in[:, data_in.shape[1]//3*2:]-offset_s)/scaling_s), 1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition, regressor)
        if t is None or z is None:
            if sample and not self.use_knn:
                t = reparameterize(mu_t, std_t)
                z = reparameterize(mu_z, std_z)
            else:
                t = mu_t
                z = mu_z
        if use_input_time is None:
            use_input_time = self.use_knn
        if t1 is not None:
            chat, uhat, shat, t_, kc, rho, vc, vu, vs = self.decoder.forward(t,
                                                                             z,
                                                                             c0,
                                                                             u0,
                                                                             s0,
                                                                             t0,
                                                                             condition,
                                                                             regressor,
                                                                             neg_slope=self.config["neg_slope"],
                                                                             sample=sample,
                                                                             return_velocity=True,
                                                                             use_input_time=use_input_time)
            chat_fw, uhat_fw, shat_fw, _, kc, rho, vc_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                                              z,
                                                                                              chat,
                                                                                              uhat,
                                                                                              shat,
                                                                                              t_,
                                                                                              condition,
                                                                                              regressor,
                                                                                              neg_slope=self.config["neg_slope"],
                                                                                              sample=sample,
                                                                                              return_velocity=True,
                                                                                              use_input_time=use_input_time)
        else:
            out = self.decoder.forward(t,
                                       z,
                                       c0,
                                       u0,
                                       s0,
                                       t0,
                                       condition,
                                       regressor,
                                       neg_slope=self.config["neg_slope"],
                                       sample=sample,
                                       four_basis=self.config["four_basis"] if not self.use_knn else False,
                                       return_velocity=return_velocity,
                                       use_input_time=use_input_time)
            if return_velocity:
                chat, uhat, shat, t_, kc, rho, vc, vu, vs = out
                chat_fw, uhat_fw, shat_fw, vc_fw, vu_fw, vs_fw = None, None, None, None, None, None
            else:
                chat, uhat, shat, t_, kc, rho = out
                chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = None, None, None, None, None, None, None, None, None
        return mu_t, std_t, mu_z, std_z, chat, uhat, shat, t_, kc, rho, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw

    def vae_risk(self,
                 q_tx,
                 p_t,
                 q_zx,
                 p_z,
                 c,
                 u,
                 s,
                 chat,
                 uhat,
                 shat,
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
            if not self.config['rna_only']:
                logp += torch.log(sigma_c/self.config['weight_c'])
            logp += torch.log(sigma_u)
            logp += torch.log(sigma_s*2*np.pi)
            logp = self.clip_fn(logp)
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
            logp = torch.sum(pw*logp, 1)
        else:
            logp = 0.5*((c - chat)/sigma_c*self.config['weight_c']).pow(2)
            logp += 0.5*((u - uhat)/sigma_u).pow(2)
            logp += 0.5*((s - shat)/sigma_s).pow(2)
            if not self.config['rna_only']:
                logp += torch.log(sigma_c/self.config['weight_c'])
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

        err_rec_n = logp.mean(1) * self.adata.n_vars
        if condition is not None:
            err_rec_n = err_rec_n / (condition * self.batch_count_balance).sum(1)
        err_rec = err_rec_n.mean()

        loss = err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz

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
            scaling_c = self.decoder._get_param('scaling_c', condition, False, False, mask_idx=self.rna_only_idx, mask_to=1)
            scaling_u = self.decoder._get_param('scaling_u', condition, False, False)
            scaling_s = self.decoder._get_param('scaling_s', condition, False, False, mask_idx=self.ref_batch, mask_to=1)
            offset_c = self.decoder._get_param('offset_c', condition, False, False, mask_idx=np.append(self.rna_only_idx, self.ref_batch), mask_to=0, enforce_positive=False)
            offset_u = self.decoder._get_param('offset_u', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            offset_s = self.decoder._get_param('offset_s', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
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
            scaling_u = self.decoder._get_param('scaling_u', condition, False, False)
            scaling_s = self.decoder._get_param('scaling_s', condition, False, False, mask_idx=self.ref_batch)
            offset_u = self.decoder._get_param('offset_u', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            offset_s = self.decoder._get_param('offset_s', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            _, _, beta, gamma, _, _, _ = self.decoder.reparameterize(condition, sample)
            cos_sim = cosine_similarity((uhat-offset_u)/scaling_u, (shat-offset_s)/scaling_s, beta, gamma, s_knn, w_hvg=w_hvg)
            loss = loss - self.config["reg_cos"]*cos_sim

        return loss, err_rec, self.config["kl_t"]*kldt, self.config["kl_z"]*kldz

    def pred_all(self, data, mode='test', output=["chat", "uhat", "shat", "t", "z", "kc", "rho"], gene_idx=None, batch=None, var_to_regress=None):
        N, G = data.shape[0], data.shape[1]//3
        if gene_idx is None:
            gene_idx = np.array(range(G))
        elbo = 0
        rec = 0
        klt = 0
        klz = 0
        save_chat_fw = "chat_fw" in output and self.use_knn and self.config["velocity_continuity"]
        save_uhat_fw = "uhat_fw" in output and self.use_knn and self.config["velocity_continuity"]
        save_shat_fw = "shat_fw" in output and self.use_knn and self.config["velocity_continuity"]
        if batch is None:
            batch = self.batch
        if var_to_regress is None:
            var_to_regress = self.var_to_regress
        if "chat" in output:
            chat_res = np.zeros((N, len(gene_idx)))
        if save_chat_fw:
            chat_fw_res = np.zeros((N, len(gene_idx)))
        if "uhat" in output:
            uhat_res = np.zeros((N, len(gene_idx)))
        if save_uhat_fw:
            uhat_fw_res = np.zeros((N, len(gene_idx)))
        if "shat" in output:
            shat_res = np.zeros((N, len(gene_idx)))
        if save_shat_fw:
            shat_fw_res = np.zeros((N, len(gene_idx)))
        if "t" in output:
            if self.config['t_network']:
                mu_t_out = np.zeros((N, self.config['dim_z']))
                std_t_out = np.zeros((N, self.config['dim_z']))
            else:
                mu_t_out = np.zeros((N, 1))
                std_t_out = np.zeros((N, 1))
            time_out = np.zeros((N))
        if "z" in output:
            mu_z_out = np.zeros((N, self.config['dim_z']))
            std_z_out = np.zeros((N, self.config['dim_z']))
        if "kc" in output:
            kc_res = np.zeros((N, len(gene_idx)))
        if "rho" in output:
            rho_res = np.zeros((N, len(gene_idx)))
        if "v" in output:
            vc_res = np.zeros((N, len(gene_idx)))
            vu_res = np.zeros((N, len(gene_idx)))
            vs_res = np.zeros((N, len(gene_idx)))
        with torch.no_grad():
            B = min(N//5, 5000)
            Nb = N // B
            if Nb*B < N:
                Nb += 1

            w_hard = F.one_hot(torch.argmax(self.decoder.logit_pw, 1), num_classes=4).T
            for n in range(Nb):
                i = n*B
                j = min([(n+1)*B, N])
                data_in = data[i:j]
                if mode == "test":
                    idx = self.test_idx[i:j]
                elif mode == "train":
                    idx = self.train_idx[i:j]
                else:
                    idx = torch.arange(i, j, device=self.device)

                c0 = self.c0[idx] if self.use_knn else None
                u0 = self.u0[idx] if self.use_knn else None
                s0 = self.s0[idx] if self.use_knn else None
                t0 = self.t0[idx] if self.use_knn else None
                t1 = self.t1[idx] if self.use_knn and self.config['velocity_continuity'] else None
                t = self.t[idx] if self.use_knn else None
                z = self.z[idx] if self.use_knn else None
                p_t = self.p_t[:, idx, :]
                p_z = self.p_z[:, idx, :]
                onehot = F.one_hot(batch[idx], self.n_batch).float() if self.enable_cvae else None
                regressor = var_to_regress[idx] if var_to_regress is not None else None
                out = self.forward(data_in, t, z, c0, u0, s0, t0, t1, onehot, regressor, sample=False, return_velocity=("v" in output))
                mu_tx, std_tx, mu_zx, std_zx, chat, uhat, shat, t_, kc, rho, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = out

                loss = self.vae_risk((mu_tx, std_tx),
                                     p_t,
                                     (mu_zx, std_zx),
                                     p_z,
                                     data_in[:, :G],
                                     data_in[:, G:G*2],
                                     data_in[:, G*2:],
                                     chat,
                                     uhat,
                                     shat,
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
                                     onehot,
                                     sample=False)

                elbo = elbo - ((j-i)/N)*loss[0].detach().cpu().item()
                rec = rec - ((j-i)/N)*loss[1].detach().cpu().item()
                klt = klt - ((j-i)/N)*loss[2].detach().cpu().item()
                klz = klz - ((j-i)/N)*loss[3].detach().cpu().item()
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
                        mu_t_out[i:j] = mu_tx.detach().cpu().squeeze().numpy()
                        std_t_out[i:j] = std_tx.detach().cpu().squeeze().numpy()
                    else:
                        mu_t_out[i:j] = mu_tx.detach().cpu().squeeze().numpy()[:, None]
                        std_t_out[i:j] = std_tx.detach().cpu().squeeze().numpy()[:, None]
                    time_out[i:j] = t_.detach().cpu().squeeze().numpy()
                if "z" in output:
                    mu_z_out[i:j] = mu_zx.detach().cpu().numpy()
                    std_z_out[i:j] = std_zx.detach().cpu().numpy()
                if "kc" in output:
                    kc_res[i:j] = kc[:, gene_idx].detach().cpu().numpy()
                if "rho" in output:
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
        if "kc" in output:
            out['kc'] = kc_res
        if "rho" in output:
            out['rho'] = rho_res
        if "v" in output:
            out["vc"] = vc_res
            out["vu"] = vu_res
            out["vs"] = vs_res

        return out, elbo, rec, klt, klz

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
        out, elbo, rec, klt, klz = self.pred_all(dataset.data, mode, out_type, gind)
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
            onehot = F.one_hot(self.batch[cell_idx], self.n_batch).float() if self.enable_cvae else None
            scaling_c = self.decoder._get_param('scaling_c', onehot, False, False, mask_idx=self.rna_only_idx, mask_to=1, detach=True).cpu().numpy()
            scaling_u = self.decoder._get_param('scaling_u', onehot, False, False, detach=True).cpu().numpy()
            scaling_s = self.decoder._get_param('scaling_s', onehot, False, False, mask_idx=self.ref_batch, mask_to=1, detach=True).cpu().numpy()
            offset_c = self.decoder._get_param('offset_c', onehot, False, False, mask_idx=np.append(self.rna_only_idx, self.ref_batch), mask_to=0, enforce_positive=False, detach=True).cpu().numpy()
            offset_u = self.decoder._get_param('offset_u', onehot, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False, detach=True).cpu().numpy()
            offset_s = self.decoder._get_param('offset_s', onehot, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False, detach=True).cpu().numpy()

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
        return elbo, rec, klt, klz

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

    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, optimizer3=None, k=0, net='both'):
        B = len(train_loader)
        self.set_mode('train', net)
        stop_training = False

        for i, batch in enumerate(train_loader):
            if self.counter % self.config["test_iter"] == (2 if self.use_knn else 1):
                elbo_test, rec_test, klt_test, klz_test = self.eval(test_set,
                                                                    None,
                                                                    self.counter,
                                                                    True)

                if len(self.loss_test) > 0:
                    if self.config["early_stop_thred"] is not None and (self.loss_test[-1] + elbo_test <= self.config["early_stop_thred"]):
                        self.n_drop = self.n_drop + 1
                    else:
                        self.n_drop = 0
                self.save_state_dict(-elbo_test)
                self.rec_test.append(-rec_test)
                self.klt_test.append(-klt_test)
                self.klz_test.append(-klz_test)
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

            xbatch, idx = batch[0], batch[1]
            batch_idx = self.train_idx[idx]

            c0 = self.c0[batch_idx] if self.use_knn else None
            u0 = self.u0[batch_idx] if self.use_knn else None
            s0 = self.s0[batch_idx] if self.use_knn else None
            t0 = self.t0[batch_idx] if self.use_knn else None
            t1 = self.t1[batch_idx] if self.use_knn and self.config['velocity_continuity'] else None
            t = self.t[batch_idx] if self.use_knn else None
            z = self.z[batch_idx] if self.use_knn else None
            p_t = self.p_t[:, batch_idx, :]
            p_z = self.p_z[:, batch_idx, :]
            onehot = F.one_hot(self.batch[batch_idx], self.n_batch).float() if self.enable_cvae else None
            regressor = self.var_to_regress[batch_idx] if self.var_to_regress is not None else None
            out = self.forward(xbatch, t, z, c0, u0, s0, t0, t1, onehot, regressor)
            mu_tx, std_tx, mu_zx, std_zx, chat, uhat, shat, _, _, _, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = out

            loss = self.vae_risk((mu_tx, std_tx),
                                 p_t,
                                 (mu_zx, std_zx),
                                 p_z,
                                 xbatch[:, :xbatch.size()[1]//3],
                                 xbatch[:, xbatch.size()[1]//3:xbatch.size()[1]//3*2],
                                 xbatch[:, xbatch.size()[1]//3*2:],
                                 chat,
                                 uhat,
                                 shat,
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
                                 self.c1[batch_idx] if self.use_knn and self.config["velocity_continuity"] else None,
                                 self.u1[batch_idx] if self.use_knn and self.config["velocity_continuity"] else None,
                                 self.s1[batch_idx] if self.use_knn and self.config["velocity_continuity"] else None,
                                 self.s_knn[batch_idx] if self.reg_velocity else None,
                                 onehot)

            loss[0].backward()
            torch.nn.utils.clip_grad_value_(self.encoder.parameters(), 1e7)
            torch.nn.utils.clip_grad_value_(self.decoder.parameters(), 1e7)
            if k == 0:
                optimizer.step()
                if optimizer2 is not None:
                    optimizer2.step()
                if optimizer3 is not None:
                    optimizer3.step()
            else:
                if optimizer2 is not None and ((i+1) % (k+1) == 0 or i == B-1):
                    optimizer2.step()
                    if optimizer3 is not None:
                        optimizer3.step()
                else:
                    optimizer.step()

            self.loss_train.append(loss[0].detach().cpu().item())
            self.rec_train.append(loss[1].detach().cpu().item())
            self.klt_train.append(loss[2].detach().cpu().item())
            self.klz_train.append(loss[3].detach().cpu().item())
            self.counter = self.counter + 1

        return stop_training

    def update_x0(self, c, u, s, save=True, ref_batch=False):
        start = time.time()
        self.set_mode('eval')
        out, _, _, _, _ = self.pred_all(torch.tensor(np.concatenate((c, u, s), 1), dtype=torch.float, device=self.device), "both")
        chat, uhat, shat, t, mu_t, z = out["chat"], out["uhat"], out["shat"], out["t"], out["mu_t"], out["mu_z"]
        kc, rho = out["kc"], out["rho"]
        if self.enable_cvae:
            out, _, _, _, _ = self.pred_all(torch.tensor(np.concatenate((c, u, s), 1), dtype=torch.float, device=self.device), "both", batch=torch.full((self.adata.n_obs,), self.ref_batch, dtype=int, device=self.device))
            _, _, _, t_, mu_t_, z_ = out["chat"], out["uhat"], out["shat"], out["t"], out["mu_t"], out["mu_z"]
            if ref_batch:  # use batch correction for predictions
                chat, uhat, shat, t, mu_t, z = out["chat"], out["uhat"], out["shat"], out["t"], out["mu_t"], out["mu_z"]
        if self.x0_index is None:
            self.t = torch.reshape(torch.tensor(t, dtype=torch.float, device=self.device), (-1, 1))
            self.z = torch.tensor(z, dtype=torch.float, device=self.device)
            if save:
                key = self.config['key']
                self.adata.obs[f"{key}_time_1step"] = t
                self.adata.obsm[f"{key}_t_1step"] = mu_t
                self.adata.obsm[f"{key}_z_1step"] = z
            if self.enable_cvae:
                self.t_ = torch.reshape(torch.tensor(t_, dtype=torch.float, device=self.device), (-1, 1))
                self.z_ = torch.tensor(z_, dtype=torch.float, device=self.device)
                if save:
                    self.adata.obs[f"{key}_time_1step_batch"] = t_
                    self.adata.obsm[f"{key}_t_1step_batch"] = mu_t_
                    self.adata.obsm[f"{key}_z_1step_batch"] = z_

        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        train_idx = self.train_idx.detach().cpu().numpy()

        init_mask = (t <= np.quantile(t, 0.02))
        onehot = F.one_hot(self.batch[init_mask], self.n_batch).float() if self.enable_cvae else None
        alpha_c, alpha, beta, gamma, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s = self.decoder.reparameterize(onehot, False, False)
        scaling_c = scaling_c.detach().cpu().numpy()
        scaling_u = scaling_u.detach().cpu().numpy()
        scaling_s = scaling_s.detach().cpu().numpy()
        offset_c = offset_c.detach().cpu().numpy()
        offset_u = offset_u.detach().cpu().numpy()
        offset_s = offset_s.detach().cpu().numpy()
        c0_init, u0_init, s0_init = pred_exp_numpy_backward((t[init_mask] - (t.min()-dt[0]))[:, None],
                                                            (c[init_mask]-offset_c)/scaling_c,
                                                            (u[init_mask]-offset_u)/scaling_u,
                                                            (s[init_mask]-offset_s)/scaling_s,
                                                            kc[init_mask],
                                                            alpha_c.detach().cpu().numpy(),
                                                            rho[init_mask],
                                                            alpha.detach().cpu().numpy(),
                                                            beta.detach().cpu().numpy(),
                                                            gamma.detach().cpu().numpy())
        for x in [c0_init, u0_init, s0_init]:
            x[np.isnan(x)] = 0
            x[np.isinf(x)] = 0
        c0_init = np.mean(c0_init * scaling_c + offset_c, 0)
        u0_init = np.mean(u0_init * scaling_u + offset_u, 0)
        s0_init = np.mean(s0_init * scaling_s + offset_s, 0)

        if len(self.rna_only_idx) > 0:
            use_index = train_idx[~np.isin(self.batch_[train_idx], self.rna_only_idx)]
        else:
            use_index = train_idx
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
        c0, u0, s0, t0 = get_x0(chat[use_index],
                                uhat[use_index],
                                shat[use_index],
                                t[use_index],
                                dt,
                                self.x0_index,
                                c0_init,
                                u0_init,
                                s0_init)
        if self.config["velocity_continuity"]:
            if self.x1_index is None:
                self.x1_index = knnx0_index(t[use_index],
                                            z[use_index],
                                            t,
                                            z,
                                            dt,
                                            self.config["n_neighbors"],
                                            bins=self.config["n_lineages"],
                                            forward=True,
                                            hist_eq=True)
            c1, u1, s1, t1 = get_x0(chat[use_index],
                                    uhat[use_index],
                                    shat[use_index],
                                    t[use_index],
                                    dt,
                                    self.x1_index,
                                    None,
                                    None,
                                    None,
                                    forward=True)
        self.c0 = c0
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0.reshape(-1, 1)
        if self.config['velocity_continuity']:
            self.c1 = c1
            self.u1 = u1
            self.s1 = s1
            self.t1 = t1.reshape(-1, 1)
        print(f"Finished. Actual Time: {convert_time(time.time()-start)}")
        return t

    def update_std_noise(self, dataset):
        G = dataset.data.shape[1]//3
        out, _, _, _, _ = self.pred_all(dataset.data,
                                        mode='train',
                                        output=["chat", "uhat", "shat"],
                                        gene_idx=np.array(range(G)),
                                        batch=dataset.batch)
        std_c = np.clip((out["chat"] - dataset.data[:, :G].cpu().numpy()).std(0), 0.01, None)
        std_u = np.clip((out["uhat"] - dataset.data[:, G:G*2].cpu().numpy()).std(0), 0.01, None)
        std_s = np.clip((out["shat"] - dataset.data[:, G*2:].cpu().numpy()).std(0), 0.01, None)
        std_c[np.isnan(std_c)] = 1
        std_u[np.isnan(std_u)] = 1
        std_s[np.isnan(std_s)] = 1
        self.decoder.register_buffer('sigma_c', torch.tensor(std_c, dtype=torch.float, device=self.device))
        self.decoder.register_buffer('sigma_u', torch.tensor(std_u, dtype=torch.float, device=self.device))
        self.decoder.register_buffer('sigma_s', torch.tensor(std_s, dtype=torch.float, device=self.device))

    def train(self,
              config={},
              plot=False,
              gene_plot=[],
              figure_path="figures",
              embed="umap"):
        self.update_config(config)
        start = time.time()
        self.global_counter = 0

        print("--------------------------- Train a VeloVAE ---------------------------")
        X = np.concatenate((self.adata_atac.layers['Mc'].A if issparse(self.adata_atac.layers['Mc']) else self.adata_atac.layers['Mc'],
                            self.adata.layers['Mu'].A if issparse(self.adata.layers['Mu']) else self.adata.layers['Mu'],
                            self.adata.layers['Ms'].A if issparse(self.adata.layers['Ms']) else self.adata.layers['Ms']), 1).astype(float)

        try:
            Xembed = self.adata.obsm[f"X_{embed}"]
            Xembed_train = Xembed[self.train_idx]
            Xembed_test = Xembed[self.test_idx]
        except KeyError:
            logger.warn(f"Embedding X_{embed} not found! Set to None.")
            Xembed = np.nan*np.ones((self.adata.n_obs, 2))
            Xembed_train = Xembed[self.train_idx]
            Xembed_test = Xembed[self.test_idx]
            plot = False

        G = X.shape[1]//3

        print("*********      Creating Training and Validation Datasets      *********")
        train_set = SCData(X[self.train_idx], device=self.device)

        test_set = None
        if len(self.test_idx) > 0:
            test_set = SCData(X[self.test_idx], device=self.device)

        data_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
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
        param_x0 = [self.decoder.c0,
                    self.decoder.u0,
                    self.decoder.s0]
        param_ode_init = [self.decoder.alpha,
                          self.decoder.gamma]
        param_ode = [self.decoder.alpha_c,
                     self.decoder.alpha,
                     self.decoder.beta,
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
        optimizer_x0 = torch.optim.AdamW(param_x0, lr=self.config["learning_rate_x0"], weight_decay=self.config["lambda"])
        optimizer_ode_init = torch.optim.AdamW(param_ode_init, lr=self.config["learning_rate_ode"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.AdamW(param_ode, lr=self.config["learning_rate_ode"], weight_decay=self.config["lambda"])

        for epoch in range(self.config["n_epochs"]):
            if epoch < self.config["n_warmup"]:
                stop_training = self.train_epoch(data_loader,
                                                 test_set,
                                                 optimizer,
                                                 k=self.config["k_alt"])
            elif epoch <= self.config["n_warmup2"]:
                stop_training = self.train_epoch(data_loader,
                                                 test_set,
                                                 optimizer,
                                                 optimizer_x0,
                                                 optimizer_ode_init,
                                                 k=self.config["k_alt"])
            else:
                stop_training = self.train_epoch(data_loader,
                                                 test_set,
                                                 optimizer,
                                                 optimizer_x0,
                                                 optimizer_ode,
                                                 k=self.config["k_alt"])

            if epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0:
                elbo_train, _, _, _ = self.eval(train_set,
                                                Xembed_train,
                                                f"train-{epoch+1}",
                                                False,
                                                gind,
                                                gene_plot,
                                                plot,
                                                figure_path)
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
            self.set_mode('eval', 'encoder')
            self.save_state_dict(reset=True)
            self.config['test_iter'] = self.config['test_iter'] // (3 if self.enable_cvae else 2)
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
            x0_change_prev = np.inf

            if self.enable_cvae:
                self.config['early_stop'] *= self.n_batch

            for r in range(self.config['n_refine']):
                self.r = r
                stop_training = (x0_change - x0_change_prev >= -0.01 and r > 1) or (x0_change < 0.01)
                if self.config['loss_std_type'] == 'deming' and noise_change > 0.001 and r < self.config['n_refine']-1:
                    self.update_std_noise(train_set)
                    stop_training = False
                if stop_training:
                    print(f"*********      Stage 2: Early Stop Triggered at round {r}.      *********")
                    break

                t = self.update_x0(X[:, :G], X[:, G:G*2], X[:, G*2:])
                if plot:
                    if np.sum(np.isnan(self.t0)) > 0:
                        print('t0 contains nan.')
                    if np.sum(np.isnan(self.c0)) > 0 or np.sum(np.isnan(self.u0)) > 0 or np.sum(np.isnan(self.s0)) > 0:
                        print('c0, u0, or s0 contains nan. Consider raising early stopping threshold.')
                    if r == 0:
                        plot_time(self.t0.squeeze(), Xembed, save=f"{figure_path}/timet0-train-test.png")
                        plot_time(t, Xembed, save=f"{figure_path}/time-train-test.png")
                    train_idx = self.train_idx.detach().cpu().numpy()
                    t0_plot = self.t0[train_idx].squeeze()
                    for i in range(len(gind)):
                        idx = gind[i]
                        c0_plot = self.c0[train_idx, idx]
                        u0_plot = self.u0[train_idx, idx]
                        s0_plot = self.s0[train_idx, idx]
                        c_plot = X[:, :G][train_idx, idx]
                        u_plot = X[:, G:G*2][train_idx, idx]
                        s_plot = X[:, G*2:][train_idx, idx]
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
                self.c0 = torch.tensor(self.c0, dtype=torch.float, device=self.device)
                self.u0 = torch.tensor(self.u0, dtype=torch.float, device=self.device)
                self.s0 = torch.tensor(self.s0, dtype=torch.float, device=self.device)
                self.t0 = torch.tensor(self.t0, dtype=torch.float, device=self.device)
                if self.config["velocity_continuity"]:
                    self.c1 = torch.tensor(self.c1, dtype=torch.float, device=self.device)
                    self.u1 = torch.tensor(self.u1, dtype=torch.float, device=self.device)
                    self.s1 = torch.tensor(self.s1, dtype=torch.float, device=self.device)
                    self.t1 = torch.tensor(self.t1, dtype=torch.float, device=self.device)

                if r == 0:
                    self.use_knn = True
                    self.decoder.logit_pw.requires_grad = False
                    self.decoder.init_weights(reinit_t=False)

                self.n_drop = 0
                self.loss_idx_start = len(self.loss_test)
                print(f"*********             Velocity Refinement Round {r+1}             *********")

                for epoch in range(self.config["n_epochs_post"]):
                    if epoch == 0:
                        elbo_train, _, _, _ = self.eval(train_set,
                                                        Xembed_train,
                                                        f"train-round{r+1}-first",
                                                        False,
                                                        gind,
                                                        gene_plot,
                                                        plot,
                                                        figure_path)
                        self.set_mode('train', 'decoder')
                    if epoch < self.config["n_warmup_post"]:
                        stop_training = self.train_epoch(data_loader,
                                                         test_set,
                                                         optimizer_post,
                                                         k=self.config["k_alt"],
                                                         net='decoder')
                    else:
                        stop_training = self.train_epoch(data_loader,
                                                         test_set,
                                                         optimizer_post,
                                                         optimizer_ode,
                                                         k=self.config["k_alt"],
                                                         net='decoder')

                    if stop_training or epoch == self.config["n_epochs_post"]:
                        elbo_train, _, _, _ = self.eval(train_set,
                                                        Xembed_train,
                                                        f"train-round{r+1}-last",
                                                        False,
                                                        gind,
                                                        gene_plot,
                                                        plot,
                                                        figure_path)
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
                    x0_change_prev = x0_change
                    norm_delta_x0 = np.sqrt(((self.c0.detach().cpu().numpy() - c0_prev)**2 + (self.u0.detach().cpu().numpy() - u0_prev)**2 + (self.s0.detach().cpu().numpy() - s0_prev)**2).sum(1).mean())
                    std_x = np.sqrt((self.c0.detach().cpu().numpy().var(0) + self.u0.detach().cpu().numpy().var(0) + self.s0.detach().cpu().numpy().var(0)).sum())
                    x0_change = norm_delta_x0/std_x
                    print(f"Change in x0: {x0_change:.4f}")
                c0_prev = self.c0.detach().cpu().numpy()
                u0_prev = self.u0.detach().cpu().numpy()
                s0_prev = self.s0.detach().cpu().numpy()

            if plot and self.config['n_refine'] > 1:
                t0_plot = self.t0[self.train_idx].detach().cpu().numpy().squeeze()
                for i in range(len(gind)):
                    idx = gind[i]
                    c0_plot = self.c0[self.train_idx, idx].detach().cpu().numpy()
                    u0_plot = self.u0[self.train_idx, idx].detach().cpu().numpy()
                    s0_plot = self.s0[self.train_idx, idx].detach().cpu().numpy()
                    plot_sig_(t0_plot,
                              c0_plot,
                              u0_plot,
                              s0_plot,
                              cell_labels=self.cell_labels_raw[self.train_idx.detach().cpu().numpy()],
                              cell_type_colors=self.cell_type_colors,
                              title=gene_plot[i],
                              save=f"{figure_path}/sigx0-{gene_plot[i].replace('.', '-')}-updated.png")

        elbo_train, rec_train, klt_train, klz_train = self.eval(train_set,
                                                                Xembed_train,
                                                                "train-final",
                                                                False,
                                                                gind,
                                                                gene_plot,
                                                                True,
                                                                figure_path)
        elbo_test, rec_test, klt_test, klz_test = self.eval(test_set,
                                                            Xembed_test,
                                                            "test-final",
                                                            True,
                                                            gind,
                                                            gene_plot,
                                                            True,
                                                            figure_path)
        self.loss_train.append(-elbo_train)
        self.rec_train.append(-rec_train)
        self.klt_train.append(-klt_train)
        self.klz_train.append(-klz_train)

        self.save_state_dict(-elbo_test)
        self.rec_test.append(-rec_test)
        self.klt_test.append(-klt_test)
        self.klz_test.append(-klz_test)
        if plot:
            plot_train_loss_log(self.loss_train, range(1, len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            plot_train_loss_log(self.rec_train, range(1, len(self.rec_train)+1), save=f'{figure_path}/train_loss_rec_velovae.png')
            plot_train_loss_log(self.klt_train, range(1, len(self.klt_train)+1), save=f'{figure_path}/train_loss_klt_velovae.png')
            plot_train_loss_log(self.klz_train, range(1, len(self.klz_train)+1), save=f'{figure_path}/train_loss_klz_velovae.png')
            if self.config["test_iter"] > 0:
                plot_test_loss_log(self.loss_test, self.loss_test_iter, color=self.loss_test_color, save=f'{figure_path}/test_loss_velovae.png')
                plot_test_loss_log(self.rec_test, self.loss_test_iter, color=self.loss_test_color, save=f'{figure_path}/test_loss_rec_velovae.png')
                plot_test_loss_log(self.klt_test, self.loss_test_iter, color=self.loss_test_color, save=f'{figure_path}/test_loss_klt_velovae.png')
                plot_test_loss_log(self.klz_test, self.loss_test_iter, color=self.loss_test_color, save=f'{figure_path}/test_loss_klz_velovae.png')
        print(f"Final: Train ELBO = {elbo_train:.3f},\tTest ELBO = {elbo_test:.3f}")
        print(f"*********      Finished. Total Time = {convert_time(time.time()-start)}     *********")

    def test(self, c, u, s, batch=None, covar=None, k=1, sample=False, seed=0):
        c = c.A if issparse(c) else c
        u = u.A if issparse(u) else u
        s = s.A if issparse(s) else s
        if c.shape != u.shape or c.shape != s.shape:
            raise ValueError("c, u, and s must have the same shape.")
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if u.ndim == 1:
            u = u.reshape(1, -1)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        x = np.concatenate((c, u, s), 1).astype(float)
        self.set_mode('eval')

        if not self.enable_cvae:
            out, _, _, _, _ = self.pred_all(torch.tensor(x, dtype=torch.float, device=self.device), "both")
        else:
            if batch is not None:
                print('Using supplied batch list for latent variable computation.')
                batch_ = np.array([self.batch_dic[x] for x in np.array(batch)])
                batch_ = torch.tensor(batch_, dtype=int, device=self.device)
                out, _, _, _, _ = self.pred_all(torch.tensor(x, dtype=torch.float, device=self.device), "both", batch=batch_)
            else:
                print('Using reference batch for latent variable computation.')
                batch_ = torch.full((x.shape[0],), self.ref_batch, dtype=int, device=self.device)
                out, _, _, _, _ = self.pred_all(torch.tensor(x, dtype=torch.float, device=self.device), "both", batch=batch_)

        c0_test, u0_test, s0_test, t0_test = np.empty_like(c), np.empty_like(u), np.empty_like(s), np.empty(x.shape[0])
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
                var_to_regress = np.empty((x.shape[0], self.var_to_regress.shape[1]))

        z_mu, z_std, t_mu, t_std = out["mu_z"], out["std_z"], out["mu_t"], out["std_t"]
        if sample:
            torch.manual_seed(seed)
            z_test = reparameterize(torch.tensor(z_mu, dtype=torch.float, device=self.device), torch.tensor(z_std, dtype=torch.float, device=self.device))
            t_test = reparameterize(torch.tensor(t_mu, dtype=torch.float, device=self.device), torch.tensor(t_std, dtype=torch.float, device=self.device))
        else:
            z_test, t_test = z_mu, out["t"]
            z_test, t_test = torch.tensor(z_test, dtype=torch.float, device=self.device), torch.tensor(t_test[:, None], dtype=torch.float, device=self.device)

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
        x = torch.tensor(x, dtype=torch.float, device=self.device)

        N, G = x.shape[0], x.shape[1]//3
        chat_res = np.zeros((N, G))
        uhat_res = np.zeros((N, G))
        shat_res = np.zeros((N, G))
        vc_res = np.zeros((N, G))
        vu_res = np.zeros((N, G))
        vs_res = np.zeros((N, G))
        time_out = np.zeros((N))
        kc_res = np.zeros((N, G))
        rho_res = np.zeros((N, G))
        with torch.no_grad():
            B = min(N//5, 5000)
            Nb = N // B
            if Nb*B < N:
                Nb += 1
            for n in range(Nb):
                i = n*B
                j = min([(n+1)*B, N])
                data_in = x[i:j]
                idx = torch.arange(i, j, device=self.device)

                c0 = c0_test[idx]
                u0 = u0_test[idx]
                s0 = s0_test[idx]
                t0 = t0_test[idx]
                t = t_test[idx]
                z = z_test[idx]
                onehot = F.one_hot(batch_[idx], self.n_batch).float() if self.enable_cvae else None
                regressor = var_to_regress[idx] if var_to_regress is not None else None
                out = self.forward(data_in, t, z, c0, u0, s0, t0, None, onehot, regressor, sample=False, return_velocity=True, use_input_time=(not sample))
                _, _, _, _, chat, uhat, shat, t_, kc, rho, _, _, _, vc, vu, vs, _, _, _ = out
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

    def save_model(self, file_path, enc_name='encoder', dec_name='decoder'):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), f"{file_path}/{enc_name}.pt")
        torch.save(self.decoder.state_dict(), f"{file_path}/{dec_name}.pt")

    def save_anndata(self, file_path=None, file_name=None):
        self.set_mode('eval')
        if file_path is not None:
            os.makedirs(file_path, exist_ok=True)

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

        X = np.concatenate((self.adata_atac.layers['Mc'].A if issparse(self.adata_atac.layers['Mc']) else self.adata_atac.layers['Mc'],
                            self.adata.layers['Mu'].A if issparse(self.adata.layers['Mu']) else self.adata.layers['Mu'],
                            self.adata.layers['Ms'].A if issparse(self.adata.layers['Ms']) else self.adata.layers['Ms']), 1).astype(float)
        G = X.shape[1]//3
        out, _, _, _, _ = self.pred_all(torch.tensor(X, dtype=torch.float, device=self.device), "both")
        chat, uhat, shat, t, std_t, t_, z, std_z, kc, rho = out["chat"], out["uhat"], out["shat"], out["mu_t"], out["std_t"], out["t"], out["mu_z"], out["std_z"], out["kc"], out["rho"]
        std_c = np.clip(np.nanstd(X[:, :G], axis=0), 0.01, None)
        std_u = np.clip(np.nanstd(X[:, G:G*2], axis=0), 0.01, None)
        std_s = np.clip(np.nanstd(X[:, G*2:], axis=0), 0.01, None)
        diff_c = (X[:, :G] - chat) / std_c
        diff_u = (X[:, G:G*2] - uhat) / std_u
        diff_s = (X[:, G*2:] - shat) / std_s
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
            out = self.test(X[:, :G], X[:, G:G*2], X[:, G*2:], covar=self.var_to_regress.detach().cpu().numpy() if self.var_to_regress is not None else None)
            chat, uhat, shat, _, _, _, z, std_z, t, std_t, t_, kc, rho = out

        self.adata.layers[f"{key}_chat"] = chat
        self.adata.layers[f"{key}_uhat"] = uhat
        self.adata.layers[f"{key}_shat"] = shat
        self.adata.obs[f"{key}_time"] = t_
        self.adata.obsm[f"{key}_t"] = t
        self.adata.obsm[f"{key}_std_t"] = std_t
        self.adata.obsm[f"{key}_z"] = z
        self.adata.obsm[f"{key}_std_z"] = std_z
        self.adata.layers[f"{key}_kc"] = kc
        self.adata.layers[f"{key}_rho"] = rho

        self.adata.obs[f"{key}_t0"] = self.t0.detach().cpu().numpy().squeeze() if self.t0 is not None else 0
        self.adata.layers[f"{key}_c0"] = self.c0.detach().cpu().numpy() if self.c0 is not None else np.zeros_like(self.adata.layers['Mu'])
        self.adata.layers[f"{key}_u0"] = self.u0.detach().cpu().numpy() if self.u0 is not None else np.zeros_like(self.adata.layers['Mu'])
        self.adata.layers[f"{key}_s0"] = self.s0.detach().cpu().numpy() if self.s0 is not None else np.zeros_like(self.adata.layers['Mu'])

        self.adata.uns[f"{key}_train_idx"] = self.train_idx.detach().cpu().numpy()
        self.adata.uns[f"{key}_test_idx"] = self.test_idx.detach().cpu().numpy()

        print("Computing velocity.")
        rna_velocity_vae(self.adata, self.adata_atac, key,
                         batch_key=self.config['batch_key'],
                         ref_batch=self.ref_batch,
                         batch_hvg_key=self.config['batch_hvg_key'],
                         batch_correction=self.enable_cvae,
                         rna_only=self.config['rna_only'])

        if file_name is not None:
            print("Writing anndata output to file.")
            if file_path is None:
                file_path = '.'
            self.adata.write_h5ad(f"{file_path}/{file_name}")
