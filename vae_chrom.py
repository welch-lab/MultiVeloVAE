import os
import time
import anndata as ad
from scipy.sparse import issparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from velovae.plotting import plot_train_loss, plot_test_loss
from velovae.plotting_chrom import plot_sig_, plot_sig, plot_vel, plot_phase, plot_time
from .model_util import hist_equal, convert_time, get_gene_index
from .model_util import elbo_collapsed_categorical, assign_gene_mode_tprior
from .model_util_chrom import pred_exp, pred_exp_backward, init_params, get_ts_global, reinit_params
from .model_util_chrom import kl_gaussian, reparameterize, softplusinv, knnx0_index, get_x0
from .model_util_chrom import cosine_similarity, find_dirichlet_param, assign_gene_mode
from .transition_graph import encode_type
from .training_data_chrom import SCData
from .velocity_chrom import rna_velocity_vae

P_MAX = 1e30
GRAD_MAX = 1e7


##############################################################
# VAE
##############################################################
class Encoder(nn.Module):
    def __init__(self,
                 Cin,
                 dim_z,
                 dim_cond=0,
                 N1=256,
                 t_network=False,
                 checkpoint=None):
        super(Encoder, self).__init__()
        self.t_network = t_network

        self.fc1 = nn.Linear(Cin+dim_cond, N1)
        self.bn1 = nn.BatchNorm1d(N1)
        self.dpt1 = nn.Dropout(p=0.2)
        self.net1 = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1)

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
        for m in self.net1.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in [self.fc_mu_t, self.fc_std_t, self.fc_mu_z, self.fc_std_z]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, data_in, condition=None):
        if condition is not None:
            data_in = torch.cat((data_in, condition), 1)
        h = self.net1(data_in)
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
                 batch_idx=None,
                 ref_batch=None,
                 N1=256,
                 parallel_arch=True,
                 t_network=False,
                 full_vb=False,
                 global_std=False,
                 log_params=False,
                 rna_only=False,
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
        self.reinit = reinit
        self.init_ton_zero = init_ton_zero
        self.init_method = init_method
        self.init_key = init_key
        self.checkpoint = checkpoint
        self.tmax = tmax
        self.construct_nn(dim_z, dim_cond, N1, p)

    def construct_nn(self, dim_z, dim_cond, N1, p):
        G = self.adata.n_vars
        self.set_shape(G, dim_cond)

        # self.fc1 = nn.Linear(dim_z+dim_cond, N1)
        # self.bn1 = nn.BatchNorm1d(num_features=N1)
        # self.dpt1 = nn.Dropout(p=0.25)
        # self.fc2 = nn.Linear(N1, N2)
        # self.bn2 = nn.BatchNorm1d(num_features=N2)
        # self.dpt2 = nn.Dropout(p=0.25)
        # self.fc_out1 = nn.Linear(N2, G)
        # self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
        #                              self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2,
        #                              self.fc_out1, nn.Hardtanh(0, 1))

        # self.fc3 = nn.Linear(dim_z+dim_cond if self.parallel_arch else G, N1)
        # self.bn3 = nn.BatchNorm1d(num_features=N1)
        # self.dpt3 = nn.Dropout(p=0.25)
        # self.fc4 = nn.Linear(N1, N2)
        # self.bn4 = nn.BatchNorm1d(num_features=N2)
        # self.dpt4 = nn.Dropout(p=0.25)
        # self.fc_out2 = nn.Linear(N2, G)
        # self.net_kc = nn.Sequential(self.fc3, self.bn3, nn.LeakyReLU(), self.dpt3,
        #                             self.fc4, self.bn4, nn.LeakyReLU(), self.dpt4,
        #                             self.fc_out2, nn.Hardtanh(0, 1))

        self.fc1 = nn.Linear(dim_z+dim_cond, N1)
        self.bn1 = nn.BatchNorm1d(N1)
        self.dpt1 = nn.Dropout(p=0.2)
        self.fc_out1 = nn.Linear(N1, G)
        self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                     self.fc_out1, nn.Sigmoid())

        self.fc2 = nn.Linear(dim_z+dim_cond if self.parallel_arch else G, N1)
        self.bn2 = nn.BatchNorm1d(N1)
        self.dpt2 = nn.Dropout(p=0.2)
        self.fc_out2 = nn.Linear(N1, G)
        self.net_kc = nn.Sequential(self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2,
                                    self.fc_out2, nn.Sigmoid())

        if self.t_network:
            self.net_t = nn.Linear(dim_z, 1)

        self.rho, self.kc = None, None

        if self.checkpoint is not None:
            self.alpha_c = nn.Parameter(torch.empty(self.params_shape))
            self.alpha = nn.Parameter(torch.empty(self.params_shape))
            self.beta = nn.Parameter(torch.empty(self.params_shape))
            self.gamma = nn.Parameter(torch.empty(self.params_shape))
            self.register_buffer('sigma_c', torch.empty(G))
            self.register_buffer('sigma_u', torch.empty(G))
            self.register_buffer('sigma_s', torch.empty(G))
            self.register_buffer('zero_vec', torch.empty(G))

            self.ton = nn.Parameter(torch.empty(G))
            self.toff = nn.Parameter(torch.empty(G))
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

    def to_param(self, x):
        if self.is_full_vb:
            if self.cvae:
                return nn.Parameter(torch.tensor(np.tile(np.stack([np.log(x) if self.log_params else softplusinv(x), np.log(0.05)*np.ones(self.params_shape[2])]), (self.params_shape[0], 1, 1))))
            else:
                return nn.Parameter(torch.tensor(np.stack([np.log(x) if self.log_params else softplusinv(x), np.log(0.05)*np.ones(self.params_shape[1])])))
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
        print("Initialization using the steady-state and dynamical models.")
        out = init_params(c, u, s, p, fit_scaling=True, global_std=self.global_std, tmax=self.tmax, rna_only=self.rna_only)
        alpha_c, alpha, beta, gamma, scaling_c, scaling_u, toff, c0, u0, s0, sigma_c, sigma_u, sigma_s, t, cpred, upred, spred = out
        scaling_s = np.ones_like(scaling_u)
        offset_c, offset_u, offset_s = np.zeros_like(scaling_u), np.zeros_like(scaling_u), np.zeros_like(scaling_u)

        if self.init_method == 'tprior':
            w = assign_gene_mode_tprior(self.adata, self.init_key, self.train_idx)
        else:
            dyn_mask = (t > self.tmax*0.01) & (np.abs(t-toff) > self.tmax*0.01)
            w = np.sum(((t < toff) & dyn_mask), 0) / (np.sum(dyn_mask, 0) + 1e-10)
            w = assign_gene_mode(self.adata, w, 'auto', 0.05, 0.1, 7)
            sigma_c = np.clip(sigma_c, 0.01, None)
            sigma_u = np.clip(sigma_u, 0.01, None)
            sigma_s = np.clip(sigma_s, 0.01, None)
            sigma_c[np.isnan(sigma_c)] = 1
            sigma_u[np.isnan(sigma_u)] = 1
            sigma_s[np.isnan(sigma_s)] = 1
            sigma_c = np.clip(sigma_c, np.min(sigma_c[self.adata.var['quantile_genes']]), np.max(sigma_c[self.adata.var['quantile_genes']]))
            sigma_u = np.clip(sigma_u, np.min(sigma_u[self.adata.var['quantile_genes']]), np.max(sigma_u[self.adata.var['quantile_genes']]))
            sigma_s = np.clip(sigma_s, np.min(sigma_s[self.adata.var['quantile_genes']]), np.max(sigma_s[self.adata.var['quantile_genes']]))
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
            scaling_c[self.ref_batch] = np.clip(np.nanpercentile(ci, 99.5, axis=0), 1e-3, None)
            scaling_u[self.ref_batch] = np.clip(std_u_ref / std_s_ref, 1e-6, 1e6)
            scaling_s[self.ref_batch] = 1.0
            for i in range(self.dim_cond):
                if i != self.ref_batch:
                    ci = c[self.batch[self.train_idx] == i]
                    ui = u[self.batch[self.train_idx] == i]
                    si = s[self.batch[self.train_idx] == i]
                    filt = (si > 0) * (ui > 0) * (ci > 0)
                    ci[~filt] = np.nan
                    ui[~filt] = np.nan
                    si[~filt] = np.nan
                    std_u, std_s = np.nanstd(ui, axis=0), np.nanstd(si, axis=0)
                    scaling_c[i] = np.clip(np.nanpercentile(ci, 99.5, axis=0), 1e-3, None)
                    scaling_u[i] = np.clip(std_u / (std_s_ref*(~np.isnan(std_s_ref)) + std_s*np.isnan(std_s_ref)), 1e-6, 1e6)
                    scaling_s[i] = np.clip(std_s / (std_s_ref*(~np.isnan(std_s_ref)) + std_s*np.isnan(std_s_ref)), 1e-6, 1e6)
            offset_c = np.zeros((self.dim_cond, G))
            offset_u = np.zeros((self.dim_cond, G))
            offset_s = np.zeros((self.dim_cond, G))
        if np.any(np.isnan(scaling_u)):
            print('scaling_u invalid nan')
        if np.any(np.isinf(scaling_u)):
            print('scaling_u invalid inf')
        if np.any(np.isnan(scaling_s)):
            print('scaling_s invalid nan')
        if np.any(np.isinf(scaling_s)):
            print('scaling_s invalid inf')
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

        self.c0 = nn.Parameter(torch.tensor(np.log(c0+1e-10) if self.log_params else c0))
        self.u0 = nn.Parameter(torch.tensor(np.log(u0+1e-10) if self.log_params else u0))
        self.s0 = nn.Parameter(torch.tensor(np.log(s0+1e-10) if self.log_params else s0))

        if self.init_ton_zero or (not self.reinit):
            self.ton = nn.Parameter(torch.zeros(G))
        else:
            self.ton = nn.Parameter(torch.tensor(ton+1e-10))
        if self.rna_only:
            sigma_c = np.full_like(sigma_c, 1.0)
        self.register_buffer('sigma_c', torch.tensor(sigma_c))
        self.register_buffer('sigma_u', torch.tensor(sigma_u))
        self.register_buffer('sigma_s', torch.tensor(sigma_s))
        self.register_buffer('zero_vec', torch.zeros_like(self.u0))
        if self.cvae:
            self.register_buffer('one_mat', torch.ones_like(self.scaling_u))
            self.register_buffer('zero_mat', torch.zeros_like(self.scaling_u))
        if self.rna_only:
            self.alpha_c.requires_grad = False
            self.scaling_c.requires_grad = False
            self.offset_c.requires_grad = False
            self.c0.requires_grad = False

    def get_param(self, x):
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
        return out

    def get_param_1d(self, x, condition=None, four_basis=False, is_full_vb=False, sample=True, mask_idx=None, mask_to=1, enforce_positive=True, detach=False):
        param = self.get_param(x)
        if detach:
            param = param.detach()
        if is_full_vb:
            if sample:
                G = self.adata.n_vars
                eps = torch.randn(G, device=self.alpha_c.device)
                if condition is not None:
                    y = param[:, 0] + eps*(param[:, 1].exp())
                else:
                    y = param[0] + eps*(param[1].exp())
            else:
                if condition is not None:
                    y = param[:, 0]
                else:
                    y = param[0]
        else:
            y = param

        if enforce_positive:
            if self.log_params:
                y = y.exp()
            else:
                y = F.softplus(y)

        if condition is not None:
            if mask_idx is not None:
                mask = torch.ones_like(condition)
                mask[:, mask_idx] = 0
                mask_flip = (~mask.bool()).int()
                y = torch.mm(condition * mask, y) + torch.mm(condition * mask_flip, self.one_mat if mask_to == 1 else self.zero_mat)
            else:
                y = torch.mm(condition, y)

        if condition is not None and four_basis:
            y = y.unsqueeze(1)
        return y

    def reparameterize(self, condition=None, sample=True, four_basis=False):
        alpha_c = self.get_param_1d('alpha_c', condition, four_basis, self.is_full_vb, sample)
        alpha = self.get_param_1d('alpha', condition, four_basis, self.is_full_vb, sample)
        beta = self.get_param_1d('beta', condition, four_basis, self.is_full_vb, sample)
        gamma = self.get_param_1d('gamma', condition, four_basis, self.is_full_vb, sample)
        scaling_c = self.get_param_1d('scaling_c', condition, four_basis)
        scaling_u = self.get_param_1d('scaling_u', condition, four_basis)
        scaling_s = self.get_param_1d('scaling_s', condition, four_basis, mask_idx=self.ref_batch)
        offset_c = self.get_param_1d('offset_c', condition, four_basis, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
        offset_u = self.get_param_1d('offset_u', condition, four_basis, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
        offset_s = self.get_param_1d('offset_s', condition, four_basis, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)

        return alpha_c, alpha, beta, gamma, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s

    def forward(self, t, z, c0=None, u0=None, s0=None, t0=None, condition=None, neg_slope=0.0, sample=True, four_basis=False, return_velocity=False, use_input_time=False, backward=False):
        alpha_c, alpha, beta, gamma, scaling_c, scaling_u, scaling_s, offset_c, offset_u, offset_s = self.reparameterize(condition, sample, four_basis)
        if condition is None:
            self.rho = self.net_rho(z)
            if self.rna_only:
                self.kc = torch.full_like(self.rho, 1.0)
            else:
                self.kc = self.net_kc(z if self.parallel_arch else self.rho)
        else:
            self.rho = self.net_rho(torch.cat((z, condition), 1))
            if self.rna_only:
                self.kc = torch.full_like(self.rho, 1.0)
            else:
                self.kc = self.net_kc(torch.cat((z, condition), 1) if self.parallel_arch else self.rho)
        t_ = self.net_t(t) if self.t_network and (not use_input_time) else t
        if backward:
            pred_func = pred_exp_backward
        else:
            pred_func = pred_exp
        if (c0 is None) or (u0 is None) or (s0 is None) or (t0 is None):
            if four_basis:
                zero_mtx = torch.zeros_like(self.rho)
                kc = torch.stack([self.kc, self.kc, zero_mtx, zero_mtx], 1)
                rho = torch.stack([self.rho, zero_mtx, self.rho, zero_mtx], 1)
                c0 = torch.stack([self.zero_vec, self.zero_vec, self.c0.exp() if self.log_params else self.c0, self.c0.exp() if self.log_params else self.c0])
                u0 = torch.stack([self.zero_vec, self.u0.exp() if self.log_params else self.u0, self.zero_vec, self.u0.exp() if self.log_params else self.u0])
                s0 = torch.stack([self.zero_vec, self.s0.exp() if self.log_params else self.s0, self.zero_vec, self.s0.exp() if self.log_params else self.s0])
                tau = torch.stack([F.leaky_relu(t_ - self.ton, neg_slope) for i in range(4)], 1)
            elif backward:
                kc = self.kc
                rho = self.rho
                c0 = torch.ones_like(alpha) * 1e-10
                u0 = torch.ones_like(alpha) * 1e-10
                s0 = torch.ones_like(alpha) * 1e-10
                tau = 1 - F.leaky_relu(t_ - self.ton, neg_slope)
            else:
                kc = self.kc
                rho = self.rho
                c0 = self.zero_vec
                u0 = self.zero_vec
                s0 = self.zero_vec
                tau = F.leaky_relu(t_ - self.ton, neg_slope)

            if self.rna_only:
                c0 = torch.ones_like(c0)

            chat, uhat, shat = pred_func(tau,
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
            chat, uhat, shat = pred_func(F.leaky_relu(t_ - t0, neg_slope),
                                         (c0-offset_c)/scaling_c,
                                         (u0-offset_u)/scaling_u,
                                         (s0-offset_s)/scaling_s,
                                         self.kc,
                                         alpha_c,
                                         self.rho,
                                         alpha,
                                         beta,
                                         gamma)
        chat = chat * scaling_c + offset_c
        uhat = uhat * scaling_u + offset_u
        shat = shat * scaling_s + offset_s

        if return_velocity:
            vc = self.kc * alpha_c - alpha_c * chat
            vu = self.rho * alpha * chat - beta * uhat
            vs = beta * uhat - gamma * shat
            return chat, uhat, shat, t_, vc, vu, vs
        else:
            return chat, uhat, shat, t_


class VAEChrom():
    def __init__(self,
                 adata,
                 adata_atac=None,
                 dim_z=None,
                 batch_key=None,
                 ref_batch=None,
                 device='cpu',
                 hidden_size=256,
                 full_vb=False,
                 parallel_arch=True,
                 t_network=False,
                 velocity_continuity=False,
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
        if dim_z is None:
            dim_z = 5 if rna_only else 7
        if ('Mc' not in adata_atac.layers) or ('Mu' not in adata.layers) or ('Ms' not in adata.layers):
            print('Chromatin/Unspliced/Spliced count matrices not found in the layers! Exiting the program...')
            return
        if issparse(adata_atac.layers['Mc']):
            adata_atac.layers['Mc'] = adata_atac.layers['Mc'].A
        if issparse(adata.layers['Mu']):
            adata.layers['Mu'] = adata.layers['Mu'].A
        if issparse(adata.layers['Ms']):
            adata.layers['Ms'] = adata.layers['Ms'].A
        if np.any(adata_atac.layers['Mc'] < 0):
            print('Warning: negative expression values detected in layers["Mc"]. Please make sure all values are non-negative.')
        if np.any(adata.layers['Mu'] < 0):
            print('Warning: negative expression values detected in layers["Mu"]. Please make sure all values are non-negative.')
        if np.any(adata.layers['Ms'] < 0):
            print('Warning: negative expression values detected in layers["Ms"]. Please make sure all values are non-negative.')

        self.adata = adata
        self.adata_atac = adata_atac

        self.config = {
            # model parameters
            "dim_z": dim_z,
            "hidden_size": hidden_size,
            "key": 'fullvb' if full_vb else 'vae',
            "indicator_arch": 'parallel' if parallel_arch else 'series',
            "t_network": t_network,
            "batch_key": batch_key,
            "ref_batch": ref_batch,
            "reinit_params": reinit_params,
            "init_ton_zero": init_ton_zero,
            "unit_scale": unit_scale,
            'loss_std_type': 'deming' if deming_std else 'global',
            "log_params": log_params,
            "rna_only": rna_only,
            "tmax": tmax,
            "init_method": init_method,
            "init_key": init_key,
            "tprior": tprior,
            "std_z_prior": 0.05,
            "tail": 0.01,
            "std_t_scaling": 0.05,
            "n_neighbors": 4,
            "dt": (0.04, 0.08),

            # training parameters
            "n_epochs": 2000,
            "n_epochs_post": 500,
            "n_refine": 20 if refine_velocity else 1,
            "batch_size": 256,
            "learning_rate": None,
            "learning_rate_ode": None,
            "learning_rate_x0": None,
            "learning_rate_post": None,
            "lambda": 1e-3,
            "lambda_post": 1e-3,
            "kl_t": 1.0,
            "kl_z": 1.0,
            "kl_w": 0.01,
            "kl_param": 1.0,
            "reg_forward": 10.0,
            "reg_cos": 0.0,
            "test_iter": None,
            "save_epoch": 50,
            "n_warmup": 6,
            "n_warmup_post": 4,
            "weight_c": 0.6 if not rna_only else 0.4,
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
            "velocity_continuity": velocity_continuity,
            "four_basis": four_basis,

            # plotting
            "sparsify": 1}

        self.set_device(device)
        self.split_train_test(adata.n_obs)
        self.encode_batch(adata)
        self.set_lr(adata, adata_atac, learning_rate)
        self.get_prior(adata)

        self.encoder = Encoder(3*self.adata.n_vars,
                               dim_z,
                               dim_cond=self.n_batch,
                               N1=hidden_size,
                               t_network=t_network,
                               checkpoint=checkpoints[0]).float().to(self.device)

        self.decoder = Decoder(self.adata,
                               self.adata_atac,
                               self.train_idx,
                               dim_z,
                               dim_cond=self.n_batch,
                               batch_idx=self.batch_,
                               ref_batch=self.ref_batch,
                               N1=hidden_size,
                               parallel_arch=parallel_arch,
                               t_network=t_network,
                               full_vb=full_vb,
                               global_std=(not deming_std),
                               log_params=log_params,
                               rna_only=rna_only,
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
        self.c0 = None
        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.x0_index = None
        self.c1 = None
        self.u1 = None
        self.s1 = None
        self.t1 = None
        self.x1_index = None
        self.t = None
        self.z = None

        self.loss_train, self.loss_test = [], []
        self.rec_train, self.klt_train, self.klz_train = [], [], []
        self.rec_test, self.klt_test, self.klz_test = [], [], []
        self.counter = 0
        self.n_drop = 0

        self.clip_fn = nn.Hardtanh(-P_MAX, P_MAX)
        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)

        if full_vb:
            self.p_log_alpha_c = torch.tensor([[-1.0], [1.0]], dtype=torch.float, device=self.device)
            self.p_log_alpha = torch.tensor([[4.0], [2.0]], dtype=torch.float, device=self.device)
            self.p_log_beta = torch.tensor([[2.0], [1.0]], dtype=torch.float, device=self.device)
            self.p_log_gamma = torch.tensor([[2.0], [1.0]], dtype=torch.float, device=self.device)
            self.p_params = [self.p_log_alpha_c, self.p_log_alpha, self.p_log_beta, self.p_log_gamma]

        if plot_init:
            self.plot_initial(gene_plot, cluster_key, figure_path, embed)

    def set_device(self, device):
        if 'cuda' in device:
            if torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                print('Warning: GPU not detected. Using CPU as the device.')
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
            print("Warning: mode not recognized. Must be 'train' or 'test'!")

    def split_train_test(self, N):
        rand_perm = np.random.permutation(N)
        n_train = int(N*self.config["train_test_split"])
        self.train_idx = rand_perm[:n_train]
        self.test_idx = rand_perm[n_train:]

    def encode_batch(self, adata):
        self.n_batch = 0
        self.batch = None
        self.batch_ = None
        batch_count = None
        self.ref_batch = self.config['ref_batch']
        if self.config['batch_key'] is not None and self.config['batch_key'] in adata.obs:
            print('CVAE enabled. Performing batch effect correction.')
            batch_raw = adata.obs[self.config['batch_key']].to_numpy()
            batch_names_raw, batch_count = np.unique(batch_raw, return_counts=True)
            self.batch_dic, self.batch_dic_rev = encode_type(batch_names_raw)
            self.n_batch = len(batch_names_raw)
            self.batch_ = np.array([self.batch_dic[x] for x in batch_raw])
            self.batch = torch.tensor(self.batch_, dtype=int, device=self.device)
            self.batch_names = np.array([self.batch_dic[batch_names_raw[i]] for i in range(self.n_batch)])
        if isinstance(self.ref_batch, int):
            if self.ref_batch >= self.n_batch:
                self.ref_batch = self.n_batch - 1
            elif self.ref_batch < -self.n_batch:
                self.ref_batch = 0
            print(f'Reference batch set to {self.ref_batch} ({batch_names_raw[self.ref_batch]}).')
            if np.issubdtype(batch_names_raw.dtype, np.number) and 0 not in batch_names_raw:
                print('Warning: integer batch names do not start from 0. Reference batch index may not match the actual batch name!')
        elif isinstance(self.ref_batch, str):
            if self.config['ref_batch'] in batch_names_raw:
                self.ref_batch = self.batch_dic[self.config['ref_batch']]
                print(f'Reference batch set to {self.ref_batch} ({batch_names_raw[self.ref_batch]}).')
            else:
                raise ValueError('Reference batch not found in the provided batch field!')
        elif batch_count is not None:
            self.ref_batch = self.batch_names[np.argmax(batch_count)]
            print(f'Reference batch set to {self.ref_batch} ({batch_names_raw[self.ref_batch]}).')
        self.enable_cvae = self.n_batch > 0
        if self.enable_cvae and 2*self.n_batch > self.config['dim_z']:
            print('Warning: number of batch classes is larger than half of dim_z. Consider increasing dim_z.')
        if self.enable_cvae and 10*self.n_batch < self.config['dim_z']:
            print('Warning: number of batch classes is smaller than 1/10 of dim_z. Consider decreasing dim_z.')

    def set_lr(self, adata, adata_atac, learning_rate):
        if learning_rate is None:
            p = np.sum(adata_atac.layers['Mc'] > 0) + np.sum(adata.layers['Mu'] > 0) + (np.sum(adata.layers['Ms'] > 0))
            p /= adata.n_obs * adata.n_vars * 3
            self.config["learning_rate"] = 10**(p-4)
            print(f'Learning rate set to {self.config["learning_rate"]*10000:.1f}e-4 based on data sparsity.')
        else:
            self.config["learning_rate"] = learning_rate
        self.config["learning_rate_post"] = self.config["learning_rate"]
        self.config["learning_rate_ode"] = 10*self.config["learning_rate"]
        self.config["learning_rate_t0"] = self.config["learning_rate"]
        if self.config['early_stop_thred'] is None:
            if self.config['rna_only']:
                self.config['early_stop_thred'] = 0.4
            else:
                if self.enable_cvae:
                    self.config['early_stop_thred'] = 0.4 + 0.2 * (self.n_batch-1)
                else:
                    self.config['early_stop_thred'] = 0.4
            print(f"Early stop threshold set to {self.config['early_stop_thred']}.")

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
                print(f"Warning: unknown hyperparameter: {key}")

    def plot_initial(self, gene_plot, cluster_key="clusters", figure_path="figures", embed=None):
        cell_labels_raw = self.adata.obs[cluster_key].to_numpy() if cluster_key in self.adata.obs else np.array(['Unknown' for i in range(self.adata.n_obs)])
        cell_types_raw = np.unique(cell_labels_raw)
        label_dic, _ = encode_type(cell_types_raw)

        cell_labels = np.array([label_dic[x] for x in cell_labels_raw])[self.train_idx]

        gind, gene_plot = get_gene_index(self.adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        onehot = F.one_hot(self.batch.detach()[self.train_idx], self.n_batch).float() if self.enable_cvae else None
        scaling_c = self.decoder.get_param_1d('scaling_c', onehot, False, False, detach=True).cpu().numpy()
        scaling_u = self.decoder.get_param_1d('scaling_u', onehot, False, False, detach=True).cpu().numpy()
        scaling_s = self.decoder.get_param_1d('scaling_s', onehot, False, False, detach=True).cpu().numpy()
        offset_c = self.decoder.get_param_1d('offset_c', onehot, False, False, enforce_positive=False, detach=True).cpu().numpy()
        offset_u = self.decoder.get_param_1d('offset_u', onehot, False, False, enforce_positive=False, detach=True).cpu().numpy()
        offset_s = self.decoder.get_param_1d('offset_s', onehot, False, False, enforce_positive=False, detach=True).cpu().numpy()

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
                     cell_labels_raw[self.train_idx],
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
                     cell_labels_raw[self.train_idx],
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
                       cell_labels,
                       cell_types_raw,
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
                       cell_labels,
                       cell_types_raw,
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
                       cell_labels,
                       cell_types_raw,
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
                       cell_labels,
                       cell_types_raw,
                       save=f"{figure_path}/phasescaled-{gene_plot[i].replace('.', '-')}-init-us.png")

    def forward(self, data_in, t=None, z=None, c0=None, u0=None, s0=None, t0=None, t1=None, condition=None, sample=True):
        if self.config['unit_scale']:
            sigma_c = self.decoder.sigma_c
            sigma_u = self.decoder.sigma_u
            sigma_s = self.decoder.sigma_s
            data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//3]/sigma_c,
                                       data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]/sigma_u,
                                       data_in[:, data_in.shape[1]//3*2:]/sigma_s), 1)
        else:
            scaling_c = self.decoder.get_param_1d('scaling_c', condition, False, False)
            scaling_u = self.decoder.get_param_1d('scaling_u', condition, False, False)
            scaling_s = self.decoder.get_param_1d('scaling_s', condition, False, False, mask_idx=self.ref_batch)
            offset_c = self.decoder.get_param_1d('offset_c', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            offset_u = self.decoder.get_param_1d('offset_u', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            offset_s = self.decoder.get_param_1d('offset_s', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            data_in_scale = torch.cat(((data_in[:, :data_in.shape[1]//3]-offset_c)/scaling_c,
                                       (data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]-offset_u)/scaling_u,
                                       (data_in[:, data_in.shape[1]//3*2:]-offset_s)/scaling_s), 1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        if t is None or z is None:
            if sample and not self.use_knn:
                t = reparameterize(mu_t, std_t)
                z = reparameterize(mu_z, std_z)
            else:
                t = mu_t
                z = mu_z

        if t1 is not None:
            chat, uhat, shat, t_, vc, vu, vs = self.decoder.forward(t,
                                                                    z,
                                                                    c0,
                                                                    u0,
                                                                    s0,
                                                                    t0,
                                                                    condition,
                                                                    neg_slope=self.config["neg_slope"],
                                                                    sample=sample,
                                                                    return_velocity=True,
                                                                    use_input_time=self.use_knn)
            chat_fw, uhat_fw, shat_fw, _, vc_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                                     z,
                                                                                     chat,
                                                                                     uhat,
                                                                                     shat,
                                                                                     t_,
                                                                                     condition,
                                                                                     neg_slope=self.config["neg_slope"],
                                                                                     sample=sample,
                                                                                     return_velocity=True,
                                                                                     use_input_time=True)
        else:
            chat, uhat, shat, t_ = self.decoder.forward(t,
                                                        z,
                                                        c0,
                                                        u0,
                                                        s0,
                                                        t0,
                                                        condition,
                                                        neg_slope=self.config["neg_slope"],
                                                        four_basis=self.config["four_basis"],
                                                        sample=sample,
                                                        use_input_time=self.use_knn)
            chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = None, None, None, None, None, None, None, None, None
        return mu_t, std_t, mu_z, std_z, chat, uhat, shat, t_, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw

    def loss_vel(self, c0, chat, vc, u0, uhat, vu, s0, shat, vs):
        delta_x = torch.cat([chat-c0, uhat-u0, shat-s0], 1)
        v = torch.cat([vc, vu, vs], 1)
        return self.cossim(delta_x, v).mean()

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
            logp += torch.log(sigma_c/self.config['weight_c'])
            logp += torch.log(sigma_u)
            logp += torch.log(sigma_s*2*np.pi)
            logp = self.clip_fn(logp)

        if chat_fw is not None and uhat_fw is not None and shat_fw is not None:
            logp += 0.5*((c1 - chat_fw)/sigma_c*self.config['weight_c']).pow(2)
            logp += 0.5*((u1 - uhat_fw)/sigma_u).pow(2)
            logp += 0.5*((s1 - shat_fw)/sigma_s).pow(2)

        err_rec = torch.mean(torch.sum(logp, 1))

        loss = err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz

        self.global_counter += 1

        if not self.use_knn and self.config["four_basis"]:
            kldw = elbo_collapsed_categorical(self.decoder.logit_pw, self.alpha_w, 4, self.decoder.scaling_u.shape[0])
            loss += self.config["kl_w"]*kldw

        if self.config["key"] == 'fullvb':
            kld_params = None
            for i, x in enumerate(['alpha_c', 'alpha', 'beta', 'gamma']):
                if self.enable_cvae:
                    for j in range(self.n_batch):
                        mu_param = self.decoder.get_param(x)[j, 0].view(1, -1)
                        if kld_params is None:
                            kld_params = kl_gaussian(mu_param if self.config['log_params'] else mu_param.log(), self.decoder.get_param(x)[j, 1].exp().view(1, -1), self.p_params[i][0], self.p_params[i][1])
                        else:
                            kld_params += kl_gaussian(mu_param if self.config['log_params'] else mu_param.log(), self.decoder.get_param(x)[j, 1].exp().view(1, -1), self.p_params[i][0], self.p_params[i][1])

                else:
                    mu_param = self.decoder.get_param(x)[0].view(1, -1)
                    if kld_params is None:
                        kld_params = kl_gaussian(mu_param if self.config['log_params'] else mu_param.log(), self.decoder.get_param(x)[1].exp().view(1, -1), self.p_params[i][0], self.p_params[i][1])
                    else:
                        kld_params += kl_gaussian(mu_param if self.config['log_params'] else mu_param.log(), self.decoder.get_param(x)[1].exp().view(1, -1), self.p_params[i][0], self.p_params[i][1])
            kld_params /= u.shape[0]
            loss = loss + self.config["kl_param"]*kld_params

        if self.use_knn and self.config["velocity_continuity"]:
            scaling_c = self.decoder.get_param_1d('scaling_c', condition, False, False)
            scaling_u = self.decoder.get_param_1d('scaling_u', condition, False, False)
            scaling_s = self.decoder.get_param_1d('scaling_s', condition, False, False, mask_idx=self.ref_batch)
            offset_c = self.decoder.get_param_1d('offset_c', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            offset_u = self.decoder.get_param_1d('offset_u', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            offset_s = self.decoder.get_param_1d('offset_s', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            forward_loss = (self.loss_vel((c0-offset_c)/scaling_c, (chat-offset_c)/scaling_c, vc,
                                          (u0-offset_u)/scaling_u, (uhat-offset_u)/scaling_u, vu,
                                          (s0-offset_s)/scaling_s, (shat-offset_s)/scaling_s, vs)
                            + self.loss_vel((chat-offset_c)/scaling_c, (chat_fw-offset_c)/scaling_c, vc_fw,
                                            (uhat-offset_u)/scaling_u, (uhat_fw-offset_u)/scaling_u, vu_fw,
                                            (shat-offset_s)/scaling_s, (shat_fw-offset_s)/scaling_s, vs_fw))
            loss = loss - self.config["reg_forward"]*forward_loss

        if self.reg_velocity:
            scaling_u = self.decoder.get_param_1d('scaling_u', condition, False, False)
            scaling_s = self.decoder.get_param_1d('scaling_s', condition, False, False, mask_idx=self.ref_batch)
            offset_u = self.decoder.get_param_1d('offset_u', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            offset_s = self.decoder.get_param_1d('offset_s', condition, False, False, mask_idx=self.ref_batch, mask_to=0, enforce_positive=False)
            _, _, beta, gamma, _, _, _ = self.decoder.reparameterize(condition, sample)
            cos_sim = cosine_similarity((uhat-offset_u)/scaling_u, (shat-offset_s)/scaling_s, beta, gamma, s_knn)
            loss = loss - self.config["reg_cos"]*cos_sim

        return loss, err_rec, self.config["kl_t"]*kldt, self.config["kl_z"]*kldz

    def pred_all(self, data, mode='test', output=["chat", "uhat", "shat", "t", "z"], gene_idx=None, batch=None):
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
                data_in = torch.tensor(data[i:j], dtype=torch.float, device=self.device)
                if mode == "test":
                    batch_idx = self.test_idx[i:j]
                elif mode == "train":
                    batch_idx = self.train_idx[i:j]
                else:
                    batch_idx = torch.arange(i, j, device=self.device)

                c0 = self.c0[batch_idx] if self.use_knn else None
                u0 = self.u0[batch_idx] if self.use_knn else None
                s0 = self.s0[batch_idx] if self.use_knn else None
                t0 = self.t0[batch_idx] if self.use_knn else None
                t1 = self.t1[batch_idx] if self.use_knn and self.config['velocity_continuity'] else None
                t = self.t[batch_idx] if self.use_knn else None
                z = self.z[batch_idx] if self.use_knn else None
                p_t = self.p_t[:, batch_idx, :]
                p_z = self.p_z[:, batch_idx, :]
                onehot = F.one_hot(batch[batch_idx], self.n_batch).float() if self.enable_cvae else None
                out = self.forward(data_in, t, z, c0, u0, s0, t0, t1, onehot, sample=False)
                mu_tx, std_tx, mu_zx, std_zx, chat, uhat, shat, t_, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = out

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
                                     self.c1[batch_idx] if self.use_knn and self.config['velocity_continuity'] else None,
                                     self.u1[batch_idx] if self.use_knn and self.config['velocity_continuity'] else None,
                                     self.s1[batch_idx] if self.use_knn and self.config['velocity_continuity'] else None,
                                     self.s_knn[batch_idx] if self.reg_velocity else None,
                                     onehot,
                                     sample=False)

                elbo = elbo - ((j-i)/N)*loss[0].detach().cpu().item()
                rec = rec - ((j-i)/N)*loss[1].detach().cpu().item()
                klt = klt - ((j-i)/N)*loss[2].detach().cpu().item()
                klz = klz - ((j-i)/N)*loss[3].detach().cpu().item()
                if "chat" in output and gene_idx is not None:
                    if chat.ndim == 3:
                        chat = torch.sum(chat*w_hard, 1)
                    chat_res[i:j] = chat[:, gene_idx].detach().cpu().numpy()
                if "uhat" in output and gene_idx is not None:
                    if uhat.ndim == 3:
                        uhat = torch.sum(uhat*w_hard, 1)
                    uhat_res[i:j] = uhat[:, gene_idx].detach().cpu().numpy()
                if "shat" in output and gene_idx is not None:
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
                if "v" in output:
                    vc_res[i:j] = vc[:, gene_idx].detach().cpu().numpy()
                    vu_res[i:j] = vu[:, gene_idx].detach().cpu().numpy()
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
        if "v" in output:
            out["vc"] = vc_res
            out["vu"] = vu_res
            out["vs"] = vs_res

        return out, elbo, rec, klt, klz

    def test(self,
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
        out_type = ["chat", "uhat", "shat", "t"]
        if self.use_knn and self.config["velocity_continuity"]:
            out_type.extend(["chat_fw", "uhat_fw", "shat_fw", "v"])
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
            onehot = F.one_hot(self.batch.detach()[cell_idx], self.n_batch).float() if self.enable_cvae else None
            scaling_c = self.decoder.get_param_1d('scaling_c', onehot, False, False, detach=True).cpu().numpy()
            scaling_u = self.decoder.get_param_1d('scaling_u', onehot, False, False, detach=True).cpu().numpy()
            scaling_s = self.decoder.get_param_1d('scaling_s', onehot, False, False, detach=True).cpu().numpy()
            offset_c = self.decoder.get_param_1d('offset_c', onehot, False, False, enforce_positive=False, detach=True).cpu().numpy()
            offset_u = self.decoder.get_param_1d('offset_u', onehot, False, False, enforce_positive=False, detach=True).cpu().numpy()
            offset_s = self.decoder.get_param_1d('offset_s', onehot, False, False, enforce_positive=False, detach=True).cpu().numpy()

            if not self.use_knn:
                plot_time(t_, Xembed, save=f"{path}/time-{testid_str}.png")

            for i in range(len(gind)):
                idx = gind[i]
                if np.any(np.isnan(chat[:, i])):
                    print(gene_plot[i], chat[:, i])
                if np.any(np.isnan(uhat[:, i])):
                    print(gene_plot[i], uhat[:, i])
                if np.any(np.isnan(shat[:, i])):
                    print(gene_plot[i], shat[:, i])

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
                         np.array([self.label_dic_rev[x] for x in dataset.labels]),
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
                         np.array([self.label_dic_rev[x] for x in dataset.labels]),
                         gene_plot[i],
                         save=f"{path}/sigscaled-{gene_plot[i].replace('.', '-')}-{testid_str}.png",
                         sparsify=self.config['sparsify'])

                if self.use_knn and self.config['velocity_continuity']:
                    plot_vel(t_.squeeze(),
                             (chat[:, i] - offset_c_) / scaling_c_,
                             (uhat[:, i] - offset_u_) / scaling_u_,
                             (shat[:, i] - offset_s_) / scaling_s_,
                             out["vc"][:, i], out["vu"][:, i], out["vs"][:, i],
                             self.t0[cell_idx].squeeze().detach().cpu().numpy(),
                             (self.c0[cell_idx, idx].detach().cpu().numpy() - offset_c_) / scaling_c_,
                             (self.u0[cell_idx, idx].detach().cpu().numpy() - offset_u_) / scaling_u_,
                             (self.s0[cell_idx, idx].detach().cpu().numpy() - offset_s_) / scaling_s_,
                             title=gene_plot[i],
                             save=f"{path}/vel-{gene_plot[i].replace('.', '-')}-{testid_str}.png")

                    plot_sig(t_.squeeze(),
                             dataset.data[:, idx].cpu().numpy(),
                             dataset.data[:, idx+G].cpu().numpy(),
                             dataset.data[:, idx+G*2].cpu().numpy(),
                             out["chat_fw"][:, i],
                             out["uhat_fw"][:, i],
                             out["shat_fw"][:, i],
                             np.array([self.label_dic_rev[x] for x in dataset.labels]),
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
                             np.array([self.label_dic_rev[x] for x in dataset.labels]),
                             gene_plot[i],
                             save=f"{path}/sigscaled-{gene_plot[i].replace('.', '-')}-{testid_str}-bw.png",
                             sparsify=self.config['sparsify'])
        return elbo, rec, klt, klz

    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, k=1, net='both'):
        B = len(train_loader)
        self.set_mode('train', net)
        stop_training = False

        for i, batch in enumerate(train_loader):
            if self.counter == 1 or self.counter % self.config["test_iter"] == 0:
                elbo_test, rec_test, klt_test, klz_test = self.test(test_set,
                                                                    None,
                                                                    self.counter,
                                                                    True)

                if len(self.loss_test) > 0:
                    if self.config["early_stop_thred"] is not None and (self.loss_test[-1] + elbo_test <= self.config["early_stop_thred"]):
                        self.n_drop = self.n_drop + 1
                    else:
                        self.n_drop = 0
                self.loss_test.append(-elbo_test)
                self.rec_test.append(-rec_test)
                self.klt_test.append(-klt_test)
                self.klz_test.append(-klz_test)
                self.set_mode('train', net)

                if (self.n_drop >= self.config["early_stop"]) and (self.config["early_stop"] > 0):
                    stop_training = True
                    break

            optimizer.zero_grad()
            if optimizer2 is not None:
                optimizer2.zero_grad()

            xbatch, idx = batch[0], batch[2]
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
            onehot = F.one_hot(batch[3], self.n_batch).float() if self.enable_cvae else None
            out = self.forward(xbatch, t, z, c0, u0, s0, t0, t1, onehot)
            mu_tx, std_tx, mu_zx, std_zx, chat, uhat, shat, t_, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = out

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
            torch.nn.utils.clip_grad_value_(self.encoder.parameters(), GRAD_MAX)
            torch.nn.utils.clip_grad_value_(self.decoder.parameters(), GRAD_MAX)
            if k == 0:
                optimizer.step()
                if optimizer2 is not None:
                    optimizer2.step()
            else:
                if optimizer2 is not None and ((i+1) % (k+1) == 0 or i == B-1):
                    optimizer2.step()
                else:
                    optimizer.step()

            self.loss_train.append(loss[0].detach().cpu().item())
            self.rec_train.append(loss[1].detach().cpu().item())
            self.klt_train.append(loss[2].detach().cpu().item())
            self.klz_train.append(loss[3].detach().cpu().item())
            self.counter = self.counter + 1

        return stop_training

    def update_x0(self, c, u, s):
        start = time.time()
        self.set_mode('eval')
        out, _, _, _, _ = self.pred_all(np.concatenate((c, u, s), 1), "both")
        chat, uhat, shat, t, z = out["chat"], out["uhat"], out["shat"], out["t"], out["mu_z"]
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        train_idx = self.train_idx.detach().cpu().numpy()
        init_mask = (t <= np.quantile(t, 0.01))
        c0_init = np.mean(c[init_mask], 0)
        u0_init = np.mean(u[init_mask], 0)
        s0_init = np.mean(s[init_mask], 0)
        c_in = chat[train_idx]
        u_in = uhat[train_idx]
        s_in = shat[train_idx]
        print("Cell-wise KNN estimation.")
        if self.x0_index is None:
            self.x0_index = knnx0_index(t[train_idx],
                                        z[train_idx],
                                        t,
                                        z,
                                        dt,
                                        self.config["n_neighbors"],
                                        hist_eq=False)
        c0, u0, s0, t0 = get_x0(c_in,
                                u_in,
                                s_in,
                                t[train_idx],
                                dt,
                                self.x0_index,
                                c0_init,
                                u0_init,
                                s0_init)
        if self.config["velocity_continuity"]:
            if self.x1_index is None:
                self.x1_index = knnx0_index(t[train_idx],
                                            z[train_idx],
                                            t,
                                            z,
                                            dt,
                                            self.config["n_neighbors"],
                                            forward=True,
                                            hist_eq=True)
            c1, u1, s1, t1 = get_x0(c_in,
                                    u_in,
                                    s_in,
                                    t[train_idx],
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
        return t, z

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
              cluster_key="clusters",
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
            print(f"Embedding X_{embed} not found! Set to None.")
            Xembed = np.nan*np.ones((self.adata.n_obs, 2))
            Xembed_train = Xembed[self.train_idx]
            Xembed_test = Xembed[self.test_idx]
            plot = False

        G = X.shape[1]

        if cluster_key is not None:
            cell_labels_raw = self.adata.obs[cluster_key].to_numpy() if cluster_key in self.adata.obs else np.array(['Unknown' for i in range(self.adata.n_obs)])
            self.cell_types_raw = np.unique(cell_labels_raw)
            self.label_dic, self.label_dic_rev = encode_type(self.cell_types_raw)

            self.n_type = len(self.cell_types_raw)
            self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
            self.cell_types = np.array([self.label_dic[self.cell_types_raw[i]] for i in range(self.n_type)])
        else:
            self.cell_labels = np.full(self.adata.n_obs, '')

        print("*********        Creating Training and Validation Datasets        *********")
        train_set = SCData(X[self.train_idx],
                           self.cell_labels[self.train_idx],
                           batch=self.batch[self.train_idx] if self.enable_cvae else None,
                           device=self.device)

        test_set = None
        if len(self.test_idx) > 0:
            test_set = SCData(X[self.test_idx],
                              self.cell_labels[self.test_idx],
                              batch=self.batch[self.test_idx] if self.enable_cvae else None,
                              device=self.device)

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
        param_nn += list(self.decoder.net_kc.parameters())
        if self.config['t_network']:
            param_nn += list(self.decoder.net_t.parameters())
        param_ode = [self.decoder.alpha_c,
                     self.decoder.alpha,
                     self.decoder.beta,
                     self.decoder.gamma]
        if self.config['train_scaling']:
            param_ode.extend([self.decoder.scaling_c,
                              self.decoder.scaling_u])
            if self.enable_cvae:
                param_ode.extend([self.decoder.scaling_s])
        if self.config['train_offset'] and self.enable_cvae:
            param_ode.extend([self.decoder.offset_c,
                              self.decoder.offset_u,
                              self.decoder.offset_s])
        if self.config['four_basis']:
            param_ode.extend([self.decoder.c0,
                              self.decoder.u0,
                              self.decoder.s0,
                              self.decoder.logit_pw])
        if self.config['train_ton']:
            param_ode.append(self.decoder.ton)

        optimizer = torch.optim.AdamW(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.AdamW(param_ode, lr=self.config["learning_rate_ode"], weight_decay=self.config["lambda"])

        for epoch in range(self.config["n_epochs"]):
            if epoch >= self.config["n_warmup"]:
                stop_training = self.train_epoch(data_loader,
                                                 test_set,
                                                 optimizer,
                                                 optimizer_ode,
                                                 self.config["k_alt"])
            else:
                stop_training = self.train_epoch(data_loader,
                                                 test_set,
                                                 optimizer,
                                                 None,
                                                 self.config["k_alt"])

            if epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0:
                elbo_train, _, _, _ = self.test(train_set,
                                                Xembed_train,
                                                f"train-{epoch+1}",
                                                False,
                                                gind,
                                                gene_plot,
                                                plot,
                                                figure_path)
                self.set_mode('train')
                elbo_test = -self.loss_test[-1] if len(self.loss_test) > 0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f} \t\t Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                break

        if self.config["run_2nd_stage"]:
            print("*********                      Stage  2                       *********")
            n_stage1 = epoch+1
            count_epoch = n_stage1
            n_test1 = len(self.loss_test)
            self.set_mode('eval', 'encoder')
            param_post = list(self.decoder.net_rho.parameters())
            param_post += list(self.decoder.net_kc.parameters())
            # if self.config['t_network']:
            #     param_post += list(self.decoder.net_t.parameters())
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

            for r in range(self.config['n_refine']):
                stop_training = (x0_change - x0_change_prev >= -0.01 and r > 1) or (x0_change < 0.01)
                if self.config['loss_std_type'] == 'deming' and noise_change > 0.001 and r < self.config['n_refine']-1:
                    self.update_std_noise(train_set)
                    stop_training = False
                if stop_training:
                    print(f"Stage 2: Early Stop Triggered at round {r}.")
                    break

                t, z = self.update_x0(X[:, :G//3], X[:, G//3:G//3*2], X[:, G//3*2:])
                if plot:
                    if np.sum(np.isnan(self.t0)) > 0:
                        print('t0 contains nan')
                    if np.sum(np.isnan(self.c0)) > 0 or np.sum(np.isnan(self.u0)) > 0 or np.sum(np.isnan(self.s0)) > 0:
                        print('c0, u0, or s0 contains nan')
                    if r == 0:
                        plot_time(self.t0.squeeze(), Xembed, save=f"{figure_path}/timet0-round{r+1}.png")
                        plot_time(t, Xembed, save=f"{figure_path}/time-round{r+1}.png")
                    train_idx = self.train_idx.detach().cpu().numpy()
                    t0_plot = self.t0[train_idx].squeeze()
                    for i in range(len(gind)):
                        idx = gind[i]
                        c0_plot = self.c0[train_idx, idx]
                        u0_plot = self.u0[train_idx, idx]
                        s0_plot = self.s0[train_idx, idx]
                        c_plot = X[:, :G//3][train_idx, idx]
                        u_plot = X[:, G//3:G//3*2][train_idx, idx]
                        s_plot = X[:, G//3*2:][train_idx, idx]
                        plot_sig_(t[train_idx],
                                  c_plot,
                                  u_plot,
                                  s_plot,
                                  cell_labels=cell_labels_raw[train_idx],
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
                self.t = torch.reshape(torch.tensor(t, dtype=torch.float, device=self.device), (-1, 1))
                self.z = torch.tensor(z, dtype=torch.float, device=self.device)
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
                print(f"*********             Velocity Refinement Round {r+1}             *********")

                for epoch in range(self.config["n_epochs_post"]):
                    if epoch == 0:
                        elbo_train, _, _, _ = self.test(train_set,
                                                        Xembed_train,
                                                        f"train-round{r+1}-first",
                                                        False,
                                                        gind,
                                                        gene_plot,
                                                        plot,
                                                        figure_path)
                        self.set_mode('train', 'decoder')
                    if epoch >= self.config["n_warmup_post"]:
                        stop_training = self.train_epoch(data_loader,
                                                         test_set,
                                                         optimizer_post,
                                                         optimizer_ode,
                                                         self.config["k_alt"],
                                                         'decoder')
                    else:
                        stop_training = self.train_epoch(data_loader,
                                                         test_set,
                                                         optimizer_post,
                                                         None,
                                                         self.config["k_alt"],
                                                         'decoder')

                    if stop_training or epoch == self.config["n_epochs_post"]:
                        elbo_train, _, _, _ = self.test(train_set,
                                                        Xembed_train,
                                                        f"train-round{r+1}-last",
                                                        False,
                                                        gind,
                                                        gene_plot,
                                                        plot,
                                                        figure_path)
                        self.set_mode('train', 'decoder')
                        elbo_test = -self.loss_test[-1] if len(self.loss_test) > n_test1 else -np.inf
                        print(f"Epoch {epoch+count_epoch+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f} \t\t Total Time = {convert_time(time.time()-start)}")

                    if stop_training:
                        print(f"*********       Round {r+1}: Early Stop Triggered at epoch {epoch+count_epoch+1}.       *********")
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
                plot_time(self.t0.detach().cpu().numpy().squeeze(), Xembed, save=f"{figure_path}/timet0-updated.png")
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
                              cell_labels=cell_labels_raw[self.train_idx.detach().cpu().numpy()],
                              title=gene_plot[i],
                              save=f"{figure_path}/sigx0-{gene_plot[i].replace('.', '-')}-updated.png")

        elbo_train, rec_train, klt_train, klz_train = self.test(train_set,
                                                                Xembed_train,
                                                                "train-final",
                                                                False,
                                                                gind,
                                                                gene_plot,
                                                                True,
                                                                figure_path)
        elbo_test, rec_test, klt_test, klz_test = self.test(test_set,
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

        self.loss_test.append(-elbo_test)
        self.rec_test.append(-rec_test)
        self.klt_test.append(-klt_test)
        self.klz_test.append(-klz_test)
        if plot:
            plot_train_loss(self.loss_train, range(1, len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            plot_train_loss(self.rec_train, range(1, len(self.rec_train)+1), save=f'{figure_path}/train_loss_rec_velovae.png')
            plot_train_loss(self.klt_train, range(1, len(self.klt_train)+1), save=f'{figure_path}/train_loss_klt_velovae.png')
            plot_train_loss(self.klz_train, range(1, len(self.klz_train)+1), save=f'{figure_path}/train_loss_klz_velovae.png')
            if self.config["test_iter"] > 0:
                plot_test_loss(self.loss_test, [i*self.config["test_iter"] for i in range(1, len(self.loss_test)+1)], save=f'{figure_path}/test_loss_velovae.png')
                plot_test_loss(self.rec_test, [i*self.config["test_iter"] for i in range(1, len(self.rec_test)+1)], save=f'{figure_path}/test_loss_rec_velovae.png')
                plot_test_loss(self.klt_test, [i*self.config["test_iter"] for i in range(1, len(self.klt_test)+1)], save=f'{figure_path}/test_loss_klt_velovae.png')
                plot_test_loss(self.klz_test, [i*self.config["test_iter"] for i in range(1, len(self.klz_test)+1)], save=f'{figure_path}/test_loss_klz_velovae.png')
        print(f"Final: Train ELBO = {elbo_train:.3f},\tTest ELBO = {elbo_test:.3f}")
        print(f"*********              Finished. Total Time = {convert_time(time.time()-start)}             *********")

    def save_model(self, file_path, enc_name='encoder', dec_name='decoder'):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), f"{file_path}/{enc_name}.pt")
        torch.save(self.decoder.state_dict(), f"{file_path}/{dec_name}.pt")

    def save_anndata(self, file_path, file_name=None):
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)

        key = self.config['key']
        if key == 'fullvb':
            if self.enable_cvae:
                for i in range(self.n_batch):
                    if self.config['log_params']:
                        self.adata.var[f"{key}_alpha_c_{i}"] = np.exp(self.decoder.alpha_c[i, 0].detach().cpu().numpy())
                        self.adata.var[f"{key}_alpha_{i}"] = np.exp(self.decoder.alpha[i, 0].detach().cpu().numpy())
                        self.adata.var[f"{key}_beta_{i}"] = np.exp(self.decoder.beta[i, 0].detach().cpu().numpy())
                        self.adata.var[f"{key}_gamma_{i}"] = np.exp(self.decoder.gamma[i, 0].detach().cpu().numpy())
                    else:
                        self.adata.var[f"{key}_alpha_c_{i}"] = F.softplus(self.decoder.alpha_c[i, 0].detach().cpu()).numpy()
                        self.adata.var[f"{key}_alpha_{i}"] = F.softplus(self.decoder.alpha[i, 0].detach().cpu()).numpy()
                        self.adata.var[f"{key}_beta_{i}"] = F.softplus(self.decoder.beta[i, 0].detach().cpu()).numpy()
                        self.adata.var[f"{key}_gamma_{i}"] = F.softplus(self.decoder.gamma[i, 0].detach().cpu()).numpy()
                    self.adata.var[f"{key}_logstd_alpha_c_{i}"] = np.exp(self.decoder.alpha_c[i, 1].detach().cpu().numpy())
                    self.adata.var[f"{key}_logstd_alpha_{i}"] = np.exp(self.decoder.alpha[i, 1].detach().cpu().numpy())
                    self.adata.var[f"{key}_logstd_beta_{i}"] = np.exp(self.decoder.beta[i, 1].detach().cpu().numpy())
                    self.adata.var[f"{key}_logstd_gamma_{i}"] = np.exp(self.decoder.gamma[i, 1].detach().cpu().numpy())
            else:
                if self.config['log_params']:
                    self.adata.var[f"{key}_alpha_c"] = np.exp(self.decoder.alpha_c[0].detach().cpu().numpy())
                    self.adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha[0].detach().cpu().numpy())
                    self.adata.var[f"{key}_beta"] = np.exp(self.decoder.beta[0].detach().cpu().numpy())
                    self.adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma[0].detach().cpu().numpy())
                else:
                    self.adata.var[f"{key}_alpha_c"] = F.softplus(self.decoder.alpha_c[0].detach().cpu()).numpy()
                    self.adata.var[f"{key}_alpha"] = F.softplus(self.decoder.alpha[0].detach().cpu()).numpy()
                    self.adata.var[f"{key}_beta"] = F.softplus(self.decoder.beta[0].detach().cpu()).numpy()
                    self.adata.var[f"{key}_gamma"] = F.softplus(self.decoder.gamma[0].detach().cpu()).numpy()
                self.adata.var[f"{key}_logstd_alpha_c"] = np.exp(self.decoder.alpha_c[1].detach().cpu().numpy())
                self.adata.var[f"{key}_logstd_alpha"] = np.exp(self.decoder.alpha[1].detach().cpu().numpy())
                self.adata.var[f"{key}_logstd_beta"] = np.exp(self.decoder.beta[1].detach().cpu().numpy())
                self.adata.var[f"{key}_logstd_gamma"] = np.exp(self.decoder.gamma[1].detach().cpu().numpy())
        else:
            if self.enable_cvae:
                for i in range(self.n_batch):
                    if self.config['log_params']:
                        self.adata.var[f"{key}_alpha_c_{i}"] = np.exp(self.decoder.alpha_c[i].detach().cpu().numpy())
                        self.adata.var[f"{key}_alpha_{i}"] = np.exp(self.decoder.alpha[i].detach().cpu().numpy())
                        self.adata.var[f"{key}_beta_{i}"] = np.exp(self.decoder.beta[i].detach().cpu().numpy())
                        self.adata.var[f"{key}_gamma_{i}"] = np.exp(self.decoder.gamma[i].detach().cpu().numpy())
                    else:
                        self.adata.var[f"{key}_alpha_c_{i}"] = F.softplus(self.decoder.alpha_c[i].detach().cpu()).numpy()
                        self.adata.var[f"{key}_alpha_{i}"] = F.softplus(self.decoder.alpha[i].detach().cpu()).numpy()
                        self.adata.var[f"{key}_beta_{i}"] = F.softplus(self.decoder.beta[i].detach().cpu()).numpy()
                        self.adata.var[f"{key}_gamma_{i}"] = F.softplus(self.decoder.gamma[i].detach().cpu()).numpy()
            else:
                if self.config['log_params']:
                    self.adata.var[f"{key}_alpha_c"] = np.exp(self.decoder.alpha_c.detach().cpu().numpy())
                    self.adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
                    self.adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
                    self.adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
                else:
                    self.adata.var[f"{key}_alpha_c"] = F.softplus(self.decoder.alpha_c.detach().cpu()).numpy()
                    self.adata.var[f"{key}_alpha"] = F.softplus(self.decoder.alpha.detach().cpu()).numpy()
                    self.adata.var[f"{key}_beta"] = F.softplus(self.decoder.beta.detach().cpu()).numpy()
                    self.adata.var[f"{key}_gamma"] = F.softplus(self.decoder.gamma.detach().cpu()).numpy()
        if self.enable_cvae:
            for i in range(self.n_batch):
                if self.config['log_params']:
                    self.adata.var[f"{key}_scaling_c_{i}"] = np.exp(self.decoder.scaling_c[i].detach().cpu().numpy())
                    self.adata.var[f"{key}_scaling_u_{i}"] = np.exp(self.decoder.scaling_u[i].detach().cpu().numpy())
                    self.adata.var[f"{key}_scaling_s_{i}"] = np.exp(self.decoder.scaling_s[i].detach().cpu().numpy())
                else:
                    self.adata.var[f"{key}_scaling_c_{i}"] = F.softplus(self.decoder.scaling_c[i].detach().cpu()).numpy()
                    self.adata.var[f"{key}_scaling_u_{i}"] = F.softplus(self.decoder.scaling_u[i].detach().cpu()).numpy()
                    self.adata.var[f"{key}_scaling_s_{i}"] = F.softplus(self.decoder.scaling_s[i].detach().cpu()).numpy()
                self.adata.var[f"{key}_offset_c_{i}"] = self.decoder.offset_c[i].detach().cpu().numpy()
                self.adata.var[f"{key}_offset_u_{i}"] = self.decoder.offset_u[i].detach().cpu().numpy()
                self.adata.var[f"{key}_offset_s_{i}"] = self.decoder.offset_s[i].detach().cpu().numpy()
        else:
            if self.config['log_params']:
                self.adata.var[f"{key}_scaling_c"] = np.exp(self.decoder.scaling_c.detach().cpu().numpy())
                self.adata.var[f"{key}_scaling_u"] = np.exp(self.decoder.scaling_u.detach().cpu().numpy())
                self.adata.var[f"{key}_scaling_s"] = np.exp(self.decoder.scaling_s.detach().cpu().numpy())
            else:
                self.adata.var[f"{key}_scaling_c"] = F.softplus(self.decoder.scaling_c.detach().cpu()).numpy()
                self.adata.var[f"{key}_scaling_u"] = F.softplus(self.decoder.scaling_u.detach().cpu()).numpy()
                self.adata.var[f"{key}_scaling_s"] = F.softplus(self.decoder.scaling_s.detach().cpu()).numpy()
            self.adata.var[f"{key}_offset_c"] = self.decoder.offset_c.detach().cpu().numpy()
            self.adata.var[f"{key}_offset_u"] = self.decoder.offset_u.detach().cpu().numpy()
            self.adata.var[f"{key}_offset_s"] = self.decoder.offset_s.detach().cpu().numpy()
        self.adata.var[f"{key}_sigma_c"] = self.decoder.sigma_c.detach().cpu().numpy()
        self.adata.var[f"{key}_sigma_u"] = self.decoder.sigma_u.detach().cpu().numpy()
        self.adata.var[f"{key}_sigma_s"] = self.decoder.sigma_s.detach().cpu().numpy()
        self.adata.var[f"{key}_ton"] = self.decoder.ton.detach().cpu().numpy()
        self.adata.varm[f"{key}_basis"] = F.softmax(self.decoder.logit_pw, 1).detach().cpu().numpy()

        x = np.concatenate((self.adata_atac.layers['Mc'].A if issparse(self.adata_atac.layers['Mc']) else self.adata_atac.layers['Mc'],
                            self.adata.layers['Mu'].A if issparse(self.adata.layers['Mu']) else self.adata.layers['Mu'],
                            self.adata.layers['Ms'].A if issparse(self.adata.layers['Ms']) else self.adata.layers['Ms']), 1).astype(float)
        G = x.shape[1]//3
        out, _, _, _, _ = self.pred_all(x, "both")
        chat, uhat, shat, t, std_t, t_, z, std_z = out["chat"], out["uhat"], out["shat"], out["mu_t"], out["std_t"], out["t"], out["mu_z"], out["std_z"]
        std_c = np.clip(np.nanstd(x[:, :G], axis=0), 0.01, None)
        std_u = np.clip(np.nanstd(x[:, G:G*2], axis=0), 0.01, None)
        std_s = np.clip(np.nanstd(x[:, G*2:], axis=0), 0.01, None)
        diff_c = (x[:, :G] - chat) / std_c
        diff_u = (x[:, G:G*2] - uhat) / std_u
        diff_s = (x[:, G*2:] - shat) / std_s
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

        if self.enable_cvae and self.ref_batch >= 0:
            self.adata.obs[f"{key}_time_batch"] = t_
            self.adata.obsm[f"{key}_t_batch"] = t
            self.adata.obsm[f"{key}_std_t_batch"] = std_t
            self.adata.obsm[f"{key}_z_batch"] = z
            self.adata.obsm[f"{key}_std_z_batch"] = std_z
            out, _, _, _, _ = self.pred_all(x, "both", batch=torch.full((self.adata.n_obs,), self.ref_batch, dtype=int, device=self.device))
            chat, uhat, shat, t, std_t, t_, z, std_z = out["chat"], out["uhat"], out["shat"], out["mu_t"], out["std_t"], out["t"], out["mu_z"], out["std_z"]
        self.adata.obs[f"{key}_time"] = t_
        self.adata.obsm[f"{key}_t"] = t
        self.adata.obsm[f"{key}_std_t"] = std_t
        self.adata.obsm[f"{key}_z"] = z
        self.adata.obsm[f"{key}_std_z"] = std_z
        self.adata.layers[f"{key}_chat"] = chat
        self.adata.layers[f"{key}_uhat"] = uhat
        self.adata.layers[f"{key}_shat"] = shat

        rho = np.zeros_like(uhat)
        kc = np.zeros_like(uhat)
        with torch.no_grad():
            B = min(uhat.shape[0]//10, 1000)
            Nb = uhat.shape[0] // B
            if Nb*B < uhat.shape[0]:
                Nb += 1
            for n in range(Nb):
                i = n*B
                j = min([(n+1)*B, uhat.shape[0]])
                if self.enable_cvae:
                    y_onehot = F.one_hot(self.batch.detach()[i:j], self.n_batch).float()
                    z_onehot = torch.cat((torch.tensor(z[i:j], dtype=torch.float, device=self.device), y_onehot), 1)
                    rho_batch = self.decoder.net_rho(z_onehot)
                    if self.config['rna_only']:
                        kc_batch = torch.full_like(rho_batch, 1.0)
                    else:
                        kc_batch = self.decoder.net_kc(z_onehot if self.config["indicator_arch"] == 'parallel' else rho_batch)
                else:
                    rho_batch = self.decoder.net_rho(torch.tensor(z[i:j], dtype=torch.float, device=self.device))
                    if self.config['rna_only']:
                        kc_batch = torch.full_like(rho_batch, 1.0)
                    else:
                        kc_batch = self.decoder.net_kc(torch.tensor(z[i:j], dtype=torch.float, device=self.device) if self.config["indicator_arch"] == 'parallel' else rho_batch)
                rho[i:j] = rho_batch.detach().cpu().numpy()
                kc[i:j] = kc_batch.detach().cpu().numpy()

        self.adata.layers[f"{key}_rho"] = rho
        self.adata.layers[f"{key}_kc"] = kc

        self.adata.obs[f"{key}_t0"] = self.t0.detach().cpu().numpy().squeeze() if self.t0 is not None else 0
        self.adata.layers[f"{key}_c0"] = self.c0.detach().cpu().numpy() if self.c0 is not None else np.zeros_like(self.adata.layers['Mu'])
        self.adata.layers[f"{key}_u0"] = self.u0.detach().cpu().numpy() if self.u0 is not None else np.zeros_like(self.adata.layers['Mu'])
        self.adata.layers[f"{key}_s0"] = self.s0.detach().cpu().numpy() if self.s0 is not None else np.zeros_like(self.adata.layers['Mu'])

        self.adata.uns[f"{key}_train_idx"] = self.train_idx.detach().cpu().numpy()
        self.adata.uns[f"{key}_test_idx"] = self.test_idx.detach().cpu().numpy()

        print("Computing velocity.")
        rna_velocity_vae(self.adata, self.adata_atac, key, batch_key=self.config['batch_key'], use_raw=False, rna_only=self.config['rna_only'])

        if file_name is not None:
            print("Writing anndata output to file.")
            self.adata.write_h5ad(f"{file_path}/{file_name}")
