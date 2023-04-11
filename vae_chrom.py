import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from velovae.plotting import plot_time, plot_train_loss, plot_test_loss
from velovae.plotting_chrom import plot_sig_, plot_sig, plot_vel, plot_phase
from .model_util import hist_equal, convert_time, get_gene_index, elbo_collapsed_categorical, assign_gene_mode, find_dirichlet_param
from .model_util_chrom import pred_exp, ode_numpy, init_params, get_ts_global, reinit_params
from .model_util_chrom import kl_gaussian, reparameterize, knn_approx, knnx0, cosine_similarity
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
                 N1=512,
                 N2=256,
                 checkpoint=None):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(Cin+dim_cond, N1)
        self.bn1 = nn.BatchNorm1d(num_features=N1)
        self.dpt1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(N1, N2)
        self.bn2 = nn.BatchNorm1d(num_features=N2)
        self.dpt2 = nn.Dropout(p=0.2)

        self.net = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                 self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2,
                                 )

        self.fc_mu_t = nn.Linear(N2, 1)
        self.spt1 = nn.Softplus()
        self.fc_std_t = nn.Linear(N2, 1)
        self.spt2 = nn.Softplus()
        self.fc_mu_z = nn.Linear(N2, dim_z)
        self.fc_std_z = nn.Linear(N2, dim_z)
        self.spt3 = nn.Softplus()

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

    def forward(self, data_in, condition=None):
        if condition is not None:
            data_in = torch.cat((data_in, condition), 1)
        h = self.net(data_in)
        mu_tx, std_tx = self.spt1(self.fc_mu_t(h)), self.spt2(self.fc_std_t(h))
        mu_zx, std_zx = self.fc_mu_z(h), self.spt3(self.fc_std_z(h))
        return mu_tx, std_tx, mu_zx, std_zx


class Decoder(nn.Module):
    def __init__(self,
                 adata,
                 adata_atac,
                 train_idx,
                 dim_z,
                 dim_cond=0,
                 N1=256,
                 N2=512,
                 indicator_arch='parallel',
                 t_network=False,
                 params_mult=1,
                 p=98,
                 tmax=1,
                 reinit=False,
                 init_ton_zero=False,
                 init_method='steady',
                 init_key=None,
                 init_type=None,
                 cluster_key='clusters',
                 assign_type='auto',
                 ks_test_thred=0.05,
                 n_cluster_thred=3,
                 std_alpha_prior=0.1,
                 checkpoint=None):
        super(Decoder, self).__init__()
        self.adata = adata
        self.adata_atac = adata_atac
        self.train_idx = train_idx
        self.dim_cond = dim_cond
        if dim_cond == 1:
            dim_cond = 0
        self.cvae = True if dim_cond > 1 else False
        self.indicator_arch = indicator_arch
        self.t_network = t_network
        self.params_mult = params_mult
        self.reinit = reinit
        self.init_ton_zero = init_ton_zero
        self.init_method = init_method
        self.init_key = init_key
        self.init_type = init_type
        self.cluster_key = cluster_key
        self.assign_type = assign_type
        self.ks_test_thred = ks_test_thred
        self.n_cluster_thred = n_cluster_thred
        self.std_alpha_prior = std_alpha_prior
        self.checkpoint = checkpoint
        self.construct_nn(dim_z, dim_cond, N1, N2, p, tmax)

    def construct_nn(self, dim_z, dim_cond, N1, N2, p, tmax):
        G = self.adata.n_vars
        self.set_shape(G, dim_cond)

        self.fc1 = nn.Linear(dim_z+dim_cond, N1)
        self.bn1 = nn.BatchNorm1d(num_features=N1)
        self.dpt1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(N1, N2)
        self.bn2 = nn.BatchNorm1d(num_features=N2)
        self.dpt2 = nn.Dropout(p=0.2)

        self.fc_out1 = nn.Linear(N2, G)

        self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                     self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)

        self.fc3 = nn.Linear(dim_z+dim_cond if self.indicator_arch == 'parallel' else G, N1)
        self.bn3 = nn.BatchNorm1d(num_features=N1)
        self.dpt3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(N1, N2)
        self.bn4 = nn.BatchNorm1d(num_features=N2)
        self.dpt4 = nn.Dropout(p=0.2)

        self.fc_out2 = nn.Linear(N2, G)

        self.net_kc = nn.Sequential(self.fc3, self.bn3, nn.LeakyReLU(), self.dpt3,
                                    self.fc4, self.bn4, nn.LeakyReLU(), self.dpt4)

        if self.t_network:
            self.fct = nn.Linear(1+dim_cond, 1)
            self.net_t = self.fct
        self.rho, self.kc = None, None

        if self.checkpoint is not None:
            self.alpha_c = nn.Parameter(torch.empty(self.params_shape))
            self.alpha = nn.Parameter(torch.empty(self.params_shape))
            self.beta = nn.Parameter(torch.empty(self.params_shape))
            self.gamma = nn.Parameter(torch.empty(self.params_shape))

            self.register_buffer('scaling_c', torch.empty(G))
            self.register_buffer('scaling', torch.empty(G))
            self.register_buffer('sigma_c', torch.empty(G))
            self.register_buffer('sigma_u', torch.empty(G))
            self.register_buffer('sigma_s', torch.empty(G))
            self.register_buffer('zero_vec', torch.empty(G))

            self.ton = nn.Parameter(torch.empty(G))
            self.toff = nn.Parameter(torch.empty(G))
            self.c0 = nn.Parameter(torch.empty(G))
            self.u0 = nn.Parameter(torch.empty(G))
            self.s0 = nn.Parameter(torch.empty(G))

            self.load_state_dict(torch.load(self.checkpoint))
        else:
            c = self.adata_atac.layers['Mc'][self.train_idx]
            u = self.adata.layers['Mu'][self.train_idx]
            s = self.adata.layers['Ms'][self.train_idx]
            self.init_weights()
            self.init_ode(c, u, s, p, tmax)
            self.init_x0(c, s, s)

    def set_shape(self, G, dim_cond):
        if isinstance(self.params_mult, (tuple, list, np.ndarray)) and (len(self.params_mult) > 1):
            if self.cvae:
                self.params_shape = (dim_cond, len(self.params_mult), G)
            else:
                self.params_shape = (len(self.params_mult), G)
        else:
            if self.cvae:
                self.params_shape = (dim_cond, G)
            else:
                self.params_shape = G
        if isinstance(self.params_mult, int):
            self.params_mult = [self.params_mult]

    def get_tensor(self, x):
        if len(self.params_mult) > 1:
            if self.cvae:
                return torch.tensor(np.tile(np.stack([self.params_mult[0]*np.log(x), self.params_mult[1]*np.ones(self.params_shape[2])]), (self.params_shape[0], 1, 1)))
            else:
                return torch.tensor(np.stack([self.params_mult[0]*np.log(x), self.params_mult[1]*np.ones(self.params_shape[1])]))
        else:
            if self.cvae:
                return torch.tensor(np.tile(self.params_mult[0]*np.log(x), (self.params_shape[0], 1)))
            else:
                return torch.tensor(self.params_mult[0]*np.log(x))

    def init_weights(self):
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
        if self.t_network:
            for m in self.net_t.modules():
                if isinstance(m, nn.Linear):
                    nn.init.eye_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

        for m in [self.fc_out1, self.fc_out2]:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def init_ode(self, c, u, s, p, tmax):
        G = self.adata.n_vars
        x = np.concatenate((c, u, s), 1)
        out = init_params(x, p, fit_scaling=True)
        alpha_c, alpha, beta, gamma, scaling_c, scaling, toff, c0, u0, s0, sigma_c, sigma_u, sigma_s, t = out

        dyn_mask = (t > tmax*0.01) & (np.abs(t-toff) > tmax*0.01)
        w = np.sum(((t < toff) & dyn_mask), 0) / (np.sum(dyn_mask, 0) + 1e-10)
        w = assign_gene_mode(self.adata, w, self.assign_type, self.ks_test_thred, self.std_alpha_prior, self.n_cluster_thred)
        self.adata.var["w_init"] = w
        logit_pw = 0.5*(np.log(w+1e-10) - np.log(1-w-1e-10))
        logit_pw = np.stack([logit_pw, -logit_pw], 1)

        print(f"Initial induction: {np.sum(w >= 0.5)}, repression: {np.sum(w < 0.5)} out of {G}.")

        if self.init_method == "random":
            print("Random Initialization.")
            self.alpha_c = nn.Parameter(torch.normal(0.0, 0.5, size=self.params_shape))
            self.alpha = nn.Parameter(torch.normal(0.0, 1.0, size=self.params_shape))
            self.beta = nn.Parameter(torch.normal(0.0, 0.5, size=self.params_shape))
            self.gamma = nn.Parameter(torch.normal(0.0, 0.5, size=self.params_shape))
            self.c0 = nn.Parameter(torch.tensor(np.log(c0+1e-10)))
            self.u0 = nn.Parameter(torch.tensor(np.log(u0+1e-10)))
            self.s0 = nn.Parameter(torch.tensor(np.log(s0+1e-10)))
            self.ton = nn.Parameter(torch.ones(G)*(-10))
            self.toff = nn.Parameter(torch.tensor(G)*(tmax/2))

            self.register_buffer('scaling_c', torch.tensor(np.log(scaling_c)))
            self.register_buffer('scaling', torch.tensor(np.log(scaling)))
            self.register_buffer('sigma_c', torch.tensor(np.log(sigma_c)))
            self.register_buffer('sigma_u', torch.tensor(np.log(sigma_u)))
            self.register_buffer('sigma_s', torch.tensor(np.log(sigma_s)))
        elif self.init_method == "tprior":
            print("Initialization using prior time.")
            self.alpha_c_ = alpha_c
            self.alpha_ = alpha
            self.beta_ = beta
            self.gamma_ = gamma
            self.scaling_c_ = scaling_c
            self.scaling_ = scaling
            self.toff_ = toff
            self.c0_ = c0
            self.u0_ = u0
            self.s0_ = s0
            self.sigma_c_ = sigma_c
            self.sigma_u_ = sigma_u
            self.sigma_s_ = sigma_s
            self.t_ = t

            t_prior = self.adata.obs[self.init_key].to_numpy()
            t_prior = t_prior[self.train_idx]
            std_t = (np.std(t_prior)+1e-3)*0.2
            self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
            self.t_init -= self.t_init.min()
            self.t_init = self.t_init
            self.t_init = self.t_init/self.t_init.max()*tmax
            if self.reinit:
                toff = get_ts_global(self.t_init, u/scaling, s, 95)
                alpha_c, alpha, beta, gamma, ton = reinit_params(c/scaling_c, u/scaling, s, self.t_init, toff)

            self.alpha_c = nn.Parameter(self.get_tensor(alpha_c))
            self.alpha = nn.Parameter(self.get_tensor(alpha))
            self.beta = nn.Parameter(self.get_tensor(beta))
            self.gamma = nn.Parameter(self.get_tensor(gamma))

            self.c0 = nn.Parameter(torch.tensor(np.log(c0+1e-10)))
            self.u0 = nn.Parameter(torch.tensor(np.log(u0+1e-10)))
            self.s0 = nn.Parameter(torch.tensor(np.log(s0+1e-10)))

            if self.init_ton_zero or (not self.reinit):
                self.ton = nn.Parameter((torch.ones(G)*(-10)))
            else:
                self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10)))
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10)))

            self.register_buffer('scaling_c', torch.tensor(np.log(scaling_c)))
            self.register_buffer('scaling', torch.tensor(np.log(scaling)))
            self.register_buffer('sigma_c', torch.tensor(np.log(sigma_c)))
            self.register_buffer('sigma_u', torch.tensor(np.log(sigma_u)))
            self.register_buffer('sigma_s', torch.tensor(np.log(sigma_s)))
        else:
            print("Initialization using the steady-state and dynamical models.")
            self.alpha_c_ = alpha_c
            self.alpha_ = alpha
            self.beta_ = beta
            self.gamma_ = gamma
            self.scaling_c_ = scaling_c
            self.scaling_ = scaling
            self.toff_ = toff
            self.c0_ = c0
            self.u0_ = u0
            self.s0_ = s0
            self.sigma_c_ = sigma_c
            self.sigma_u_ = sigma_u
            self.sigma_s_ = sigma_s
            self.t_ = t

            if self.reinit:
                if self.init_key is not None:
                    self.t_init = self.adata.obs[self.init_key].to_numpy()[self.train_idx]
                else:
                    t = t+np.random.rand(t.shape[0], t.shape[1]) * 1e-3
                    t_eq = np.zeros(t.shape)
                    n_bin = t.shape[0]//50+1
                    for i in range(t.shape[1]):
                        t_eq[:, i] = hist_equal(t[:, i], tmax, 0.9, n_bin)
                    self.t_init = np.quantile(t_eq, 0.5, 1)
                toff = get_ts_global(self.t_init, c/scaling_c, u/scaling, s, 95)
                alpha_c, alpha, beta, gamma, ton = reinit_params(c/scaling_c, u/scaling, s, self.t_init, toff)

            self.alpha_c = nn.Parameter(self.get_tensor(alpha_c))
            self.alpha = nn.Parameter(self.get_tensor(alpha))
            self.beta = nn.Parameter(self.get_tensor(beta))
            self.gamma = nn.Parameter(self.get_tensor(gamma))

            self.c0 = nn.Parameter(torch.tensor(np.log(c0+1e-10)))
            self.u0 = nn.Parameter(torch.tensor(np.log(u0+1e-10)))
            self.s0 = nn.Parameter(torch.tensor(np.log(s0+1e-10)))

            if self.init_ton_zero or (not self.reinit):
                self.ton = nn.Parameter((torch.ones(G)*(-10)))
            else:
                self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10)))
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10)))

            self.register_buffer('scaling_c', torch.tensor(np.log(scaling_c)))
            self.register_buffer('scaling', torch.tensor(np.log(scaling)))
            self.register_buffer('sigma_c', torch.tensor(np.log(sigma_c)))
            self.register_buffer('sigma_u', torch.tensor(np.log(sigma_u)))
            self.register_buffer('sigma_s', torch.tensor(np.log(sigma_s)))
            self.logit_pw = nn.Parameter(torch.tensor(logit_pw))

        self.register_buffer('zero_vec', torch.zeros_like(self.u0))

    def init_x0(self, c, u, s):
        G = self.adata.n_vars
        if self.init_type is None:
            return
        elif self.init_type == "random":
            rv_c = stats.gamma(1.0, 0, 4.0)
            rv_u = stats.gamma(1.0, 0, 4.0)
            rv_s = stats.gamma(1.0, 0, 4.0)
            r_c_gamma = rv_c.rvs(size=(G))
            r_u_gamma = rv_u.rvs(size=(G))
            r_s_gamma = rv_s.rvs(size=(G))
            r_c_bern = stats.bernoulli(0.02).rvs(size=(G))
            r_u_bern = stats.bernoulli(0.02).rvs(size=(G))
            r_s_bern = stats.bernoulli(0.02).rvs(size=(G))
            c_top = np.quantile(c, 0.99, 0)
            u_top = np.quantile(u, 0.99, 0)
            s_top = np.quantile(s, 0.99, 0)
            c0, u0, s0 = c_top*r_c_gamma*r_c_bern, u_top*r_u_gamma*r_u_bern, s_top*r_s_gamma*r_s_bern
            self.c0 = nn.Parameter(torch.tensor(np.log(c0+1e-10)))
            self.u0 = nn.Parameter(torch.tensor(np.log(u0+1e-10)))
            self.s0 = nn.Parameter(torch.tensor(np.log(s0+1e-10)))
        else:
            cell_labels = self.adata.obs[self.cluster_key].to_numpy()[self.train_idx]
            cell_mask = cell_labels == self.init_type
            self.c0 = nn.Parameter(torch.tensor(np.log(c[cell_mask].mean(0)+1e-10)))
            self.u0 = nn.Parameter(torch.tensor(np.log(u[cell_mask].mean(0)+1e-10)))
            self.s0 = nn.Parameter(torch.tensor(np.log(s[cell_mask].mean(0)+1e-10)))

    def get_param(self, x):
        if x == 'ton':
            return self.ton
        elif x == 'c0':
            return self.c0
        elif x == 'u0':
            return self.u0
        elif x == 's0':
            return self.s0
        elif x == 'alpha_c':
            return self.alpha_c
        elif x == 'alpha':
            return self.alpha
        elif x == 'beta':
            return self.beta
        elif x == 'gamma':
            return self.gamma

    def reparameterize(self, condition=None, sample=True):
        if len(self.params_mult) > 1:
            if sample:
                G = self.adata.n_vars
                eps = torch.randn((4, G), device=self.alpha_c.device)
                if self.cvae:
                    alpha_c = torch.exp(self.alpha_c[:, 0] + eps[0]*(self.alpha_c[:, 1].exp()))
                    alpha = torch.exp(self.alpha[:, 0] + eps[1]*(self.alpha[:, 1].exp()))
                    beta = torch.exp(self.beta[:, 0] + eps[2]*(self.beta[:, 1].exp()))
                    gamma = torch.exp(self.gamma[:, 0] + eps[3]*(self.gamma[:, 1].exp()))
                else:
                    alpha_c = torch.exp(self.alpha_c[0] + eps[0]*(self.alpha_c[1].exp()))
                    alpha = torch.exp(self.alpha[0] + eps[1]*(self.alpha[1].exp()))
                    beta = torch.exp(self.beta[0] + eps[2]*(self.beta[1].exp()))
                    gamma = torch.exp(self.gamma[0] + eps[3]*(self.gamma[1].exp()))
            else:
                if self.cvae:
                    alpha_c = self.alpha_c[:, 0].exp()
                    alpha = self.alpha[:, 0].exp()
                    beta = self.beta[:, 0].exp()
                    gamma = self.gamma[:, 0].exp()
                else:
                    alpha_c = self.alpha_c[0].exp()
                    alpha = self.alpha[0].exp()
                    beta = self.beta[0].exp()
                    gamma = self.gamma[0].exp()
        else:
            alpha_c = self.alpha_c.exp()
            alpha = self.alpha.exp()
            beta = self.beta.exp()
            gamma = self.gamma.exp()

        if condition is not None:
            alpha_c = torch.mm(condition, alpha_c)
            alpha = torch.mm(condition, alpha)
            beta = torch.mm(condition, beta)
            gamma = torch.mm(condition, gamma)

        return alpha_c, alpha, beta, gamma

    def forward(self, t, z, c0=None, u0=None, s0=None, t0=None, condition=None, neg_slope=0.0, sample=True, two_basis=False, return_velocity=False):
        alpha_c, alpha, beta, gamma = self.reparameterize(condition, sample)
        if condition is None:
            self.rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
            self.kc = torch.sigmoid(self.fc_out2(self.net_kc(z if self.indicator_arch == 'parallel' else self.rho)))
            t_ = F.softplus(self.net_t(t)) if self.t_network else t
        else:
            self.rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z, condition), 1))))
            self.kc = torch.sigmoid(self.fc_out2(self.net_kc(torch.cat((z, condition), 1) if self.indicator_arch == 'parallel' else torch.cat((self.rho, condition), 1))))
            t_ = F.softplus(self.net_t(torch.cat((t, condition), 1))) if self.t_network else t
        if (c0 is None) or (u0 is None) or (s0 is None) or (t0 is None):
            if two_basis:
                zero_mtx = torch.zeros_like(self.rho)
                kc = torch.stack([self.kc, zero_mtx], 1)
                rho = torch.stack([self.rho, zero_mtx], 1)
                c0 = torch.stack([self.zero_vec, self.c0.exp()])
                u0 = torch.stack([self.zero_vec, self.u0.exp()])
                s0 = torch.stack([self.zero_vec, self.s0.exp()])
                tau = torch.stack([F.leaky_relu(t_ - self.ton.exp(), neg_slope) for i in range(2)], 1)
            else:
                kc = self.kc
                rho = self.rho
                c0 = self.zero_vec
                u0 = self.zero_vec
                s0 = self.zero_vec
                tau = F.leaky_relu(t_ - self.ton.exp(), neg_slope)
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
            chat, uhat, shat = pred_exp(F.leaky_relu(t_ - t0, neg_slope),
                                        c0/self.scaling_c.exp(),
                                        u0/self.scaling.exp(),
                                        s0,
                                        self.kc,
                                        alpha_c,
                                        self.rho,
                                        alpha,
                                        beta,
                                        gamma)
        chat = chat * torch.exp(self.scaling_c)
        uhat = uhat * torch.exp(self.scaling)
        if return_velocity:
            vc = self.kc * alpha_c - alpha_c * chat
            vu = self.rho * alpha * chat - beta * uhat
            vs = beta * uhat - gamma * shat
            return F.relu(chat), F.relu(uhat), F.relu(shat), t_, vc, vu, vs
        else:
            return F.relu(chat), F.relu(uhat), F.relu(shat), t_


class VAEChrom():
    def __init__(self,
                 adata,
                 adata_atac,
                 dim_z=20,
                 batch_key=None,
                 ref_batch=None,
                 device='cpu',
                 hidden_size=(512, 256, 256, 512),
                 full_vb=False,
                 indicator_arch='parallel',
                 t_network=False,
                 train_x0=False,
                 knn_use_pred=True,
                 train_3rd_stage=False,
                 velocity_continuity=False,
                 two_basis=False,
                 tmax=1,
                 reinit_params=False,
                 init_method='steady',
                 init_key=None,
                 tprior=None,
                 init_type=None,
                 init_ton_zero=True,
                 unit_scale=False,
                 learning_rate=None,
                 checkpoints=[None, None],
                 plot_init=False,
                 gene_plot=[],
                 cluster_key='clusters',
                 figure_path='figures'):
        if ('Mc' not in adata_atac.layers) or ('Mu' not in adata.layers) or ('Ms') not in adata.layers:
            print('Chromatin/Unspliced/Spliced count matrices not found in the layers! Exit the program...')
            return

        self.is_full_vb = full_vb
        self.config = {
            # model parameters
            "dim_z": dim_z,
            "hidden_size": hidden_size,
            "mode": 'fullvb' if full_vb else 'vae',
            "indicator_arch": indicator_arch,
            "t_network": t_network,
            "batch_key": batch_key,
            "ref_batch": ref_batch,
            "reinit_params": reinit_params,
            "init_ton_zero": init_ton_zero,
            "unit_scale": unit_scale,
            "tmax": tmax,
            "init_method": init_method,
            "init_key": init_key,
            "tprior": tprior,
            "init_type": init_type,
            "std_z_prior": 0.01,
            "tail": 0.01,
            "std_t_scaling": 0.05,
            "n_neighbors": 10,
            "dt": (0.03, 0.06),

            # training parameters
            "n_epochs": 1000,
            "n_epochs_post": 1000,
            "n_epochs_final": 1000,
            "batch_size": 256,
            "learning_rate": None,
            "learning_rate_ode": None,
            "learning_rate_post": None,
            "learning_rate_final": None,
            "lambda": 1e-3,
            "lambda_rho": 1e-3,
            "kl_t": 1.0,
            "kl_z": 1.0,
            "kl_w": 0.01,
            "kl_param": 1.0,
            "reg_forward": 0.1,
            "reg_velo": 0.1,
            "max_rate": 1e4,
            "test_iter": None,
            "save_epoch": 100,
            "n_warmup": 5,
            "n_warmup2": 50,
            "early_stop": 5,
            "early_stop_thred": adata.n_vars*1e-3,
            "train_test_split": 0.7,
            "neg_slope": 0.0,
            "k_alt": 1,
            "train_scaling": False,
            "train_std": False,
            "train_ton": init_method != 'tprior',
            "train_x0": train_x0,
            "weight_sample": False,
            "always_update_t": False,
            "knn_use_pred": knn_use_pred,
            "run_3rd_stage": train_3rd_stage,
            "velocity_continuity": velocity_continuity,
            "two_basis": two_basis,

            # plotting
            "sparsify": 1}

        self.set_device(device)
        self.set_lr(adata, learning_rate)
        self.split_train_test(adata.n_obs)
        self.get_prior(adata, tmax, tprior)
        self.encode_batch(adata, batch_key, ref_batch)

        self.encoder = Encoder(3*adata.n_vars,
                               dim_z,
                               dim_cond=self.n_batch,
                               N1=hidden_size[0],
                               N2=hidden_size[1],
                               checkpoint=checkpoints[0]).float().to(self.device)

        self.decoder = Decoder(adata,
                               adata_atac,
                               self.train_idx,
                               dim_z,
                               dim_cond=self.n_batch,
                               N1=hidden_size[2],
                               N2=hidden_size[3],
                               indicator_arch=indicator_arch,
                               t_network=t_network,
                               params_mult=(1, np.log(0.05)) if full_vb else 1,
                               p=98,
                               tmax=tmax,
                               reinit=reinit_params,
                               init_ton_zero=init_ton_zero,
                               init_method=init_method,
                               init_key=init_key,
                               init_type=init_type,
                               cluster_key=cluster_key,
                               checkpoint=checkpoints[1]).float().to(self.device)

        self.p_z = torch.stack([torch.zeros(adata.shape[0], dim_z), torch.ones(adata.shape[0], dim_z)*self.config["std_z_prior"]]).float().to(self.device)

        self.alpha_w = torch.tensor(find_dirichlet_param(0.5, 0.05), dtype=torch.float, device=self.device)

        self.use_knn = False
        self.reg_velocity = False
        self.c0 = None
        self.u0 = None
        self.s0 = None
        self.t0 = None

        self.loss_train, self.loss_test = [], []
        self.counter = 0
        self.n_drop = 0

        if unit_scale:
            self.scale_c = torch.tensor(np.clip(np.std(adata_atac.layers['Mc'][self.train_idx, :], 0), 1e-6, None), dtype=torch.float, device=self.device)
            self.scale_u = torch.tensor(np.clip(np.std(adata.layers['Mu'][self.train_idx, :], 0), 1e-6, None), dtype=torch.float, device=self.device)
            self.scale_s = torch.tensor(np.clip(np.std(adata.layers['Ms'][self.train_idx, :], 0), 1e-6, None), dtype=torch.float, device=self.device)

        if full_vb:
            self.p_log_alpha_c = torch.tensor([[0.0], [0.5]], dtype=torch.float, device=self.device)
            self.p_log_alpha = torch.tensor([[0.0], [1.0]], dtype=torch.float, device=self.device)
            self.p_log_beta = torch.tensor([[0.0], [0.5]], dtype=torch.float, device=self.device)
            self.p_log_gamma = torch.tensor([[0.0], [0.5]], dtype=torch.float, device=self.device)
            self.p_params = [self.p_log_alpha_c, self.p_log_alpha, self.p_log_beta, self.p_log_gamma]

        if plot_init:
            self.plot_initial(adata, adata_atac, gene_plot, cluster_key, figure_path)

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

    def set_lr(self, adata, learning_rate):
        if learning_rate is None:
            p = np.sum(adata.layers["unspliced"].A > 0) + (np.sum(adata.layers["spliced"].A > 0))
            p /= adata.n_obs * adata.n_vars * 2
            self.config["learning_rate"] = 10**(-4*p-3)
            print(f'Learning rate set to {self.config["learning_rate"]:.4f} based on data sparsity.')
        else:
            self.config["learning_rate"] = learning_rate
        self.config["learning_rate_post"] = self.config["learning_rate"]
        self.config["learning_rate_final"] = self.config["learning_rate"]
        self.config["learning_rate_ode"] = 4*self.config["learning_rate"]

    def encode_batch(self, adata, batch_key, ref_batch):
        self.n_batch = 0
        self.batch = None
        batch_count = None
        if batch_key is not None and batch_key in adata.obs:
            batch_raw = adata.obs[batch_key].to_numpy()
            batch_names_raw, batch_count = np.unique(batch_raw, return_counts=True)
            self.batch_dic, self.batch_dic_rev = encode_type(batch_names_raw)
            self.n_batch = len(batch_names_raw)
            self.batch = np.array([self.batch_dic[x] for x in batch_raw])
            self.batch = torch.tensor(self.batch, dtype=int, device=self.device)
            self.batch_names = np.array([self.batch_dic[batch_names_raw[i]] for i in range(self.n_batch)])
        if isinstance(ref_batch, int):
            self.ref_batch = ref_batch
            if self.ref_batch >= self.n_batch:
                self.ref_batch = self.n_batch - 1
            elif self.ref_batch < -self.n_batch:
                self.ref_batch = 0
            print(f'Reference batch set to {self.ref_batch} ({batch_names_raw[self.ref_batch]}).')
            if np.issubdtype(batch_names_raw, np.number) and 0 not in batch_names_raw:
                print('Warning: integer batch names do not start from 0. Reference batch index may not match the actual batch name!')
        elif isinstance(ref_batch, str):
            if ref_batch in batch_names_raw:
                self.ref_batch = self.batch_dic[ref_batch]
                print(f'Reference batch set to {self.ref_batch} ({batch_names_raw[self.ref_batch]}).')
            else:
                raise ValueError('Reference batch not found in the provided batch field!')
        elif batch_count is not None:
            self.ref_batch = self.batch_names[np.argmax(batch_count)]
            print(f'Reference batch set to {self.ref_batch} ({batch_names_raw[self.ref_batch]}).')
        self.enable_cvae = self.n_batch > 0

    def get_prior(self, adata, tmax, tprior=None):
        print("Gaussian Prior.")
        if tprior is None:
            self.p_t = torch.stack([torch.ones(adata.n_obs, 1)*tmax*0.5, torch.ones(adata.n_obs, 1)*tmax*self.config["std_t_scaling"]]).float().to(self.device)
        else:
            print('Using informative time prior.')
            t = adata.obs[tprior].to_numpy()
            t = t/t.max()*tmax
            t_cap = np.sort(np.unique(t))

            std_t = np.zeros((len(t)))
            std_t[t == t_cap[0]] = (t_cap[1] - t_cap[0])*(0.5+0.5*self.config["time_overlap"])
            for i in range(1, len(t_cap)-1):
                std_t[t == t_cap[i]] = 0.5*(t_cap[i] - t_cap[i-1])*(0.5+0.5*self.config["time_overlap"]) + 0.5*(t_cap[i+1] - t_cap[i])*(0.5+0.5*self.config["time_overlap"])
            std_t[t == t_cap[-1]] = (t_cap[-1] - t_cap[-2])*(0.5+0.5*self.config["time_overlap"])

            self.p_t = torch.stack([torch.tensor(t).view(-1, 1), torch.tensor(std_t).view(-1, 1)]).float().to(self.device)

    def plot_initial(self, adata, adata_atac, gene_plot, cluster_key="clusters", figure_path="figures"):
        cell_labels_raw = adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs else np.array(['Unknown' for i in range(adata.n_obs)])
        cell_types_raw = np.unique(cell_labels_raw)
        label_dic, _ = encode_type(cell_types_raw)

        n_type = len(cell_types_raw)
        cell_labels = np.array([label_dic[x] for x in cell_labels_raw])[self.train_idx]
        cell_types = np.array([label_dic[cell_types_raw[i]] for i in range(n_type)])

        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        c = adata_atac.layers['Mc'][self.train_idx, :][:, gind]
        u = adata.layers['Mu'][self.train_idx, :][:, gind]
        s = adata.layers['Ms'][self.train_idx, :][:, gind]

        alpha_c = self.decoder.alpha_c_[gind]
        alpha = self.decoder.alpha_[gind]
        beta = self.decoder.beta_[gind]
        gamma = self.decoder.gamma_[gind]
        scaling_c = self.decoder.scaling_c_[gind]
        scaling = self.decoder.scaling_[gind]
        toff = self.decoder.toff_[gind]
        t = self.decoder.t_[:, gind]
        chat, uhat, shat = ode_numpy(t, alpha_c, alpha, beta, gamma, 0, toff)

        for i in range(len(gind)):
            plot_sig(t[:, i].squeeze(),
                     c[:, i]/scaling_c[i],
                     u[:, i]/scaling[i],
                     s[:, i],
                     chat[:, i],
                     uhat[:, i],
                     shat[:, i],
                     cell_labels,
                     gene_plot[i],
                     save=f"{figure_path}/sig-{gene_plot[i]}-init.png",
                     sparsify=self.config['sparsify'])

            plot_phase(c[:, i]/scaling_c[i],
                       u[:, i]/scaling[i],
                       s[:, i],
                       chat[:, i],
                       uhat[:, i],
                       shat[:, i],
                       gene_plot[i],
                       'cu',
                       None,
                       cell_labels,
                       cell_types,
                       save=f"{figure_path}/phase-{gene_plot[i]}-init-cu.png")

            plot_phase(c[:, i]/scaling_c[i],
                       u[:, i]/scaling[i],
                       s[:, i],
                       chat[:, i],
                       uhat[:, i],
                       shat[:, i],
                       gene_plot[i],
                       'us',
                       None,
                       cell_labels,
                       cell_types,
                       save=f"{figure_path}/phase-{gene_plot[i]}-init-us.png")

    def forward(self, data_in, c0=None, u0=None, s0=None, t0=None, t1=None, condition=None, sample=True):
        if self.config['unit_scale']:
            data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//3]/self.scale_c,
                                       data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]/self.scale_u,
                                       data_in[:, data_in.shape[1]//3*2:]/self.scale_s), 1)
        else:
            data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//3]/self.decoder.scaling_c.exp(),
                                       data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]/self.decoder.scaling.exp(),
                                       data_in[:, data_in.shape[1]//3*2:]), 1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        if sample:
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
                                                                    neg_slope=self.config["neg_slope"] if sample else 0.0,
                                                                    sample=sample,
                                                                    return_velocity=True)
            chat_fw, uhat_fw, shat_fw, _, vc_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                                     z,
                                                                                     chat,
                                                                                     uhat,
                                                                                     shat,
                                                                                     t_,
                                                                                     condition,
                                                                                     neg_slope=self.config["neg_slope"] if sample else 0.0,
                                                                                     sample=sample,
                                                                                     return_velocity=True)
        else:
            chat, uhat, shat, t_ = self.decoder.forward(t,
                                                        z,
                                                        c0,
                                                        u0,
                                                        s0,
                                                        t0,
                                                        condition,
                                                        neg_slope=self.config["neg_slope"],
                                                        two_basis=self.config["two_basis"],
                                                        sample=sample)
            chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = None, None, None, None, None, None, None, None, None
        return mu_t, std_t, mu_z, std_z, chat, uhat, shat, t_, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw

    def loss_vel(self, x0, xhat, v):
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos_sim(xhat-x0, v).mean()

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
                 sigma_c,
                 sigma_u,
                 sigma_s,
                 chat_fw=None,
                 uhat_fw=None,
                 shat_fw=None,
                 c0=None,
                 u0=None,
                 s0=None,
                 c1=None,
                 u1=None,
                 s1=None,
                 vc=None,
                 vu=None,
                 vs=None,
                 vc_fw=None,
                 vu_fw=None,
                 vs_fw=None,
                 s_knn=None,
                 onehot=None,
                 weight=None,
                 sample=True):

        kldt = kl_gaussian(q_tx[0], q_tx[1], p_t[0], p_t[1])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])

        kldw = elbo_collapsed_categorical(self.decoder.logit_pw, self.alpha_w, 2, self.decoder.scaling.shape[0]) if not self.use_knn and self.config["two_basis"] else 0

        clip_fn = nn.Hardtanh(-P_MAX, P_MAX)

        if uhat.ndim == 3:
            logp = -0.5*((c.unsqueeze(1)-chat)/sigma_c).pow(2)
            logp -= 0.5*((u.unsqueeze(1)-uhat)/sigma_u).pow(2)
            logp -= 0.5*((s.unsqueeze(1)-shat)/sigma_s).pow(2)
            logp -= torch.log(sigma_c)
            logp -= torch.log(sigma_u)
            logp -= torch.log(sigma_s*2*np.pi)
            logp = clip_fn(logp)
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
            logp = torch.sum(pw*logp, 1)
        else:
            logp = -0.5*((c-chat)/sigma_c).pow(2)
            logp -= 0.5*((u-uhat)/sigma_u).pow(2)
            logp -= 0.5*((s-shat)/sigma_s).pow(2)
            logp -= torch.log(sigma_c)
            logp -= torch.log(sigma_u)
            logp -= torch.log(sigma_s*2*np.pi)
            logp = clip_fn(logp)

        if chat_fw is not None and uhat_fw is not None and shat_fw is not None:
            logp -= 0.5*((c1-chat_fw)/sigma_c).pow(2)
            logp -= 0.5*((u1-uhat_fw)/sigma_u).pow(2)
            logp -= 0.5*((s1-shat_fw)/sigma_s).pow(2)

        if weight is not None:
            logp = logp*weight

        err_rec = torch.mean(torch.sum(logp, 1))

        loss = - err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz + self.config["kl_w"]*kldw

        if self.is_full_vb:
            kld_params = None
            for i, x in enumerate(['alpha_c', 'alpha', 'beta', 'gamma']):
                if self.enable_cvae:
                    for j in range(self.n_batch):
                        if kld_params is None:
                            kld_params = kl_gaussian(self.decoder.get_param(x)[j, 0].view(1, -1), self.decoder.get_param(x)[j, 1].exp().view(1, -1), self.p_params[i][0], self.p_params[i][1])
                        else:
                            kld_params += kl_gaussian(self.decoder.get_param(x)[j, 0].view(1, -1), self.decoder.get_param(x)[j, 1].exp().view(1, -1), self.p_params[i][0], self.p_params[i][1])
                else:
                    if kld_params is None:
                        kld_params = kl_gaussian(self.decoder.get_param(x)[0].view(1, -1), self.decoder.get_param(x)[1].exp().view(1, -1), self.p_params[i][0], self.p_params[i][1])
                    else:
                        kld_params += kl_gaussian(self.decoder.get_param(x)[0].view(1, -1), self.decoder.get_param(x)[1].exp().view(1, -1), self.p_params[i][0], self.p_params[i][1])
            kld_params /= u.shape[0]
            loss = loss + self.config["kl_param"]*kld_params

        if self.use_knn and self.config["velocity_continuity"]:
            loss = loss - self.config["reg_forward"] * (self.loss_vel(c0/self.decoder.scaling_c.exp(), chat/self.decoder.scaling_c.exp(), vc)
                                                        + self.loss_vel(u0/self.decoder.scaling.exp(), uhat/self.decoder.scaling.exp(), vu)
                                                        + self.loss_vel(s0, shat, vs)
                                                        + self.loss_vel(chat/self.decoder.scaling_c.exp(), chat_fw/self.decoder.scaling_c.exp(), vc_fw)
                                                        + self.loss_vel(uhat/self.decoder.scaling.exp(), uhat_fw/self.decoder.scaling.exp(), vu_fw)
                                                        + self.loss_vel(shat, shat_fw, vs_fw))

        if self.reg_velocity:
            _, _, beta, gamma = self.decoder.reparameterize(onehot, sample)
            cos_sim = cosine_similarity(uhat/self.decoder.scaling.exp(), shat, beta, gamma, s_knn)
            loss = loss - self.config["reg_velo"] * cos_sim

        return loss

    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, optimizer3=None, k=1, train_net='both', drop=True):
        B = len(train_loader)
        self.set_mode('train', train_net)
        stop_training = False

        for i, batch in enumerate(train_loader):
            if self.counter == 1 or self.counter % self.config["test_iter"] == 0:
                elbo_test = self.test(test_set,
                                      None,
                                      self.counter,
                                      True)

                if len(self.loss_test) > 0:
                    if (elbo_test - self.loss_test[-1] <= self.config["early_stop_thred"]) and drop:
                        self.n_drop = self.n_drop + 1
                    else:
                        self.n_drop = 0
                self.loss_test.append(elbo_test)
                self.set_mode('train', train_net)

                if (self.n_drop >= self.config["early_stop"]) and (self.config["early_stop"] > 0):
                    stop_training = True
                    break

            optimizer.zero_grad()
            if optimizer2 is not None:
                optimizer2.zero_grad()
            if optimizer3 is not None:
                optimizer3.zero_grad()

            xbatch, idx = batch[0], batch[2]
            batch_idx = self.train_idx[idx]

            c0 = self.c0[batch_idx] if self.use_knn else None
            u0 = self.u0[batch_idx] if self.use_knn else None
            s0 = self.s0[batch_idx] if self.use_knn else None
            t0 = self.t0[batch_idx] if self.use_knn else None
            t1 = self.t1[batch_idx] if self.use_knn and self.config['velocity_continuity'] else None
            p_t = self.p_t[:, batch_idx, :]
            p_z = self.p_z[:, batch_idx, :]
            onehot = F.one_hot(batch[3], self.n_batch).float() if self.enable_cvae else None
            out = self.forward(xbatch, c0, u0, s0, t0, t1, onehot)
            mu_tx, std_tx, mu_zx, std_zx, chat, uhat, shat, _, chat_fw, uhat_fw, shat_fw, vc, vu, vs, vc_fw, vu_fw, vs_fw = out

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
                                 self.decoder.sigma_c.exp(),
                                 self.decoder.sigma_u.exp(),
                                 self.decoder.sigma_s.exp(),
                                 chat_fw,
                                 uhat_fw,
                                 shat_fw,
                                 c0,
                                 u0,
                                 s0,
                                 self.c1[batch_idx] if self.use_knn and self.config["velocity_continuity"] else None,
                                 self.u1[batch_idx] if self.use_knn and self.config["velocity_continuity"] else None,
                                 self.s1[batch_idx] if self.use_knn and self.config["velocity_continuity"] else None,
                                 vc,
                                 vu,
                                 vs,
                                 vc_fw,
                                 vu_fw,
                                 vs_fw,
                                 self.s_knn[batch_idx] if self.reg_velocity else None,
                                 onehot,
                                 weight=None)

            loss.backward()
            torch.nn.utils.clip_grad_value_(self.encoder.parameters(), GRAD_MAX)
            torch.nn.utils.clip_grad_value_(self.decoder.parameters(), GRAD_MAX)
            if k == 0:
                optimizer.step()
                if optimizer2 is not None:
                    optimizer2.step()
                if optimizer3 is not None:
                    optimizer3.step()
            else:
                if ((i+1) % (k+1) == 0) or (i == B-1):
                    if optimizer2 is not None:
                        optimizer2.step()
                    if optimizer3 is not None:
                        optimizer3.step()
                else:
                    optimizer.step()

            self.loss_train.append(loss.detach().cpu().item())
            self.counter = self.counter + 1
        return stop_training

    def update_x0(self, c, u, s):
        start = time.time()
        self.set_mode('eval')
        out, _ = self.pred_all(np.concatenate((c, u, s), 1), "both")
        chat, uhat, shat, t, z = out["chat"], out["uhat"], out["shat"], out["t"], out["mu_z"]
        t_ = t.copy()
        t = np.clip(t, 0, np.quantile(t, 0.99))
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        train_idx = self.train_idx.cpu().numpy()
        with torch.no_grad():
            init_mask = (t <= np.quantile(t, 0.01))
            c0_init = np.mean(c[init_mask], 0)
            u0_init = np.mean(u[init_mask], 0)
            s0_init = np.mean(s[init_mask], 0)
            if self.config["knn_use_pred"]:
                c_in = chat[train_idx]
                u_in = uhat[train_idx]
                s_in = shat[train_idx]
            else:
                c_in = c[train_idx]
                u_in = u[train_idx]
                s_in = s[train_idx]
        print("Cell-wise KNN Estimation.")
        c0, u0, s0, t0 = knnx0(c_in,
                               u_in,
                               s_in,
                               t[train_idx],
                               z[train_idx],
                               t,
                               z,
                               dt,
                               self.config["n_neighbors"],
                               c0_init,
                               u0_init,
                               s0_init,
                               hist_eq=True)
        if self.config["velocity_continuity"]:
            c1, u1, s1, t1 = knnx0(c_in,
                                   u_in,
                                   s_in,
                                   t[train_idx],
                                   z[train_idx],
                                   t,
                                   z,
                                   dt,
                                   self.config["n_neighbors"],
                                   c0_init,
                                   u0_init,
                                   s0_init,
                                   forward=True,
                                   hist_eq=True)
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
        return t_

    def predict_knn(self, c, u, s, c0, u0, s0, sigma_c, sigma_u, sigma_s):
        if self.config["knn_use_pred"]:
            out, _ = self.pred_all(np.concatenate((c, u, s), 1), "both")
            chat, uhat, shat = out["chat"], out["uhat"], out["shat"]
            knn = knn_approx(chat/sigma_c, uhat/sigma_u, shat/sigma_s, c0/sigma_c, u0/sigma_u, s0/sigma_s, self.config["n_neighbors"])
            s_knn = shat[knn]
        else:
            knn = knn_approx(c/sigma_c, u/sigma_u, s/sigma_s, c0/sigma_c, u0/sigma_u, s0/sigma_s, self.config["n_neighbors"])
            s_knn = s[knn]
        self.s_knn = torch.tensor(s_knn, dtype=torch.float, device=self.device)

    def update_config(self, config):
        for key in config:
            if key in self.config:
                self.config[key] = config[key]
            else:
                self.config[key] = config[key]
                print(f"Warning: unknown hyperparameter: {key}")

    def split_train_test(self, N):
        rand_perm = np.random.permutation(N)
        n_train = int(N*self.config["train_test_split"])
        self.train_idx = rand_perm[:n_train]
        self.test_idx = rand_perm[n_train:]

    def train(self,
              adata,
              adata_atac,
              config={},
              plot=False,
              gene_plot=[],
              cluster_key="clusters",
              figure_path="figures",
              embed="umap",
              use_raw=False):
        self.update_config(config)
        start = time.time()

        print("--------------------------- Train a VeloVAE ---------------------------")

        if use_raw:
            X = np.concatenate((adata_atac.X.todense()), np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense()), 1).astype(int)
        else:
            X = np.concatenate((adata_atac.layers['Mc'], adata.layers['Mu'], adata.layers['Ms']), 1).astype(float)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
            Xembed_train = Xembed[self.train_idx]
            Xembed_test = Xembed[self.test_idx]
        except KeyError:
            print(f"Embedding X_{embed} not found! Set to None.")
            Xembed, Xembed_train, Xembed_test = None, None, None
            plot = False

        cell_labels_raw = adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs else np.array(['Unknown' for i in range(adata.n_obs)])

        cell_types_raw = np.unique(cell_labels_raw)
        self.label_dic, self.label_dic_rev = encode_type(cell_types_raw)
        self.n_type = len(cell_types_raw)
        self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
        self.cell_types = np.array([self.label_dic[cell_types_raw[i]] for i in range(self.n_type)])

        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

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

        print("*********                      Stage  1                       *********")
        param_nn = list(self.encoder.parameters())
        param_nn += list(self.decoder.net_rho.parameters())+list(self.decoder.fc_out1.parameters())
        param_nn += list(self.decoder.net_kc.parameters())+list(self.decoder.fc_out2.parameters())
        if self.config['t_network']:
            param_nn += list(self.decoder.net_t.parameters())
        param_ode = [self.decoder.alpha_c,
                     self.decoder.alpha,
                     self.decoder.beta,
                     self.decoder.gamma,
                     self.decoder.c0,
                     self.decoder.u0,
                     self.decoder.s0,
                     self.decoder.toff,
                     self.decoder.logit_pw]
        if self.config['train_ton']:
            param_ode.append(self.decoder.ton)

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])

        # from torch.profiler import profile, ProfilerActivity
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        # self.config["n_epochs"] = 20
        for epoch in range(self.config["n_epochs"]):
            if self.config["k_alt"] is None:
                stop_training = self.train_epoch(data_loader, test_set, optimizer)

                if epoch >= self.config["n_warmup"]:
                    stop_training_ode = self.train_epoch(data_loader, test_set, optimizer_ode)
                    if stop_training_ode:
                        print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                        break
            else:
                if epoch >= self.config["n_warmup"]:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_ode, optimizer, None, self.config["k_alt"])
                else:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer, None, None, self.config["k_alt"])

            if epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0:
                elbo_train = self.test(train_set,
                                       Xembed_train,
                                       f"train{epoch+1}",
                                       False,
                                       gind,
                                       gene_plot,
                                       plot,
                                       figure_path)
                self.set_mode('train')
                elbo_test = self.loss_test[-1] if len(self.loss_test) > 0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                break
        # return prof

        print("*********                      Stage  2                       *********")
        n_stage1 = epoch+1
        n_test1 = len(self.loss_test)

        self.set_mode('eval', 'encoder')
        t_ = self.update_x0(X[:, :X.shape[1]//3], X[:, X.shape[1]//3:X.shape[1]//3*2], X[:, X.shape[1]//3*2:])
        c0_ = self.c0.copy()
        u0_ = self.u0.copy()
        s0_ = self.s0.copy()
        t0_ = self.t0.copy()
        if self.config['velocity_continuity']:
            c1_ = self.c1.copy()
            u1_ = self.u1.copy()
            s1_ = self.s1.copy()
            t1_ = self.t1.copy()

        self.use_knn = True
        self.decoder.init_weights()

        if plot:
            plot_time(self.t0.squeeze(), Xembed, save=f"{figure_path}/t0.png")
            t0_plot = self.t0[self.train_idx.cpu().numpy()].squeeze()
            for i in range(len(gind)):
                idx = gind[i]
                c0_plot = self.c0[self.train_idx.cpu().numpy(), idx]
                u0_plot = self.u0[self.train_idx.cpu().numpy(), idx]
                s0_plot = self.s0[self.train_idx.cpu().numpy(), idx]
                plot_sig_(t0_plot,
                          c0_plot,
                          u0_plot,
                          s0_plot,
                          cell_labels=cell_labels_raw[self.train_idx.cpu().numpy()],
                          title=gene_plot[i],
                          save=f"{figure_path}/{gene_plot[i]}-x0.png")

        self.n_drop = 0
        param_post = list(self.decoder.net_rho.parameters()) + list(self.decoder.fc_out1.parameters())
        param_post += list(self.decoder.net_kc.parameters()) + list(self.decoder.fc_out2.parameters())
        if self.config['t_network'] and self.config['always_update_t']:
            param_post += list(self.decoder.net_t.parameters())
        optimizer_post = torch.optim.Adam(param_post, lr=self.config["learning_rate_post"], weight_decay=self.config["lambda_rho"])

        self.c0 = torch.tensor(self.c0, dtype=torch.float, device=self.device, requires_grad=True)
        self.u0 = torch.tensor(self.u0, dtype=torch.float, device=self.device, requires_grad=True)
        self.s0 = torch.tensor(self.s0, dtype=torch.float, device=self.device, requires_grad=True)
        self.t0 = torch.tensor(self.t0, dtype=torch.float, device=self.device, requires_grad=True)
        if self.config["velocity_continuity"]:
            self.c1 = torch.tensor(self.c1, dtype=torch.float, device=self.device, requires_grad=True)
            self.u1 = torch.tensor(self.u1, dtype=torch.float, device=self.device, requires_grad=True)
            self.s1 = torch.tensor(self.s1, dtype=torch.float, device=self.device, requires_grad=True)
            self.t1 = torch.tensor(self.t1, dtype=torch.float, device=self.device, requires_grad=True)

        if self.config["train_x0"]:
            optimizer_x0 = torch.optim.Adam([self.c0, self.u0, self.s0, self.t0], lr=self.config["learning_rate_ode"])
        else:
            optimizer_x0 = None

        for epoch in range(self.config["n_epochs_post"]):
            if self.config["k_alt"] is None:
                stop_training = self.train_epoch(data_loader, test_set, optimizer_post)

                if epoch >= self.config["n_warmup"]:
                    stop_training_ode = self.train_epoch(data_loader, test_set, optimizer_ode, optimizer_x0, train_net='decoder')
                    if stop_training_ode:
                        print(f"*********       Stage 2: Early Stop Triggered at epoch {epoch+n_stage1+1}.       *********")
                        break
            else:
                if epoch >= self.config["n_warmup2"]:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_post, optimizer_ode, optimizer_x0, self.config["k_alt"], train_net='decoder')
                elif epoch >= self.config["n_warmup"]:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_post, optimizer_ode, None, self.config["k_alt"], train_net='decoder', drop=False)
                else:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_post, None, None, self.config["k_alt"], train_net='decoder', drop=False)

            if epoch == 0 or (epoch+n_stage1+1) % self.config["save_epoch"] == 0:
                elbo_train = self.test(train_set,
                                       Xembed_train,
                                       f"train{epoch+n_stage1+1}",
                                       False,
                                       gind,
                                       gene_plot,
                                       plot,
                                       figure_path)
                self.set_mode('train', 'decoder')
                elbo_test = self.loss_test[-1] if len(self.loss_test) > n_test1 else -np.inf
                print(f"Epoch {epoch+n_stage1+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********       Stage 2: Early Stop Triggered at epoch {epoch+n_stage1+1}.       *********")
                break

        if self.config["run_3rd_stage"]:
            print("*********                      Stage  3                       *********")
            n_stage2 = epoch+n_stage1+1
            n_test2 = len(self.loss_test)

            self.set_mode('train', 'encoder')
            self.set_mode('eval', 'decoder')
            self.predict_knn(X[:, :X.shape[1]//3],
                             X[:, X.shape[1]//3:X.shape[1]//3*2],
                             X[:, X.shape[1]//3*2:],
                             self.c0.detach().cpu().numpy(),
                             self.u0.detach().cpu().numpy(),
                             self.s0.detach().cpu().numpy(),
                             torch.exp(self.decoder.sigma_c).detach().cpu().numpy(),
                             torch.exp(self.decoder.sigma_u).detach().cpu().numpy(),
                             torch.exp(self.decoder.sigma_s).detach().cpu().numpy())

            if self.config['reg_velo'] > 0:
                self.reg_velocity = True

            if plot:
                plot_time(self.t0.detach().cpu().numpy().squeeze(), Xembed, save=f"{figure_path}/t0_updated.png")
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
                              cell_labels=cell_labels_raw[self.train_idx.cpu().numpy()],
                              title=gene_plot[i],
                              save=f"{figure_path}/{gene_plot[i]}-x0-updated.png")

            self.n_drop = 0
            param_final = list(self.encoder.parameters())
            optimizer_final = torch.optim.Adam(param_final, lr=self.config["learning_rate_final"], weight_decay=self.config["lambda_rho"])

            for epoch in range(self.config["n_epochs_final"]):
                if self.config["k_alt"] is None:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_final, train_net='encoder')
                else:
                    if epoch >= self.config["n_warmup2"]:
                        stop_training = self.train_epoch(data_loader, test_set, optimizer_final, None, None, self.config["k_alt"], train_net='encoder')
                    else:
                        stop_training = self.train_epoch(data_loader, test_set, optimizer_final, None, None, self.config["k_alt"], train_net='encoder', drop=False)

                if epoch == 0 or (epoch+n_stage2+1) % self.config["save_epoch"] == 0:
                    elbo_train = self.test(train_set,
                                           Xembed_train,
                                           f"train{epoch+n_stage2+1}",
                                           False,
                                           gind,
                                           gene_plot,
                                           plot,
                                           figure_path)
                    self.set_mode('train', 'encoder')
                    elbo_test = self.loss_test[-1] if len(self.loss_test) > n_test2 else -np.inf
                    print(f"Epoch {epoch+n_stage2+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")

                if stop_training:
                    print(f"*********       Stage 3: Early Stop Triggered at epoch {epoch+n_stage2+1}.       *********")
                    break

        elbo_train = self.test(train_set,
                               Xembed_train,
                               "final-train",
                               False,
                               gind,
                               gene_plot,
                               True,
                               figure_path)
        elbo_test = self.test(test_set,
                              Xembed_test,
                              "final-test",
                              True,
                              gind,
                              gene_plot,
                              True,
                              figure_path)
        self.loss_train.append(elbo_train)
        self.loss_test.append(elbo_test)
        if plot:
            plot_train_loss(self.loss_train, range(1, len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            if self.config["test_iter"] > 0:
                plot_test_loss(self.loss_test, [i*self.config["test_iter"] for i in range(1, len(self.loss_test)+1)], save=f'{figure_path}/test_loss_velovae.png')
        print(f"Final: Train ELBO = {elbo_train:.3f},\tTest ELBO = {elbo_test:.3f}")
        print(f"*********              Finished. Total Time = {convert_time(time.time()-start)}             *********")
        if self.config['velocity_continuity']:
            return c0_, u0_, s0_, t0_, t_, c1_, u1_, s1_, t1_
        else:
            return c0_, u0_, s0_, t0_, t_

    def pred_all(self, data, mode='test', output=["chat", "uhat", "shat", "t", "z"], gene_idx=None, batch=None):
        N, G = data.shape[0], data.shape[1]//3
        if gene_idx is None:
            gene_idx = np.array(range(G))
        elbo = 0
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
            mu_t_out = np.zeros((N))
            std_t_out = np.zeros((N))
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

            w_hard = F.one_hot(torch.argmax(self.decoder.logit_pw, 1), num_classes=2).T
            for n in range(Nb):
                i = n*B
                j = min([(n+1)*B, N])
                data_in = torch.tensor(data[i:j], dtype=torch.float, device=self.device)
                if mode == "test":
                    batch_idx = self.test_idx[i:j]
                elif mode == "train":
                    batch_idx = self.train_idx[i:j]
                else:
                    batch_idx = np.arange(i, j)

                c0 = self.c0[batch_idx] if self.use_knn else None
                u0 = self.u0[batch_idx] if self.use_knn else None
                s0 = self.s0[batch_idx] if self.use_knn else None
                t0 = self.t0[batch_idx] if self.use_knn else None
                t1 = self.t1[batch_idx] if self.use_knn and self.config['velocity_continuity'] else None
                p_t = self.p_t[:, batch_idx, :]
                p_z = self.p_z[:, batch_idx, :]
                onehot = F.one_hot(batch[batch_idx], self.n_batch).float() if self.enable_cvae else None
                out = self.forward(data_in, c0, u0, s0, t0, t1, onehot, sample=False)
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
                                     self.decoder.sigma_c.exp(),
                                     self.decoder.sigma_u.exp(),
                                     self.decoder.sigma_s.exp(),
                                     chat_fw,
                                     uhat_fw,
                                     shat_fw,
                                     c0,
                                     u0,
                                     s0,
                                     self.c1[batch_idx] if self.use_knn and self.config['velocity_continuity'] else None,
                                     self.u1[batch_idx] if self.use_knn and self.config['velocity_continuity'] else None,
                                     self.s1[batch_idx] if self.use_knn and self.config['velocity_continuity'] else None,
                                     vc,
                                     vu,
                                     vs,
                                     vc_fw,
                                     vu_fw,
                                     vs_fw,
                                     self.s_knn[batch_idx] if self.reg_velocity else None,
                                     onehot,
                                     weight=None,
                                     sample=False)

                elbo = elbo - (B/N)*loss
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
                    mu_t_out[i:j] = mu_tx.detach().cpu().squeeze().numpy()
                    std_t_out[i:j] = std_tx.detach().cpu().squeeze().numpy()
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

        return out, elbo.cpu().item()

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
        out, elbo = self.pred_all(dataset.data, mode, out_type, gind)
        chat, uhat, shat, t_ = out["chat"], out["uhat"], out["shat"], out["t"]

        G = dataset.data.shape[1]//3

        testid_str = str(testid)
        if testid_str.startswith('train'):
            id_num = testid_str[5:]
            testid_str = 'train' + '0'*(9-len(testid_str)) + id_num
        elif testid_str.startswith('test'):
            id_num = testid_str[4:]
            testid_str = 'test' + '0'*(8-len(testid_str)) + id_num

        if plot:
            plot_time(t_, Xembed, save=f"{path}/time-{testid_str}-velovae.png")

            for i in range(len(gind)):
                idx = gind[i]
                if np.any(np.isnan(chat[:, i])):
                    print(gene_plot[i], chat[:, i])
                if np.any(np.isnan(uhat[:, i])):
                    print(gene_plot[i], uhat[:, i])
                if np.any(np.isnan(shat[:, i])):
                    print(gene_plot[i], shat[:, i])

                plot_sig(t_.squeeze(),
                         dataset.data[:, idx].cpu().numpy(),
                         dataset.data[:, idx+G].cpu().numpy(),
                         dataset.data[:, idx+G*2].cpu().numpy(),
                         chat[:, i],
                         uhat[:, i],
                         shat[:, i],
                         np.array([self.label_dic_rev[x] for x in dataset.labels]),
                         gene_plot[i],
                         save=f"{path}/sig-{gene_plot[i]}-{testid_str}.png",
                         sparsify=self.config['sparsify'])

                if self.use_knn and self.config['velocity_continuity']:
                    scaling_c = self.decoder.scaling_c[i].detach().cpu().exp().numpy()
                    scaling = self.decoder.scaling[i].detach().cpu().exp().numpy()
                    cell_idx = self.test_idx if test_mode else self.train_idx
                    plot_vel(t_.squeeze(),
                             chat[:, i]/scaling_c, uhat[:, i]/scaling, shat[:, i],
                             out["vc"][:, i], out["vu"][:, i], out["vs"][:, i],
                             self.t0[cell_idx].squeeze().detach().cpu().numpy(),
                             self.c0[cell_idx, idx].detach().cpu().numpy()/scaling_c,
                             self.u0[cell_idx, idx].detach().cpu().numpy()/scaling,
                             self.s0[cell_idx, idx].detach().cpu().numpy(),
                             title=gene_plot[i],
                             save=f"{path}/vel-{gene_plot[i]}-{testid_str}.png")

                    plot_sig(t_.squeeze(),
                             dataset.data[:, idx].cpu().numpy(),
                             dataset.data[:, idx+G].cpu().numpy(),
                             dataset.data[:, idx+G*2].cpu().numpy(),
                             out["chat_fw"][:, i],
                             out["uhat_fw"][:, i],
                             out["shat_fw"][:, i],
                             np.array([self.label_dic_rev[x] for x in dataset.labels]),
                             gene_plot[i],
                             save=f"{path}/sig-{gene_plot[i]}-{testid_str}-bw.png",
                             sparsify=self.config['sparsify'])

        return elbo

    def save_model(self, file_path, enc_name='encoder', dec_name='decoder'):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), f"{file_path}/{enc_name}.pt")
        torch.save(self.decoder.state_dict(), f"{file_path}/{dec_name}.pt")

    def save_anndata(self, adata, adata_atac, file_path, file_name=None):
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)

        mode = self.config['mode']
        if mode == 'fullvb':
            if self.enable_cvae:
                for i in range(self.n_batch):
                    adata.var[f"{mode}_alpha_c_{i}"] = np.exp(self.decoder.alpha_c[i, 0].detach().cpu().numpy())
                    adata.var[f"{mode}_alpha_{i}"] = np.exp(self.decoder.alpha[i, 0].detach().cpu().numpy())
                    adata.var[f"{mode}_beta_{i}"] = np.exp(self.decoder.beta[i, 0].detach().cpu().numpy())
                    adata.var[f"{mode}_gamma_{i}"] = np.exp(self.decoder.gamma[i, 0].detach().cpu().numpy())
                    adata.var[f"{mode}_logstd_alpha_c_{i}"] = np.exp(self.decoder.alpha_c[i, 1].detach().cpu().numpy())
                    adata.var[f"{mode}_logstd_alpha_{i}"] = np.exp(self.decoder.alpha[i, 1].detach().cpu().numpy())
                    adata.var[f"{mode}_logstd_beta_{i}"] = np.exp(self.decoder.beta[i, 1].detach().cpu().numpy())
                    adata.var[f"{mode}_logstd_gamma_{i}"] = np.exp(self.decoder.gamma[i, 1].detach().cpu().numpy())
            else:
                adata.var[f"{mode}_alpha_c"] = np.exp(self.decoder.alpha_c[0].detach().cpu().numpy())
                adata.var[f"{mode}_alpha"] = np.exp(self.decoder.alpha[0].detach().cpu().numpy())
                adata.var[f"{mode}_beta"] = np.exp(self.decoder.beta[0].detach().cpu().numpy())
                adata.var[f"{mode}_gamma"] = np.exp(self.decoder.gamma[0].detach().cpu().numpy())
                adata.var[f"{mode}_logstd_alpha_c"] = np.exp(self.decoder.alpha_c[1].detach().cpu().numpy())
                adata.var[f"{mode}_logstd_alpha"] = np.exp(self.decoder.alpha[1].detach().cpu().numpy())
                adata.var[f"{mode}_logstd_beta"] = np.exp(self.decoder.beta[1].detach().cpu().numpy())
                adata.var[f"{mode}_logstd_gamma"] = np.exp(self.decoder.gamma[1].detach().cpu().numpy())
        else:
            if self.enable_cvae:
                for i in range(self.n_batch):
                    adata.var[f"{mode}_alpha_c_{i}"] = np.exp(self.decoder.alpha_c[i].detach().cpu().numpy())
                    adata.var[f"{mode}_alpha_{i}"] = np.exp(self.decoder.alpha[i].detach().cpu().numpy())
                    adata.var[f"{mode}_beta_{i}"] = np.exp(self.decoder.beta[i].detach().cpu().numpy())
                    adata.var[f"{mode}_gamma_{i}"] = np.exp(self.decoder.gamma[i].detach().cpu().numpy())
            else:
                adata.var[f"{mode}_alpha_c"] = np.exp(self.decoder.alpha_c.detach().cpu().numpy())
                adata.var[f"{mode}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
                adata.var[f"{mode}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
                adata.var[f"{mode}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{mode}_ton"] = np.exp(self.decoder.ton.detach().cpu().numpy())
        adata.var[f"{mode}_scaling_c"] = np.exp(self.decoder.scaling_c.detach().cpu().numpy())
        adata.var[f"{mode}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{mode}_sigma_c"] = np.exp(self.decoder.sigma_c.detach().cpu().numpy())
        adata.var[f"{mode}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{mode}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        adata.varm[f"{mode}_basis"] = F.softmax(self.decoder.logit_pw, 1).detach().cpu().numpy()

        c, u, s = adata_atac.layers['Mc'], adata.layers['Mu'], adata.layers['Ms']

        if self.enable_cvae and self.ref_batch >= 0:
            out, _ = self.pred_all(np.concatenate((c, u, s), 1), "both", batch=torch.full((adata.n_obs,), self.ref_batch, dtype=int, device=self.device))
        else:
            out, _ = self.pred_all(np.concatenate((c, u, s), 1), "both")
        chat, uhat, shat, t, std_t, t_, z, std_z = out["chat"], out["uhat"], out["shat"], out["mu_t"], out["std_t"], out["t"], out["mu_z"], out["std_z"]

        adata.obs[f"{mode}_time"] = t_
        adata.obs[f"{mode}_t"] = t
        adata.obs[f"{mode}_std_t"] = std_t
        adata.obsm[f"{mode}_z"] = z
        adata.obsm[f"{mode}_std_z"] = std_z
        adata.layers[f"{mode}_chat"] = chat
        adata.layers[f"{mode}_uhat"] = uhat
        adata.layers[f"{mode}_shat"] = shat
        sigma_c, sigma_u, sigma_s = adata.var[f"{mode}_sigma_c"].to_numpy(), adata.var[f"{mode}_sigma_u"].to_numpy(), adata.var[f"{mode}_sigma_s"].to_numpy()
        adata.var[f"{mode}_likelihood"] = np.mean(-0.5*((c-chat)/sigma_c)**2-0.5*((u-uhat)/sigma_u)**2-0.5*((s-shat)/sigma_s)**2-np.log(sigma_c)-np.log(sigma_u)-np.log(sigma_s)-np.log(2*np.pi), 0)

        rho = np.zeros(u.shape)
        kc = np.zeros(u.shape)
        with torch.no_grad():
            B = min(u.shape[0]//10, 1000)
            Nb = u.shape[0] // B
            if Nb*B < u.shape[0]:
                Nb += 1
            for n in range(Nb):
                i = n*B
                j = min([(n+1)*B, u.shape[0]])
                if self.enable_cvae:
                    y_onehot = F.one_hot(self.batch[i:j], self.n_batch).float()
                    rho_batch = torch.sigmoid(self.decoder.fc_out1(self.decoder.net_rho(torch.cat((torch.tensor(z[i:j], dtype=torch.float, device=self.device), y_onehot), 1))))
                    rho[i:j] = rho_batch.cpu().numpy()
                    if self.config['indicator_arch'] == 'parallel':
                        kc_batch = torch.sigmoid(self.decoder.fc_out2(self.decoder.net_kc(torch.cat((torch.tensor(z[i:j], dtype=torch.float, device=self.device), y_onehot), 1))))
                    else:
                        kc_batch = torch.sigmoid(self.decoder.fc_out2(self.decoder.net_kc(torch.cat((torch.tensor(rho[i:j], dtype=torch.float, device=self.device), y_onehot), 1))))
                    kc[i:j] = kc_batch.cpu().numpy()
                else:
                    rho_batch = torch.sigmoid(self.decoder.fc_out1(self.decoder.net_rho(torch.tensor(z[i:j], dtype=torch.float, device=self.device))))
                    rho[i:j] = rho_batch.cpu().numpy()
                    if self.config['indicator_arch'] == 'parallel':
                        kc_batch = torch.sigmoid(self.decoder.fc_out2(self.decoder.net_kc(torch.tensor(z[i:j], dtype=torch.float, device=self.device))))
                    else:
                        kc_batch = torch.sigmoid(self.decoder.fc_out2(self.decoder.net_kc(torch.tensor(rho[i:j], dtype=torch.float, device=self.device))))
                    kc[i:j] = kc_batch.cpu().numpy()

        adata.layers[f"{mode}_rho"] = rho
        adata.layers[f"{mode}_kc"] = kc

        adata.obs[f"{mode}_t0"] = self.t0.detach().cpu().numpy().squeeze()
        adata.layers[f"{mode}_c0"] = self.c0.detach().cpu().numpy()
        adata.layers[f"{mode}_u0"] = self.u0.detach().cpu().numpy()
        adata.layers[f"{mode}_s0"] = self.s0.detach().cpu().numpy()

        adata.uns[f"{mode}_train_idx"] = self.train_idx.cpu().numpy()
        adata.uns[f"{mode}_test_idx"] = self.test_idx.cpu().numpy()

        rna_velocity_vae(adata, adata_atac, mode, batch_key=self.config['batch_key'], use_raw=False, use_scv_genes=False)

        if file_name is not None:
            adata.write_h5ad(f"{file_path}/{file_name}")
