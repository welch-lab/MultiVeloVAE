import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from velovae.plotting import plot_time, plot_train_loss, plot_test_loss
from velovae.plotting_chrom import plot_sig, plot_sig_, plot_phase
from .model_util import hist_equal, convert_time, get_gene_index
from .model_util_chrom import pred_exp, ode_numpy, init_params, get_ts_global, reinit_params, kl_uniform, kl_gaussian, knn_approx, knnx0, knnx0_bin, cosine_similarity
from .TransitionGraph import encode_type
from .TrainingDataChrom import SCData
from .velocity_chrom import rna_velocity_vae


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
                 device=torch.device('cpu'),
                 checkpoint=None):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(Cin+dim_cond, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)

        self.net = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                 self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2,
                                 )

        self.fc_mu_t, self.spt1 = nn.Linear(N2, 1).to(device), nn.Softplus()
        self.fc_std_t, self.spt2 = nn.Linear(N2, 1).to(device), nn.Softplus()
        self.fc_mu_z = nn.Linear(N2, dim_z).to(device)
        self.fc_std_z, self.spt3 = nn.Linear(N2, dim_z).to(device), nn.Softplus()

        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint, map_location=device))
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
                 device=torch.device('cpu'),
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
        self.device = device
        self.init_method = init_method
        self.init_key = init_key
        self.init_type = init_type
        self.cluster_key = cluster_key
        self.device = device
        self.checkpoint = checkpoint
        self.construct_nn(dim_z, dim_cond, N1, N2, p, tmax)

    def construct_nn(self, dim_z, dim_cond, N1, N2, p, tmax):
        G = self.adata.n_vars
        self.set_shape(G, dim_cond)

        self.fc1 = nn.Linear(dim_z+dim_cond, N1).to(self.device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(self.device)
        self.dpt1 = nn.Dropout(p=0.2).to(self.device)
        self.fc2 = nn.Linear(N1, N2).to(self.device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(self.device)
        self.dpt2 = nn.Dropout(p=0.2).to(self.device)

        self.fc_out1 = nn.Linear(N2, G).to(self.device)

        self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                     self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)

        self.fc3 = nn.Linear(dim_z+dim_cond, N1).to(self.device)
        self.bn3 = nn.BatchNorm1d(num_features=N1).to(self.device)
        self.dpt3 = nn.Dropout(p=0.2).to(self.device)
        self.fc4 = nn.Linear(N1, N2).to(self.device)
        self.bn4 = nn.BatchNorm1d(num_features=N2).to(self.device)
        self.dpt4 = nn.Dropout(p=0.2).to(self.device)

        self.fc_out2 = nn.Linear(N2, G).to(self.device)

        self.net_rho2 = nn.Sequential(self.fc3, self.bn3, nn.LeakyReLU(), self.dpt3,
                                      self.fc4, self.bn4, nn.LeakyReLU(), self.dpt4)

        self.fc5 = nn.Linear(dim_z+dim_cond if self.indicator_arch == 'parallel' else G, N1).to(self.device)
        self.bn5 = nn.BatchNorm1d(num_features=N1).to(self.device)
        self.dpt5 = nn.Dropout(p=0.2).to(self.device)
        self.fc6 = nn.Linear(N1, N2).to(self.device)
        self.bn6 = nn.BatchNorm1d(num_features=N2).to(self.device)
        self.dpt6 = nn.Dropout(p=0.2).to(self.device)

        self.fc_out3 = nn.Linear(N2, G).to(self.device)

        self.net_kc = nn.Sequential(self.fc5, self.bn5, nn.LeakyReLU(), self.dpt5,
                                    self.fc6, self.bn6, nn.LeakyReLU(), self.dpt6)

        self.fc7 = nn.Linear(dim_z+dim_cond if self.indicator_arch == 'parallel' else G, N1).to(self.device)
        self.bn7 = nn.BatchNorm1d(num_features=N1).to(self.device)
        self.dpt7 = nn.Dropout(p=0.2).to(self.device)
        self.fc8 = nn.Linear(N1, N2).to(self.device)
        self.bn8 = nn.BatchNorm1d(num_features=N2).to(self.device)
        self.dpt8 = nn.Dropout(p=0.2).to(self.device)

        self.fc_out4 = nn.Linear(N2, G).to(self.device)

        self.net_kc2 = nn.Sequential(self.fc7, self.bn7, nn.LeakyReLU(), self.dpt7,
                                     self.fc8, self.bn8, nn.LeakyReLU(), self.dpt8)

        if self.t_network:
            self.fct = nn.Linear(1+dim_cond, 1).to(self.device)
            self.net_t = self.fct
        self.rho, self.kc = None, None

        if self.checkpoint is not None:
            self.alpha_c = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.alpha = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.beta = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.gamma = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.scaling_c = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.scaling = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.ton = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.sigma_c = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.sigma_u = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.sigma_s = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.c0 = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.u0 = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())
            self.s0 = nn.Parameter(torch.empty(self.params_shape, device=self.device).float())

            self.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        else:
            C, U, S = self.adata_atac.layers['Mc'][self.train_idx], self.adata.layers['Mu'][self.train_idx], self.adata.layers['Ms'][self.train_idx]
            self.init_weights()
            self.init_ode(C, U, S, p, tmax)
            self.init_x0(C, U, S)

        self.scaling_c.requires_grad = False
        self.scaling.requires_grad = False
        self.sigma_c.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False

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
                return torch.tensor(np.tile(np.stack([self.params_mult[0]*np.log(x), self.params_mult[1]*np.ones(self.params_shape[2])]), (self.params_shape[0], 1, 1)), device=self.device).float()
            else:
                return torch.tensor(np.stack([self.params_mult[0]*np.log(x), self.params_mult[1]*np.ones(self.params_shape[2])]), device=self.device).float()
        else:
            if self.cvae:
                return torch.tensor(np.tile(self.params_mult[0]*np.log(x), (self.params_shape[0], 1)), device=self.device).float()
            else:
                return torch.tensor(self.params_mult[0]*np.log(x), device=self.device).float()

    def init_weights(self):
        for m in self.net_rho.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        for m in self.net_rho2.modules():
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
        for m in self.net_kc2.modules():
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

        for m in [self.fc_out1, self.fc_out2, self.fc_out3, self.fc_out4]:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def init_ode(self, C, U, S, p, tmax):
        X = np.concatenate((C, U, S), 1)
        device = self.device

        if self.init_method == "random":
            print("Random Initialization.")
            alpha_c, alpha, beta, gamma, scaling_c, scaling, toff, c0, u0, s0, sigma_c, sigma_u, sigma_s, T = init_params(X, p, fit_scaling=True)
            self.alpha_c = nn.Parameter(torch.normal(0.0, 0.5, size=self.params_shape, device=device).float())
            self.alpha = nn.Parameter(torch.normal(0.0, 1.0, size=self.params_shape, device=device).float())
            self.beta = nn.Parameter(torch.normal(0.0, 0.5, size=self.params_shape, device=device).float())
            self.gamma = nn.Parameter(torch.normal(0.0, 0.5, size=self.params_shape, device=device).float())
            self.ton = torch.nn.Parameter(torch.ones(self.params_shape, device=self.device).float()*(-10))
            self.scaling_c = nn.Parameter(torch.tensor(np.log(scaling_c), device=device).float())
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            self.sigma_c = nn.Parameter(torch.tensor(np.log(sigma_c), device=device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
        elif self.init_method == "tprior":
            print("Initialization using prior time.")
            alpha_c, alpha, beta, gamma, scaling_c, scaling, toff, c0, u0, s0, sigma_c, sigma_u, sigma_s, T = init_params(X, p, fit_scaling=True)
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
            self.T_ = T

            t_prior = self.adata.obs[self.init_key].to_numpy()
            t_prior = t_prior[self.train_idx]
            std_t = (np.std(t_prior)+1e-3)*0.2
            self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
            self.t_init -= self.t_init.min()
            self.t_init = self.t_init
            self.t_init = self.t_init/self.t_init.max()*tmax
            if self.reinit:
                toff = get_ts_global(self.t_init, U/scaling, S, 95)
                alpha_c, alpha, beta, gamma, ton = reinit_params(C/scaling_c, U/scaling, S, self.t_init, toff)

            self.alpha_c = nn.Parameter(self.get_tensor(alpha_c))
            self.alpha = nn.Parameter(self.get_tensor(alpha))
            self.beta = nn.Parameter(self.get_tensor(beta))
            self.gamma = nn.Parameter(self.get_tensor(gamma))
            if self.init_ton_zero or (not self.reinit):
                self.ton = nn.Parameter((torch.ones(self.params_shape, device=device)*(-10)).float())
            else:
                self.ton = nn.Parameter(self.get_tensor(ton+1e-10))

            self.scaling_c = nn.Parameter(torch.tensor(np.log(scaling_c), device=device).float())
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            self.sigma_c = nn.Parameter(torch.tensor(np.log(sigma_c), device=device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
        else:
            print("Initialization using the steady-state and dynamical models.")
            alpha_c, alpha, beta, gamma, scaling_c, scaling, toff, c0, u0, s0, sigma_c, sigma_u, sigma_s, T = init_params(X, p, fit_scaling=True, tmax=tmax)
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
            self.T_ = T

            if self.reinit:
                if self.init_key is not None:
                    self.t_init = self.adata.obs[self.init_key].to_numpy()[self.train_idx]
                else:
                    T = T+np.random.rand(T.shape[0], T.shape[1]) * 1e-3
                    T_eq = np.zeros(T.shape)
                    Nbin = T.shape[0]//50+1
                    for i in range(T.shape[1]):
                        T_eq[:, i] = hist_equal(T[:, i], tmax, 0.9, Nbin)
                    self.t_init = np.quantile(T_eq, 0.5, 1)
                toff = get_ts_global(self.t_init, C/scaling_c, U/scaling, S, 95)
                alpha_c, alpha, beta, gamma, ton = reinit_params(C/scaling_c, U/scaling, S, self.t_init, toff)

            self.alpha_c = nn.Parameter(self.get_tensor(alpha_c))
            self.alpha = nn.Parameter(self.get_tensor(alpha))
            self.beta = nn.Parameter(self.get_tensor(beta))
            self.gamma = nn.Parameter(self.get_tensor(gamma))
            if self.init_ton_zero or (not self.reinit):
                self.ton = nn.Parameter((torch.ones(self.params_shape, device=device)*(-10)).float())
            else:
                self.ton = nn.Parameter(self.get_tensor(ton+1e-10))

            self.scaling_c = nn.Parameter(torch.tensor(np.log(scaling_c), device=self.device).float())
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=self.device).float())
            self.sigma_c = nn.Parameter(torch.tensor(np.log(sigma_c), device=self.device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=self.device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=self.device).float())

    def init_x0(self, C, U, S):
        G = self.adata.n_vars
        device = self.device
        if self.init_type is None:
            self.c0 = nn.Parameter(torch.ones(self.params_shape, device=device).float()*(-10))
            self.u0 = nn.Parameter(torch.ones(self.params_shape, device=device).float()*(-10))
            self.s0 = nn.Parameter(torch.ones(self.params_shape, device=device).float()*(-10))
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
            c_top = np.quantile(C, 0.99, 0)
            u_top = np.quantile(U, 0.99, 0)
            s_top = np.quantile(S, 0.99, 0)

            c0, u0, s0 = c_top*r_c_gamma*r_c_bern, u_top*r_u_gamma*r_u_bern, s_top*r_s_gamma*r_s_bern
            self.c0 = nn.Parameter(self.get_tensor(c0+1e-10))
            self.u0 = nn.Parameter(self.get_tensor(u0+1e-10))
            self.s0 = nn.Parameter(self.get_tensor(s0+1e-10))
        else:
            cell_labels = self.adata.obs[self.cluster_key].to_numpy()[self.train_idx]
            cell_mask = cell_labels == self.init_type
            self.c0 = nn.Parameter(self.get_tensor(C[cell_mask].mean(0)+1e-10))
            self.u0 = nn.Parameter(self.get_tensor(U[cell_mask].mean(0)+1e-10))
            self.s0 = nn.Parameter(self.get_tensor(S[cell_mask].mean(0)+1e-10))

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

    def reparameterize(self, condition=None, sample=True, compute_x0=True):
        if len(self.params_mult) > 1:
            if sample:
                G = self.adata.n_vars
                eps = torch.normal(mean=torch.zeros((8, G)), std=torch.ones((8, G))).to(self.device)
                if self.cvae:
                    if compute_x0:
                        ton = torch.exp(self.ton[:, 0] + eps[0]*(self.ton[:, 1].exp()))
                        c0 = torch.exp(self.c0[:, 0] + eps[1]*(self.c0[:, 1].exp()))
                        u0 = torch.exp(self.u0[:, 0] + eps[2]*(self.u0[:, 1].exp()))
                        s0 = torch.exp(self.s0[:, 0] + eps[3]*(self.s0[:, 1].exp()))
                    alpha_c = torch.exp(self.alpha_c[:, 0] + eps[4]*(self.alpha_c[:, 1].exp()))
                    alpha = torch.exp(self.alpha[:, 0] + eps[5]*(self.alpha[:, 1].exp()))
                    beta = torch.exp(self.beta[:, 0] + eps[6]*(self.beta[:, 1].exp()))
                    gamma = torch.exp(self.gamma[:, 0] + eps[7]*(self.gamma[:, 1].exp()))
                else:
                    if compute_x0:
                        ton = torch.exp(self.ton[0] + eps[0]*(self.ton[1].exp()))
                        c0 = torch.exp(self.c0[0] + eps[1]*(self.c0[1].exp()))
                        u0 = torch.exp(self.u0[0] + eps[2]*(self.u0[1].exp()))
                        s0 = torch.exp(self.s0[0] + eps[3]*(self.s0[1].exp()))
                    alpha_c = torch.exp(self.alpha_c[0] + eps[4]*(self.alpha_c[1].exp()))
                    alpha = torch.exp(self.alpha[0] + eps[5]*(self.alpha[1].exp()))
                    beta = torch.exp(self.beta[0] + eps[6]*(self.beta[1].exp()))
                    gamma = torch.exp(self.gamma[0] + eps[7]*(self.gamma[1].exp()))
            else:
                if self.cvae:
                    if compute_x0:
                        ton = self.ton[:, 0].exp()
                        c0 = self.c0[:, 0].exp()
                        u0 = self.u0[:, 0].exp()
                        s0 = self.s0[:, 0].exp()
                    alpha_c = self.alpha_c[:, 0].exp()
                    alpha = self.alpha[:, 0].exp()
                    beta = self.beta[:, 0].exp()
                    gamma = self.gamma[:, 0].exp()
                else:
                    if compute_x0:
                        ton = self.ton[0].exp()
                        c0 = self.c0[0].exp()
                        u0 = self.u0[0].exp()
                        s0 = self.s0[0].exp()
                    alpha_c = self.alpha_c[0].exp()
                    alpha = self.alpha[0].exp()
                    beta = self.beta[0].exp()
                    gamma = self.gamma[0].exp()
        else:
            if compute_x0:
                ton = self.ton.exp()
                c0 = self.c0.exp()
                u0 = self.u0.exp()
                s0 = self.s0.exp()
            alpha_c = self.alpha_c.exp()
            alpha = self.alpha.exp()
            beta = self.beta.exp()
            gamma = self.gamma.exp()

        if condition is not None:
            if compute_x0:
                ton = torch.mm(condition, ton)
                c0 = torch.mm(condition, c0)
                u0 = torch.mm(condition, u0)
                s0 = torch.mm(condition, s0)
            alpha_c = torch.mm(condition, alpha_c)
            alpha = torch.mm(condition, alpha)
            beta = torch.mm(condition, beta)
            gamma = torch.mm(condition, gamma)

        if compute_x0:
            return ton, c0, u0, s0, alpha_c, alpha, beta, gamma
        else:
            return alpha_c, alpha, beta, gamma

    def forward(self, t, z, condition=None, c0=None, u0=None, s0=None, t0=None, neg_slope=0.0, sample=True):
        if (c0 is None) or (u0 is None) or (s0 is None) or (t0 is None):
            ton, c0_, u0_, s0_, alpha_c, alpha, beta, gamma = self.reparameterize(condition, sample)
            if condition is None:
                self.rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
                self.kc = torch.sigmoid(self.fc_out3(self.net_kc(z if self.indicator_arch == 'parallel' else self.rho)))
                t_ = F.softplus(self.net_t(t)) if self.t_network else t
            else:
                self.rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z, condition), 1))))
                self.kc = torch.sigmoid(self.fc_out3(self.net_kc(torch.cat((z, condition), 1) if self.indicator_arch == 'parallel' else torch.cat((self.rho, condition), 1))))
                t_ = F.softplus(self.net_t(torch.cat((t, condition), 1))) if self.t_network else t

            chat, uhat, shat = pred_exp(t_,
                                        ton,
                                        neg_slope,
                                        c0_/self.scaling_c.exp(),
                                        u0_/self.scaling.exp(),
                                        s0_,
                                        self.kc,
                                        alpha_c,
                                        self.rho,
                                        alpha,
                                        beta,
                                        gamma)
        else:
            alpha_c, alpha, beta, gamma = self.reparameterize(condition, sample, False)
            if condition is None:
                self.rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
                self.kc = torch.sigmoid(self.fc_out4(self.net_kc2(z if self.indicator_arch == 'parallel' else self.rho)))
                t_ = F.softplus(self.net_t(t)) if self.t_network else t
            else:
                self.rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z, condition), 1))))
                self.kc = torch.sigmoid(self.fc_out4(self.net_kc2(torch.cat((z, condition), 1) if self.indicator_arch == 'parallel' else torch.cat((self.rho, condition), 1))))
                t_ = F.softplus(self.net_t(torch.cat((t, condition), 1))) if self.t_network else t

            chat, uhat, shat = pred_exp(t_,
                                        t0,
                                        neg_slope,
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
        return F.relu(chat), F.relu(uhat), F.relu(shat), t_


class VAEChrom():
    def __init__(self,
                 adata,
                 adata_atac,
                 dim_z=20,
                 batch_key=None,
                 ref_batch=-1,
                 device='cpu',
                 hidden_size=(512, 256, 256, 512),
                 mode='vae',
                 indicator_arch='parallel',
                 t_network=False,
                 train_x0=False,
                 train_3rd_stage=False,
                 tmax=1,
                 reinit_params=False,
                 init_method='steady',
                 init_key=None,
                 tprior=None,
                 init_type=None,
                 init_ton_zero=True,
                 unit_scale=False,
                 time_distribution='gaussian',
                 std_z_prior=0.01,
                 checkpoints=[None, None],
                 plot_init=False,
                 gene_plot=[],
                 cluster_key='clusters',
                 figure_path='figures'):
        if ('Mc' not in adata.atac.layers) or ('Mu' not in adata.layers) or ('Ms') not in adata.layers:
            print('Chromatin/Unspliced/Spliced count matrices not found in the layers! Exit the program...')
            return

        self.config = {
            # model parameters
            "dim_z": dim_z,
            "hidden_size": hidden_size,
            "mode": mode,
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
            "std_z_prior": std_z_prior,
            "tail": 0.01,
            "time_overlap": 0.5,
            "n_neighbors": 10,
            "dt": (0.03, 0.06),
            "n_bin": None,

            # training parameters
            "n_epochs": 1000,
            "n_epochs_post": 1000,
            "n_epochs_final": 1000,
            "batch_size": 128,
            "learning_rate": 2e-4,
            "learning_rate_ode": 5e-4,
            "learning_rate_post": 2e-4,
            "learning_rate_final": 2e-4,
            "lambda": 1e-3,
            "lambda_rho": 1e-3,
            "kl_t": 1.0,
            "kl_z": 1.0,
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
            "train_ton": True,
            "train_x0": train_x0,
            "weight_sample": False,
            "always_update_t": False,
            "knn_use_pred": True,
            "run_3rd_stage": train_3rd_stage,
            "reg_velo": True,

            # plotting
            "sparsify": 1}

        self.set_device(device)
        self.split_train_test(adata.n_obs)

        G = adata.n_vars
        self.dim_z = dim_z

        self.n_batch = 0
        self.ref_batch = int(ref_batch)
        self.batch = None
        if batch_key is not None and batch_key in adata.obs:
            batch_raw = adata.obs[batch_key].to_numpy()
            batch_names_raw = np.unique(batch_raw)
            self.batch_dic, self.batch_dic_rev = encode_type(batch_names_raw)
            self.n_batch = len(batch_names_raw)
            self.batch = np.array([self.batch_dic[x] for x in batch_raw])
            self.batch_names = np.array([self.batch_dic[batch_names_raw[i]] for i in range(self.n_batch)])
            if self.ref_batch >= self.n_batch:
                self.ref_batch = self.n_batch - 1
        self.enable_cvae = self.n_batch > 0

        self.encoder = Encoder(3*G,
                               dim_z,
                               dim_cond=self.n_batch,
                               N1=hidden_size[0],
                               N2=hidden_size[1],
                               device=self.device,
                               checkpoint=checkpoints[0]).float()

        self.decoder = Decoder(adata,
                               adata_atac,
                               self.train_idx,
                               dim_z,
                               dim_cond=self.n_batch,
                               N1=hidden_size[2],
                               N2=hidden_size[3],
                               indicator_arch=indicator_arch,
                               t_network=t_network,
                               params_mult=(1, np.log(0.05)) if mode == 'fullvb' else 1,
                               p=98,
                               tmax=tmax,
                               reinit=reinit_params,
                               init_ton_zero=init_ton_zero,
                               init_method=init_method,
                               init_key=init_key,
                               init_type=init_type,
                               cluster_key=cluster_key,
                               device=self.device,
                               checkpoint=checkpoints[1]).float()
        self.tmax = tmax
        self.time_distribution = time_distribution
        self.get_prior(adata, time_distribution, tmax, tprior)

        self.p_z = torch.stack([torch.zeros(adata.layers['Mu'].shape[0], dim_z), torch.ones(adata.layers['Mu'].shape[0], dim_z)*self.config["std_z_prior"]]).float().to(self.device)

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
            self.scale_c = torch.tensor(np.clip(np.std(adata_atac.layers['Mc'][self.train_idx, :], 0), 1e-6, None)).float().to(self.device)
            self.scale_u = torch.tensor(np.clip(np.std(adata.layers['Mu'][self.train_idx, :], 0), 1e-6, None)).float().to(self.device)
            self.scale_s = torch.tensor(np.clip(np.std(adata.layers['Ms'][self.train_idx, :], 0), 1e-6, None)).float().to(self.device)

        if mode == 'fullvb':
            self.p_log_ton = torch.tensor([[0.0], [0.1]]).to(self.device)
            self.p_log_c0 = torch.tensor([[0.0], [0.1]]).to(self.device)
            self.p_log_u0 = torch.tensor([[0.0], [0.1]]).to(self.device)
            self.p_log_s0 = torch.tensor([[0.0], [0.1]]).to(self.device)
            self.p_log_alpha_c = torch.tensor([[0.0], [0.5]]).to(self.device)
            self.p_log_alpha = torch.tensor([[0.0], [1.0]]).to(self.device)
            self.p_log_beta = torch.tensor([[0.0], [0.5]]).to(self.device)
            self.p_log_gamma = torch.tensor([[0.0], [0.5]]).to(self.device)
            self.p_params = [self.p_log_ton, self.p_log_c0, self.p_log_u0, self.p_log_s0, self.p_log_alpha_c, self.p_log_alpha, self.p_log_beta, self.p_log_gamma]

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

    def set_mode(self, mode):
        if mode == 'train':
            self.encoder.train()
            self.decoder.train()
        elif mode == 'eval':
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")

    def reparameterize(self, mu, std):
        eps = torch.normal(mean=torch.zeros(mu.shape), std=torch.ones(mu.shape)).to(self.device)
        return std*eps + mu

    def reparameterize_uniform(self, mu, std):
        eps = torch.rand(mu.shape).to(self.device)
        return np.sqrt(12)*std*eps + (mu - np.sqrt(3)*std)

    def get_prior(self, adata, time_distribution, tmax, tprior=None):
        if time_distribution == "gaussian":
            print("Gaussian Prior.")
            self.kl_time = kl_gaussian
            self.sample = self.reparameterize
            if tprior is None:
                self.p_t = torch.stack([torch.ones(adata.n_obs, 1)*tmax*0.5, torch.ones(adata.n_obs, 1)*tmax*self.config["time_overlap"]]).float().to(self.device)
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
        else:
            print("Tailed Uniform Prior.")
            self.kl_time = kl_uniform
            self.sample = self.reparameterize_uniform
            if tprior is None:
                self.p_t = torch.stack([torch.zeros(adata.n_obs, 1), torch.ones(adata.n_obs, 1)*tmax]).float().to(self.device)
            else:
                print('Using informative time prior.')
                t = adata.obs[tprior].to_numpy()
                t = t/t.max()*tmax
                t_cap = np.sort(np.unique(t))
                t_start = np.zeros((len(t)))
                t_end = np.zeros((len(t)))
                for i in range(len(t_cap)-1):
                    t_end[t == t_cap[i]] = t_cap[i] + (t_cap[i+1] - t_cap[i])*(0.5+0.5*self.config["time_overlap"])
                t_end[t == t_cap[-1]] = t_cap[-1] + (t_cap[-1] - t_cap[-2])*(0.5+0.5*self.config["time_overlap"])

                for i in range(1, len(t_cap)):
                    t_start[t == t_cap[i]] = max(0, t_cap[i] - (t_cap[i] - t_cap[i-1])*(0.5+0.5*self.config["time_overlap"]))
                t_start[t == t_cap[0]] = max(0, t_cap[0] - (t_cap[1] - t_cap[0])*(0.5+0.5*self.config["time_overlap"]))

                self.p_t = torch.stack([torch.tensor(t).unsqueeze(-1), torch.tensor(t_end).unsqueeze(-1)]).float().to(self.device)

    def plot_initial(self, adata, adata_atac, gene_plot, cluster_key="clusters", figure_path="figures"):
        cell_labels_raw = adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs else np.array(['Unknown' for i in range(adata.n_obs)])
        cell_types_raw = np.unique(cell_labels_raw)
        label_dic, _ = encode_type(cell_types_raw)

        n_type = len(cell_types_raw)
        cell_labels = np.array([label_dic[x] for x in cell_labels_raw])[self.train_idx]
        cell_types = np.array([label_dic[cell_types_raw[i]] for i in range(n_type)])

        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        C, U, S = adata_atac.layers['Mc'][self.train_idx, :][:, gind], adata.layers['Mu'][self.train_idx, :][:, gind], adata.layers['Ms'][self.train_idx, :][:, gind]

        alpha_c = self.decoder.alpha_c_[gind]
        alpha = self.decoder.alpha_[gind]
        beta = self.decoder.beta_[gind]
        gamma = self.decoder.gamma_[gind]
        scaling_c = self.decoder.scaling_c_[gind]
        scaling = self.decoder.scaling_[gind]
        toff = self.decoder.toff_[gind]
        T = self.decoder.T_[:, gind]
        chat, uhat, shat = ode_numpy(T, alpha_c, alpha, beta, gamma, 0, toff)

        for i in range(len(gind)):
            if np.any(np.isnan(chat[:, i])):
                print(gene_plot[i], chat[:, i])
            if np.any(np.isnan(uhat[:, i])):
                print(gene_plot[i], uhat[:, i])
            if np.any(np.isnan(shat[:, i])):
                print(gene_plot[i], shat[:, i])

            plot_sig(T[:, i].squeeze(),
                     C[:, i]/scaling_c[i],
                     U[:, i]/scaling[i],
                     S[:, i],
                     chat[:, i],
                     uhat[:, i],
                     shat[:, i],
                     'us',
                     cell_labels,
                     gene_plot[i],
                     save=f"{figure_path}/sig-{gene_plot[i]}-init-us.png",
                     sparsify=self.config['sparsify'])

            plot_sig(T[:, i].squeeze(),
                     C[:, i]/scaling_c[i],
                     U[:, i]/scaling[i],
                     S[:, i],
                     chat[:, i],
                     uhat[:, i],
                     shat[:, i],
                     'cu',
                     cell_labels,
                     gene_plot[i],
                     save=f"{figure_path}/sig-{gene_plot[i]}-init-cu.png",
                     sparsify=self.config['sparsify'])

            plot_phase(C[:, i]/scaling_c[i],
                       U[:, i]/scaling[i],
                       S[:, i],
                       chat[:, i],
                       uhat[:, i],
                       shat[:, i],
                       gene_plot[i],
                       'us',
                       None,
                       cell_labels,
                       cell_types,
                       save=f"{figure_path}/phase-{gene_plot[i]}-init-us.png")

            plot_phase(C[:, i]/scaling_c[i],
                       U[:, i]/scaling[i],
                       S[:, i],
                       chat[:, i],
                       uhat[:, i],
                       shat[:, i],
                       gene_plot[i],
                       'cu',
                       None,
                       cell_labels,
                       cell_types,
                       save=f"{figure_path}/phase-{gene_plot[i]}-init-cu.png")

    def forward(self, data_in, c0=None, u0=None, s0=None, t0=None, condition=None):
        if self.config['unit_scale']:
            data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//3]/self.scale_c,
                                      data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]/self.scale_u,
                                      data_in[:, data_in.shape[1]//3*2:]/self.scale_s), 1)
        else:
            data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//3]/torch.exp(self.decoder.scaling_c),
                                      data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]/torch.exp(self.decoder.scaling),
                                      data_in[:, data_in.shape[1]//3*2:]), 1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)

        chat, uhat, shat, t_ = self.decoder.forward(t, z, condition, c0, u0, s0, t0, neg_slope=self.config["neg_slope"])

        return mu_t, std_t, mu_z, std_z, t, z, chat, uhat, shat, t_

    def eval_model(self, data_in, c0=None, u0=None, s0=None, t0=None, condition=None):
        if self.config['unit_scale']:
            data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//3]/self.scale_c,
                                      data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]/self.scale_u,
                                      data_in[:, data_in.shape[1]//3*2:]/self.scale_s), 1)
        else:
            data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//3]/torch.exp(self.decoder.scaling_c),
                                      data_in[:, data_in.shape[1]//3:data_in.shape[1]//3*2]/torch.exp(self.decoder.scaling),
                                      data_in[:, data_in.shape[1]//3*2:]), 1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)

        chat, uhat, shat, t_ = self.decoder.forward(mu_t, mu_z, condition, c0=c0, u0=u0, s0=s0, t0=t0, neg_slope=0.0, sample=False)

        return mu_t, std_t, mu_z, std_z, chat, uhat, shat, t_

    def vae_risk(self,
                 q_tx,
                 p_t,
                 q_zx,
                 p_z,
                 C,
                 U,
                 S,
                 chat,
                 uhat,
                 shat,
                 sigma_c,
                 sigma_u,
                 sigma_s,
                 S_knn=None,
                 onehot=None,
                 weight=None):

        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        kld_params = torch.tensor(0).float().to(self.device)
        if self.config['mode'] == 'fullvb':
            for x in ['alpha_c', 'alpha', 'beta', 'gamma'] if self.use_knn else ['ton', 'c0', 'u0', 's0', 'alpha_c', 'alpha', 'beta', 'gamma']:
                if self.enable_cvae:
                    for i in range(self.n_batch):
                        kld_params += kl_gaussian(self.decoder.get_param(x)[i, 0].view(1, -1), self.decoder.get_param(x)[i, 1].exp().view(1, -1), self.p_log_ton[0], self.p_log_ton[1])
                else:
                    kld_params += kl_gaussian(self.decoder.get_param(x)[0].view(1, -1), self.decoder.get_param(x)[1].exp().view(1, -1), self.p_log_ton[0], self.p_log_ton[1])
            kld_params /= U.shape[0]

        logp = -0.5*((C-chat)/sigma_c).pow(2)
        logp -= 0.5*((U-uhat)/sigma_u).pow(2)
        logp -= 0.5*((S-shat)/sigma_s).pow(2)
        logp -= torch.log(sigma_c)
        logp -= torch.log(sigma_u)
        logp -= torch.log(sigma_s*2*np.pi)

        if weight is not None:
            logp = logp*weight

        err_rec = torch.mean(torch.sum(logp, 1))

        if S_knn is not None:
            cos_sim = cosine_similarity(uhat/self.decoder.scaling.exp(), shat, self.decoder.beta.exp(), self.decoder.gamma.exp(), S_knn, onehot)
            loss = - err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz + self.config["kl_param"]*kld_params - cos_sim
        else:
            loss = - err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz + self.config["kl_param"]*kld_params

        return loss

    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, optimizer3=None, K=1, train_net='both', drop=True):
        B = len(train_loader)
        if train_net == 'both':
            self.set_mode('train')
        elif train_net == 'encoder':
            self.encoder.train()
        elif train_net == 'decoder':
            self.decoder.train()
        stop_training = False

        for i, batch in enumerate(train_loader):
            if (self.counter == 1) or (self.counter % self.config["test_iter"] == 0):
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
                if train_net == 'both':
                    self.set_mode('train')
                elif train_net == 'encoder':
                    self.encoder.train()
                elif train_net == 'decoder':
                    self.decoder.train()

                if (self.n_drop >= self.config["early_stop"]) and (self.config["early_stop"] > 0):
                    stop_training = True
                    break

            optimizer.zero_grad()
            if optimizer2 is not None:
                optimizer2.zero_grad()
            if optimizer3 is not None:
                optimizer3.zero_grad()

            xbatch, idx = batch[0].float().to(self.device), batch[3]
            c = xbatch[:, :xbatch.shape[1]//3]
            u = xbatch[:, xbatch.shape[1]//3:xbatch.shape[1]//3*2]
            s = xbatch[:, xbatch.shape[1]//3*2:]

            condition = F.one_hot(batch[4].to(self.device), self.n_batch).float() if self.enable_cvae else None
            mu_tx, std_tx, mu_zx, std_zx, t, z, chat, uhat, shat, t_ = self.forward(xbatch,
                                                                                    self.c0[self.train_idx[idx]] if self.use_knn else None,
                                                                                    self.u0[self.train_idx[idx]] if self.use_knn else None,
                                                                                    self.s0[self.train_idx[idx]] if self.use_knn else None,
                                                                                    self.t0[self.train_idx[idx]] if self.use_knn else None,
                                                                                    condition)

            loss = self.vae_risk((mu_tx, std_tx),
                                 self.p_t[:, self.train_idx[idx], :],
                                 (mu_zx, std_zx),
                                 self.p_z[:, self.train_idx[idx], :],
                                 c,
                                 u,
                                 s,
                                 chat,
                                 uhat,
                                 shat,
                                 torch.exp(self.decoder.sigma_c),
                                 torch.exp(self.decoder.sigma_u),
                                 torch.exp(self.decoder.sigma_s),
                                 torch.tensor(self.S_knn[self.train_idx[idx]], dtype=torch.float, device=self.device, requires_grad=False) if self.reg_velocity else None,
                                 F.one_hot(torch.tensor(self.batch[self.train_idx[idx]], dtype=int, device=self.device), self.n_batch).float() if self.enable_cvae else None,
                                 None)

            loss.backward()
            if K == 0:
                optimizer.step()
                if optimizer2 is not None:
                    optimizer2.step()
                if optimizer3 is not None:
                    optimizer3.step()
            else:
                if ((i+1) % (K+1) == 0) or (i == B-1):
                    if optimizer2 is not None:
                        optimizer2.step()
                    if optimizer3 is not None:
                        optimizer3.step()
                else:
                    optimizer.step()

            self.loss_train.append(loss.detach().cpu().item())

            self.counter = self.counter + 1
        return stop_training

    def update_x0(self, C, U, S):
        start = time.time()
        self.set_mode('eval')
        out, _ = self.pred_all(np.concatenate((C, U, S), 1), "both", gene_idx=np.array(range(U.shape[1])))
        chat, uhat, shat, t, z = out[0], out[1], out[2], out[5], out[6]
        t_ = t.copy()
        t = np.clip(t, 0, np.quantile(t, 0.99))
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        if self.config["n_bin"] is None:
            print("Cell-wise KNN Estimation.")
            if self.config["knn_use_pred"]:
                c0, u0, s0, t0, _ = knnx0(chat[self.train_idx], uhat[self.train_idx], shat[self.train_idx], t[self.train_idx], z[self.train_idx], t, z, dt, self.config["n_neighbors"])
            else:
                c0, u0, s0, t0, _ = knnx0(C[self.train_idx], U[self.train_idx], S[self.train_idx], t[self.train_idx], z[self.train_idx], t, z, dt, self.config["n_neighbors"])
        else:
            n_bin = self.config["n_bin"]
            print(f"Fast KNN Estimation with {n_bin} time bins.")
            if self.config["knn_use_pred"]:
                c0, u0, s0, t0, _ = knnx0_bin(chat[self.train_idx], uhat[self.train_idx], shat[self.train_idx], t[self.train_idx], z[self.train_idx], t, z, dt, self.config["n_neighbors"], n_bin)
            else:
                c0, u0, s0, t0, _ = knnx0_bin(C[self.train_idx], U[self.train_idx], S[self.train_idx], t[self.train_idx], z[self.train_idx], t, z, dt, self.config["n_neighbors"], n_bin)
        t0 = t0.reshape(-1, 1)
        print(f"Finished. Actual Time: {convert_time(time.time()-start)}")
        return c0, u0, s0, t0, t_

    def predict_knn(self, C, U, S, c0, u0, s0, sigma_c, sigma_u, sigma_s):
        if self.config["knn_use_pred"]:
            out, _ = self.pred_all(np.concatenate((C, U, S), 1), "both", gene_idx=np.array(range(U.shape[1])))
            chat, uhat, shat = out[0], out[1], out[2]
            knn = knn_approx(chat/sigma_c, uhat/sigma_u, shat/sigma_s, c0/sigma_c, u0/sigma_u, s0/sigma_s, self.config["n_neighbors"])
            C_knn, U_knn, S_knn = chat[knn], uhat[knn], shat[knn]
        else:
            knn = knn_approx(C/sigma_c, U/sigma_u, S/sigma_s, c0/sigma_c, u0/sigma_u, s0/sigma_s, self.config["n_neighbors"])
            C_knn, U_knn, S_knn = C[knn], U[knn], S[knn]
        return C_knn, U_knn, S_knn

    def load_config(self, config):
        for key in config:
            if key in self.config:
                self.config[key] = config[key]
            else:
                self.config[key] = config[key]
                print(f"Warning: unknown hyperparameter: {key}")
        if self.config["train_scaling"]:
            self.decoder.scaling_c.requires_grad = True
            self.decoder.scaling.requires_grad = True
        if self.config["train_std"]:
            self.decoder.sigma_c.requires_grad = True
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True

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
        self.load_config(config)
        start = time.time()

        print("--------------------------- Train a VeloVAE ---------------------------")

        if use_raw:
            C = np.array(adata_atac.X.todense())
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
            X = np.concatenate((C, U, S), 1).astype(int)
        else:
            X = np.concatenate((adata_atac.layers['Mc'], adata.layers['Mu'], adata.layers['Ms']), 1).astype(float)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Set to None.")
            Xembed = None
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
                           weight=self.decoder.Rscore[self.train_idx] if self.config['weight_sample'] else None,
                           batch=self.batch[self.train_idx] if self.enable_cvae else None)

        test_set = None
        if len(self.test_idx) > 0:
            test_set = SCData(X[self.test_idx],
                              self.cell_labels[self.test_idx],
                              weight=self.decoder.Rscore[self.test_idx] if self.config['weight_sample'] else None,
                              batch=self.batch[self.test_idx] if self.enable_cvae else None)

        data_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
        if self.config["test_iter"] is None:
            self.config["test_iter"] = len(self.train_idx)//self.config["batch_size"]*2
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")
        print("*********                      Finished.                      *********")

        print("*********                      Stage  1                       *********")
        param_nn = list(self.encoder.parameters())+list(self.decoder.net_rho.parameters())+list(self.decoder.fc_out1.parameters())
        param_nn += list(self.decoder.net_kc.parameters())+list(self.decoder.fc_out3.parameters())
        if self.config['t_network']:
            param_nn += list(self.decoder.net_t.parameters())
        param_ode = [self.decoder.alpha_c, self.decoder.alpha, self.decoder.beta, self.decoder.gamma, self.decoder.c0, self.decoder.u0, self.decoder.s0]  #
        if self.config['train_ton']:
            param_ode.append(self.decoder.ton)
        if self.config['train_scaling']:
            self.decoder.scaling_c.requires_grad = True
            self.decoder.scaling.requires_grad = True
            param_ode += [self.decoder.scaling_c]+[self.decoder.scaling]
        if self.config['train_std']:
            self.decoder.sigma_c.requires_grad = True
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True
            param_ode += [self.decoder.sigma_c, self.decoder.sigma_u, self.decoder.sigma_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])

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

            if plot and (epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+1}",
                                       False,
                                       gind,
                                       gene_plot,
                                       True,
                                       figure_path)
                self.set_mode('train')
                elbo_test = self.loss_test[-1] if len(self.loss_test) > 0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                break
        self.save_model("checkpoints/HSPC/FullVB3Stage", "encoder_fullvb", "decoder_fullvb")

        print("*********                      Stage  2                       *********")
        n_stage1 = epoch+1
        n_test1 = len(self.loss_test)

        self.encoder.eval()
        self.c0, self.u0, self.s0, self.t0, t_ = self.update_x0(X[:, :X.shape[1]//3], X[:, X.shape[1]//3:X.shape[1]//3*2], X[:, X.shape[1]//3*2:])
        c0_ = self.c0.copy()
        u0_ = self.u0.copy()
        s0_ = self.s0.copy()
        t0_ = self.t0.copy()

        self.use_knn = True
        self.decoder.init_weights()

        if plot:
            plot_time(self.t0.squeeze(), Xembed, save=f"{figure_path}/t0.png")
            t0_plot = self.t0[self.train_idx].squeeze()
            for i in range(len(gind)):
                idx = gind[i]
                c0_plot = self.c0[self.train_idx, idx]
                u0_plot = self.u0[self.train_idx, idx]
                s0_plot = self.s0[self.train_idx, idx]
                plot_sig_(t0_plot,
                          c0_plot,
                          u0_plot, s0_plot,
                          cell_labels=cell_labels_raw[self.train_idx],
                          by='us',
                          title=gene_plot[i],
                          save=f"{figure_path}/{gene_plot[i]}-x0-us.png")
                plot_sig_(t0_plot,
                          c0_plot,
                          u0_plot, s0_plot,
                          cell_labels=cell_labels_raw[self.train_idx],
                          by='cu',
                          title=gene_plot[i],
                          save=f"{figure_path}/{gene_plot[i]}-x0-cu.png")

        self.n_drop = 0
        param_post = list(self.decoder.net_rho2.parameters()) + list(self.decoder.fc_out2.parameters())
        param_post += list(self.decoder.net_kc2.parameters()) + list(self.decoder.fc_out4.parameters())
        if self.config['t_network'] and self.config['always_update_t']:
            param_post += list(self.decoder.net_t.parameters())
        optimizer_post = torch.optim.Adam(param_post, lr=self.config["learning_rate_post"], weight_decay=self.config["lambda_rho"])

        self.c0 = torch.tensor(self.c0, dtype=torch.float, device=self.device, requires_grad=True)
        self.u0 = torch.tensor(self.u0, dtype=torch.float, device=self.device, requires_grad=True)
        self.s0 = torch.tensor(self.s0, dtype=torch.float, device=self.device, requires_grad=True)
        self.t0 = torch.tensor(self.t0, dtype=torch.float, device=self.device, requires_grad=True)
        if self.config["train_x0"]:
            optimizer_x0 = torch.optim.Adam([self.s0, self.u0, self.c0, self.t0], lr=self.config["learning_rate_ode"])
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

            if plot and (epoch == 0 or (epoch+n_stage1+1) % self.config["save_epoch"] == 0):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+n_stage1+1}",
                                       False,
                                       gind,
                                       gene_plot,
                                       True,
                                       figure_path)
                self.decoder.train()
                elbo_test = self.loss_test[-1] if len(self.loss_test) > n_test1 else -np.inf
                print(f"Epoch {epoch+n_stage1+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********       Stage 2: Early Stop Triggered at epoch {epoch+n_stage1+1}.       *********")
                break

        if self.config["run_3rd_stage"]:
            print("*********                      Stage  3                       *********")
            n_stage2 = epoch+n_stage1+1
            n_test2 = len(self.loss_test)

            self.encoder.train()
            self.decoder.eval()
            self.C_knn, self.U_knn, self.S_knn = self.predict_knn(X[:, :X.shape[1]//3],
                                                                  X[:, X.shape[1]//3:X.shape[1]//3*2],
                                                                  X[:, X.shape[1]//3*2:],
                                                                  self.c0.detach().cpu().numpy(),
                                                                  self.u0.detach().cpu().numpy(),
                                                                  self.s0.detach().cpu().numpy(),
                                                                  torch.exp(self.decoder.sigma_c).detach().cpu().numpy(),
                                                                  torch.exp(self.decoder.sigma_u).detach().cpu().numpy(),
                                                                  torch.exp(self.decoder.sigma_s).detach().cpu().numpy())
            if self.config['reg_velo']:
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
                              cell_labels=cell_labels_raw[self.train_idx],
                              by='us',
                              title=gene_plot[i],
                              save=f"{figure_path}/{gene_plot[i]}-x0-us_updated.png")
                    plot_sig_(t0_plot,
                              c0_plot,
                              u0_plot, s0_plot,
                              cell_labels=cell_labels_raw[self.train_idx],
                              by='cu',
                              title=gene_plot[i],
                              save=f"{figure_path}/{gene_plot[i]}-x0-cu_updated.png")

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

                if plot and (epoch == 0 or (epoch+n_stage2+1) % self.config["save_epoch"] == 0):
                    elbo_train = self.test(train_set,
                                           Xembed[self.train_idx],
                                           f"train{epoch+n_stage2+1}",
                                           False,
                                           gind,
                                           gene_plot,
                                           True,
                                           figure_path)
                    self.encoder.train()
                    elbo_test = self.loss_test[-1] if len(self.loss_test) > n_test2 else -np.inf
                    print(f"Epoch {epoch+n_stage2+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")

                if stop_training:
                    print(f"*********       Stage 3: Early Stop Triggered at epoch {epoch+n_stage2+1}.       *********")
                    break

        print(f"*********              Finished. Total Time = {convert_time(time.time()-start)}             *********")

        if plot:
            elbo_train = self.test(train_set,
                                   Xembed[self.train_idx],
                                   "final-train",
                                   False,
                                   gind,
                                   gene_plot,
                                   True,
                                   figure_path)
            elbo_test = self.test(test_set,
                                  Xembed[self.test_idx],
                                  "final-test",
                                  True,
                                  gind,
                                  gene_plot,
                                  True,
                                  figure_path)
            plot_train_loss(self.loss_train, range(1, len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            if self.config["test_iter"] > 0:
                plot_test_loss(self.loss_test, [i*self.config["test_iter"] for i in range(1, len(self.loss_test)+1)], save=f'{figure_path}/test_loss_velovae.png')
        return c0_, u0_, s0_, t0_, t_

    def pred_all(self, data, mode='test', output=["chat", "uhat", "shat", "t", "z"], gene_idx=None, batch=None):
        N, G = data.shape[0], data.shape[1]//3
        elbo = 0
        if batch is None:
            batch = self.batch
        if "chat" in output:
            chat = None if gene_idx is None else np.zeros((N, len(gene_idx)))
        if "uhat" in output:
            uhat = None if gene_idx is None else np.zeros((N, len(gene_idx)))
        if "shat" in output:
            shat = None if gene_idx is None else np.zeros((N, len(gene_idx)))
        if "t" in output:
            t_out = np.zeros((N))
            std_t_out = np.zeros((N))
            time_out = np.zeros((N))
        if "z" in output:
            z_out = np.zeros((N, self.dim_z))
            std_z_out = np.zeros((N, self.dim_z))

        with torch.no_grad():
            B = min(N//5, 5000)
            Nb = N // B
            if Nb*B < N:
                Nb += 1
            for n in range(Nb):
                i = n*B
                j = min([(n+1)*B, N])
                data_in = torch.tensor(data[i:j]).float().to(self.device)
                if mode == "test":
                    c0 = self.c0[self.test_idx[i:j]] if self.use_knn else None
                    u0 = self.u0[self.test_idx[i:j]] if self.use_knn else None
                    s0 = self.s0[self.test_idx[i:j]] if self.use_knn else None
                    t0 = self.t0[self.test_idx[i:j]] if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(batch[self.test_idx[i:j]], dtype=int, device=self.device), self.n_batch).float() if self.enable_cvae else None
                    S_knn_sub = torch.tensor(self.S_knn[self.test_idx[i:j]], dtype=torch.float, device=self.device) if self.reg_velocity else None
                    p_t = self.p_t[:, self.test_idx[i:j], :]
                    p_z = self.p_z[:, self.test_idx[i:j], :]
                    mu_tx, std_tx, mu_zx, std_zx, chat, uhat, shat, t_ = self.eval_model(data_in, c0, u0, s0, t0, y_onehot)
                elif mode == "train":
                    c0 = self.c0[self.train_idx[i:j]] if self.use_knn else None
                    u0 = self.u0[self.train_idx[i:j]] if self.use_knn else None
                    s0 = self.s0[self.train_idx[i:j]] if self.use_knn else None
                    t0 = self.t0[self.train_idx[i:j]] if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(batch[self.train_idx[i:j]], dtype=int, device=self.device), self.n_batch).float() if self.enable_cvae else None
                    S_knn_sub = torch.tensor(self.S_knn[self.train_idx[i:j]], dtype=torch.float, device=self.device) if self.reg_velocity else None
                    p_t = self.p_t[:, self.train_idx[i:j], :]
                    p_z = self.p_z[:, self.train_idx[i:j], :]
                    mu_tx, std_tx, mu_zx, std_zx, chat, uhat, shat, t_ = self.eval_model(data_in, c0, u0, s0, t0, y_onehot)
                else:
                    c0 = self.c0[i:j] if self.use_knn else None
                    u0 = self.u0[i:j] if self.use_knn else None
                    s0 = self.s0[i:j] if self.use_knn else None
                    t0 = self.t0[i:j] if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(batch[i:j], dtype=int, device=self.device), self.n_batch).float() if self.enable_cvae else None
                    S_knn_sub = torch.tensor(self.S_knn[i:j], dtype=torch.float, device=self.device) if self.reg_velocity else None
                    p_t = self.p_t[:, i:j, :]
                    p_z = self.p_z[:, i:j, :]
                    mu_tx, std_tx, mu_zx, std_zx, chat, uhat, shat, t_ = self.eval_model(data_in, c0, u0, s0, t0, y_onehot)
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
                                     torch.exp(self.decoder.sigma_c),
                                     torch.exp(self.decoder.sigma_u),
                                     torch.exp(self.decoder.sigma_s),
                                     S_knn_sub,
                                     y_onehot,
                                     None)
                elbo = elbo - (B/N)*loss
                if "chat" in output and gene_idx is not None:
                    chat[i:j] = chat[:, gene_idx].cpu().numpy()
                if "uhat" in output and gene_idx is not None:
                    uhat[i:j] = uhat[:, gene_idx].cpu().numpy()
                if "shat" in output and gene_idx is not None:
                    shat[i:j] = shat[:, gene_idx].cpu().numpy()
                if "t" in output:
                    t_out[i:j] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[i:j] = std_tx.cpu().squeeze().numpy()
                    time_out[i:j] = t_.cpu().squeeze().numpy()
                if "z" in output:
                    z_out[i:j] = mu_zx.cpu().numpy()
                    std_z_out[i:j] = std_zx.cpu().numpy()
        out = []
        if "chat" in output:
            out.append(chat)
        if "uhat" in output:
            out.append(uhat)
        if "shat" in output:
            out.append(shat)
        if "t" in output:
            out.append(t_out)
            out.append(std_t_out)
            out.append(time_out)
        if "z" in output:
            out.append(z_out)
            out.append(std_z_out)

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
        out, elbo = self.pred_all(dataset.data, mode, ["chat", "uhat", "shat", "t"], gind)
        chat, uhat, shat, t_ = out[0], out[1], out[2], out[5]

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
                         dataset.data[:, idx],
                         dataset.data[:, idx+G],
                         dataset.data[:, idx+G*2],
                         chat[:, i],
                         uhat[:, i],
                         shat[:, i],
                         'us',
                         np.array([self.label_dic_rev[x] for x in dataset.labels]),
                         gene_plot[i],
                         save=f"{path}/sig-{gene_plot[i]}-{testid_str}-us.png",
                         sparsify=self.config['sparsify'])

                plot_sig(t_.squeeze(),
                         dataset.data[:, idx],
                         dataset.data[:, idx+G],
                         dataset.data[:, idx+G*2],
                         chat[:, i],
                         uhat[:, i],
                         shat[:, i],
                         'cu',
                         np.array([self.label_dic_rev[x] for x in dataset.labels]),
                         gene_plot[i],
                         save=f"{path}/sig-{gene_plot[i]}-{testid_str}-cu.png",
                         sparsify=self.config['sparsify'])

        return elbo

    def save_anndata(self, adata, adata_atac, key, file_path, file_name=None):
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)

        if self.config['mode'] == 'fullvb':
            if self.enable_cvae:
                for i in range(self.n_batch):
                    adata.var[f"{key}_alpha_c_{i}"] = np.exp(self.decoder.alpha_c[i, 0].detach().cpu().numpy())
                    adata.var[f"{key}_alpha_{i}"] = np.exp(self.decoder.alpha[i, 0].detach().cpu().numpy())
                    adata.var[f"{key}_beta_{i}"] = np.exp(self.decoder.beta[i, 0].detach().cpu().numpy())
                    adata.var[f"{key}_gamma_{i}"] = np.exp(self.decoder.gamma[i, 0].detach().cpu().numpy())
                    adata.var[f"{key}_ton_{i}"] = np.exp(self.decoder.ton[i, 0].detach().cpu().numpy())
                    adata.var[f"{key}_logstd_alpha_c_{i}"] = np.exp(self.decoder.alpha_c[i, 1].detach().cpu().numpy())
                    adata.var[f"{key}_logstd_alpha_{i}"] = np.exp(self.decoder.alpha[i, 1].detach().cpu().numpy())
                    adata.var[f"{key}_logstd_beta_{i}"] = np.exp(self.decoder.beta[i, 1].detach().cpu().numpy())
                    adata.var[f"{key}_logstd_gamma_{i}"] = np.exp(self.decoder.gamma[i, 1].detach().cpu().numpy())
                    adata.var[f"{key}_logstd_ton_{i}"] = np.exp(self.decoder.ton[i, 1].detach().cpu().numpy())
            else:
                adata.var[f"{key}_alpha_c"] = np.exp(self.decoder.alpha_c[0].detach().cpu().numpy())
                adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha[0].detach().cpu().numpy())
                adata.var[f"{key}_beta"] = np.exp(self.decoder.beta[0].detach().cpu().numpy())
                adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma[0].detach().cpu().numpy())
                adata.var[f"{key}_ton"] = np.exp(self.decoder.ton[0].detach().cpu().numpy())
                adata.var[f"{key}_logstd_alpha_c"] = np.exp(self.decoder.alpha_c[1].detach().cpu().numpy())
                adata.var[f"{key}_logstd_alpha"] = np.exp(self.decoder.alpha[1].detach().cpu().numpy())
                adata.var[f"{key}_logstd_beta"] = np.exp(self.decoder.beta[1].detach().cpu().numpy())
                adata.var[f"{key}_logstd_gamma"] = np.exp(self.decoder.gamma[1].detach().cpu().numpy())
                adata.var[f"{key}_logstd_ton"] = np.exp(self.decoder.ton[1].detach().cpu().numpy())
        else:
            if self.enable_cvae:
                for i in range(self.n_batch):
                    adata.var[f"{key}_alpha_c_{i}"] = np.exp(self.decoder.alpha_c[i].detach().cpu().numpy())
                    adata.var[f"{key}_alpha_{i}"] = np.exp(self.decoder.alpha[i].detach().cpu().numpy())
                    adata.var[f"{key}_beta_{i}"] = np.exp(self.decoder.beta[i].detach().cpu().numpy())
                    adata.var[f"{key}_gamma_{i}"] = np.exp(self.decoder.gamma[i].detach().cpu().numpy())
                    adata.var[f"{key}_ton_{i}"] = np.exp(self.decoder.ton[i].detach().cpu().numpy())
            else:
                adata.var[f"{key}_alpha_c"] = np.exp(self.decoder.alpha_c.detach().cpu().numpy())
                adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
                adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
                adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
                adata.var[f"{key}_ton"] = np.exp(self.decoder.ton.detach().cpu().numpy())
        adata.var[f"{key}_scaling_c"] = np.exp(self.decoder.scaling_c.detach().cpu().numpy())
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_c"] = np.exp(self.decoder.sigma_c.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())

        C, U, S = adata_atac.layers['Mc'], adata.layers['Mu'], adata.layers['Ms']

        if self.enable_cvae and self.ref_batch >= 0:
            out, _ = self.pred_all(np.concatenate((C, U, S), 1), "both", gene_idx=np.array(range(adata.n_vars)), batch=np.full(adata.n_obs, self.ref_batch))
        else:
            out, _ = self.pred_all(np.concatenate((C, U, S), 1), "both", gene_idx=np.array(range(adata.n_vars)))
        chat, uhat, shat, t, std_t, t_, z, std_z = out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]

        adata.obs[f"{key}_time"] = t_
        adata.obs[f"{key}_t"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        adata.layers[f"{key}_chat"] = chat
        adata.layers[f"{key}_uhat"] = uhat
        adata.layers[f"{key}_shat"] = shat
        sigma_c, sigma_u, sigma_s = adata.var[f"{key}_sigma_c"].to_numpy(), adata.var[f"{key}_sigma_u"].to_numpy(), adata.var[f"{key}_sigma_s"].to_numpy()
        adata.var[f"{key}_likelihood"] = np.mean(-0.5*((C-chat)/sigma_c)**2-0.5*((U-uhat)/sigma_u)**2-0.5*((S-shat)/sigma_s)**2-np.log(sigma_c)-np.log(sigma_u)-np.log(sigma_s)-np.log(2*np.pi), 0)

        rho = np.zeros(U.shape)
        kc = np.zeros(U.shape)
        indicator_arch = self.config['indicator_arch']
        with torch.no_grad():
            B = min(U.shape[0]//10, 1000)
            Nb = U.shape[0] // B
            if Nb*B < U.shape[0]:
                Nb += 1
            for n in range(Nb):
                i = n*B
                j = min([(n+1)*B, U.shape[0]])
                if self.enable_cvae:
                    y_onehot = F.one_hot(torch.tensor(self.batch[i:j], dtype=int, device=self.device), self.n_batch).float()
                    rho_batch = torch.sigmoid(self.decoder.fc_out2(self.decoder.net_rho2(torch.cat((torch.tensor(z[i:j]).to(self.device), y_onehot), 1).float())))
                    rho[i:j] = rho_batch.cpu().numpy()
                    if indicator_arch == 'parallel':
                        kc_batch = torch.sigmoid(self.decoder.fc_out4(self.decoder.net_kc2(torch.cat((torch.tensor(z[i:j]).to(self.device), y_onehot), 1).float())))
                    else:
                        kc_batch = torch.sigmoid(self.decoder.fc_out4(self.decoder.net_kc2(torch.cat((torch.tensor(rho[i:j]).to(self.device), y_onehot), 1).float())))
                    kc[i:j] = kc_batch.cpu().numpy()
                else:
                    rho_batch = torch.sigmoid(self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[i:j]).float().to(self.device))))
                    rho[i:j] = rho_batch.cpu().numpy()
                    if indicator_arch == 'parallel':
                        kc_batch = torch.sigmoid(self.decoder.fc_out4(self.decoder.net_kc2(torch.tensor(z[i:j]).float().to(self.device))))
                    else:
                        kc_batch = torch.sigmoid(self.decoder.fc_out4(self.decoder.net_kc2(torch.tensor(rho[i:j]).float().to(self.device))))
                    kc[i:j] = kc_batch.cpu().numpy()

        adata.layers[f"{key}_rho"] = rho
        adata.layers[f"{key}_kc"] = kc

        adata.obs[f"{key}_t0"] = self.t0.detach().cpu().numpy().squeeze()
        adata.layers[f"{key}_c0"] = self.c0.detach().cpu().numpy()
        adata.layers[f"{key}_u0"] = self.u0.detach().cpu().numpy()
        adata.layers[f"{key}_s0"] = self.s0.detach().cpu().numpy()

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx

        rna_velocity_vae(adata, adata_atac, key, batch_key=self.config['batch_key'], use_raw=False, use_scv_genes=False, full_vb=self.config['mode'] == 'fullvb')

        if file_name is not None:
            adata.write_h5ad(f"{file_path}/{file_name}")
