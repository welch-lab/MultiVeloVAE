import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import time
#from velovae.plotting import plot_phase, plot_sig, plot_time, plot_train_loss, plot_test_loss
from velovae.plotting import plot_time, plot_train_loss, plot_test_loss #
from velovae.plotting_chrom import plot_sig, plot_phase #

#from .model_util import hist_equal, init_params, get_ts_global, reinit_params, convert_time, ode, get_gene_index
from .model_util import hist_equal, convert_time, get_gene_index #
from .model_util_chrom import ode, init_params, get_ts_global, reinit_params #
#from .TrainingData import SCData
from .TrainingDataChrom import SCData #
#from .velocity import rna_velocity_vanillavae
from .velocity_chrom import rna_velocity_vanillavae #


############################################################
#KL Divergence
############################################################
def kl_uniform(mu_t, std_t, t_start, t_end, **kwargs):
    #KL Divergence for the 1D near-uniform model
    #KL(q||p) where
    #q = uniform(t0, t0+dt)
    #p = uniform(t_start, t_end) with exponential decays on both sides
    
    tail = kwargs["tail"] if "tail" in kwargs else 0.05
    t0 = mu_t - np.sqrt(3)*std_t
    dt = np.sqrt(12)*std_t
    C = 1/((t_end-t_start)*(1+tail))
    lamb = 2/(tail*(t_end-t_start))
    
    t1 = t0+dt
    dt1_til = nn.functional.relu(torch.minimum(t_start, t1) - t0)
    dt2_til = nn.functional.relu(t1 - torch.maximum(t_end, t0))
    
    term1 = -lamb*(dt1_til.pow(2)+dt2_til.pow(2))/(2*dt)
    term2 = lamb*((t_start-t0)*dt1_til+(t1-t_end)*dt2_til)/dt
    
    return torch.mean(term1 + term2 - torch.log(C*dt))

def kl_gaussian(mu1, std1, mu2, std2, **kwargs):
    #Compute the KL divergence between two Gaussian distributions with diagonal covariance
    
    return torch.mean(torch.sum(torch.log(std2/std1)+std1.pow(2)/(2*std2.pow(2))-0.5+(mu1-mu2).pow(2)/(2*std2.pow(2)),1))


##############################################################
# Vanilla VAE
##############################################################
class encoder(nn.Module):
    """Encoder of the vanilla VAE
    """
    def __init__(self, Cin, N1=500, N2=250, device=torch.device('cpu'), checkpoint=None):
        super(encoder, self).__init__()
        self.fc1 = nn.Linear(Cin, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)
        
        self.net = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                 self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)
        
        self.fc_mu, self.spt1 = nn.Linear(N2,1).to(device), nn.Softplus()
        self.fc_std, self.spt2 = nn.Linear(N2,1).to(device), nn.Softplus()
        
        if(checkpoint is not None):
            self.load_state_dict(torch.load(checkpoint,map_location=device))
        else:
            self.init_weights()

    def init_weights(self):
        for m in self.net.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in [self.fc_mu, self.fc_std]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, data_in):
        z = self.net(data_in)
        mu_zx, std_zx = self.spt1(self.fc_mu(z)), self.spt2(self.fc_std(z))
        return mu_zx, std_zx

class decoder(nn.Module):
    def __init__(self, 
                 adata, 
                 adata_atac, #
                 tmax, 
                 train_idx, 
                 p=98, 
                 device=torch.device('cpu'), 
                 init_method="steady", 
                 init_key=None):
        super(decoder,self).__init__()
        C = adata_atac.layers['Mc'][train_idx]
        U,S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
        X = np.concatenate((U,S),1)
        N,G = U.shape
        #Dynamical Model Parameters
        if(init_method == "random"):
            print("Random Initialization.")
            #alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
            alpha_c, alpha, beta, gamma, scaling_c, scaling, toff, c0, u0, s0, sigma_c, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True) #
            self.alpha_c = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float()) #
            self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.beta =  nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.ton = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.toff = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float()+self.ton.detach())
            self.scaling_c = nn.Parameter(torch.tensor(np.log(scaling_c), device=device).float()) #
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            self.sigma_c = nn.Parameter(torch.tensor(np.log(sigma_c), device=device).float()) #
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
        elif(init_method == "tprior"):
            print("Initialization using prior time.")
            #alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
            alpha_c, alpha, beta, gamma, scaling_c, scaling, toff, c0, u0, s0, sigma_c, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True) #
            t_prior = adata.obs[init_key].to_numpy()
            t_prior = t_prior[train_idx]
            std_t = (np.std(t_prior)+1e-3)*0.2
            self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
            self.t_init -= self.t_init.min()
            self.t_init = self.t_init
            self.t_init = self.t_init/self.t_init.max()*tmax
            #toff = get_ts_global(self.t_init, U/scaling, S, 95)
            toff = get_ts_global(self.t_init, C/scaling_c, U/scaling, S, 95)
            #alpha, beta, gamma, ton = reinit_params(U/scaling, S, self.t_init, toff)
            alpha_c, alpha, beta, gamma, ton = reinit_params(C/scaling_c, U/scaling, S, self.t_init, toff) #
            
            self.alpha_c = nn.Parameter(torch.tensor(np.log(alpha_c), device=device).float()) #
            self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
            self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
            self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
            self.scaling_c = nn.Parameter(torch.tensor(np.log(scaling_c), device=device).float()) #
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
            self.sigma_c = nn.Parameter(torch.tensor(np.log(sigma_c), device=device).float()) #
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
        else:
            print("Initialization using the steady-state and dynamical models.")
            #alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
            alpha_c, alpha, beta, gamma, scaling_c, scaling, toff, c0, u0, s0, sigma_c, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True) #
            if(init_key is not None):
                t_init = adata.obs['init_key'].to_numpy()
            else:
                T = T+np.random.rand(T.shape[0],T.shape[1]) * 1e-3
                T_eq = np.zeros(T.shape)
                Nbin = T.shape[0]//50+1
                for i in range(T.shape[1]):
                    T_eq[:, i] = hist_equal(T[:, i], tmax, 0.9, Nbin)
                self.t_init = np.quantile(T_eq,0.5,1)
            #toff = get_ts_global(self.t_init, U/scaling, S, 95)
            toff = get_ts_global(self.t_init, C/scaling_c, U/scaling, S, 95) #
            #alpha, beta, gamma,ton = reinit_params(U/scaling, S, self.t_init, toff)
            alpha_c, alpha, beta, gamma, ton = reinit_params(C/scaling_c, U/scaling, S, self.t_init, toff) #
            
            self.alpha_c = nn.Parameter(torch.tensor(np.log(alpha_c), device=device).float()) #
            self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
            self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
            self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
            self.scaling_c = nn.Parameter(torch.tensor(np.log(scaling_c), device=device).float()) #
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
            self.sigma_c = nn.Parameter(torch.tensor(np.log(sigma_c), device=device).float()) #
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())

        self.scaling_c.requires_grad = False #
        self.scaling.requires_grad = False
        self.sigma_c.requires_grad = False #
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
    
    def forward(self, t, neg_slope=0.0):
        #Uhat, Shat = ode(t, torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gamma), torch.exp(self.ton), torch.exp(self.toff), neg_slope=neg_slope)
        Chat, Uhat, Shat = ode(t, torch.exp(self.alpha_c), torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gamma), torch.exp(self.ton), torch.exp(self.toff), neg_slope=neg_slope) #
        Chat = Chat * torch.exp(self.scaling_c) #
        Uhat = Uhat * torch.exp(self.scaling)
        #return nn.functional.relu(Uhat), nn.functional.relu(Shat)
        return nn.functional.relu(Chat), nn.functional.relu(Uhat), nn.functional.relu(Shat) #
    
    #def pred_su(self, t, gidx=None):
    def pred_exp(self, t, gidx=None):
        scaling_c = torch.exp(self.scaling_c) #
        scaling = torch.exp(self.scaling)
        if(gidx is not None):
            #Uhat, Shat = ode(t, torch.exp(self.alpha[gidx]), torch.exp(self.beta[gidx]), torch.exp(self.gamma[gidx]), torch.exp(self.ton[gidx]), torch.exp(self.toff[gidx]), neg_slope=0.0)
            Chat, Uhat, Shat = ode(t, torch.exp(self.alpha_c[gidx]), torch.exp(self.alpha[gidx]), torch.exp(self.beta[gidx]), torch.exp(self.gamma[gidx]), torch.exp(self.ton[gidx]), torch.exp(self.toff[gidx]), neg_slope=0.0)
            return nn.functional.relu(Chat*scaling_c[gidx]), nn.functional.relu(Uhat*scaling[gidx]), nn.functional.relu(Shat)
        #Uhat, Shat = ode(t, torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gamma), torch.exp(self.ton), torch.exp(self.toff), neg_slope=0.0)
        Chat, Uhat, Shat = ode(t, torch.exp(self.alpha_c), torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gamma), torch.exp(self.ton), torch.exp(self.toff), neg_slope=0.0)
        #return nn.functional.relu(Uhat*scaling), nn.functional.relu(Shat)
        return nn.functional.relu(Chat*scaling), nn.functional.relu(Uhat*scaling), nn.functional.relu(Shat)

class VanillaVAEChrom():
    def __init__(self, 
                 adata, 
                 adata_atac, #
                 tmax, 
                 device='cpu', 
                 hidden_size=(500, 250), 
                 init_method="steady",
                 init_key=None,
                 tprior=None, 
                 time_distribution="gaussian",
                 checkpoints=None):
        """VeloVAE with latent time only
        
        Arguments 
        ---------
        adata : :class:`anndata.AnnData`
        tmax : float
            Time Range 
        device : {'cpu','gpu'}, optional
        hidden_size : tuple, optional
            Width of the first and second hidden layer
        init_type : str, optional
            The stem cell type. Used to estimated the initial conditions.
            This is not commonly used in practice and please consider leaving it to default.
        init_key : str, optional
            column in the AnnData object containing the capture time
        tprior : str, optional
            key in adata.obs that stores the capture time.
            Used for informative time prior
        time_distribution : str, optional
            Should be either "gaussian" or "uniform.
        checkpoints : string list
            Contains the path to saved encoder and decoder models. Should be a .pt file.
        """
        #Extract Input Data
        try:
            C = adata_atac.layers['Mc'],
            U,S = adata.layers['Mu'], adata.layers['Ms']
        except KeyError:
            print('Unspliced/Spliced count matrices not found in the layers! Exit the program...')
            print('Chromatin/Unspliced/Spliced count matrices not found in the layers! Exit the program...')
        
        #Default Training Configuration
        self.config = {
            #Model Parameters
            "tmax":tmax,
            "hidden_size":hidden_size,
            "init_method": init_method,
            "init_key": init_key,
            "tprior":tprior,
            "tail":0.01,
            "time_overlap":0.5,

            #Training Parameters
            "n_epochs":2000, 
            "batch_size":128,
            "learning_rate":2e-4, 
            "learning_rate_ode":5e-4, 
            "lambda":1e-3, 
            "kl_t":1.0, 
            "test_iter":None, 
            "save_epoch":100,
            "n_warmup":5,
            "early_stop":5,
            "early_stop_thred":1e-3*adata.n_vars,
            "train_test_split":0.7,
            "k_alt":1,
            "neg_slope":0.0,
            "train_scaling":False, 
            "train_std":False, 
            "weight_sample":False,
            
            #Plotting
            "sparsify":1
        }
        
        self.set_device(device)
        self.split_train_test(adata.n_obs)
        
        G = adata.n_vars
        #Create an encoder
        try:
            #self.encoder = encoder(2*G, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints).float()
            self.encoder = encoder(3*G, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints).float()
        except IndexError:
            print('Please provide two dimensions!')
        #Create a decoder
        self.decoder = decoder(adata, 
                               adata_atac, #
                               tmax, 
                               self.train_idx, 
                               device=self.device, 
                               init_method = init_method,
                               init_key = init_key).float()
        self.tmax=torch.tensor(tmax).to(self.device)
        self.time_distribution = time_distribution
        #Time prior
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        #class attributes for training
        self.loss_train, self.loss_test = [], []
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive epochs with negative/low ELBO gain
        
    #def get_prior(self, adata, time_distribution, tmax, tprior=None):
    def get_prior(self, adata, adata_atac, time_distribution, tmax, tprior=None):
        """Compute the parameters of time prior distribution
        
        Arguments
        ---------
        adata : :class:`anndata.AnnData`
        time_distribution : {'gaussian', 'uniform'}
        tmax : float
            Maximum time
        tprior : str, optional
            Key in adata.obs storing the capture time
        """
        if(time_distribution=="gaussian"):
            print("Gaussian Prior.")
            self.kl_time = kl_gaussian
            self.sample = self.reparameterize
            if(tprior is None):
                self.p_t = torch.stack([torch.ones(adata.n_obs,1)*tmax*0.5,torch.ones(adata.n_obs,1)*tmax*self.config["time_overlap"]]).float().to(self.device)
            else:
                print('Using informative time prior.')
                t = adata.obs[tprior].to_numpy()
                t = t/t.max()*tmax
                t_cap = np.sort(np.unique(t))
                
                std_t = np.zeros((len(t)))
                std_t[t==t_cap[0]] = (t_cap[1] - t_cap[0])*(0.5+0.5*self.config["time_overlap"])
                for i in range(1, len(t_cap)-1):
                    std_t[t==t_cap[i]] = 0.5*(t_cap[i] - t_cap[i-1])*(0.5+0.5*self.config["time_overlap"]) + 0.5*(t_cap[i+1] - t_cap[i])*(0.5+0.5*self.config["time_overlap"])
                std_t[t==t_cap[-1]] = (t_cap[-1] - t_cap[-2])*(0.5+0.5*self.config["time_overlap"])
                
                self.p_t = torch.stack( [torch.tensor(t).view(-1,1),torch.tensor(std_t).view(-1,1)] ).float().to(self.device)
        else:
            print("Tailed Uniform Prior.")
            self.kl_time = kl_uniform
            self.sample = self.reparameterize_uniform
            if(tprior is None):
                self.p_t = torch.stack([torch.zeros(adata.n_obs,1),torch.ones(adata.n_obs,1)*tmax]).float().to(self.device)
            else:
                print('Using informative time prior.')
                t = adata.obs[tprior].to_numpy()
                t = t/t.max()*tmax
                t_cap = np.sort(np.unique(t))
                t_start = np.zeros((len(t)))
                t_end = np.zeros((len(t)))
                for i in range(len(t_cap)-1):
                    t_end[t==t_cap[i]] = t_cap[i] + (t_cap[i+1] - t_cap[i])*(0.5+0.5*self.config["time_overlap"])
                t_end[t==t_cap[-1]] = t_cap[-1] + (t_cap[-1] - t_cap[-2])*(0.5+0.5*self.config["time_overlap"])
                
                for i in range(1, len(t_cap)):
                    t_start[t==t_cap[i]] = max(0, t_cap[i] - (t_cap[i] - t_cap[i-1])*(0.5+0.5*self.config["time_overlap"]))
                t_start[t==t_cap[0]] = max(0, t_cap[0] - (t_cap[1] - t_cap[0])*(0.5+0.5*self.config["time_overlap"]))
                
                self.p_t = torch.stack( [torch.tensor(t).unsqueeze(-1),torch.tensor(t_end).unsqueeze(-1)] ).float().to(self.device)
    
    def set_device(self, device):
        """Set the device of the model.
        """
        if('cuda' in device):
            if(torch.cuda.is_available()):
                self.device = torch.device(device)
            else:
                print('Warning: GPU not detected. Using CPU as the device.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
    
    def reparameterize(self, mu, std):
        #Apply the reparameterization trick for Gaussian random variables.
        
        eps = torch.normal(mean=torch.zeros(mu.shape),std=torch.ones(mu.shape)).to(self.device)
        return std*eps+mu
    
    def reparameterize_uniform(self, mu, std):
        #Apply the reparameterization trick for uniform random variables.
        
        eps = torch.rand(mu.shape).to(self.device)
        return np.sqrt(12)*std*eps + (mu - np.sqrt(3)*std)
    
    def forward(self, data_in):
        #data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//3]/torch.exp(self.decoder.scaling_c), data_in[:,data_in.shape[1]//3:data_in.shape[1]//3*2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//3*2:]),1)
        mu_t, std_t = self.encoder.forward(data_in_scale)
        t_global = self.reparameterize(mu_t, std_t)
         
        #uhat, shat = self.decoder.forward(t_global, neg_slope=self.config["neg_slope"]) #uhat is scaled
        chat, uhat, shat = self.decoder.forward(t_global, neg_slope=self.config["neg_slope"]) #uhat is scaled
        #return mu_t, std_t, t_global, uhat, shat
        return mu_t, std_t, t_global, chat, uhat, shat
    
    def eval_model(self, data_in):
        #data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//3]/torch.exp(self.decoder.scaling_c), data_in[:,data_in.shape[1]//3:data_in.shape[1]//3*2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//3*2:]),1)
        mu_t, std_t = self.encoder.forward(data_in_scale)
        
        #uhat, shat = self.decoder.pred_su(mu_t) #uhat is scaled
        chat, uhat, shat = self.decoder.pred_exp(mu_t) #uhat is scaled
        #return mu_t, std_t, uhat, shat
        return mu_t, std_t, chat, uhat, shat
        
    def set_mode(self,mode):
        #Set the model to either training or evaluation mode.
        
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")
    
    ############################################################
    #Training Objective
    ############################################################
    #def vae_risk(self, q_tx, p_t, u, s, uhat, shat, sigma_u, sigma_s, weight=None, b=1.0):
    def vae_risk(self, q_tx, p_t, c, u, s, chat, uhat, shat, sigma_c, sigma_u, sigma_s, weight=None, b=1.0):
        #This is the negative ELBO.
        
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        
        #u and sigma_u has the original scale
        #logp = -0.5*((u-uhat)/sigma_u).pow(2)-0.5*((s-shat)/sigma_s).pow(2)-torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
        logp = -0.5*((c-chat)/sigma_c).pow(2)-0.5*((u-uhat)/sigma_u).pow(2)-0.5*((s-shat)/sigma_s).pow(2)-torch.log(sigma_c)-torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
        
        if( weight is not None):
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp,1))
        
        return (- err_rec + b*(kldt))
        
    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1):
        ##########################################################################
        #Training in each epoch.
        #Early stopping if enforced by default. 
        #< Input Arguments >
        #1.  train_loader [torch.utils.data.DataLoader]
        #    Data loader of the input data.
        #2.  test_set [torch.utils.data.Dataset]
        #    Validation dataset
        #3.  optimizer  [optimizer from torch.optim]
        #4.  optimizer2 [optimizer from torch.optim]
        #    (Optional) A second optimizer.
        #    This is used when we optimize NN and ODE simultaneously in one epoch.
        #    By default, VeloVAE performs alternating optimization in each epoch.
        #    The argument will be set to proper value automatically.
        #5.  K [int]
        #    Alternating update period.
        #    For every K updates of optimizer, there's one update for optimizer2.
        #    If set to 0, optimizer2 will be ignored and only optimizer will be
        #    updated. Users can set it to 0 if they want to update sorely NN in one 
        #    epoch and ODE in the next epoch. 
        #< Output >
        #1.  stop_training [bool]
        #    Whether to stop training based on the early stopping criterium.
        ##########################################################################
        B = len(train_loader)
        self.set_mode('train')
        stop_training = False
        
        for i, batch in enumerate(train_loader):
            if( self.counter==1 or self.counter % self.config["test_iter"] == 0):
                elbo_test = self.test(test_set, None, self.counter)
                if(len(self.loss_test)>0):
                    if(elbo_test - self.loss_test[-1] <= self.config["early_stop_thred"]):
                        self.n_drop = self.n_drop+1
                    else:
                        self.n_drop = 0
                self.loss_test.append(elbo_test)
                self.set_mode('train')
                if(self.n_drop>=self.config["early_stop"] and self.config["early_stop"]>0):
                    stop_training=True
                    break
            
            optimizer.zero_grad()
            if(optimizer2 is not None):
                optimizer2.zero_grad()
            
            xbatch, weight, idx = batch[0].float().to(self.device), batch[2].float().to(self.device), batch[3]
            #u = xbatch[:,:xbatch.shape[1]//2]
            #s = xbatch[:,xbatch.shape[1]//2:]
            c = xbatch[:,:xbatch.shape[1]//3]
            u = xbatch[:,xbatch.shape[1]//3:xbatch.shape[1]//3*2]
            s = xbatch[:,xbatch.shape[1]//3*2:]
            mu_tx, std_tx, t_global, chat, uhat, shat = self.forward(xbatch)
            
            loss = self.vae_risk((mu_tx,std_tx), 
                                 self.p_t[:,self.train_idx[idx],:], 
                                 c, 
                                 u, s, 
                                 chat, 
                                 uhat, shat, 
                                 torch.exp(self.decoder.sigma_c), 
                                 torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                 None,
                                 self.config["kl_t"])
            
            loss.backward()
            if(K==0):
                optimizer.step()
                if( optimizer2 is not None ):
                    optimizer2.step()
            else:
                if( optimizer2 is not None and ((i+1) % (K+1) == 0 or i==B-1)):
                    optimizer2.step()
                else:
                    optimizer.step()
            
            self.loss_train.append(loss.detach().cpu().item())
            self.counter = self.counter + 1
        return stop_training
    
    def load_config(self, config):
        #Update hyper-parameters
        #We don't have to specify all the hyperparameters. Just pass the ones we want to modify.
        
        for key in config:
            if(key in self.config):
                self.config[key] = config[key]
            else:
                self.config[key] = config[key]
                print(f"Warning: unknown hyperparameter: {key}")
        if(self.config["train_scaling"]):
            self.decoder.scaling_c.requires_grad = True #
            self.decoder.scaling.requires_grad = True
        if(self.config["train_std"]):
            self.decoder.sigma_c.requires_grad = True #
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True
    
    def split_train_test(self, N):
        #Randomly select indices as training samples.
        
        rand_perm = np.random.permutation(N)
        n_train = int(N*self.config["train_test_split"])
        self.train_idx = rand_perm[:n_train]
        self.test_idx = rand_perm[n_train:]
        
        return
    
    def train(self, 
              adata, 
              adata_atac, #
              config={}, 
              plot=False, 
              gene_plot=[], 
              cluster_key="clusters",
              figure_path="figures", 
              embed="umap"):
        """The high-level API for training.
        
        Arguments
        ---------
        adata : :class:`anndata.AnnData`
            AnnData Object
        config : dictionary, optional
            Contains all hyper-parameters.
        plot : bool, optional
            Whether to plot some sample genes during training. Used for debugging.
        gene_plot : string list, optional
            List of gene names to plot. Used only if plot==True
        cluster_key : str, optional
            Key in adata.obs storing the cell type annotation
        figure_path : str, optional
            Path to the folder for saving plots
        embed : str, optional
            Low dimensional embedding in adata.obsm. The actual key storing the embedding should be f'X_{embed}'
        """
        self.load_config(config)
        
        print("------------------------- Train a Vanilla VAE -------------------------")
        #Get data loader
        C = adata_atac.layers['Mu']
        U,S = adata.layers['Mu'], adata.layers['Ms']
        #X = np.concatenate((U,S), 1)
        X = np.concatenate((C,U,S), 1)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Please run the corresponding preprocessing step!")
        
        cell_labels_raw = adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs else np.array(['Unknown' for i in range(adata.n_obs)])
        
        print("*********        Creating Training/Validation Datasets        *********")
        train_set = SCData(X[self.train_idx], cell_labels_raw[self.train_idx], weight=self.decoder.Rscore[self.train_idx]) if self.config['weight_sample'] else SCData(X[self.train_idx], cell_labels_raw[self.train_idx])
        test_set = None
        if(len(self.test_idx)>0):
            test_set = SCData(X[self.test_idx], cell_labels_raw[self.test_idx], weight=self.decoder.Rscore[self.test_idx]) if self.config['weight_sample'] else SCData(X[self.test_idx], cell_labels_raw[self.test_idx])
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
        #Automatically set test iteration if not given
        if(self.config["test_iter"] is None):
            self.config["test_iter"] = len(self.train_idx)//self.config["batch_size"]*2
        print("*********                      Finished.                      *********")
        
        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        
        os.makedirs(figure_path, exist_ok=True)
        
        #define optimizer
        print("*********                 Creating optimizers                 *********")
        param_nn = list(self.encoder.parameters())
        #param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma, self.decoder.ton, self.decoder.toff] 
        param_ode = [self.decoder.alpha_c, self.decoder.alpha, self.decoder.beta, self.decoder.gamma, self.decoder.ton, self.decoder.toff] #
        if(self.config['train_scaling']):
            #param_ode = param_ode+[self.decoder.scaling]
            param_ode += [self.decoder.scaling_c, self.decoder.scaling]
        if(self.config['train_std']):
            #param_ode = param_ode+[self.decoder.sigma_u, self.decoder.sigma_s]
            param_ode += [self.decoder.sigma_c, self.decoder.sigma_u, self.decoder.sigma_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")
      
        #Main Training Process
        print("*********                    Start training                   *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")
        
        n_epochs, n_save = self.config["n_epochs"], self.config["save_epoch"]
        n_warmup = self.config["n_warmup"]
        
        start = time.time()
        for epoch in range(n_epochs):
            #Train the encoder
            if(self.config["k_alt"] is None):
                stop_training = self.train_epoch(data_loader, test_set, optimizer)
                if(epoch>=n_warmup):
                    stop_training_ode = self.train_epoch(data_loader, test_set, optimizer_ode)
                    if(stop_training_ode):
                        print(f"********* Early Stop Triggered at epoch {epoch+1}. *********")
                        break
            else:
                if(epoch>=n_warmup):
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_ode, optimizer, self.config["k_alt"])
                else:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer, None, self.config["k_alt"])
            
            if(plot and (epoch==0 or (epoch+1) % self.config["save_epoch"] == 0)):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+1}", 
                                       False,
                                       gind, 
                                       gene_plot,
                                       True, 
                                       figure_path)
                self.set_mode('train')
                elbo_test = self.loss_test[-1] if len(self.loss_test)>0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")
                
            
            
            if(stop_training):
                print(f"********* Early Stop Triggered at epoch {epoch+1}. *********")
                break
                
                
        print(f"*********              Finished. Total Time = {convert_time(time.time()-start)}             *********")
        if(plot):
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
            plot_train_loss(self.loss_train, range(1,len(self.loss_train)+1), save=f'{figure_path}/train_loss_vanilla.png')
            if(self.config["test_iter"]>0):
                plot_test_loss(self.loss_test, [i*self.config["test_iter"] for i in range(1,len(self.loss_test)+1)], save=f'{figure_path}/test_loss_vanilla.png')
        return
    
    #def pred_all(self, data, mode='test', output=["uhat", "shat", "t"], gene_idx=None):
    def pred_all(self, data, mode='test', output=["chat", "uhat", "shat", "t"], gene_idx=None): #
        #N, G = data.shape[0], data.shape[1]//2
        N, G = data.shape[0], data.shape[1]//3 #
        if("chat" in output): #
            Chat = None if gene_idx is None else np.zeros((N,len(gene_idx))) #
        if("uhat" in output):
            Uhat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
        if("shat" in output):
            Shat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
        if("t" in output):
            t_out = np.zeros((N))
            std_t_out = np.zeros((N))
        elbo = 0
        with torch.no_grad():
            B = min(N//10, 1000)
            Nb = N // B
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
                #mu_tx, std_tx, uhat, shat = self.eval_model(data_in)
                mu_tx, std_tx, chat, uhat, shat = self.eval_model(data_in) #
                if(mode=="test"):
                    p_t = self.p_t[:,self.test_idx[i*B:(i+1)*B],:]
                elif(mode=="train"):
                    p_t = self.p_t[:,self.train_idx[i*B:(i+1)*B],:]
                else:
                    p_t = self.p_t[:,i*B:(i+1)*B,:]
                loss = self.vae_risk((mu_tx, std_tx), 
                                     p_t,
                                     #data_in[:,:G], data_in[:,G:], 
                                     data_in[:,:G], data_in[:,G:G*2], data_in[:,G*2:], #
                                     chat, #
                                     uhat, shat, 
                                     torch.exp(self.decoder.sigma_c), #
                                     torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                     None,
                                     1.0)
                elbo = elbo-loss*B
                if("chat" in output and gene_idx is not None):
                    Chat[i*B:(i+1)*B] = chat[:,gene_idx].cpu().numpy()
                if("uhat" in output and gene_idx is not None):
                    Uhat[i*B:(i+1)*B] = uhat[:,gene_idx].cpu().numpy()
                if("shat" in output and gene_idx is not None):
                    Shat[i*B:(i+1)*B] = shat[:,gene_idx].cpu().numpy()
                if("t" in output):
                    t_out[i*B:(i+1)*B] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[i*B:(i+1)*B] = std_tx.cpu().squeeze().numpy()
            if(N > B*Nb):
                data_in = torch.tensor(data[B*Nb:]).float().to(self.device)
                #mu_tx, std_tx, uhat, shat = self.eval_model(data_in)
                mu_tx, std_tx, chat, uhat, shat = self.eval_model(data_in) #
                if(mode=="test"):
                    p_t = self.p_t[:,self.test_idx[B*Nb:],:]
                elif(mode=="train"):
                    p_t = self.p_t[:,self.train_idx[B*Nb:],:]
                else:
                    p_t = self.p_t[:,B*Nb:,:]
                loss = self.vae_risk((mu_tx, std_tx), 
                                     p_t,
                                     #data_in[:,:G], data_in[:,G:], 
                                     data_in[:,:G], data_in[:,G:G*2], data_in[:,G*2:], #
                                     chat, #
                                     uhat, shat, 
                                     torch.exp(self.decoder.sigma_c), #
                                     torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                     None,
                                     1.0)
                elbo = elbo-loss*(N-B*Nb)
                if("chat" in output and gene_idx is not None): #
                    Chat[Nb*B:] = chat[:,gene_idx].cpu().numpy() #
                if("uhat" in output and gene_idx is not None):
                    Uhat[Nb*B:] = uhat[:,gene_idx].cpu().numpy()
                if("shat" in output and gene_idx is not None):
                    Shat[Nb*B:] = shat[:,gene_idx].cpu().numpy()
                if("t" in output):
                    t_out[Nb*B:] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[Nb*B:] = std_tx.cpu().squeeze().numpy()
        out = []
        if("chat" in output): #
            out.append(Chat) #
        if("uhat" in output):
            out.append(Uhat)
        if("shat" in output):
            out.append(Shat)
        if("t" in output):
            out.append(t_out)
            out.append(std_t_out)
        return out, elbo.cpu().item()/N
    
    def test(self,
             test_set, 
             Xembed,
             testid=0, 
             test_mode=True,
             gind=None, 
             gene_plot=None,
             plot=False, 
             path='figures', 
             **kwargs):
        """Evaluate the model upon training/test dataset.
        
        Arguments
        ---------
        test_set : `torch.utils.data.Dataset`
            Training or validation dataset
        Xembed : `numpy array`
            Low-dimensional embedding for plotting
        testid : string or int, optional
            Used to name the figures
        gind : `numpy array`
            Index of genes in adata.var_names. Used for plotting.
        gene_plot : `numpy array`, optional
            Gene names.
        plot : bool, optional
            Whether to generate plots.
        path : str, optional
            Saving path.
        
        Returns
        -------
        elbo : float
        """
        self.set_mode('eval')
        data = test_set.data
        mode = "test" if test_mode else "train"
        out, elbo = self.pred_all(data, mode, gene_idx=gind)
        #Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3]
        Chat, Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3], out[4]
        
        #G = data.shape[1]//2
        G = data.shape[1]//3
        if(plot):
            ton, toff = np.exp(self.decoder.ton.detach().cpu().numpy()), np.exp(self.decoder.toff.detach().cpu().numpy())
            state = np.ones(toff.shape)*(t.reshape(-1,1)>toff)+np.ones(ton.shape)*2*(t.reshape(-1,1)<ton)
            #Plot Time
            plot_time(t, Xembed, save=f"{path}/time-{testid}-vanilla.png")
            
            #Plot u/s-t and phase portrait for each gene
            for i in range(len(gind)):
                idx = gind[i]
                
                plot_phase(data[:,idx], data[:,idx+G], data[:,idx+G*2], 
                           Chat[:,i], Uhat[:,i], Shat[:,i], 
                           gene_plot[i], 
                           'us', #
                           None, 
                           state[:,idx], 
                           ['Induction', 'Repression', 'Off'],
                           save=f"{path}/phase-{gene_plot[i]}-{testid}-vanilla.png")
                
                plot_sig(t.squeeze(), 
                         #data[:,idx], data[:,idx+G], 
                         data[:,idx], data[:,idx+G], data[:,idx+G*2], #
                         Chat[:,i], 
                         Uhat[:,i], Shat[:,i], 
                         test_set.labels,
                         gene_plot[i], 
                         save=f"{path}/sig-{gene_plot[i]}-{testid}-vanilla.png",
                         sparsify=self.config["sparsify"])
        
        return elbo
        
        
    def save_model(self, file_path, enc_name='encoder_vanilla', dec_name='decoder_vanilla'):
        """Save the encoder parameters to a .pt file.
        
        Arguments
        ---------
        file_path : str
            Path to the folder for saving model parameters
        enc_name : str, optional
            Name of the .pt file containing encoder parameters
        dec_name : str, optional
            Name of the .pt file containing decoder parameters
        """
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), f"{file_path}/{enc_name}.pt")
        torch.save(self.decoder.state_dict(), f"{file_path}/{dec_name}.pt")
        
    #def save_anndata(self, adata, key, file_path, file_name=None):
    def save_anndata(self, adata, adata_atac, key, file_path, file_name=None):
        """Save the ODE parameters and cell time to the anndata object and write it to disk.
        
        Arguments
        ---------
        adata : :class:`anndata.AnnData`
        key : str
            Used to store all parameters of the model.
        file_path : str
            Saving path.
        file_name : str, optional
            If set to a string ending with .h5ad, the updated anndata object will be written to disk.
        """
        os.makedirs(file_path, exist_ok=True)
        
        self.set_mode('eval')
        adata.var[f"{key}_alpha_c"] = np.exp(self.decoder.alpha.detach().cpu().numpy()) #
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_toff"] = np.exp(self.decoder.toff.detach().cpu().numpy())
        adata.var[f"{key}_ton"] = np.exp(self.decoder.ton.detach().cpu().numpy())
        adata.var[f"{key}_scaling_c"] = np.exp(self.decoder.scaling.detach().cpu().numpy()) #
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_c"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy()) #
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        #scaling = adata.var[f"{key}_scaling"].to_numpy()
        
        #out, elbo = self.pred_all(np.concatenate((adata.layers['Mu'], adata.layers['Ms']),axis=1), mode="both", gene_idx=np.array(range(adata.n_vars)))
        out, elbo = self.pred_all(np.concatenate((adata_atac.layers['Mc'], adata.layers['Mu'], adata.layers['Ms']),axis=1), mode="both", gene_idx=np.array(range(adata.n_vars))) #
        Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3]
        Chat, Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3], out[4]
        
        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.layers[f"{key}_chat"] = Chat
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat
        
        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        #rna_velocity_vanillavae(adata, key)
        rna_velocity_vanillavae(adata, adata_atac, key) #
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
