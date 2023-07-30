import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import igraph as ig
import pynndescent
import umap
from sklearn.neighbors import NearestNeighbors
from loess import loess_1d


#######################################################################################
#Default colors and markers for plotting
#######################################################################################
TAB10 = list(plt.get_cmap("tab10").colors)
TAB20 = list(plt.get_cmap("tab20").colors)
TAB20B = list(plt.get_cmap("tab20b").colors)
TAB20C = list(plt.get_cmap("tab20c").colors)
RAINBOW = [plt.cm.rainbow(i) for i in range(256)]

markers = ["o","x","s","v","+","d","1","*","^","p","h","8","1","2","|"]

def get_colors(n, color_map=None):
    """Get colors for plotting cell clusters.
    
    Arguments
    ---------
    
    n : int
        Number of cell clusters
    color_map : str, optional
        User-defined colormap. If not set, the colors will be chosen as the colors for tabular data in matplotlib.
    """
    if(color_map is None): #default color based on 
        if(n<=10):
            return TAB10[:n]
        elif(n<=20):
            return TAB20[:n]
        elif(n<=40):
            TAB40 = TAB20B+TAB20C
            return TAB40[:n]
        else:
            print("Warning: Number of colors exceeds the maximum (40)! Use a continuous colormap (256) instead.")
            return RAINBOW[:n]
    else:
        color_map_obj = list(plt.get_cmap(color_map).colors)
        k = len(color_map_obj)//n
        colors = [color_map_obj(i) for i in range(0,len(color_map_obj),k)] if k>0 else [color_map_obj(i) for i in range(len(color_map_obj))]
    return colors

def save_fig(fig, save, bbox_extra_artists=None):
    if(save is not None):
        try:
            idx = save.find('.')
            fig.savefig(save,bbox_extra_artists=bbox_extra_artists, format=save[idx+1:], bbox_inches='tight')
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
        plt.close(fig)

############################################################
# Functions used in debugging.
############################################################
def plot_sig_(t, 
              c, u, s, 
              cell_labels, 
              tpred=None, 
              cpred=None, upred=None, spred=None, 
              by='us', 
              type_specific=False, 
              title='Gene', 
              save=None, 
              **kwargs):
    fig, ax = plt.subplots(2,1,figsize=(15,12),facecolor='white')
    D = kwargs['sparsify'] if('sparsify' in kwargs) else 1
    cell_types = np.unique(cell_labels)
    colors = get_colors(len(cell_types), None)
    mod1 = u if by == 'us' else c
    mod2 = s if by == 'us' else u
    mod_pred1 = upred if by == 'us' else cpred
    mod_pred2 = spred if by == 'us' else upred
    
    for i, type_ in enumerate(cell_types):
        mask_type = cell_labels==type_
        ax[0].plot(t[mask_type][::D], mod1[mask_type][::D],'.',color=colors[i%len(colors)], alpha=0.7, label=type_)
        ax[1].plot(t[mask_type][::D], mod2[mask_type][::D],'.',color=colors[i%len(colors)], alpha=0.7, label=type_)
    
    if((tpred is not None) and (mod_pred1 is not None) and (mod_pred2 is not None)):
        if(type_specific):
            for i, type_ in enumerate(cell_types):
                mask_type = cell_labels==type_
                order = np.argsort(tpred[mask_type])
                ax[0].plot(tpred[mask_type][order], mod_pred1[mask_type][order], '-', color=colors[i%len(colors)], label=type_, linewidth=1.5)
                ax[1].plot(tpred[mask_type][order], mod_pred2[mask_type][order], '-', color=colors[i%len(colors)], label=type_, linewidth=1.5)
        else:
            order = np.argsort(tpred)
            ax[0].plot(tpred[order], mod_pred1[order], 'k-', linewidth=1.5)
            ax[1].plot(tpred[order], mod_pred2[order], 'k-', linewidth=1.5)
    
    if('ts' in kwargs and 't_trans' in kwargs):
        ts = kwargs['ts']
        t_trans = kwargs['t_trans']
        for i, type_ in enumerate(cell_types):
            ax[0].plot([t_trans[i],t_trans[i]], [0, mod1.max()], '-x', color=colors[i%len(colors)])
            ax[0].plot([ts[i],ts[i]], [0, mod1.max()], '--x', color=colors[i%len(colors)])
            ax[1].plot([t_trans[i],t_trans[i]], [0, mod2.max()], '-x', color=colors[i%len(colors)])
            ax[1].plot([ts[i],ts[i]], [0, mod2.max()], '--x', color=colors[i%len(colors)])
    
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("U" if by == 'us' else "C", fontsize=18)
    
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("S" if by == 'us' else "U", fontsize=18)
    handles, labels = ax[1].get_legend_handles_labels()
    
    ax[0].set_title('Unspliced, VAE' if by == 'us' else 'Chromatin, VAE')
    ax[1].set_title('Spliced, VAE' if by == 'us' else 'Unspliced, VAE')
    
    lgd=fig.legend(handles, labels, fontsize=15, markerscale=5, bbox_to_anchor=(1.0,1.0), loc='upper left')
    fig.suptitle(title, fontsize=28)
    plt.tight_layout()
    
    save_fig(fig, save, (lgd,))
    return

def plot_sig(t, 
             c, u, s, 
             cpred, upred, spred, 
             by='us', 
             cell_labels=None, 
             title="Gene", 
             save=None, 
             **kwargs):
    """Generate a 2x2 u/s-t plot for a single gene
    The first row shows the original data, while the second row overlaps prediction with original data because VeloVAE outputs a point cloud instead of line fitting.
    
    ArgumentS
    ---------
    
    t : `numpy array`
        Cell time, (N,)
    u, s : `numpy array`
        Original unspliced and spliced counts of a single gene, (N,)
    upred, spred : `numpy array`
        Predicted unspliced and spliced counts of a single gene, (N,)
    cell_labels : `numpy array`, optional
        Cell type annotation, (N,)
    title : str, optional
        Title of the figure
    save : str
        Figure name for saving (including path)
    """
    mod1 = u if by == 'us' else c
    mod2 = s if by == 'us' else u
    mod_pred1 = upred if by == 'us' else cpred
    mod_pred2 = spred if by == 'us' else upred
    
    D = kwargs['sparsify'] if('sparsify' in kwargs) else 1
    tscv = kwargs['tscv'] if 'tscv' in kwargs else t
    tdemo = kwargs["tdemo"] if "tdemo" in kwargs else t
    if('cell_labels' == None):
        fig, ax = plt.subplots(2,1,figsize=(15,12),facecolor='white')
        #order = np.argsort(t)
        ax[0].plot(t[::D], mod1[::D],'b.',label="raw")
        ax[1].plot(t[::D], mod2[::D],'b.',label="raw")
        ax[0].plot(tdemo, mod_pred1, '.', color='lawngreen', label='Prediction', linewidth=2.0)
        ax[1].plot(tdemo, mod_pred2, '.', color='lawngreen', label="Prediction", linewidth=2.0)

        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("U" if by == 'us' else "C", fontsize=18)
        
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("S" if by == 'us' else "U", fontsize=18)
        
        #fig.subplots_adjust(right=0.7)
        handles, labels = ax[1].get_legend_handles_labels()
    else:
        fig, ax = plt.subplots(2,2,figsize=(24,12),facecolor='white')
        labels_pred = kwargs['labels_pred'] if 'labels_pred' in kwargs else []
        labels_demo = kwargs['labels_demo'] if 'labels_demo' in kwargs else None
        cell_types = np.unique(cell_labels)
        colors = get_colors(len(cell_types), None)
        
        #Plot the input data in the true labels
        for i, type_ in enumerate(cell_types):
            mask_type = cell_labels==type_
            ax[0,0].scatter(tscv[mask_type][::D], mod1[mask_type][::D],s=8.0, color=colors[i%len(colors)], alpha=0.7, label=type_, edgecolors='none')
            ax[0,1].scatter(tscv[mask_type][::D], mod2[mask_type][::D],s=8.0, color=colors[i%len(colors)], alpha=0.7, label=type_, edgecolors='none')
            if(len(labels_pred) > 0):
                mask_mytype = labels_pred==type_
                ax[1,0].scatter(t[mask_mytype][::D], mod1[mask_mytype][::D],s=8.0,color=colors[i%len(colors)], alpha=0.7, label=type_, edgecolors='none')
                ax[1,1].scatter(t[mask_mytype][::D], mod2[mask_mytype][::D],s=8.0,color=colors[i%len(colors)], alpha=0.7, label=type_, edgecolors='none')
            else:
                ax[1,0].scatter(t[mask_type][::D], mod1[mask_type][::D],s=8.0,color=colors[i%len(colors)], alpha=0.7, label=type_, edgecolors='none')
                ax[1,1].scatter(t[mask_type][::D], mod2[mask_type][::D],s=8.0,color=colors[i%len(colors)], alpha=0.7, label=type_, edgecolors='none')
            
        if(labels_demo is not None):
            for i, type_ in enumerate(cell_types):
                mask_mytype = labels_demo==type_
                order = np.argsort(tdemo[mask_mytype])
                ax[1,0].plot(tdemo[mask_mytype][order], mod_pred1[mask_mytype][order], color=colors[i%len(colors)], linewidth=2.0)
                ax[1,1].plot(tdemo[mask_mytype][order], mod_pred2[mask_mytype][order], color=colors[i%len(colors)], linewidth=2.0)
        else:
            order = np.argsort(tdemo)
            ax[1,0].plot(tdemo[order], mod_pred1[order], 'k.', linewidth=2.0)
            ax[1,1].plot(tdemo[order], mod_pred2[order], 'k.', linewidth=2.0)

        if('t_trans' in kwargs and by == 'us'):
            t_trans = kwargs['t_trans']
            for i, type_ in enumerate(cell_types):
                ax[1,0].plot([t_trans[i],t_trans[i]], [0, u.max()], '-x', color=colors[i%len(colors)])
                ax[1,1].plot([t_trans[i],t_trans[i]], [0, s.max()], '-x', color=colors[i%len(colors)])
        for j in range(2): 
            ax[j,0].set_xlabel("Time")
            ax[j,0].set_ylabel("U" if by == 'us' else "C", fontsize=18)
            
            ax[j,1].set_xlabel("Time")
            ax[j,1].set_ylabel("S" if by == 'us' else "U", fontsize=18)
            handles, labels = ax[1,0].get_legend_handles_labels()
           
        if by == 'us':
            if('subtitles' in kwargs):
                ax[0,0].set_title(f'Unspliced, {subtitle[0]}')
                ax[0,1].set_title(f'Spliced, {subtitle[0]}')
                ax[1,0].set_title(f'Unspliced, {subtitle[1]}')
                ax[1,1].set_title(f'Spliced, {subtitle[1]}')
            else:
                ax[0,0].set_title('Unspliced, True Label')
                ax[0,1].set_title('Spliced, True Label')
                ax[1,0].set_title('Unspliced, VAE')
                ax[1,1].set_title('Spliced, VAE')
        else:
            if('subtitles' in kwargs):
                ax[0,0].set_title(f'Chromatin, {subtitle[0]}')
                ax[0,1].set_title(f'Unspliced, {subtitle[0]}')
                ax[1,0].set_title(f'Chromatin, {subtitle[1]}')
                ax[1,1].set_title(f'Unspliced, {subtitle[1]}')
            else:
                ax[0,0].set_title('Chromatin, True Label')
                ax[0,1].set_title('Unpliced, True Label')
                ax[1,0].set_title('Chromatin, VAE')
                ax[1,1].set_title('Unspliced, VAE')
    
    lgd=fig.legend(handles, labels, fontsize=15, markerscale=5, ncol=4, bbox_to_anchor=(0.0, 1.0, 1.0, 0.25), loc='center')
    fig.suptitle(title, fontsize=28)
    plt.tight_layout()
    
    save_fig(fig, save, (lgd,))
    return

def plot_phase(c, u, s, 
               cpred, upred, spred, 
               title, 
               by='us', 
               track_idx=None, 
               labels=None, # array/list of integer
               types=None,  # array/list of string
               save=None, 
               plot_pred=True):
    """Plot the phase portrait of a gene
    
    Arguments
    ---------
    
    u, s : :class:`numpy array`
        Original unpsliced and spliced counts, (N,)
    upred, spred : :class:`numpy array`
        Predicted u,s values, (N,)
    title : str
        Figure title
    track_idx : `numpy array`, optional
        Indices of cells with lines connecting input data and prediction, (N,)
    labels : `numpy array`, optional 
        Cell type annotation
    types : `numpy array`, optional
        Unique cell types
    save : str
        Figure name for saving (including path)
    """
    fig, ax = plt.subplots(figsize=(6,6),facecolor='white')
    if(labels is None or types is None):
        if by == 'us':
            ax.scatter(s,u,c="b",alpha=0.5)
        else:
            ax.scatter(u,c,c="b",alpha=0.5)
    else:
        colors = get_colors(len(types), None)
        for i, type_ in enumerate(types):
            if by == 'us':
                ax.scatter(s[labels==i],u[labels==i],color=colors[i%len(colors)],alpha=0.3,label=type_)
            else:
                ax.scatter(u[labels==i],c[labels==i],color=colors[i%len(colors)],alpha=0.3,label=type_)
    if by == 'us':
        ax.plot(spred,upred,'k.',label="ode")
    else:
        ax.plot(upred,cpred,'k.',label="ode")
    #Plot the correspondence
    if plot_pred:
        if(track_idx is None):
            rng = np.random.default_rng()
            perm = rng.permutation(len(s))
            Nsample = 50
            s_comb = np.stack([s[perm[:Nsample]],spred[perm[:Nsample]]]).ravel('F')
            u_comb = np.stack([u[perm[:Nsample]],upred[perm[:Nsample]]]).ravel('F')
            c_comb = np.stack([c[perm[:Nsample]],cpred[perm[:Nsample]]]).ravel('F')
        else:
            s_comb = np.stack([s[track_idx],spred[track_idx]]).ravel('F')
            u_comb = np.stack([u[track_idx],upred[track_idx]]).ravel('F')
            c_comb = np.stack([c[track_idx],cpred[track_idx]]).ravel('F')
            
        for i in range(0, len(s_comb), 2):
            if by == 'us':
                ax.plot(s_comb[i:i+2], u_comb[i:i+2], 'k-', linewidth=0.8)
            else:
                ax.plot(u_comb[i:i+2], c_comb[i:i+2], 'k-', linewidth=0.8)
    ax.set_xlabel("S" if by=='us' else "U", fontsize=18)
    ax.set_ylabel("U" if by=='us' else "C", fontsize=18)
    
    handles, labels = ax.get_legend_handles_labels()
    lgd=fig.legend(handles, labels, fontsize=15, markerscale=5, ncol=4, bbox_to_anchor=(0.0, 1.0, 1.0, 0.25), loc='center')
    fig.suptitle(title)
    
    save_fig(fig, save, (lgd,))
