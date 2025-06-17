import logging
from scipy import sparse
import numpy as np
import scvelo as scv
import matplotlib
import matplotlib.pyplot as plt
from .model.model_util_chrom import velocity_graph
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel
from scipy.stats.distributions import chi2
import statsmodels.api as sm
logger = logging.getLogger(__name__)

#######################################################################################
# Default colors and markers for plotting
#######################################################################################
TAB10 = list(plt.get_cmap("tab10").colors)
TAB20 = list(plt.get_cmap("tab20").colors)
TAB20B = list(plt.get_cmap("tab20b").colors)
TAB20C = list(plt.get_cmap("tab20c").colors)
RAINBOW = [plt.cm.rainbow(i) for i in range(256)]

markers = ["o", "x", "s", "v", "+", "d", "1", "*", "^", "p", "h", "8", "1", "2", "|"]


def get_colors(n, color_map=None):
    """Get colors for plotting cell clusters.

    Arguments
    ---------

    n : int
        Number of cell clusters
    color_map : str, optional
        User-defined colormap. If not set, the colors will be chosen as the colors for tabular data in matplotlib.
    """
    if color_map is None:  # default color
        if n <= 10:
            return TAB10[:n]
        elif n <= 20:
            return TAB20[:n]
        elif n <= 40:
            TAB40 = TAB20B+TAB20C
            return TAB40[:n]
        else:
            logger.warning("Number of colors exceeds the maximum (40)! Use a continuous colormap (256) instead.")
            return RAINBOW[:n]
    else:
        color_map_obj = list(plt.get_cmap(color_map).colors)
        k = len(color_map_obj)//n
        colors = ([color_map_obj(i) for i in range(0, len(color_map_obj), k)]
                  if k > 0 else
                  [color_map_obj(i) for i in range(len(color_map_obj))])
    return colors


def save_fig(fig, save, bbox_extra_artists=None):
    if save is not None:
        try:
            idx = save.find('.')
            fig.savefig(save, bbox_extra_artists=bbox_extra_artists, format=save[idx+1:], bbox_inches='tight')
        except FileNotFoundError:
            logger.warning("Saving failed. File path doesn't exist!")
        plt.close(fig)


def plot_sig_(t,
              c, u, s,
              cell_labels,
              cell_type_colors=None,
              tpred=None,
              cpred=None, upred=None, spred=None,
              type_specific=False,
              title='Gene',
              save=None,
              **kwargs):
    fig, ax = plt.subplots(3, 1, figsize=(15, 20), facecolor='white')
    D = kwargs['sparsify'] if 'sparsify' in kwargs else 1
    if cell_type_colors is None:
        cell_types = np.unique(cell_labels)
        colors = get_colors(len(cell_types), None)
    else:
        cell_types = np.array(list(cell_type_colors.keys()))
        colors = np.array([cell_type_colors[type_] for type_ in cell_types])
    for i, type_ in enumerate(cell_types):
        mask_type = cell_labels == type_
        ax[0].plot(t[mask_type][::D], c[mask_type][::D], '.', color=colors[i % len(colors)], alpha=0.7, label=type_)
        ax[1].plot(t[mask_type][::D], u[mask_type][::D], '.', color=colors[i % len(colors)], alpha=0.7, label=type_)
        ax[2].plot(t[mask_type][::D], s[mask_type][::D], '.', color=colors[i % len(colors)], alpha=0.7, label=type_)

    if tpred is not None and cpred is not None and upred is not None and spred is not None:
        if type_specific:
            for i, type_ in enumerate(cell_types):
                mask_type = cell_labels == type_
                order = np.argsort(tpred[mask_type])
                ax[0].plot(tpred[mask_type][order],
                           cpred[mask_type][order],
                           '.',
                           color=colors[i % len(colors)],
                           alpha=0.8,
                           label=type_)
                ax[1].plot(tpred[mask_type][order],
                           upred[mask_type][order],
                           '.',
                           color=colors[i % len(colors)],
                           alpha=0.8,
                           label=type_)
                ax[2].plot(tpred[mask_type][order],
                           spred[mask_type][order],
                           '.',
                           color=colors[i % len(colors)],
                           alpha=0.8,
                           label=type_)
        else:
            order = np.argsort(tpred)
            ax[0].plot(tpred[order], cpred[order], 'k.', alpha=0.7)
            ax[1].plot(tpred[order], upred[order], 'k.', alpha=0.7)
            ax[2].plot(tpred[order], spred[order], 'k.', alpha=0.7)

        perm = np.random.permutation(len(s))
        Nsample = len(u) // 10
        idx = perm[:Nsample]

        for i in idx:
            ax[0].plot([tpred[i], t[i]], [cpred[i], c[i]], 'k-', linewidth=0.5, alpha=0.6)
            ax[1].plot([tpred[i], t[i]], [upred[i], u[i]], 'k-', linewidth=0.5, alpha=0.6)
            ax[2].plot([tpred[i], t[i]], [spred[i], s[i]], 'k-', linewidth=0.5, alpha=0.6)

    if 'ts' in kwargs and 't_trans' in kwargs:
        ts = kwargs['ts']
        t_trans = kwargs['t_trans']
        for i, type_ in enumerate(cell_types):
            ax[0].plot([t_trans[i], t_trans[i]], [0, c.max()], '-x', color=colors[i % len(colors)])
            ax[0].plot([ts[i], ts[i]], [0, c.max()], '--x', color=colors[i % len(colors)])
            ax[1].plot([t_trans[i], t_trans[i]], [0, u.max()], '-x', color=colors[i % len(colors)])
            ax[1].plot([ts[i], ts[i]], [0, u.max()], '--x', color=colors[i % len(colors)])
            ax[2].plot([t_trans[i], t_trans[i]], [0, s.max()], '-x', color=colors[i % len(colors)])
            ax[2].plot([ts[i], ts[i]], [0, s.max()], '--x', color=colors[i % len(colors)])

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("C", fontsize=18)

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("U", fontsize=18)

    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("S", fontsize=18)
    handles, labels = ax[1].get_legend_handles_labels()

    ax[0].set_title('Chromatin, VAE')
    ax[1].set_title('Unspliced, VAE')
    ax[2].set_title('Spliced, VAE')

    lgd = fig.legend(handles,
                     labels,
                     fontsize=15,
                     markerscale=5,
                     bbox_to_anchor=(1.0, 1.0),
                     loc='upper left')
    fig.suptitle(title, y=0.99, fontsize=20)
    plt.tight_layout()

    save_fig(fig, save, (lgd,))


def plot_sig(t,
             c, u, s,
             cpred, upred, spred,
             cell_labels=None,
             cell_type_colors=None,
             title="Gene",
             save=None,
             **kwargs):
    D = kwargs['sparsify'] if 'sparsify' in kwargs else 1
    tscv = kwargs['tscv'] if 'tscv' in kwargs else t
    tdemo = kwargs["tdemo"] if "tdemo" in kwargs else t
    if cell_labels is None:
        fig, ax = plt.subplots(3, 1, figsize=(15, 24), facecolor='white')
        ax[0].plot(t[::D], c[::D], 'b.', label="raw")
        ax[1].plot(t[::D], u[::D], 'b.', label="raw")
        ax[2].plot(t[::D], s[::D], 'b.', label="raw")
        ax[0].plot(tdemo, cpred, '.', color='lawngreen', label='Prediction', linewidth=2.0)
        ax[1].plot(tdemo, upred, '.', color='lawngreen', label='Prediction', linewidth=2.0)
        ax[2].plot(tdemo, spred, '.', color='lawngreen', label="Prediction", linewidth=2.0)

        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("C", fontsize=18)

        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("U", fontsize=18)

        ax[2].set_xlabel("Time")
        ax[2].set_ylabel("S", fontsize=18)

        handles, labels = ax[1].get_legend_handles_labels()
    else:
        fig, ax = plt.subplots(2, 3, figsize=(36, 12), facecolor='white')
        labels_pred = kwargs['labels_pred'] if 'labels_pred' in kwargs else []
        labels_demo = kwargs['labels_demo'] if 'labels_demo' in kwargs else None
        if cell_type_colors is None:
            cell_types = np.unique(cell_labels)
            colors = get_colors(len(cell_types), None)
        else:
            cell_types = np.array(list(cell_type_colors.keys()))
            colors = np.array([cell_type_colors[type_] for type_ in cell_types])

        for i, type_ in enumerate(cell_types):
            mask_type = cell_labels == type_
            ax[0, 0].scatter(tscv[mask_type][::D],
                             c[mask_type][::D],
                             s=8.0,
                             color=colors[i % len(colors)],
                             alpha=0.7,
                             label=type_,
                             edgecolors='none')
            ax[0, 1].scatter(tscv[mask_type][::D],
                             u[mask_type][::D],
                             s=8.0,
                             color=colors[i % len(colors)],
                             alpha=0.7,
                             label=type_,
                             edgecolors='none')
            ax[0, 2].scatter(tscv[mask_type][::D],
                             s[mask_type][::D],
                             s=8.0,
                             color=colors[i % len(colors)],
                             alpha=0.7,
                             label=type_, edgecolors='none')
            if len(labels_pred) > 0:
                mask_mytype = labels_pred == type_
                ax[1, 0].scatter(t[mask_mytype][::D],
                                 c[mask_mytype][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
                ax[1, 1].scatter(t[mask_mytype][::D],
                                 u[mask_mytype][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
                ax[1, 2].scatter(t[mask_mytype][::D],
                                 s[mask_mytype][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
            else:
                ax[1, 0].scatter(t[mask_type][::D],
                                 c[mask_type][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
                ax[1, 1].scatter(t[mask_type][::D],
                                 u[mask_type][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
                ax[1, 2].scatter(t[mask_type][::D],
                                 s[mask_type][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')

        if labels_demo is not None:
            for i, type_ in enumerate(cell_types):
                mask_mytype = labels_demo == type_
                order = np.argsort(tdemo[mask_mytype])
                ax[1, 0].plot(tdemo[mask_mytype][order],
                              cpred[mask_mytype][order],
                              color=colors[i % len(colors)],
                              linewidth=2.0)
                ax[1, 1].plot(tdemo[mask_mytype][order],
                              upred[mask_mytype][order],
                              color=colors[i % len(colors)],
                              linewidth=2.0)
                ax[1, 2].plot(tdemo[mask_mytype][order],
                              spred[mask_mytype][order],
                              color=colors[i % len(colors)],
                              linewidth=2.0)
        else:
            order = np.argsort(tdemo)
            ax[1, 0].plot(tdemo[order], cpred[order], 'k.', linewidth=2.0)
            ax[1, 1].plot(tdemo[order], upred[order], 'k.', linewidth=2.0)
            ax[1, 2].plot(tdemo[order], spred[order], 'k.', linewidth=2.0)

        if 't_trans' in kwargs:
            t_trans = kwargs['t_trans']
            for i, type_ in enumerate(cell_types):
                ax[1, 0].plot([t_trans[i], t_trans[i]], [0, c.max()], '-x', color=colors[i % len(colors)])
                ax[1, 1].plot([t_trans[i], t_trans[i]], [0, u.max()], '-x', color=colors[i % len(colors)])
                ax[1, 2].plot([t_trans[i], t_trans[i]], [0, s.max()], '-x', color=colors[i % len(colors)])
        for j in range(2):
            ax[j, 0].set_xlabel("Time")
            ax[j, 0].set_ylabel("C", fontsize=18)

            ax[j, 1].set_xlabel("Time")
            ax[j, 1].set_ylabel("U", fontsize=18)

            ax[j, 2].set_xlabel("Time")
            ax[j, 2].set_ylabel("S", fontsize=18)
        handles, labels = ax[1, 0].get_legend_handles_labels()

        if 'subtitles' in kwargs:
            ax[0, 0].set_title(f"Chromatin, {kwargs['subtitles'][0]}")
            ax[0, 1].set_title(f"Unspliced, {kwargs['subtitles'][0]}")
            ax[0, 2].set_title(f"Spliced, {kwargs['subtitles'][0]}")
            ax[1, 0].set_title(f"Chromatin, {kwargs['subtitles'][1]}")
            ax[1, 1].set_title(f"Unspliced, {kwargs['subtitles'][1]}")
            ax[1, 2].set_title(f"Spliced, {kwargs['subtitles'][1]}")
        else:
            ax[0, 0].set_title('Chromatin, True Label')
            ax[0, 1].set_title('Unspliced, True Label')
            ax[0, 2].set_title('Spliced, True Label')
            ax[1, 0].set_title('Chromatin, VAE')
            ax[1, 1].set_title('Unspliced, VAE')
            ax[1, 2].set_title('Spliced, VAE')

    lgd = fig.legend(handles,
                     labels,
                     fontsize=15,
                     markerscale=5,
                     ncol=4,
                     bbox_to_anchor=(0.0, 1.0, 1.0, 0.25),
                     loc='center')
    fig.suptitle(title, fontsize=28)
    plt.tight_layout()

    save_fig(fig, save, (lgd,))


def plot_vel(t,
             chat, uhat, shat,
             vc, vu, vs,
             t0=None,
             c0=None,
             u0=None,
             s0=None,
             dt=0.2,
             n_sample=100,
             cell_labels=None,
             cell_type_colors=None,
             title="Gene",
             axis_on=True,
             frame_on=True,
             legend=True,
             save=None):
    fig, ax = plt.subplots(1, 3, figsize=(28, 6), facecolor='white')

    if cell_labels is None:
        ax[0].scatter(t, chat, color='grey', s=8.0, alpha=0.2)
        ax[1].scatter(t, uhat, color='grey', s=8.0, alpha=0.2)
        ax[2].scatter(t, shat, color='grey', s=8.0, alpha=0.2)
        handles, labels = ax[1].get_legend_handles_labels()
    else:
        if cell_type_colors is None:
            cell_types = np.unique(cell_labels)
            colors = get_colors(len(cell_types), None)
        else:
            cell_types = np.array(list(cell_type_colors.keys()))
            colors = np.array([cell_type_colors[type_] for type_ in cell_types])
        for i, type_ in enumerate(cell_types):
            mask_type = cell_labels == type_
            ax[0].scatter(t[mask_type], chat[mask_type], color=colors[i % len(colors)], s=20, alpha=0.5, label=type_, edgecolors='none')
            ax[1].scatter(t[mask_type], uhat[mask_type], color=colors[i % len(colors)], s=20, alpha=0.5, label=type_, edgecolors='none')
            ax[2].scatter(t[mask_type], shat[mask_type], color=colors[i % len(colors)], s=20, alpha=0.5, label=type_, edgecolors='none')
            handles, labels = ax[1].get_legend_handles_labels()

    plot_indices = np.random.choice(len(t), min(n_sample, len(t)), replace=False)
    if dt > 0:
        ax[0].quiver(t[plot_indices],
                     chat[plot_indices],
                     dt*np.ones((len(plot_indices),)),
                     vc[plot_indices]*dt,
                     angles='xy')
        ax[1].quiver(t[plot_indices],
                     uhat[plot_indices],
                     dt*np.ones((len(plot_indices),)),
                     vu[plot_indices]*dt,
                     angles='xy')
        ax[2].quiver(t[plot_indices],
                     shat[plot_indices],
                     dt*np.ones((len(plot_indices),)),
                     vs[plot_indices]*dt,
                     angles='xy')
    if t0 is not None and c0 is not None and u0 is not None and s0 is not None:
        for i, k in enumerate(plot_indices):
            if i == 0:
                ax[0].plot([t0[k], t[k]], [c0[k], chat[k]], 'r-o', label='Prediction')
            else:
                ax[0].plot([t0[k], t[k]], [c0[k], chat[k]], 'r-o')
            ax[1].plot([t0[k], t[k]], [u0[k], uhat[k]], 'r-o')
            ax[2].plot([t0[k], t[k]], [s0[k], shat[k]], 'r-o')

    ax[0].set_ylabel("C", fontsize=16)
    ax[1].set_ylabel("U", fontsize=16)
    ax[2].set_ylabel("S", fontsize=16)

    for j in range(3):
        axi = ax[j]
        if not axis_on:
            axi.xaxis.set_ticks_position('none')
            axi.yaxis.set_ticks_position('none')
            axi.get_xaxis().set_visible(False)
            axi.get_yaxis().set_visible(False)
        if not frame_on:
            axi.xaxis.set_ticks_position('none')
            axi.yaxis.set_ticks_position('none')
            axi.set_frame_on(False)

    if legend:
        lgd = fig.legend(handles,
                         labels,
                         fontsize=15,
                         markerscale=5,
                         ncol=4,
                         bbox_to_anchor=(0.0, 1.0, 1.0, 0.25),
                         loc='center')
    fig.suptitle(title, fontsize=28)
    plt.tight_layout()

    save_fig(fig, save, (lgd,) if legend else None)


def plot_phase(c, u, s,
               cpred, upred, spred,
               title,
               by='us',
               t=None,
               track_idx=None,
               cell_labels=None,
               cell_type_colors=None,
               save=None,
               plot_pred=True,
               show=False):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    if cell_labels is None:
        if by == 'us':
            ax.scatter(s, u, c="b", alpha=0.5)
        else:
            ax.scatter(u, c, c="b", alpha=0.5)
    else:
        if cell_type_colors is None:
            cell_types = np.unique(cell_labels)
            colors = get_colors(len(cell_types), None)
        else:
            cell_types = np.array(list(cell_type_colors.keys()))
            colors = np.array([cell_type_colors[type_] for type_ in cell_types])
        for i, type_ in enumerate(cell_types):
            mask_type = cell_labels == type_
            if by == 'us':
                if t is not None:
                    ax.scatter(s[mask_type], u[mask_type], color=colors[i % len(colors)], alpha=0.2, label=type_, s=10)
                else:
                    ax.scatter(s[mask_type], u[mask_type], color=colors[i % len(colors)], alpha=0.4, label=type_)
            else:
                if t is not None:
                    ax.scatter(u[mask_type], c[mask_type], color=colors[i % len(colors)], alpha=0.2, label=type_, s=10)
                else:
                    ax.scatter(u[mask_type], c[mask_type], color=colors[i % len(colors)], alpha=0.4, label=type_)
    if by == 'us':
        if t is None:
            ax.scatter(spred, upred, c='black', label="ode", s=20)
        else:
            ax.scatter(spred, upred, c=t, label="ode", cmap='RdBu_r', s=20)
    else:
        if t is None:
            ax.scatter(upred, cpred, c='black', label="ode", s=20)
        else:
            ax.scatter(upred, cpred, c=t, label="ode", cmap='RdBu_r', s=20)

    if plot_pred:
        if track_idx is None:
            perm = np.random.permutation(len(s))
            Nsample = 50
            s_comb = np.stack([s[perm[:Nsample]], spred[perm[:Nsample]]]).ravel('F')
            u_comb = np.stack([u[perm[:Nsample]], upred[perm[:Nsample]]]).ravel('F')
            c_comb = np.stack([c[perm[:Nsample]], cpred[perm[:Nsample]]]).ravel('F')
        else:
            s_comb = np.stack([s[track_idx], spred[track_idx]]).ravel('F')
            u_comb = np.stack([u[track_idx], upred[track_idx]]).ravel('F')
            c_comb = np.stack([c[track_idx], cpred[track_idx]]).ravel('F')

        for i in range(0, len(s_comb), 2):
            if by == 'us':
                ax.plot(s_comb[i:i+2], u_comb[i:i+2], 'k-', linewidth=0.5, alpha=0.5)
            else:
                ax.plot(u_comb[i:i+2], c_comb[i:i+2], 'k-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel("S" if by == 'us' else "U", fontsize=18)
    ax.set_ylabel("U" if by == 'us' else "C", fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles,
                     labels,
                     fontsize=15,
                     markerscale=5,
                     ncol=4,
                     bbox_to_anchor=(0.0, 1.0, 1.0, 0.25),
                     loc='center')
    fig.suptitle(title)

    if show:
        plt.show()
    else:
        save_fig(fig, save, (lgd,))


def _plot_heatmap(ax,
                  vals,
                  X_embed,
                  colorbar_name,
                  colorbar_ticklabels=None,
                  markersize=5,
                  cmap='plasma',
                  axis_off=True):
    ax.scatter(X_embed[:, 0],
               X_embed[:, 1],
               s=markersize,
               c=vals,
               cmap=cmap,
               edgecolors='none')
    vmin = np.quantile(vals, 0.01)
    vmax = np.quantile(vals, 0.99)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(colorbar_name, rotation=270, fontsize=15)
    if colorbar_ticklabels is not None:
        if len(colorbar_ticklabels) == 2:
            cbar.ax.get_yaxis().labelpad = 5
        cbar.set_ticks(np.linspace(vmin, vmax, len(colorbar_ticklabels)))
        cbar.ax.set_yticklabels(colorbar_ticklabels, fontsize=12)
    if axis_off:
        ax.axis("off")

    return ax


def plot_time(t_latent,
              X_embed,
              cmap='plasma',
              save=None):
    """Plots mean cell time as a heatmap.

    Arguments
    ---------

    t_latent : `numpy array`
        Mean latent time, (N,)
    X_embed : `numpy array`
        2D coordinates for visualization, (N,2)
    cmap : str, optional
        Colormap name
    save : str, optional
        Figure name for saving (including path)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_heatmap(ax, t_latent, X_embed, "Latent Time", None, cmap=cmap, axis_off=True)
    save_fig(fig, save)


def elipse_axis(x, y, xc, yc, slope):
    return (y - yc) - (slope * (x - xc))


def ellipse_fit(adata,
                genes,
                color_by='quantile',
                n_cols=8,
                title=None,
                figsize=None,
                axis_on=False,
                pointsize=2,
                linewidth=2
                ):
    """Plot ellipse fit for genes.

    Args:
        adata (:class:`anndata.AnnData`):
            Input data object.
        genes ([str, list of str]):
            Gene names to plot.
        color_by (str, optional):
            Color used for the plots. Defaults to ellipse 'quantile'.
        n_cols (int, optional):
            Number of columns to plot. Defaults to 8.
        title (str, optional):
            Title of the whole figure. Defaults to None.
        figsize (tuple, optional):
            Size of the figure. Defaults to None.
        axis_on (bool, optional):
            Whether to plot axis. Defaults to False.
        pointsize (int, optional):
            Point size. Defaults to 2.
        linewidth (int, optional):
            Line width. Defaults to 2.
    """
    by_quantile = color_by == 'quantile'
    by_quantile_score = color_by == 'quantile_scores'
    if not by_quantile and not by_quantile_score:
        types = adata.obs[color_by].cat.categories
        colors = adata.uns[f'{color_by}_colors']
    if isinstance(genes, str):
        genes = [genes]
    gn = len(genes)
    if gn < n_cols:
        n_cols = gn
    fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False, figsize=(2 * n_cols, 2.4 * (-(-gn // n_cols))) if figsize is None else figsize)
    count = 0
    for gene in genes:
        u = np.array(adata[:, gene].layers['Mu'])
        s = np.array(adata[:, gene].layers['Ms'])
        row = count // n_cols
        col = count % n_cols
        non_zero = (u > 0) & (s > 0)
        if np.sum(non_zero) < 10:
            count += 1
            fig.delaxes(axs[row, col])
            continue

        mean_u, mean_s = np.mean(u[non_zero]), np.mean(s[non_zero])
        std_u, std_s = np.std(u[non_zero]), np.std(s[non_zero])
        u_ = (u - mean_u)/std_u
        s_ = (s - mean_s)/std_s
        X = np.reshape(s_[non_zero], (-1, 1))
        Y = np.reshape(u_[non_zero], (-1, 1))

        # Ax^2 + Bxy + Cy^2 + Dx + Ey + 1 = 0
        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = -np.ones_like(X)
        x, res, _, _ = np.linalg.lstsq(A, b)
        x = x.squeeze()
        A, B, C, D, E = x
        good_fit = B**2 - 4*A*C < 0
        theta = np.arctan(B/(A - C))/2 if x[0] > x[2] else np.pi/2 + np.arctan(B/(A - C))/2
        good_fit = good_fit & (theta < np.pi/2) & (theta > 0)
        if not good_fit:
            count += 1

        x_coord = np.linspace((-mean_s)/std_s, (np.max(s)-mean_s)/std_s, 500)
        y_coord = np.linspace((-mean_u)/std_u, (np.max(u)-mean_u)/std_u, 500)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = A * X_coord**2 + B * X_coord * Y_coord + C * Y_coord**2 + D * X_coord + E * Y_coord + 1

        M0 = np.array([
             A, B/2, D/2,
             B/2, C, E/2,
             D/2, E/2, 1,
        ]).reshape(3, 3)
        M = np.array([
            A, B/2,
            B/2, C,
        ]).reshape(2, 2)
        l1, l2 = np.sort(np.linalg.eigvals(M))
        xc = (B*E - 2*C*D)/(4*A*C - B**2)
        yc = (B*D - 2*A*E)/(4*A*C - B**2)
        slope_major = np.tan(theta)
        theta2 = np.pi/2 + theta
        slope_minor = np.tan(theta2)
        a = np.sqrt(-np.linalg.det(M0)/np.linalg.det(M)/l2)
        b = np.sqrt(-np.linalg.det(M0)/np.linalg.det(M)/l1)
        xtop = xc + a*np.cos(theta)
        ytop = yc + a*np.sin(theta)
        xbot = xc - a*np.cos(theta)
        ybot = yc - a*np.sin(theta)
        xtop2 = xc + b*np.cos(theta2)
        ytop2 = yc + b*np.sin(theta2)
        xbot2 = xc - b*np.cos(theta2)
        ybot2 = yc - b*np.sin(theta2)
        mse = res[0] / np.sum(non_zero)
        major_bit = elipse_axis(s_, u_, xc, yc, slope_major)
        minor_bit = elipse_axis(s_, u_, xc, yc, slope_minor)
        quant1 = (major_bit > 0) & (minor_bit < 0)
        quant2 = (major_bit > 0) & (minor_bit > 0)
        quant3 = (major_bit < 0) & (minor_bit > 0)
        quant4 = (major_bit < 0) & (minor_bit < 0)
        if (np.sum(quant1 | quant4) < 10) or (np.sum(quant2 | quant3) < 10):
            count += 1

        if by_quantile:
            axs[row, col].scatter(s_[quant1], u_[quant1], s=pointsize, c='tab:red', alpha=0.6)
            axs[row, col].scatter(s_[quant2], u_[quant2], s=pointsize, c='tab:orange', alpha=0.6)
            axs[row, col].scatter(s_[quant3], u_[quant3], s=pointsize, c='tab:green', alpha=0.6)
            axs[row, col].scatter(s_[quant4], u_[quant4], s=pointsize, c='tab:blue', alpha=0.6)
        elif by_quantile_score:
            if 'quantile_scores' not in adata.layers:
                raise ValueError('Please run multivelo.compute_quantile_scores first to compute quantile scores.')
            axs[row, col].scatter(s_, u_, s=pointsize, c=adata[:, gene].layers['quantile_scores'], cmap='RdBu_r', alpha=0.7)
        else:
            for i in range(len(types)):
                filt = adata.obs[color_by] == types[i]
                axs[row, col].scatter(s_[filt], u_[filt], s=pointsize, c=colors[i], alpha=0.7)
        axs[row, col].contour(X_coord, Y_coord, Z_coord, levels=[0], colors=('r'), linewidths=linewidth, alpha=0.7)
        axs[row, col].scatter([xc], [yc], c='black', s=5, zorder=2)
        axs[row, col].scatter([0], [0], c='black', s=5, zorder=2)
        axs[row, col].plot([xtop, xbot], [ytop, ybot], color='b', linestyle='dashed', linewidth=linewidth, alpha=0.7)
        axs[row, col].plot([xtop2, xbot2], [ytop2, ybot2], color='g', linestyle='dashed', linewidth=linewidth, alpha=0.7)

        axs[row, col].set_title(f'{gene} {mse:.3g}')
        axs[row, col].set_xlabel('s')
        axs[row, col].set_ylabel('u')
        common_range = [np.min([(-mean_s)/std_s, (-mean_u)/std_u])-(0.05*np.max(s)/std_s), np.max([(np.max(s)-mean_s)/std_s, (np.max(u)-mean_u)/std_u])+(0.05*np.max(s)/std_s)]
        axs[row, col].set_xlim(common_range)
        axs[row, col].set_ylim(common_range)
        if not axis_on:
            axs[row, col].xaxis.set_ticks_position('none')
            axs[row, col].yaxis.set_ticks_position('none')
            axs[row, col].get_xaxis().set_visible(False)
            axs[row, col].get_yaxis().set_visible(False)
            axs[row, col].xaxis.set_ticks_position('none')
            axs[row, col].yaxis.set_ticks_position('none')
            axs[row, col].set_frame_on(False)
        count += 1

    for i in range(col+1, n_cols):
        fig.delaxes(axs[row, i])
    if title is not None:
        fig.suptitle(title, fontsize=15)
    else:
        fig.suptitle('Ellipse Fit', fontsize=15)
    fig.tight_layout(rect=[0, 0.1, 1, 0.98])


def dynamic_plot(adata,
                 adata_atac,
                 genes,
                 key='vae',
                 by='expression',
                 modalities=None,
                 modalities_pred=None,
                 color_by=None,
                 axis_on=True,
                 frame_on=True,
                 show_pred=True,
                 show_pred_only=False,
                 batch_correction=False,
                 downsample=1,
                 figsize=None,
                 pointsize=2,
                 cmap='coolwarm'
                 ):
    """Gene dynamics plot.

    This function plots accessibility, expression, or velocity by time.

    Args:
        adata :class:`anndata.AnnData`:
            Anndata result after VAE inference.
        adata_atac :class:`anndata.AnnData`:
            ATAC Anndata object.
        genes [str,  list of str]:
            List of genes to plot.
        key (str, optional):
            Key to find VAE variables. Defaults to `vae`.
        by (str, optional):
            Plot accessibilities and expressions if `expression`. Plot velocities if `velocity`.
            Defaults to `expression`.
        modalities (list, optional):
            List of modalities in adata.layers to plot. Defaults to None.
        modalities_pred (list, optional):
            List of predicted modalities in adata.layers to plot. Defaults to None.
        color_by: (str, optional):
            Color used for the plots. Defaults to None.
        axis_on (bool):
            Whether to show axis labels. Defaults to True.
        frame_on (bool):
            Whether to show plot frames. Defaults to True.
        show_pred (bool):
            Whether to show prediction. Defaults to True.
        show_pred_only (bool):
            Whether to show prediction only. Defaults to False.
        batch_correction (bool):
            Whether the output was generated with batch correction. Defaults to False.
        downsample (int):
            How much to downsample the cells. The remaining number will be 1/`downsample` of original.
            Defaults to 1.
        figsize (tuple):
            Total figure size. Defaults to None.
        pointsize (float):
            Point size for scatter plots. Defaults to 2.
        cmap: (str)
            Color map for continuous color key. Defaults to 'coolwarm'.
    """
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype
    if by not in ['expression', 'velocity']:
        raise ValueError('"by" must be either "expression" or "velocity".')
    if by == 'velocity':
        show_pred = False
    if color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
        types = None
        colors = adata.obs[color_by].values
    elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]) and color_by+'_colors' in adata.uns.keys():
        types = adata.obs[color_by].cat.categories
        colors = adata.uns[f'{color_by}_colors']
    else:
        raise ValueError('Currently, color key must be a single string of either numerical or categorical available in adata obs, '
                         'and the colors of categories can be found in adata uns.\nYou can use scvelo.pl.scatter directly for more comprehensive plots.')

    downsample = np.clip(int(downsample), 1, 10)
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        logger.warning(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    if gn == 0:
        return
    no_c = False
    if adata_atac is None or 'Mc' not in adata_atac.layers.keys():
        no_c = True
    if batch_correction:
        show_pred_only = True
    if show_pred_only:
        show_pred = False

    fig, axs = plt.subplots(gn, 3, squeeze=False, figsize=(10, 2.3*gn) if figsize is None else figsize)
    fig.patch.set_facecolor('white')
    for row, gene in enumerate(genes):
        if modalities is not None:
            u = adata[:, gene].layers[modalities[-2]].copy()
            s = adata[:, gene].layers[modalities[-1]].copy()
        elif by == 'velocity':
            u = adata[:, gene].layers[f'{key}_velocity_u'].copy()
            s = adata[:, gene].layers[f'{key}_velocity'].copy()
        elif show_pred_only:
            if modalities_pred is not None:
                u = adata[:, gene].layers[modalities_pred[-2]].copy()
                s = adata[:, gene].layers[modalities_pred[-1]].copy()
            else:
                u = adata[:, gene].layers[f'{key}_uhat'].copy()
                s = adata[:, gene].layers[f'{key}_shat'].copy()
        else:
            u = adata[:, gene].layers['Mu'].copy()
            s = adata[:, gene].layers['Ms'].copy()
        if not no_c:
            if modalities is not None:
                if modalities[0] in adata.layers.keys():
                    c = adata[:, gene].layers[modalities[0]].copy()
                else:
                    c = adata_atac[:, gene].layers[modalities[0]].copy()
            elif by == 'velocity':
                c = adata[:, gene].layers[f'{key}_velocity_c'].copy()
            elif show_pred_only:
                if modalities_pred is not None:
                    c = adata[:, gene].layers[modalities_pred[0]].copy()
                else:
                    c = adata[:, gene].layers[f'{key}_chat'].copy()
            else:
                c = adata_atac[:, gene].layers['Mc'].copy()
        if not no_c:
            c = c.toarray() if sparse.issparse(c) else c
        u = u.toarray() if sparse.issparse(u) else u
        s = s.toarray() if sparse.issparse(s) else s
        if not no_c:
            c = np.ravel(c)
        u, s = np.ravel(u), np.ravel(s)
        time = np.array(adata.obs[f'{key}_time']).copy()
        if types is not None:
            for i in range(len(types)):
                filt = adata.obs[color_by] == types[i]
                filt = np.ravel(filt)
                if np.sum(filt) > 0:
                    if not no_c:
                        axs[row, 0].scatter(time[filt][::downsample], c[filt][::downsample], s=pointsize, c=colors[i], alpha=0.6)
                    axs[row, 1].scatter(time[filt][::downsample], u[filt][::downsample], s=pointsize, c=colors[i], alpha=0.6)
                    axs[row, 2].scatter(time[filt][::downsample], s[filt][::downsample], s=pointsize, c=colors[i], alpha=0.6)
        else:
            if not no_c:
                axs[row, 0].scatter(time[::downsample], c[::downsample], s=pointsize, c=colors[::downsample], alpha=0.6, cmap=cmap)
            axs[row, 1].scatter(time[::downsample], u[::downsample], s=pointsize, c=colors[::downsample], alpha=0.6, cmap=cmap)
            axs[row, 2].scatter(time[::downsample], s[::downsample], s=pointsize, c=colors[::downsample], alpha=0.6, cmap=cmap)

        if show_pred:
            if not no_c:
                if modalities_pred is not None:
                    a_c = adata[:, gene].layers[modalities_pred[0]].ravel()
                else:
                    a_c = adata[:, gene].layers[f'{key}_chat'].ravel()
            if modalities_pred is not None:
                a_u = adata[:, gene].layers[modalities_pred[-2]].ravel()
                a_s = adata[:, gene].layers[modalities_pred[-1]].ravel()
            else:
                a_u = adata[:, gene].layers[f'{key}_uhat'].ravel()
                a_s = adata[:, gene].layers[f'{key}_shat'].ravel()
            if not no_c:
                axs[row, 0].scatter(time[::downsample], a_c[::downsample], s=pointsize/2, c='black', alpha=0.2, zorder=1000)
            axs[row, 1].scatter(time[::downsample], a_u[::downsample], s=pointsize/2, c='black', alpha=0.2, zorder=1000)
            axs[row, 2].scatter(time[::downsample], a_s[::downsample], s=pointsize/2, c='black', alpha=0.2, zorder=1000)

        axs[row, 0].set_title(f'{gene} chromatin' if by == 'expression' else f'{gene} chromatin velocity')
        axs[row, 0].set_xlabel('t')
        axs[row, 0].set_ylabel('c' if by == 'expression' else 'dc/dt')
        axs[row, 1].set_title(f'{gene} unspliced' + ('' if by == 'expression' else ' velocity'))
        axs[row, 1].set_xlabel('t')
        axs[row, 1].set_ylabel('u' if by == 'expression' else 'du/dt')
        axs[row, 2].set_title(f'{gene} spliced' + ('' if by == 'expression' else ' velocity'))
        axs[row, 2].set_xlabel('t')
        axs[row, 2].set_ylabel('s' if by == 'expression' else 'ds/dt')

        for j in range(3):
            ax = axs[row, j]
            if not axis_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if not frame_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_frame_on(False)
            if no_c:
                fig.delaxes(ax)
    fig.tight_layout()


def scatter_plot(adata,
                 adata_atac,
                 genes,
                 key='vae',
                 by='us',
                 modalities=None,
                 modalities_pred=None,
                 color_by=None,
                 n_cols=5,
                 axis_on=True,
                 frame_on=True,
                 show_pred=True,
                 show_pred_only=False,
                 batch_correction=False,
                 title_more_info=False,
                 velocity_arrows=False,
                 downsample=1,
                 figsize=None,
                 pointsize=2,
                 fontsize=11,
                 cmap='coolwarm',
                 view_3d_elev=None,
                 view_3d_azim=None,
                 full_name=False
                 ):
    """Gene scatter plot.

    This function plots phase portraits of the specified plane.

    Args:
        adata :class:`anndata.AnnData`:
            Anndata result after VAE inference.
        adata_atac :class:`anndata.AnnData`:
            ATAC Anndata object.
        genes [str,  list of str]:
            List of genes to plot.
        key (str, optional):
            Key to find VAE variables. Defaults to `vae`.
        by (str):
            Plot unspliced-spliced plane if `us`. Plot chromatin-unspliced plane if `cu`.
            Plot 3D phase portraits if `cus`. Defaults to 'us'.
        modalities (list, optional):
            List of modalities in adata.layers to plot. Defaults to None.
        modalities_pred (list, optional):
            List of predicted modalities in adata.layers to plot. Defaults to None.
        color_by: (str, optional):
            Color used for the plots. Defaults to None.
        n_cols (int):
            Number of columns to plot. Defaults to 5.
        axis_on (bool):
            Whether to show axis labels. Defaults to True.
        frame_on (bool):
            Whether to show plot frames. Defaults to True.
        show_pred (bool):
            Whether to show prediction. Defaults to True.
        show_pred_only (bool):
            Whether to show prediction only. Defaults to False.
        batch_correction (bool):
            Whether the output was generated with batch correction. Defaults to False.
        title_more_info (bool):
            Whether to show likelihood for the gene in title. Defaults to False.
        velocity_arrows (bool):
            Whether to show velocity arrows of cells on the phase portraits. Defaults to False.
        downsample (int):
            How much to downsample the cells. The remaining number will be 1/`downsample` of original.
            Defaults to 1.
        figsize (tuple):
            Total figure size. Defaults to None.
        pointsize (float):
            Point size for scatter plots. Defaults to 2.
        fontsize (int):
            Font size for title. Defaults to 11.
        cmap: (str)
            Color map for continuous color key. Defaults to 'coolwarm'.
        view_3d_elev (float):
            Matplotlib 3D plot `elev` argument. `elev=90` is the same as U-S plane, and `elev=0` is the same as C-U plane.
            Defaults to None.
        view_3d_azim (float):
            Matplotlib 3D plot `azim` argument. `azim=270` is the same as U-S plane, and `azim=0` is the same as C-U plane.
            Defaults to None.
        full_name (bool):
            Show full names for chromatin, unspliced, and spliced rather than using abbreviated terms c, u, and s.
            Defaults to False.
    """
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype
    if by not in ['us', 'cu', 'cus']:
        raise ValueError("'by' argument must be one of ['us', 'cu', 'cus']")
    if by == 'us' and color_by == 'c':
        types = None
    elif color_by in adata.obs and is_numeric_dtype(adata.obs[color_by]):
        types = None
        colors = adata.obs[color_by].values
    elif color_by in adata.obs and is_categorical_dtype(adata.obs[color_by]) and color_by+'_colors' in adata.uns.keys():
        types = adata.obs[color_by].cat.categories
        colors = adata.uns[f'{color_by}_colors']
    else:
        raise ValueError('Currently, color key must be a single string of either numerical or categorical available in adata obs, '
                         'and the colors of categories can be found in adata uns.\nYou can use scvelo.pl.scatter directly for more comprehensive plots.')

    downsample = np.clip(int(downsample), 1, 10)
    genes = np.array(genes)
    missing_genes = genes[~np.isin(genes, adata.var_names)]
    if len(missing_genes) > 0:
        logger.warning(f'{missing_genes} not found')
    genes = genes[np.isin(genes, adata.var_names)]
    gn = len(genes)
    if gn == 0:
        return
    if gn < n_cols:
        n_cols = gn
    no_c = False
    if adata_atac is None or 'Mc' not in adata_atac.layers.keys():
        no_c = True
        by = 'us'
    if batch_correction:
        show_pred_only = True
    if show_pred_only:
        show_pred = False

    if by == 'cus':
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False, figsize=(3.2*n_cols, 2.7*(-(-gn // n_cols))) if figsize is None else figsize, subplot_kw={'projection': '3d'})
    else:
        fig, axs = plt.subplots(-(-gn // n_cols), n_cols, squeeze=False, figsize=(2.7*n_cols, 2.4*(-(-gn // n_cols))) if figsize is None else figsize)
    fig.patch.set_facecolor('white')
    count = 0
    for gene in genes:
        if show_pred_only:
            if modalities_pred is not None:
                u = adata[:, gene].layers[modalities_pred[-2]].copy()
                s = adata[:, gene].layers[modalities_pred[-1]].copy()
            else:
                u = adata[:, gene].layers[f'{key}_uhat'].copy()
                s = adata[:, gene].layers[f'{key}_shat'].copy()
        else:
            if modalities is not None:
                u = adata[:, gene].layers[modalities[-2]].copy()
                s = adata[:, gene].layers[modalities[-1]].copy()
            else:
                u = adata[:, gene].layers['Mu'].copy() if 'Mu' in adata.layers else adata[:, gene].layers['unspliced'].copy()
                s = adata[:, gene].layers['Ms'].copy() if 'Ms' in adata.layers else adata[:, gene].layers['spliced'].copy()
        u = u.toarray() if sparse.issparse(u) else u
        s = s.toarray() if sparse.issparse(s) else s
        u, s = np.ravel(u), np.ravel(s)
        if not no_c:
            if show_pred_only:
                if modalities_pred is not None:
                    c = adata[:, gene].layers[modalities_pred[0]].copy()
                else:
                    c = adata[:, gene].layers[f'{key}_chat'].copy()
            else:
                if modalities is not None:
                    if modalities[0] in adata.layers.keys():
                        c = adata[:, gene].layers[modalities[0]].copy()
                    else:
                        c = adata_atac[:, gene].layers[modalities[0]].copy()
                else:
                    c = adata_atac[:, gene].layers['Mc'].copy()
            c = c.toarray() if sparse.issparse(c) else c
            c = np.ravel(c)

        if velocity_arrows:
            vu = adata[:, gene].layers[f'{key}_velocity_u'].copy()
            max_u = np.max([np.max(u), 1e-6])
            u /= max_u
            vu = np.ravel(vu)
            vu /= np.max([np.max(np.abs(vu)), 1e-6])
            vs = adata[:, gene].layers[f'{key}_velocity'].copy()
            max_s = np.max([np.max(s), 1e-6])
            s /= max_s
            vs = np.ravel(vs)
            vs /= np.max([np.max(np.abs(vs)), 1e-6])
            if not no_c:
                vc = adata[:, gene].layers[f'{key}_velocity_c'].copy()
                max_c = np.max([np.max(c), 1e-6])
                c /= max_c
                vc = np.ravel(vc)
                vc /= np.max([np.max(np.abs(vc)), 1e-6])

        row = count // n_cols
        col = count % n_cols
        ax = axs[row, col]
        if types is not None:
            for i in range(len(types)):
                filt = adata.obs[color_by] == types[i]
                filt = np.ravel(filt)
                if by == 'us':
                    if velocity_arrows:
                        ax.quiver(s[filt][::downsample], u[filt][::downsample], vs[filt][::downsample], vu[filt][::downsample],
                                  color=colors[i], alpha=0.5, scale_units='xy', scale=10, width=0.005, headwidth=4, headaxislength=5.5)
                    else:
                        ax.scatter(s[filt][::downsample], u[filt][::downsample], s=pointsize, c=colors[i], alpha=0.7)
                elif by == 'cu':
                    if velocity_arrows:
                        ax.quiver(u[filt][::downsample], c[filt][::downsample], vu[filt][::downsample], vc[filt][::downsample],
                                  color=colors[i], alpha=0.5, scale_units='xy', scale=10, width=0.005, headwidth=4, headaxislength=5.5)
                    else:
                        ax.scatter(u[filt][::downsample], c[filt][::downsample], s=pointsize, c=colors[i], alpha=0.7)
                else:
                    if velocity_arrows:
                        ax.quiver(s[filt][::downsample], u[filt][::downsample], c[filt][::downsample],
                                  vs[filt][::downsample], vu[filt][::downsample], vc[filt][::downsample],
                                  color=colors[i], alpha=0.4, length=0.1, arrow_length_ratio=0.5, normalize=True)
                    else:
                        ax.scatter(s[filt][::downsample], u[filt][::downsample], c[filt][::downsample], s=pointsize, c=colors[i], alpha=0.7)
        elif color_by == 'c':
            outlier = 99.8
            non_zero = (u > 0) & (s > 0) & (c > 0)
            non_outlier = u < np.percentile(u, outlier)
            non_outlier &= s < np.percentile(s, outlier)
            non_outlier &= c < np.percentile(c, outlier)
            c -= np.min(c)
            c /= np.max(c)
            if velocity_arrows:
                ax.quiver(s[non_zero & non_outlier][::downsample], u[non_zero & non_outlier][::downsample],
                          vs[non_zero & non_outlier][::downsample], vu[non_zero & non_outlier][::downsample],
                          np.log1p(c[non_zero & non_outlier][::downsample]), alpha=0.5,
                          scale_units='xy', scale=10, width=0.005, headwidth=4, headaxislength=5.5, cmap=cmap)
            else:
                ax.scatter(s[non_zero & non_outlier][::downsample], u[non_zero & non_outlier][::downsample], s=pointsize,
                           c=np.log1p(c[non_zero & non_outlier][::downsample]), alpha=0.8, cmap=cmap)
        else:
            if by == 'us':
                if velocity_arrows:
                    ax.quiver(s[::downsample], u[::downsample], vs[::downsample], vu[::downsample],
                              colors[::downsample], alpha=0.5, scale_units='xy', scale=10, width=0.005, headwidth=4, headaxislength=5.5, cmap=cmap)
                else:
                    ax.scatter(s[::downsample], u[::downsample], s=pointsize, c=colors[::downsample], alpha=0.7, cmap=cmap)
            elif by == 'cu':
                if velocity_arrows:
                    ax.quiver(u[::downsample], c[::downsample], vu[::downsample], vc[::downsample],
                              colors[::downsample], alpha=0.5, scale_units='xy', scale=10, width=0.005, headwidth=4, headaxislength=5.5, cmap=cmap)
                else:
                    ax.scatter(u[::downsample], c[::downsample], s=pointsize, c=colors[::downsample], alpha=0.7, cmap=cmap)
            else:
                if velocity_arrows:
                    ax.quiver(s[::downsample], u[::downsample], c[::downsample],
                              vs[::downsample], vu[::downsample], vc[::downsample],
                              colors[::downsample], alpha=0.4, length=0.1, arrow_length_ratio=0.5, normalize=True, cmap=cmap)
                else:
                    ax.scatter(s[::downsample], u[::downsample], c[::downsample], s=pointsize, c=colors[::downsample], alpha=0.7, cmap=cmap)

        if show_pred:
            if not no_c:
                if modalities_pred is not None:
                    a_c = adata[:, gene].layers[modalities_pred[0]].ravel()
                else:
                    a_c = adata[:, gene].layers[f'{key}_chat'].ravel()
            if modalities_pred is not None:
                a_u = adata[:, gene].layers[modalities_pred[-2]].ravel()
                a_s = adata[:, gene].layers[modalities_pred[-1]].ravel()
            else:
                a_u = adata[:, gene].layers[f'{key}_uhat'].ravel()
                a_s = adata[:, gene].layers[f'{key}_shat'].ravel()
            if velocity_arrows:
                if not no_c:
                    a_c /= max_c
                a_u /= max_u
                a_s /= max_s
            if by == 'us':
                ax.scatter(a_s[::downsample], a_u[::downsample], s=pointsize/2, c='black', alpha=0.2, zorder=1000)
            elif by == 'cu':
                ax.scatter(a_u[::downsample], a_c[::downsample], s=pointsize/2, c='black', alpha=0.2, zorder=1000)
            else:
                ax.scatter(a_s[::downsample], a_u[::downsample], a_c[::downsample], s=pointsize/2, c='black', alpha=0.2, zorder=1000)

        if by == 'cus' and (view_3d_elev is not None or view_3d_azim is not None):
            # US: elev=90, azim=270. CU: elev=0, azim=0.
            ax.view_init(elev=view_3d_elev, azim=view_3d_azim)
        title = gene
        if title_more_info:
            if f'{key}_likelihood' in adata.var and not np.isnan(adata[:, gene].var[f'{key}_likelihood']):
                title += f" {adata[:, gene].var[f'{key}_likelihood'].values[0]:.3g}"
        ax.set_title(f'{title}', fontsize=fontsize)
        if by == 'us':
            ax.set_xlabel('spliced' if full_name else 's')
            ax.set_ylabel('unspliced' if full_name else 'u')
        elif by == 'cu':
            ax.set_xlabel('unspliced' if full_name else 'u')
            ax.set_ylabel('chromatin' if full_name else 'c')
        elif by == 'cus':
            ax.set_xlabel('spliced' if full_name else 's')
            ax.set_ylabel('unspliced' if full_name else 'u')
            ax.set_zlabel('chromatin' if full_name else 'c')
        if by in ['us', 'cu']:
            if not axis_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if not frame_on:
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.set_frame_on(False)
        elif by == 'cus':
            if not axis_on:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.zaxis.set_ticklabels([])
            if not frame_on:
                ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
                ax.xaxis._axinfo['tick']['inward_factor'] = 0
                ax.xaxis._axinfo['tick']['outward_factor'] = 0
                ax.yaxis._axinfo['tick']['inward_factor'] = 0
                ax.yaxis._axinfo['tick']['outward_factor'] = 0
                ax.zaxis._axinfo['tick']['inward_factor'] = 0
                ax.zaxis._axinfo['tick']['outward_factor'] = 0
        count += 1
    for i in range(col+1, n_cols):
        fig.delaxes(axs[row, i])
    fig.tight_layout()


def difference(v1, v2, norm=0.1, eps=1e-8):
    return (v1 - v2) / (norm + eps)


def fold_change(v1, v2, eps=1e-8):
    return v1 / (v2 + eps)


# Taken from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
def colored_line_between_pts(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified between (x, y) points by a third value.

    It does this by creating a collection of line segments between each pair of
    neighboring points. The color of each segment is determined by the
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should have a size one less than that of x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    from matplotlib.collections import LineCollection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, **lc_kwargs)

    # Set the values used for colormapping
    lc.set_array(c)

    return ax.add_collection(lc)


def differential_dynamics_plot(adata,
                               genes,
                               group1=None,
                               group2=None,
                               var='v',
                               signed_velocity=True,
                               color_by=None,
                               color_include=None,
                               t_min=None,
                               t_max=None,
                               key='vae',
                               n_bins=50,
                               n_samples=100,
                               seed=0,
                               kernel='RBF',
                               p_value=True,
                               n_cols=5,
                               figsize=None,
                               plot_equal=True,
                               axis_on=True,
                               frame_on=True,
                               title=True,
                               legend=True):
    """Plot differential dynamics between two groups of cells.

    Args:
        adata (:class:`anndata.AnnData`):
            Anndata object with differential dynamics results stored in `adata.uns['differential_dynamics']`.
        genes (str, list of str):
            List of genes to plot.
        group1 (str, optional):
            Name of group 1. Defaults to None.
        group2 (str, optional):
            Name of group 2. Defaults to None.
        var (str, optional):
            Variable to plot. Defaults to velocity 'v'.
        signed_velocity (bool, optional):
            Whether to use original velocity values or absolute values. Defaults to True.
        color_by: (str, optional):
            Color used for the plots. Defaults to None.
        color_include (list, optional):
            List of categories such as cell types to include in the plot. Defaults to None.
        t_min (float, optional):
            Minimum time point to include in the plot. Defaults to None.
        t_max (float, optional):
            Maximum time point to include in the plot. Defaults to None.
        key (str, optional):
            Key to find VAE variables. Defaults to `vae`.
        n_bins (int, optional):
            Number of bins to divide the time points. Defaults to 50.
        n_samples (int, optional):
            Number of data points to permute in each bin for each group. Defaults to 100.
        seed (int, optional):
            Seed for random generator. Defaults to 0.
        kernel (str, optional):
            Kernel to use for Gaussian Process regression. Defaults to 'RBF'.
        p_value (bool, optional):
            Whether to show p-value in the plot. Defaults to True.
        n_cols (int, optional):
            Number of columns to plot. Defaults to 5.
        figsize (tuple, optional):
            Total figure size. Defaults to None.
        plot_equal (bool, optional):
            Whether to plot equal time points. Defaults to True.
        axis_on (bool, optional):
            Whether to show axis labels. Defaults to True.
        frame_on (bool, optional):
            Whether to show plot frames. Defaults to True.
        title (bool, optional):
            Whether to show title. Defaults to True.
        legend (bool, optional):
            Whether to show legend. Defaults to True.
    """
    eps = 1e-8
    if isinstance(genes, str) or isinstance(genes, int):
        genes = [genes]
    if group1 is None:
        group1 = '1'
    if group2 is None:
        group2 = '2'
    if var not in ['kc', 'rho', 'c', 'u', 's', 'v']:
        raise ValueError(f"Variable {var} not recognized. Must be one of ['kc', 'rho', 'c', 'u', 's', 'v'].")
    default_func = {'kc': 'ld',
                    'rho': 'ld',
                    'c': 'lfc',
                    'u': 'lfc',
                    's': 'lfc',
                    'v': 'ld' if signed_velocity else 'lfc'}
    func = default_func[var]
    if f'{var}_{group1}' not in adata.uns['differential_dynamics'].keys():
        raise ValueError(f"{var}_{group1} not found in adata.varm. Was differential_dynamics run with save_raw?")
    if f'{var}_{group2}' not in adata.uns['differential_dynamics'].keys():
        raise ValueError(f"{var}_{group2} not found in adata.varm. Was differential_dynamics run with save_raw?")
    var_g1 = adata.uns['differential_dynamics'][f'{var}_{group1}']
    var_g2 = adata.uns['differential_dynamics'][f'{var}_{group2}']
    t1 = adata.uns['differential_dynamics'][f't_{group1}']
    t2 = adata.uns['differential_dynamics'][f't_{group2}']
    t_both = np.concatenate([t1, t2])
    steps = np.quantile(t_both, np.linspace(0, 1, n_bins + 1))
    steps = steps[1:-1]
    t_bins = np.digitize(t_both, steps)
    t1_bins = np.digitize(t1, steps)
    t2_bins = np.digitize(t2, steps)
    if func == 'ld':
        if var == 'v':
            mean_norm = np.mean(adata.uns['differential_dynamics'][f's_{group2}'], 0)
        else:
            mean_norm = None

    if color_by is not None:
        from pandas.api.types import is_categorical_dtype
        if color_by not in adata.obs:
            raise ValueError(f"Color key {color_by} not found in adata.obs.")
        elif not is_categorical_dtype(adata.obs[color_by]) or color_by+'_colors' not in adata.uns.keys():
            raise ValueError(f"Color key {color_by} must be a categorical variable with colors stored in adata.uns.")
        else:
            types = adata.obs[color_by].cat.categories
            colors = adata.uns[f'{color_by}_colors']
            colors_dict = dict(zip(types, colors))
            if color_include is None:
                color_include = types
            cell_time = adata.obs[f'{key}_time'].values[np.isin(adata.obs[color_by].values, color_include)]
            color_array = np.array([colors_dict[x] for x in adata.obs[color_by].values if x in color_include])
            color_array = color_array[np.argsort(cell_time)]
            cell_time = np.sort(cell_time)

    fig, axs = plt.subplots(-(-len(genes) // n_cols), min(n_cols, len(genes)), squeeze=False, figsize=(5*min(n_cols, len(genes)), 4.4*(-(-len(genes) // n_cols))) if figsize is None else figsize)
    fig.patch.set_facecolor('white')
    count = 0
    for gene in genes:
        rng = np.random.default_rng(seed=seed)
        gene_idx = adata.var_names == gene
        var_g1_gene = var_g1[:, gene_idx]
        var_g2_gene = var_g2[:, gene_idx]
        if func == 'ld':
            mean_norm_gene = mean_norm[gene_idx] if mean_norm is not None else 1

        time_array, dd_array, bf_array = [], [], []
        for i in range(n_bins):
            time_bin = np.mean(t_both[t_bins == i])
            var_g1_bin = var_g1_gene[t1_bins == i]
            var_g2_bin = var_g2_gene[t2_bins == i]
            if len(var_g1_bin) < 10 or len(var_g2_bin) < 10:
                continue
            time_array.append(time_bin)
            var_g1_bin_perm = rng.choice(var_g1_bin, n_samples)
            var_g2_bin_perm = rng.choice(var_g2_bin, n_samples)
            if func == 'lfc':
                fc_bin = fold_change(np.abs(var_g1_bin_perm), np.abs(var_g2_bin_perm), eps)
                dd_array.append(np.mean(fc_bin))
            else:
                diff_bin = difference(var_g1_bin_perm, var_g2_bin_perm, mean_norm_gene)
                dd_array.append(np.mean(diff_bin))
            p1 = np.mean(var_g1_bin_perm > var_g2_bin_perm)
            bf_array.append(np.log(p1 + eps) - np.log(1 - p1 + eps))

        time_array = np.array(time_array)
        dd_array = np.array(dd_array)
        bounds = np.quantile(t_both, [0.005, 0.995])
        t_both_sorted = np.sort(t_both)
        t_both_sorted = t_both_sorted[(t_both_sorted >= bounds[0]) & (t_both_sorted <= bounds[1])]

        if kernel == 'RBF':
            kernel_ = 1.0 * RBF(1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
        elif kernel == 'ExpSineSquared':
            kernel_ = 1.0 * ExpSineSquared(1.0, 1.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
        elif kernel == 'RationalQuadratic':
            kernel_ = 1.0 * RationalQuadratic(1.0, 1.0, length_scale_bounds=(0.1, 10.0), alpha_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
        else:
            raise ValueError(f"Kernel {kernel} not supported. Must be one of ['RBF', 'ExpSineSquared', 'RationalQuadratic'].")
        gaussian_process = GaussianProcessRegressor(kernel=kernel_, random_state=seed, n_restarts_optimizer=10)
        gaussian_process.fit(time_array.reshape(-1, 1), dd_array.reshape(-1, 1))
        mean_prediction, std_prediction = gaussian_process.predict(t_both_sorted.reshape(-1, 1), return_std=True)
        ll = gaussian_process.log_marginal_likelihood(gaussian_process.kernel_.theta)

        if p_value:
            const = 0.0 if func == 'ld' else 1.0
            kernel_constant = ConstantKernel(const, constant_value_bounds='fixed') + WhiteKernel(0.1)
            gaussian_process_ = GaussianProcessRegressor(kernel=kernel_constant, random_state=seed, n_restarts_optimizer=10)
            gaussian_process_.fit(time_array.reshape(-1, 1), dd_array.reshape(-1, 1))
            ll_null = gaussian_process_.log_marginal_likelihood(gaussian_process_.kernel_.theta)
            lrt = -2 * (ll_null - ll)
            pval = chi2.sf(lrt, 1)

        row = count // n_cols
        col = count % n_cols
        ax = axs[row, col]

        filt = None
        if t_min is not None and t_max is not None:
            filt = (t_both_sorted > t_min) & (t_both_sorted < t_max)
        elif t_min is not None:
            filt = t_both_sorted > t_min
        elif t_max is not None:
            filt = t_both_sorted < t_max
        if np.all(np.isclose(bf_array, bf_array[0])):
            if filt is not None:
                ax.plot(t_both_sorted[filt], mean_prediction[filt], label="Mean prediction", color='black', alpha=0.6)
            else:
                ax.plot(t_both_sorted, mean_prediction, label="Mean prediction", color='black', alpha=0.6)
        else:
            bf_array_mid = (np.array(bf_array)[:-1] + np.array(bf_array)[1:]) / 2
            bf_array_mid = np.concatenate([[bf_array_mid[0]], bf_array_mid, [bf_array_mid[-1]]])
            t_sorted_bins = np.digitize(t_both_sorted, time_array)
            if filt is not None:
                colored_line_between_pts(t_both_sorted[filt], mean_prediction[filt], bf_array_mid[t_sorted_bins][filt], ax, linewidth=5, cmap="viridis", label='Mean prediction\ncolored by BF')
            else:
                colored_line_between_pts(t_both_sorted, mean_prediction, bf_array_mid[t_sorted_bins], ax, linewidth=5, cmap="viridis", label='Mean prediction\ncolored by BF')
        if plot_equal:
            if color_by is None:
                ax.plot(t_both_sorted,
                        np.zeros_like(t_both_sorted) if func == 'ld' else np.ones_like(t_both_sorted),
                        label='Zero line' if func == 'ld' else 'One line',
                        linestyle='--', color='black', alpha=0.6)
            else:
                ax.scatter(cell_time,
                           np.zeros_like(cell_time) if func == 'ld' else np.ones_like(cell_time),
                           label=f"Zero line\ncolored by {color_by}" if func == 'ld' else f"One line\ncolored by {color_by}",
                           c=color_array, s=20, marker='|')

        if filt is not None:
            ax.fill_between(
                t_both_sorted[filt],
                mean_prediction[filt] - 1.96 * std_prediction[filt],
                mean_prediction[filt] + 1.96 * std_prediction[filt],
                alpha=0.4,
                label="Credible interval",
                facecolor='gray'
            )
        else:
            ax.fill_between(
                t_both_sorted,
                mean_prediction - 1.96 * std_prediction,
                mean_prediction + 1.96 * std_prediction,
                alpha=0.4,
                label="Credible interval",
                facecolor='gray'
            )
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{'Difference' if func == 'ld' else 'Fold change'}")
        if p_value:
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.75, 1.06, f'Pval={pval:.2e}', transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
        ax.set_title(f'{gene}', fontsize=11)
        if not axis_on:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        if not frame_on:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.set_frame_on(False)
        count += 1
    if len(genes) > n_cols:
        for i in range(col+1, n_cols):
            fig.delaxes(axs[row, i])
    if title:
        plt.suptitle(f"Gaussian process regression on differential dynamics on {var}", fontsize=12+2*min(n_cols, len(genes)), y=1.01)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(-np.log10(min(n_cols, len(genes)))/2+1.5, 1),  # (1.6-0.1*min(n_cols, len(genes)), 0.8),
                   fontsize=11)
    fig.tight_layout()


def decoupling_plot(adata,
                    genes,
                    group1=None,
                    group2=None,
                    absolute=False,
                    show_coupling=True,
                    show_decoupling=True,
                    color_by=None,
                    color_include=None,
                    key='vae',
                    n_bins=50,
                    n_samples=100,
                    seed=0,
                    use_gp=False,
                    kernel='RBF',
                    n_cols=5,
                    figsize=None,
                    axis_on=True,
                    frame_on=True,
                    title=True,
                    legend=True):
    """Plot differential dynamics between two groups of cells.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object with differential dynamics results.
        genes (str or list of str):
            List of genes to plot.
        group1 (str, optional):
            Name of group 1. Defaults to None.
        group2 (str, optional):
            Name of group 2. Defaults to None.
        absolute (bool, optional):
            Whether to plot absolute (de)coupling values. Defaults to False.
        show_coupling (bool, optional):
            Whether to show coupling values in the plot. Defaults to True.
        show_decoupling (bool, optional):
            Whether to show decoupling values in the plot. Defaults to True.
        color_by (str, optional):
            Key in `adata.obs` to color the points by. Defaults to None.
        color_include (list, optional):
            List of categories to include in the color_by variable. Defaults to None.
        key (str, optional):
            Key to find VAE variables. Defaults to `vae`.
        n_bins (int, optional):
            Number of bins to divide the time points. Defaults to 50.
        n_samples (int, optional):
            Number of data points to permute in each bin for each group. Defaults to 100.
        seed (int, optional):
            Seed for random generator. Defaults to 0.
        n_cols (int, optional):
            Number of columns to plot. Defaults to 5.
        figsize (tuple, optional):
            Total figure size. Defaults to None.
        axis_on (bool, optional):
            Whether to show axis labels. Defaults to True.
        frame_on (bool, optional):
            Whether to show plot frames. Defaults to True.
        title (bool, optional):
            Whether to show title. Defaults to True.
        legend (bool, optional):
            Whether to show legend. Defaults to True.
    """
    lowess = sm.nonparametric.lowess
    if isinstance(genes, str) or isinstance(genes, int):
        genes = [genes]
    if group1 is None:
        group1 = '1'
    if group2 is None:
        group2 = '2'
    if f'decoupling_{group1}' not in adata.uns['differential_dynamics'].keys():
        raise ValueError(f"decoupling_{group1} not found in adata.varm.\nWas differential_dynamics run with save_raw and test_decoupling=True?")
    if f'decoupling_{group2}' not in adata.uns['differential_dynamics'].keys():
        raise ValueError(f"decoupling_{group2} not found in adata.varm.\nWas differential_dynamics run with save_raw and test_decoupling=True?")
    decoupling_g1 = adata.uns['differential_dynamics'][f'decoupling_{group1}']
    decoupling_g2 = adata.uns['differential_dynamics'][f'decoupling_{group2}']
    coupling_g1 = adata.uns['differential_dynamics'][f'coupling_{group1}']
    coupling_g2 = adata.uns['differential_dynamics'][f'coupling_{group2}']

    t1 = adata.uns['differential_dynamics'][f't_{group1}']
    t2 = adata.uns['differential_dynamics'][f't_{group2}']
    t_both = np.concatenate([t1, t2])
    steps = np.quantile(t_both, np.linspace(0, 1, n_bins + 1))
    steps = steps[1:-1]
    t_bins = np.digitize(t_both, steps)
    t1_bins = np.digitize(t1, steps)
    t2_bins = np.digitize(t2, steps)

    if color_by is not None:
        from pandas.api.types import is_categorical_dtype
        if color_by not in adata.obs:
            raise ValueError(f"Color key {color_by} not found in adata.obs.")
        elif not is_categorical_dtype(adata.obs[color_by]) or color_by+'_colors' not in adata.uns.keys():
            raise ValueError(f"Color key {color_by} must be a categorical variable with colors stored in adata.uns.")
        else:
            types = adata.obs[color_by].cat.categories
            colors = adata.uns[f'{color_by}_colors']
            colors_dict = dict(zip(types, colors))
            if color_include is None:
                color_include = types
            cell_time = adata.obs[f'{key}_time'].values[np.isin(adata.obs[color_by].values, color_include)]
            color_array = np.array([colors_dict[x] for x in adata.obs[color_by].values if x in color_include])
            color_array = color_array[np.argsort(cell_time)]
            cell_time = np.sort(cell_time)

    fig, axs = plt.subplots(-(-len(genes) // n_cols), min(n_cols, len(genes)), squeeze=False, figsize=(5*min(n_cols, len(genes)), 4.4*(-(-len(genes) // n_cols))) if figsize is None else figsize)
    fig.patch.set_facecolor('white')
    count = 0
    for gene in genes:
        rng = np.random.default_rng(seed=seed)
        gene_idx = adata.var_names == gene
        decoupling_g1_gene = decoupling_g1[:, gene_idx]
        decoupling_g2_gene = decoupling_g2[:, gene_idx]

        time_array, decoupling_array, coupling_array = [], [], []
        for i in range(n_bins):
            time_bin = np.mean(t_both[t_bins == i])
            decoupling_g1_bin = decoupling_g1_gene[t1_bins == i]
            decoupling_g2_bin = decoupling_g2_gene[t2_bins == i]
            if len(decoupling_g1_bin) < 10 or len(decoupling_g2_bin) < 10:
                continue
            coupling_g1_bin = coupling_g1[:, gene_idx][t1_bins == i]
            coupling_g2_bin = coupling_g2[:, gene_idx][t2_bins == i]
            if len(coupling_g1_bin) < 10 or len(coupling_g2_bin) < 10:
                continue
            time_array.append(time_bin)
            decoupling_g1_bin_perm = rng.choice(decoupling_g1_bin, n_samples)
            decoupling_g2_bin_perm = rng.choice(decoupling_g2_bin, n_samples)
            decoupling_array.append(np.mean(np.concatenate([decoupling_g1_bin_perm, decoupling_g2_bin_perm])))
            coupling_g1_bin_perm = rng.choice(coupling_g1_bin, n_samples)
            coupling_g2_bin_perm = rng.choice(coupling_g2_bin, n_samples)
            coupling_array.append(np.mean(np.concatenate([coupling_g1_bin_perm, coupling_g2_bin_perm])))

        time_array = np.array(time_array)
        decoupling_array = np.array(decoupling_array)
        coupling_array = np.array(coupling_array)
        if absolute:
            decoupling_array = np.abs(decoupling_array)
            coupling_array = np.abs(coupling_array)

        row = count // n_cols
        col = count % n_cols
        ax = axs[row, col]

        if use_gp:
            bounds = np.quantile(t_both, [0.005, 0.995])
            t_both_sorted = np.sort(t_both)
            t_both_sorted = t_both_sorted[(t_both_sorted >= bounds[0]) & (t_both_sorted <= bounds[1])]
            if kernel == 'RBF':
                kernel_ = 1.0 * RBF(1.0, length_scale_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
            elif kernel == 'ExpSineSquared':
                kernel_ = 1.0 * ExpSineSquared(1.0, 1.0, length_scale_bounds=(0.1, 10.0), periodicity_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
            elif kernel == 'RationalQuadratic':
                kernel_ = 1.0 * RationalQuadratic(1.0, 1.0, length_scale_bounds=(0.1, 10.0), alpha_bounds=(0.1, 10.0)) + WhiteKernel(0.1)
            else:
                raise ValueError(f"Kernel {kernel} not supported. Must be one of ['RBF', 'ExpSineSquared', 'RationalQuadratic'].")
            gaussian_process = GaussianProcessRegressor(kernel=kernel_, random_state=seed, n_restarts_optimizer=10)
            gaussian_process.fit(time_array.reshape(-1, 1), decoupling_array.reshape(-1, 1))
            mean_prediction_decoupling, std_prediction_decoupling = gaussian_process.predict(t_both_sorted.reshape(-1, 1), return_std=True)
            gaussian_process = GaussianProcessRegressor(kernel=kernel_, random_state=seed, n_restarts_optimizer=10)
            gaussian_process.fit(time_array.reshape(-1, 1), coupling_array.reshape(-1, 1))
            mean_prediction_coupling, std_prediction_coupling = gaussian_process.predict(t_both_sorted.reshape(-1, 1), return_std=True)

            if show_decoupling:
                ax.plot(t_both_sorted, mean_prediction_decoupling, label="Mean prediction decoupling" + ('\n(absolute)' if absolute else ''),
                        color='black', linestyle='solid', linewidth=2)
                ax.fill_between(
                    t_both_sorted,
                    mean_prediction_decoupling - 1.96 * std_prediction_decoupling,
                    mean_prediction_decoupling + 1.96 * std_prediction_decoupling,
                    alpha=0.4,
                    label="Credible interval",
                    facecolor='gray'
                )
            if show_coupling:
                ax.plot(t_both_sorted, mean_prediction_coupling, label="Mean prediction coupling" + ('\n(absolute)' if absolute else ''),
                        color='black', linestyle='dashed', linewidth=2)
                ax.fill_between(
                    t_both_sorted,
                    mean_prediction_coupling - 1.96 * std_prediction_coupling,
                    mean_prediction_coupling + 1.96 * std_prediction_coupling,
                    alpha=0.2,
                    label="Credible interval",
                    facecolor='gray'
                )
        else:
            decoupling_smooth = lowess(decoupling_array, time_array, frac=0.3)[:, 1]
            coupling_smooth = lowess(coupling_array, time_array, frac=0.3)[:, 1]
            decoupling_smooth_pos_idx = np.where(decoupling_smooth > 0)[0]
            breaks = np.where(np.diff(decoupling_smooth_pos_idx) > 1)[0] + 1
            if len(breaks) > 0:
                decoupling_smooth_pos_idx = np.array_split(decoupling_smooth_pos_idx, breaks)
                for i, x in enumerate(decoupling_smooth_pos_idx[:-1]):
                    decoupling_smooth_pos_idx[i] = np.append(x, x[-1] + 1)
                decoupling_smooth_pos_idx[-1] = np.append([decoupling_smooth_pos_idx[-1][0]-1], decoupling_smooth_pos_idx[-1])
            else:
                if len(decoupling_smooth_pos_idx) > 0 and np.min(decoupling_smooth_pos_idx) > 0:
                    decoupling_smooth_pos_idx = np.append([decoupling_smooth_pos_idx[0]-1], decoupling_smooth_pos_idx)
                if len(decoupling_smooth_pos_idx) > 0 and np.max(decoupling_smooth_pos_idx) < len(decoupling_smooth) - 1:
                    decoupling_smooth_pos_idx = np.append(decoupling_smooth_pos_idx, [decoupling_smooth_pos_idx[-1] + 1])
                decoupling_smooth_pos_idx = [decoupling_smooth_pos_idx]
            coupling_smooth_pos_idx = np.where(coupling_smooth > 0)[0]
            breaks = np.where(np.diff(coupling_smooth_pos_idx) > 1)[0] + 1
            if len(breaks) > 0:
                coupling_smooth_pos_idx = np.array_split(coupling_smooth_pos_idx, breaks)
                for i, x in enumerate(coupling_smooth_pos_idx[:-1]):
                    coupling_smooth_pos_idx[i] = np.append(x, x[-1] + 1)
            else:
                if len(coupling_smooth_pos_idx) > 0 and np.min(coupling_smooth_pos_idx) > 0:
                    coupling_smooth_pos_idx = np.append([coupling_smooth_pos_idx[0]-1], coupling_smooth_pos_idx)
                if len(coupling_smooth_pos_idx) > 0 and np.max(coupling_smooth_pos_idx) < len(coupling_smooth) - 1:
                    coupling_smooth_pos_idx = np.append(coupling_smooth_pos_idx, [coupling_smooth_pos_idx[-1] + 1])
                coupling_smooth_pos_idx = [coupling_smooth_pos_idx]
            decoupling_smooth_neg_idx = np.where(decoupling_smooth < 0)[0]
            breaks = np.where(np.diff(decoupling_smooth_neg_idx) > 1)[0] + 1
            if len(breaks) > 0:
                decoupling_smooth_neg_idx = np.array_split(decoupling_smooth_neg_idx, breaks)
                for i, x in enumerate(decoupling_smooth_neg_idx[:-1]):
                    decoupling_smooth_neg_idx[i] = np.append(x, x[-1] + 1)
            else:
                if len(decoupling_smooth_neg_idx) > 0 and np.min(decoupling_smooth_neg_idx) > 0:
                    decoupling_smooth_neg_idx = np.append([decoupling_smooth_neg_idx[0]-1], decoupling_smooth_neg_idx)
                if len(decoupling_smooth_neg_idx) > 0 and np.max(decoupling_smooth_neg_idx) < len(decoupling_smooth) - 1:
                    decoupling_smooth_neg_idx = np.append(decoupling_smooth_neg_idx, [decoupling_smooth_neg_idx[-1] + 1])
                decoupling_smooth_neg_idx = [decoupling_smooth_neg_idx]
            coupling_smooth_neg_idx = np.where(coupling_smooth < 0)[0]
            breaks = np.where(np.diff(coupling_smooth_neg_idx) > 1)[0] + 1
            if len(breaks) > 0:
                coupling_smooth_neg_idx = np.array_split(coupling_smooth_neg_idx, breaks)
                for i, x in enumerate(coupling_smooth_neg_idx[:-1]):
                    coupling_smooth_neg_idx[i] = np.append(x, x[-1] + 1)
            else:
                if len(coupling_smooth_neg_idx) > 0 and np.min(coupling_smooth_neg_idx) > 0:
                    coupling_smooth_neg_idx = np.append([coupling_smooth_neg_idx[0]-1], coupling_smooth_neg_idx)
                if len(coupling_smooth_neg_idx) > 0 and np.max(coupling_smooth_neg_idx) < len(coupling_smooth) - 1:
                    coupling_smooth_neg_idx = np.append(coupling_smooth_neg_idx, [coupling_smooth_neg_idx[-1] + 1])
                coupling_smooth_neg_idx = [coupling_smooth_neg_idx]

            if show_decoupling:
                label_ = True
                for arr in decoupling_smooth_pos_idx:
                    if label_:
                        ax.plot(time_array[arr], decoupling_smooth[arr], label="Decoupled" if absolute else "Decoupled kc > rho", color='tab:blue', linestyle='solid', linewidth=2)
                    else:
                        ax.plot(time_array[arr], decoupling_smooth[arr], label="", color='tab:blue', linestyle='solid', linewidth=2)
                    label_ = False
            if show_coupling:
                label_ = True
                for arr in coupling_smooth_pos_idx:
                    if label_:
                        ax.plot(time_array[arr], coupling_smooth[arr], label="Coupled" if absolute else "Coupled induction", color='tab:orange', linestyle='dashed', linewidth=2)
                    else:
                        ax.plot(time_array[arr], coupling_smooth[arr], label="", color='tab:orange', linestyle='dashed', linewidth=2)
                    label_ = False
            if not absolute:
                if show_decoupling:
                    label_ = True
                    for arr in decoupling_smooth_neg_idx:
                        if label_:
                            ax.plot(time_array[arr], decoupling_smooth[arr], label="Decoupled kc < rho", color='tab:green', linestyle='solid', linewidth=2)
                        else:
                            ax.plot(time_array[arr], decoupling_smooth[arr], label="", color='tab:green', linestyle='solid', linewidth=2)
                        label_ = False
                if show_coupling:
                    label_ = True
                    for arr in coupling_smooth_neg_idx:
                        if label_:
                            ax.plot(time_array[arr], coupling_smooth[arr], label="Coupled repression", color='tab:red', linestyle='dashed', linewidth=2)
                        else:
                            ax.plot(time_array[arr], coupling_smooth[arr], label="", color='tab:red', linestyle='dashed', linewidth=2)
                        label_ = False

        cell_filt = (cell_time >= np.min(time_array)) & (cell_time <= np.max(time_array))
        cell_time_ = cell_time[cell_filt]
        ax.scatter(cell_time_, np.zeros_like(cell_time_), label=f"Zero line\ncolored by {color_by}", c=color_array[cell_filt], s=20, marker='|')

        ax.set_xlabel("Time")
        if show_decoupling and not show_coupling:
            y_text = "Decoupling"
        elif not show_decoupling and show_coupling:
            y_text = "Coupling"
        else:
            y_text = "(De)coupling"
        ax.set_ylabel(y_text + (" (absolute)" if absolute else ""))
        ax.set_title(f'{gene}', fontsize=11)
        if not axis_on:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        if not frame_on:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.set_frame_on(False)
        count += 1
    if len(genes) > n_cols:
        for i in range(col+1, n_cols):
            fig.delaxes(axs[row, i])
    if title:
        if show_decoupling and not show_coupling:
            title_text = "Decoupling factors"
        elif not show_decoupling and show_coupling:
            title_text = "Coupling factors"
        else:
            title_text = "Decoupling and coupling factors"
        plt.suptitle(title_text, fontsize=12+2*min(n_cols, len(genes)), y=1.01)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(-np.log10(min(n_cols, len(genes)))/2+1.45, 1),  # (1.6-0.15*min(n_cols, len(genes)), 1),
                   fontsize=11)
    fig.tight_layout()


# Modified from MultiVelo
def velocity_embedding_stream(adata, key='vae', show=True, **kwargs):
    """Plot velocity streamplot with `scvelo.pl.velocity_embedding_stream`.

    Args:
        adata (:class:`anndata.AnnData`):
            Anndata output from VAE inference.
        key (str, optional):
            Key to find layers. Defaults to 'vae'.
        show (bool):
            Whether to show the plot. Defaults to True.
        **kwargs:
            Additional parameters passed to `scvelo.pl.velocity_embedding_stream`.

    Returns
        if not show. A matplotlib axis object.
    """
    vkey = f'{key}_velocity'
    if vkey+'_norm' not in adata.layers.keys():
        adata.layers[vkey+'_norm'] = adata.layers[vkey] / np.sum(np.abs(adata.layers[vkey]), 0)
        adata.uns[vkey+'_norm_params'] = adata.uns[vkey+'_params']
    if vkey+'_norm_genes' not in adata.var.columns:
        adata.var[vkey+'_norm_genes'] = adata.var[vkey+'_genes']
    if vkey+'_norm_graph' not in adata.uns.keys():
        velocity_graph(adata, key=key, batch_corrected='s_leveled' in adata.layers.keys())
    out = scv.pl.velocity_embedding_stream(adata, vkey=vkey+'_norm', show=show, **kwargs)
    if not show:
        return out


def plot_train_loss_log(loss, iters, save=None):
    fig, ax = plt.subplots(facecolor='white', figsize=(12, 6))
    loss_min = np.min(loss)
    loss_log = np.log1p(loss - loss_min)
    ax.plot(iters, loss_log, '.-')
    ax.set_title(f"Training Loss Log (min={loss_min})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")

    save_fig(fig, save)


def plot_test_loss_log(loss, iters, color=None, save=None):
    fig, ax = plt.subplots(facecolor='white', figsize=(12, 6))
    loss_min = np.min(loss)
    loss_log = np.log1p(loss - loss_min)
    if color is not None:
        ax.plot(iters, loss_log, '-')
        ax.scatter(iters, loss_log, c=color)
    else:
        ax.plot(iters, loss_log, '.-')
    ax.set_title(f"Testing Loss Log (min={loss_min})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    save_fig(fig, save)
