import numpy as np
import matplotlib.pyplot as plt


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
            print("Warning: Number of colors exceeds the maximum (40)! Use a continuous colormap (256) instead.")
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
              type_specific=False,
              title='Gene',
              save=None,
              **kwargs):
    fig, ax = plt.subplots(3, 1, figsize=(15, 20), facecolor='white')
    D = kwargs['sparsify'] if 'sparsify' in kwargs else 1
    cell_types = np.unique(cell_labels)
    colors = get_colors(len(cell_types), None)
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
                           '-',
                           color=colors[i % len(colors)],
                           label=type_,
                           linewidth=1.5)
                ax[1].plot(tpred[mask_type][order],
                           upred[mask_type][order],
                           '-',
                           color=colors[i % len(colors)],
                           label=type_,
                           linewidth=1.5)
                ax[2].plot(tpred[mask_type][order],
                           spred[mask_type][order],
                           '-',
                           color=colors[i % len(colors)],
                           label=type_,
                           linewidth=1.5)
        else:
            order = np.argsort(tpred)
            ax[0].plot(tpred[order], cpred[order], 'k-', linewidth=1.5)
            ax[1].plot(tpred[order], upred[order], 'k-', linewidth=1.5)
            ax[2].plot(tpred[order], spred[order], 'k-', linewidth=1.5)

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

    lgd = fig.legend(handles, labels, fontsize=15, markerscale=5, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    fig.suptitle(title, fontsize=28)
    plt.tight_layout()

    save_fig(fig, save, (lgd,))


def plot_sig(t,
             c, u, s,
             cpred, upred, spred,
             cell_labels=None,
             title="Gene",
             save=None,
             **kwargs):
    D = kwargs['sparsify'] if 'sparsify' in kwargs else 1
    tscv = kwargs['tscv'] if 'tscv' in kwargs else t
    tdemo = kwargs["tdemo"] if "tdemo" in kwargs else t
    if cell_labels is None:
        fig, ax = plt.subplots(3, 1, figsize=(15, 20), facecolor='white')
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
        cell_types = np.unique(cell_labels)
        colors = get_colors(len(cell_types), None)

        # Plot the input data in the true labels
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
             t0,
             c0, u0, s0,
             dt=0.2,
             n_sample=10,
             title="Gene",
             save=None):
    fig, ax = plt.subplots(1, 3, figsize=(28, 6), facecolor='white')

    ax[0].plot(t, chat, '.', color='grey', alpha=0.1)
    ax[1].plot(t, uhat, '.', color='grey', alpha=0.1)
    ax[1].plot(t, shat, '.', color='grey', alpha=0.1)
    plot_indices = np.random.choice(len(t), n_sample, replace=False)
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
    fig.suptitle(title, fontsize=28)
    fig.legend(loc=1, fontsize=18)
    plt.tight_layout()

    save_fig(fig, save)


def plot_phase(c, u, s,
               cpred, upred, spred,
               title,
               by='us',
               track_idx=None,
               labels=None,
               types=None,
               save=None,
               plot_pred=True):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    if labels is None or types is None:
        if by == 'us':
            ax.scatter(s, u, c="b", alpha=0.5)
        else:
            ax.scatter(u, c, c="b", alpha=0.5)
    else:
        colors = get_colors(len(types), None)
        for i, type_ in enumerate(types):
            if by == 'us':
                ax.scatter(s[labels == i], u[labels == i], color=colors[i % len(colors)], alpha=0.3, label=type_)
            else:
                ax.scatter(u[labels == i], c[labels == i], color=colors[i % len(colors)], alpha=0.3, label=type_)
    if by == 'us':
        ax.plot(spred, upred, 'k.', label="ode")
    else:
        ax.plot(upred, cpred, 'k.', label="ode")

    if plot_pred:
        if track_idx is None:
            rng = np.random.default_rng()
            perm = rng.permutation(len(s))
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
                ax.plot(s_comb[i:i+2], u_comb[i:i+2], 'k-', linewidth=0.8)
            else:
                ax.plot(u_comb[i:i+2], c_comb[i:i+2], 'k-', linewidth=0.8)
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

    save_fig(fig, save, (lgd,))
