import numpy as np
import matplotlib
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
            ax[0].plot(tpred[order], cpred[order], 'k.')
            ax[1].plot(tpred[order], upred[order], 'k.')
            ax[2].plot(tpred[order], spred[order], 'k.')

        perm = np.random.permutation(len(s))
        Nsample = 100
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

    lgd = fig.legend(handles, labels, fontsize=15, markerscale=5, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    fig.suptitle(title, y=0.99, fontsize=20)
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
             n_sample=20,
             title="Gene",
             save=None):
    fig, ax = plt.subplots(1, 3, figsize=(28, 6), facecolor='white')

    ax[0].plot(t, chat, '.', color='grey', alpha=0.1)
    ax[1].plot(t, uhat, '.', color='grey', alpha=0.1)
    ax[2].plot(t, shat, '.', color='grey', alpha=0.1)
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
               t=None,
               track_idx=None,
               labels=None,
               types=None,
               save=None,
               plot_pred=True,
               show=False):
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
                if t is not None:
                    ax.scatter(s[labels == i], u[labels == i], color=colors[i % len(colors)], alpha=0.2, label=type_, s=10)
                else:
                    ax.scatter(s[labels == i], u[labels == i], color=colors[i % len(colors)], alpha=0.4, label=type_)
            else:
                if t is not None:
                    ax.scatter(u[labels == i], c[labels == i], color=colors[i % len(colors)], alpha=0.2, label=type_, s=10)
                else:
                    ax.scatter(u[labels == i], c[labels == i], color=colors[i % len(colors)], alpha=0.4, label=type_)
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
    fig, ax = plt.subplots(figsize=(8, 6))
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
    by_quantile = color_by == 'quantile'
    by_quantile_score = color_by == 'quantile_scores'
    if not by_quantile and not by_quantile_score:
        types = adata.obs[color_by].cat.categories
        colors = adata.uns[f'{color_by}_colors']
    gn = len(genes)
    if gn < n_cols:
        n_cols = gn
    fig, axs = plt.subplots(-(-gn // n_cols), n_cols, figsize=(2 * n_cols, 2.4 * (-(-gn // n_cols))) if figsize is None else figsize)
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
            fig.delaxes(axs[row, col])
            continue
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
            continue

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
