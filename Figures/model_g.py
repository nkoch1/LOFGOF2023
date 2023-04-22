import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import string
from plotstyle import scheme_style
import pandas as pd
from matplotlib import cm

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def show_spines(ax, spines='lrtb'):
    """ Show and hide spines.

    From github.com/janscience/plottools.git  spines.py

    Parameters
    ----------
    ax: matplotlib figure, matplotlib axis, or list of matplotlib axes
        Axis on which spine and ticks visibility is manipulated.
        If figure, then apply manipulations on all axes of the figure.
        If list of axes, apply manipulations on each of the given axes.
    spines: string
        Specify which spines and ticks should be shown.
        All other ones or hidden.
        'l' is the left spine, 'r' the right spine,
        't' the top one and 'b' the bottom one.
        E.g. 'lb' shows the left and bottom spine, and hides the top
        and and right spines, as well as their tick marks and labels.
        '' shows no spines at all.
        'lrtb' shows all spines and tick marks.

    Examples
    --------
    ```py
    import matplotlib.pyplot as plt
    import plottools.spines

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.show_spines('lb')
    ax1.show_spines('bt')
    ax2.show_spines('tr')
    ```
    ![show](figures/spines-show.png)
    """
    # collect spine visibility:
    xspines = []
    if 't' in spines:
        xspines.append('top')
    if 'b' in spines:
        xspines.append('bottom')
    yspines = []
    if 'l' in spines:
        yspines.append('left')
    if 'r' in spines:
        yspines.append('right')
    # collect axes:
    if isinstance(ax, (list, tuple, np.ndarray)):
        axs = ax
    elif hasattr(ax, 'get_axes'):
        # ax is figure:
        axs = ax.get_axes()
    else:
        axs = [ax]
    if not isinstance(axs, (list, tuple)):
        axs = [axs]
    for ax in axs:
        # hide spines:
        ax.spines['top'].set_visible('top' in xspines)
        ax.spines['bottom'].set_visible('bottom' in xspines)
        ax.spines['left'].set_visible('left' in yspines)
        ax.spines['right'].set_visible('right' in yspines)
        # ticks:
        if len(xspines) == 0:
            ax.xaxis.set_ticks_position('none')
            ax.xaxis.label.set_visible(False)
            ax.xaxis._orig_major_locator = ax.xaxis.get_major_locator()
            ax.xaxis.set_major_locator(ticker.NullLocator())
        else:
            if hasattr(ax.xaxis, '_orig_major_locator'):
                ax.xaxis.set_major_locator(ax.xaxis._orig_major_locator)
                delattr(ax.xaxis, '_orig_major_locator')
            elif isinstance(ax.xaxis.get_major_locator(), ticker.NullLocator):
                ax.xaxis.set_major_locator(ticker.AutoLocator())
            if len(xspines) == 1:
                ax.xaxis.set_ticks_position(xspines[0])
                ax.xaxis.set_label_position(xspines[0])
            else:
                ax.xaxis.set_ticks_position('both')
                ax.xaxis.set_label_position('bottom')
        if len(yspines) == 0:
            ax.yaxis.set_ticks_position('none')
            ax.yaxis.label.set_visible(False)
            ax.yaxis._orig_major_locator = ax.yaxis.get_major_locator()
            ax.yaxis.set_major_locator(ticker.NullLocator())
        else:
            if hasattr(ax.yaxis, '_orig_major_locator'):
                ax.yaxis.set_major_locator(ax.yaxis._orig_major_locator)
                delattr(ax.yaxis, '_orig_major_locator')
            elif isinstance(ax.yaxis.get_major_locator(), ticker.NullLocator):
                ax.yaxis.set_major_locator(ticker.AutoLocator())
            if len(yspines) == 1:
                ax.yaxis.set_ticks_position(yspines[0])
                ax.yaxis.set_label_position(yspines[0])
            else:
                ax.yaxis.set_ticks_position('both')
                ax.yaxis.set_label_position('left')

def plot_g(ax, df, models, i, let_x, let_y, titlesize=10, letsize=12):
    '''
            Plot the g_max values from df

                Parameters
                ----------
                ax : matplotlib axis
                    axis to plot spike train on
                d : dataframe
                    dataframe wit data
                models : list
                    model list (used to get column i)
                i : int
                    column of dataframe to plot
                let_x: float
                    x shift of subplot letter
                let_y: float
                    y shift of subplot letter
                titlesize: int
                    font size of title
                letsize: int
                    font size of subplot letter
                Returns
                -------
                ax : matplotlib axis
                    updated axis with plot data
            '''

    c = [cm.gray(x) for x in np.linspace(0., 0.75, 9)]
    myorder = [0, 4, 1, 6, 2,7, 3,8]
    colors = [c[i] for i in myorder]
    df.plot.bar(y=models[i], rot=90, ax=ax, legend=False,
                ylabel='$\mathrm{g}_{\mathrm{max}}$ [$\mathrm{mS}/ \mathrm{cm}^2$]',
                color=colors)
    ax.set_title(models[i], fontsize=titlesize)
    show_spines(ax, spines='lb')
    ax.text(let_x, let_y, string.ascii_uppercase[i], transform=ax.transAxes, size=letsize, weight='bold')
    ax.set_yscale('log')
    ax.set_xlim(-0.5, 9)
    from matplotlib.ticker import ScalarFormatter
    for axis in [ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    return ax


index = ['$\mathrm{g}_{\mathrm{Na}}$', '$\mathrm{g}_{\mathrm{Kd}}$', '$\mathrm{g}_{\mathrm{K_V1.1}}$',
         '$\mathrm{g}_{\mathrm{A}}$', '$\mathrm{g}_{\mathrm{M}}$', '$\mathrm{g}_{\mathrm{L}}$',
         '$\mathrm{g}_{\mathrm{T}}$', ' $\mathrm{g}_{\mathrm{Ca,K}}$', ' $\mathrm{g}_{\mathrm{Leak}}$']
df = pd.DataFrame({'RS Pyramidal': [56, 6, 0, 0, 0.075, 0, 0, 0, 0.0205],
                   'RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': [56, 5.4, 0.6, 0, 0.075, 0, 0, 0, 0.0205],
                   'RS Inhibitory': [10, 2.1, 0, 0, 0.0098, 0, 0, 0, 0.0205],
                   'RS Inhibitory +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': [10, 1.89, 0.21, 0, 0.0098, 0, 0, 0, 0.0205],
                   'FS': [58, 3.9, 0, 0, 0.075, 0, 0, 0, 0.038],
                   'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': [58, 3.51, 0.39, 0, 0.075, 0, 0, 0, 0.038],
                   'Cb stellate': [3.4, 9.0556, 0, 15.0159, 0, 0, 0.4545, 0, 0.07407],
                   'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': [3.4, 8.15, 0.90556, 15.0159, 0, 0, 0.4545, 0, 0.07407],
                   'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': [3.4, 9.0556, 1.50159, 0, 0, 0, 0.4545, 0, 0.07407],
                   'STN': [49, 57, 0, 5, 0, 5, 5, 1, 0.035],
                   'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': [49, 56.43, 0.57, 5, 0, 5, 5, 1, 0.035],
                   'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': [49, 57, 0.5, 0, 0, 5, 5, 1, 0.035]},
                  index=index)

scheme_style()
models = ['Cb stellate', 'RS Inhibitory', 'FS', 'RS Pyramidal', 'RS Inhibitory +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN']
fig, axs = plt.subplots(4, 3, figsize=cm2inch(20, 20))  # ,  sharey=True)
plt.subplots_adjust(hspace=1.5, wspace=1.0)
let_x = -0.6
let_y = 1.2
titlesize = 9
letsize = 10
axs[0, 0] = plot_g(axs[0, 0], df, models, 0, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[0, 1] = plot_g(axs[0, 1], df, models, 1, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[0, 2] = plot_g(axs[0, 2], df, models, 2, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[1, 0] = plot_g(axs[1, 0], df, models, 3, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[1, 1] = plot_g(axs[1, 1], df, models, 4, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[1, 2] = plot_g(axs[1, 2], df, models, 5, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[2, 0] = plot_g(axs[2, 0], df, models, 6, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[2, 1] = plot_g(axs[2, 1], df, models, 7, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[2, 2] = plot_g(axs[2, 2], df, models, 8, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[3, 0] = plot_g(axs[3, 0], df, models, 9, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[3, 1] = plot_g(axs[3, 1], df, models, 10, let_x, let_y, titlesize=titlesize, letsize=letsize)
axs[3, 2] = plot_g(axs[3, 2], df, models, 11, let_x, let_y, titlesize=titlesize, letsize=letsize)

# save
fig.savefig('./Figures/model_g.jpg', dpi=300, bbox_inches='tight')  # pdf # eps
plt.show()

