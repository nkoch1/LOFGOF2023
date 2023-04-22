# -*- coding: utf-8 -*-
"""
Script to plot illustrate AUC and rheobase firing characterization - Figure 2

"""
__author__ = "Nils A. Koch"
__copyright__ = "Copyright 2022, Nils A. Koch"
__license__ = "MIT"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import colorConverter as cc
from matplotlib.colors import to_hex
import string
from plotstyle import scheme_style

colorslist = ['#40A787',  # cyan'#
              '#F0D730',  # yellow
              '#C02717',  # red
              '#007030',  # dark green
              '#AAB71B',  # lightgreen
              '#008797',  # light blue
              '#F78017',  # orange
              '#478010',  # green
              '#53379B',  # purple
              '#2060A7',  # blue
              '#873770',  # magenta
              '#D03050'   # pink
              ]

def cm2inch(*tupl):
    '''
    convert cm to inch for plots size tuple
    '''
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def show_spines(ax, spines='lrtb'):
    """ Show and hide spines.

    From github.com/janscience/plottools.git  spines.py
    under the GNU General Public License v3.0
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



def lighter(color, lightness):
    """ Make a color lighter.

    From github.com/janscience/plottools.git  colors.py
    under the GNU General Public License v3.0
    ![lighter](figures/colors-lighter.png)

    Parameters
    ----------
    color: dict or matplotlib color spec
        A matplotlib color (hex string, name color string, rgb tuple)
        or a dictionary with an 'color' or 'facecolor' key.
    lightness: float
        The smaller the lightness, the lighter the returned color.
        A lightness of 0 returns white.
        A lightness of 1 leaves the color untouched.
        A lightness of 2 returns black.

    Returns
    -------
    color: string or dict
        The lighter color as a hexadecimal RGB string (e.g. '#rrggbb').
        If `color` is a dictionary, a copy of the dictionary is returned
        with the value of 'color' or 'facecolor' set to the lighter color.

    Examples
    --------
    For 40% lightness of blue do
    ```py
    import plottools.colors as c
    colors = c.palettes['muted']
    lightblue = c.lighter(colors['blue'], 0.4)
    ```
    """
    try:
        c = color['color']
        cd = dict(**color)
        cd['color'] = lighter(c, lightness)
        return cd
    except (KeyError, TypeError):
        try:
            c = color['facecolor']
            cd = dict(**color)
            cd['facecolor'] = lighter(c, lightness)
            return cd
        except (KeyError, TypeError):
            if lightness > 2:
                lightness = 2
            if lightness > 1:
                return darker(color, 2.0-lightness)
            if lightness < 0:
                lightness = 0
            r, g, b = cc.to_rgb(color)
            rl = r + (1.0-lightness)*(1.0 - r)
            gl = g + (1.0-lightness)*(1.0 - g)
            bl = b + (1.0-lightness)*(1.0 - b)
            return to_hex((rl, gl, bl)).upper()
                

def plot_AUC(ax, a=1, b=0.2, c=180, d=0, width=0.2):
    '''

    plot the Area Under the Curve (AUC) for a square root function
    c*np.sqrt(a*(x - b)) + d

    Parameters
    ----------
    ax : matplotlib axis
        axis to plot legend on
    a : float

    b : float

    c :  float

    d : float

    width :  float
        width of AUC shaded area on x-axis
    '''
    x = np.linspace(0, 1, 1000)
    y = c*np.sqrt(a*(x - b)) + d
    ax.plot(x, y, colorslist[9])
    ax.set_xlabel('Current [nA]')
    ax.set_ylabel('Frequency [Hz]')
    ax.fill_between(x, y, where=(x<=b+width), color=lighter(colorslist[9], 0.3))
    ax.text(0.3, 15, 'AUC', ha='center')
    ax.annotate('', (0.2, 10), (0, 10), arrowprops=dict(arrowstyle="<->"))
    ax.text(0.1, 20, 'rheobase', ha='center')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

def plot_diff_sqrt(ax, a=1, b=0.2, c=100, d=0, a2=1, b2=0.2, c2=100, d2=0):
    '''
    plot 2 square root functions:
    c*np.sqrt(a*(x - b)) + d
    c2*np.sqrt(a2*(x - b2)) + d2

    Parameters
    ----------
    ax : matplotlib axis
        axis to plot legend on
    a : float
    b : float
    c : float
    d : float
    a2 : float
    b2 : float
    c2 : float
    d2 : float

    '''
    show_spines(ax, 'lb')
    x = np.linspace(0, 1, 10000)
    y = c*np.sqrt(a*(x - b)) + d
    y2 = c2 * np.sqrt(a2 * (x - b2)) + d2
    ax.plot(x, y, colorslist[9])
    ax.plot(x, y2, colorslist[2])
    ax.set_xlim(0,1)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xticks([])
    ax.set_yticks([])


def plot_quadrant(ax):
    '''
    Plot a quadrant on axis

    Parameters
    ----------
    ax : matplotlib axis
        axis to plot legend on

    '''
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.text(1.2, -0.15, '$\\Delta$ rheobase', ha='right')
    ax.text(-0.03, 0.7, '$\\Delta$ AUC', ha='right', rotation=90)
    ax.tick_params(length=0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.annotate('', (1, 0), (-1, 0), arrowprops=dict(arrowstyle="->"))
    ax.annotate('', (0, 1), (0, -1), arrowprops=dict(arrowstyle="->"))
    ax.set_xticks([-0.5, 0.5])
    a = ax.get_xticks().tolist()
    a[0] = '\u2212'
    a[1] = '+'
    ax.set_xticklabels(a)
    ax.set_yticks([-0.5, 0.5])
    b = ax.get_xticks().tolist()
    b[0] = '\u2212'
    b[1] = '+'
    ax.set_yticklabels(b)


#%%
fig = plt.figure(figsize=cm2inch(7, 12))
gs = gridspec.GridSpec(3,6, top=0.9, bottom=0.05, left=0.15, right = 0.9, hspace=0.8, wspace=0.6)
ax1 = fig.add_subplot(gs[0,1:])
show_spines(ax1, 'lb')
plot_AUC(ax1, width=0.2)
ax3 = fig.add_subplot(gs[1:, :])

# add panel letter labels
ax1.text(-0.4, 1.2, string.ascii_uppercase[0], transform=ax1.transAxes, size=16, weight='bold')
ax3.text(-0.15, 1.05, string.ascii_uppercase[1], transform=ax3.transAxes, size=16, weight='bold')
show_spines(ax3, '')
plot_quadrant(ax3) # plot delineation into quadrants


inset_ylim = (0, 100)
# top left
lfsize = 8
ax3.text(x=-0.9, y=0.7, s='$\\uparrow$ AUC\n$\\downarrow$ rheobase', fontsize=lfsize)
ax3.text(-0.95, 0.35, 'GOF', ha='right')
ax3_TL = ax3.inset_axes([0.07, 0.6, 0.3, 0.2])
plot_diff_sqrt(ax3_TL, b2=0.09, c2=200)
ax3_TL.set_ylim(inset_ylim)
ax3_TL.annotate('', (0.25, 7), (0.085, 7), arrowprops=dict(arrowstyle="<|-", color=colorslist[2],lw=0.5, mutation_scale=5), zorder=-10) #
ax3_TL.annotate('', (0.55, 55), (0.3, 90), arrowprops=dict(arrowstyle="<|-", color=colorslist[2],lw=0.5, mutation_scale=5), zorder=-10) #
# top right
ax3.text(x=0.22, y=0.7, s='$\\uparrow$ AUC\n$\\uparrow$ rheobase', fontsize=lfsize)
ax3.text(0.95, 0.35, 'LOF/\nGOF?')
ax3_TR = ax3.inset_axes([0.63, 0.6, 0.3, 0.2])
plot_diff_sqrt(ax3_TR, b2=0.4, c2=200)
ax3_TR.set_ylim(inset_ylim)
ax3_TR.annotate('', (0.17, 7), (0.41, 7), arrowprops=dict(arrowstyle="<|-", color=colorslist[2],lw=0.5, mutation_scale=5), zorder=-10) #
ax3_TR.annotate('', (0.8, 70), (0.6, 90), arrowprops=dict(arrowstyle="<|-", color=colorslist[2],lw=0.5, mutation_scale=5), zorder=-10) #

# bottom left
ax3.text(x=-0.9, y=-0.95, s='$\\downarrow$ AUC\n$\downarrow$ rheobase', fontsize=lfsize)
ax3.text(-0.95, -0.55, 'GOF/\nLOF?', ha='right')
ax3_BL = ax3.inset_axes([0.07, 0.15, 0.3, 0.2])
plot_diff_sqrt(ax3_BL, b2=0.06, c2=65)
ax3_BL.set_ylim(inset_ylim)
ax3_BL.annotate('', (0.25, 7), (0.07, 7), arrowprops=dict(arrowstyle="<|-", color=colorslist[2],lw=0.5, mutation_scale=5), zorder=-10) #
ax3_BL.annotate('', (0.8, 80), (0.85, 55), arrowprops=dict(arrowstyle="<|-", color=colorslist[2],lw=0.5, mutation_scale=5), zorder=-10) #

# bottom right
ax3.text(x=0.22, y=-0.95, s='$\\downarrow$ AUC\n$\\uparrow$ rheobase', fontsize=lfsize)
ax3.text(0.95, -0.55, 'LOF')
ax3_BR = ax3.inset_axes([0.63, 0.15, 0.3, 0.2])
plot_diff_sqrt(ax3_BR, b2=0.4, c2=60)
ax3_BR.set_ylim(inset_ylim)
ax3_BR.annotate('', (0.18, 7), (0.42, 7), arrowprops=dict(arrowstyle="<|-", color=colorslist[2],lw=0.5, mutation_scale=5), zorder=-10) #
ax3_BR.annotate('', (0.7, 75), (0.8, 37.5), arrowprops=dict(arrowstyle="<|-", color=colorslist[2],lw=0.5, mutation_scale=5), zorder=-10) #



fig.set_size_inches(cm2inch(8.5,12))
fig.savefig('./Figures/firing_characterization.png', dpi=fig.dpi, bbox_inches='tight') #, dpi=fig.dpi #pdf #eps
plt.show()
