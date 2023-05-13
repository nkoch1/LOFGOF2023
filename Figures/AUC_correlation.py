# -*- coding: utf-8 -*-
"""
Script to plot AUC correlations - Figure 4

"""
__author__ = "Nils A. Koch"
__copyright__ = "Copyright 2022, Nils A. Koch"
__license__ = "MIT"

import pandas as pd
import numpy as np
from numpy import ndarray
import string
import textwrap
import json
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms
import matplotlib.path
import matplotlib.colors
from matplotlib import ticker
from matplotlib.ticker import NullFormatter
from matplotlib.axes._axes import Axes
from matplotlib.markers import MarkerStyle
from matplotlib.collections import LineCollection
from Figures.plotstyle import corr_style


def cm2inch(*tupl):
    '''
        convert cm to inch for plots size tuple
        '''
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
#%% Modified from https://stackoverflow.com/a/52935294 (ImportanceOfBeingErnest and Miguel) under CC BY-SA 4.0 license

def GetColor2Marker(markers):
    '''
    Modified from https://stackoverflow.com/a/52935294 (ImportanceOfBeingErnest and Miguel) under CC BY-SA 4.0 license
    '''
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
                  '#D03050'  # pink
                  ]
    import matplotlib.colors
    palette = [matplotlib.colors.to_rgb(c) for c in colorslist]
    mkcolors = [(palette[i]) for i in range(len(markers))]
    return dict(zip(mkcolors,markers))

def fixlegend(ax,markers,markersize=3,**kwargs):
    '''
    Modified from https://stackoverflow.com/a/52935294 (ImportanceOfBeingErnest and Miguel) under CC BY-SA 4.0 license
    '''
    # Fix Legend
    legtitle =  ax.get_legend().get_title().get_text()
    _,l = ax.get_legend_handles_labels()
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
                  '#D03050'  # pink
                  ]
    import matplotlib.colors
    palette = [matplotlib.colors.to_rgb(c) for c in colorslist]
    mkcolors = [(palette[i]) for i in range(len(markers))]
    newHandles = [plt.Line2D([0],[0], ls="none", marker=m, color=c, mec="none", markersize=markersize,**kwargs) \
                for m,c in zip(markers, mkcolors)]
    ax.legend(newHandles,l)
    leg = ax.get_legend()
    leg.set_title(legtitle)

old_scatter = Axes.scatter
def new_scatter(self, *args, **kwargs):
    '''
    Modified from https://stackoverflow.com/a/52935294 (ImportanceOfBeingErnest and Miguel) under CC BY-SA 4.0 license
    '''
    colors = kwargs.get("c", None)
    co2mk = kwargs.pop("co2mk",None)
    FinalCollection = old_scatter(self, *args, **kwargs)
    if co2mk is not None and isinstance(colors, ndarray):
        Color2Marker = GetColor2Marker(co2mk)
        paths=[]
        for col in colors:
            mk=Color2Marker[tuple(col)]
            marker_obj = MarkerStyle(mk)
            paths.append(marker_obj.get_path().transformed(marker_obj.get_transform()))
        FinalCollection.set_paths(paths)
    return FinalCollection
Axes.scatter = new_scatter
########################################################################################################################

def gradientaxis(ax, start, end, cmap, n=100,lw=1):
    '''
    add color gradient to axis
        Parameters
        ----------
        ax : matplotlib axis
            axis to apply gradient to
        start : tuple
            start coordinates on axis
        end : tuple
            end coordinates on axis
        cmap : colormap
            colormap for gradient
        n : int
            number of segments in gradient
        lw : float
            width of axis line


        Returns
        -------
        ax : matplotlib axis
            updated axis with gradient
    '''
    x = np.linspace(start[0],end[0],n)
    y = np.linspace(start[1],end[1],n)
    points = np.array([x,y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1],points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=lw,zorder=15)
    lc.set_array(np.linspace(0,1,n))
    ax.add_collection(lc)
    return ax

def corr_with_markers(ax,max_width, alteration='shift', msize=3):
    '''
    Plot Model Correlation as markers

    Parameters
    ----------
    ax : matplotlib axis
        axis to plot on
    max_width :
        maximum width of label text
    alteration : string
        'shift', 'slope' or 'g'
    msize : float
        marker size
    '''
    hlinewidth = 0.5
    model_names = ['RS pyramidal','RS inhibitory','FS', 'RS pyramidal +$K_V1.1$','RS inhibitory +$K_V1.1$',
                'FS +$K_V1.1$','Cb stellate','Cb stellate +$K_V1.1$', 'Cb stellate $\Delta$$K_V1.1$','STN',
                   'STN +$K_V1.1$', 'STN $\Delta$$K_V1.1$']

    colorslist = ['#007030',  # dark green
                  '#F0D730',  # yellow
                  '#C02717',  # red
                    '#478010',  # green
                  '#AAB71B',  # lightgreen
                '#F78017',  # orange
                  '#40A787',  # cyan'#
                  '#008797',  # light blue
                '#2060A7',  # blue
                  '#D03050',  # pink
                  '#53379B',  # purple
                  '#873770',  # magenta
                  ]


    colors = [matplotlib.colors.to_rgb(c) for c in colorslist]
    clr_dict = {}
    for m in range(len(model_names)):
        clr_dict[model_names[m]] = colors[m]
    print(colors)
    print(clr_dict)
    Markers = ["o", "o", "o", "^", "^", "^", "D", "D", "D", "s", "s", "s"]
    if alteration=='shift':
        i = 2  # Kd act
        ax.axvspan(i - 0.4, i + 0.4, fill=False, edgecolor = 'k')
        df = pd.read_csv('./Figures/Data/AUC_shift_corr.csv')
        sns.swarmplot(y="corr", x="$\Delta V_{1/2}$", hue="model", data=df,
                      palette=clr_dict, linewidth=0, orient='v', ax=ax, size=msize,
                      order=['Na activation', 'Na inactivation', 'K activation', '$K_V1.1$ activation',
                             '$K_V1.1$ inactivation', 'A activation', 'A inactivation'],
                      hue_order=model_names, co2mk=Markers)
        lim = ax.get_xlim()
        ax.plot([lim[0], lim[1]], [0, 0], ':r',linewidth=hlinewidth)
        ax.plot([lim[0], lim[1]], [1, 1], ':k',linewidth=hlinewidth)
        ax.plot([lim[0], lim[1]], [-1, -1], ':k',linewidth=hlinewidth)
        ax.set_title("Shift ($\Delta V_{1/2}$)", y=1.05)
        ax.set_xticklabels(['Na \nactivation', 'Na \ninactivation', 'Kd \nactivation', '$K_V1.1$ \nactivation',
                            '$K_V1.1$ \ninactivation', 'A \nactivation', 'A \ninactivation'])
    elif alteration=='slope':
        i = 3  # Kv1.1 act
        ax.axvspan(i - 0.4, i + 0.4, fill=False, edgecolor='k')
        df = pd.read_csv('./Figures/Data/AUC_scale_corr.csv')

        # Add in points to show each observation
        sns.swarmplot(y="corr", x="Slope (k)", hue="model", data=df,
                      palette=clr_dict, linewidth=0, orient='v', ax=ax, size=msize,
                      order=['Na activation', 'Na inactivation', 'K activation', '$K_V1.1$ activation',
                             '$K_V1.1$ inactivation', 'A activation', 'A inactivation'],
                      hue_order=model_names, co2mk=Markers)
        lim = ax.get_xlim()
        ax.plot([lim[0], lim[1]], [0, 0], ':r',linewidth=hlinewidth)
        ax.plot([lim[0], lim[1]], [1, 1], ':k',linewidth=hlinewidth)
        ax.plot([lim[0], lim[1]], [-1, -1], ':k',linewidth=hlinewidth)
        ax.set_title("Slope (k)", y=1.05)
        ax.set_xticklabels(['Na \nactivation', 'Na \ninactivation', 'Kd \nactivation', '$K_V1.1$ \nactivation',
                            '$K_V1.1$ \ninactivation', 'A \nactivation', 'A \ninactivation'])
    elif alteration=='g':
        i = 1  # Kd
        ax.axvspan(i - 0.4, i + 0.4, fill=False, edgecolor='k')
        df = pd.read_csv('./Figures/Data/AUC_g_corr.csv')

        # Add in points to show each observation
        sns.swarmplot(y="corr", x="g", hue="model", data=df,
                      palette=clr_dict, linewidth=0, orient='v', ax=ax, size=msize,
                      order=['Na', 'K', '$K_V1.1$', 'A', 'Leak'],
                      hue_order=model_names, co2mk=Markers)
        lim = ax.get_xlim()
        # ax.plot([lim[0], lim[1]], [0,0], ':k')
        ax.plot([lim[0], lim[1]], [0, 0], ':r',linewidth=hlinewidth)
        ax.plot([lim[0], lim[1]], [1, 1], ':k',linewidth=hlinewidth)
        ax.plot([lim[0], lim[1]], [-1, -1], ':k',linewidth=hlinewidth)
        # Tweak the visual presentation
        ax.set_title("Conductance (g)", y=1.05)
        # ax.set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels())
        ax.set_xticklabels(['Na', 'Kd', '$K_V1.1$', 'A', 'Leak'])
    else:
        print('Please chose "shift", "slope" or "g"')
    ax.get_legend().remove()
    ax.xaxis.grid(False)
    sns.despine(trim=True, bottom=True, ax=ax)
    ax.set(xlabel=None, ylabel=r'Kendall $\it{\tau}$')


def model_legend(ax, marker_s_leg, pos, ncol):
    '''
    plot model legend on axis
    Parameters
    ----------
    ax : matplotlib axis
        axis to plot legend on
    marker_s_leg : int
        marker size in legend
    pos : tuple
        position in axis
    ncol : int
        number of columns in legend

    '''
    colorslist = ['#007030',  # dark green
                  '#F0D730',  # yellow
                  '#C02717',  # red
                  '#478010',  # green
                  '#AAB71B',  # lightgreen
                  '#F78017',  # orange
                  '#40A787',  # cyan'#
                  '#008797',  # light blue
                  '#2060A7',  # blue
                  '#D03050',  # pink
                  '#53379B',  # purple
                  '#873770',  # magenta
                  ]

    import matplotlib.colors
    colors = [matplotlib.colors.to_rgb(c) for c in colorslist]
    model_pos = {'Cb stellate':0, 'RS Inhibitory':1, 'FS':2, 'RS Pyramidal':3,
              'RS Inhibitory +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':4,
              'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':5, 'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':6,
              'RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':7, 'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':8,
              'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':9,
              'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':10, 'STN':11}
    Markers = ["o", "o", "o", "^", "^", "^", "D", "D", "D", "s", "s", "s"]

    RS_p = mlines.Line2D([], [], color='#007030', marker="^", markersize=marker_s_leg, linestyle='None', label='Model D')
    RS_i = mlines.Line2D([], [], color='#F0D730', marker="o", markersize=marker_s_leg, linestyle='None', label='Model B')
    FS = mlines.Line2D([], [], color='#C02717', marker="o", markersize=marker_s_leg, linestyle='None', label='Model C')
    RS_p_Kv = mlines.Line2D([], [], color='#478010', marker="D", markersize=marker_s_leg, linestyle='None', label='Model H')
    RS_i_Kv = mlines.Line2D([], [], color='#AAB71B', marker="^", markersize=marker_s_leg, linestyle='None', label='Model E')
    FS_Kv = mlines.Line2D([], [], color='#F78017', marker="D", markersize=marker_s_leg, linestyle='None', label='Model G')
    Cb = mlines.Line2D([], [], color='#40A787', marker="o", markersize=marker_s_leg, linestyle='None', label='Model A')
    Cb_pl = mlines.Line2D([], [], color='#008797', marker="^", markersize=marker_s_leg, linestyle='None', label='Model F')
    Cb_sw = mlines.Line2D([], [], color='#2060A7', marker="s", markersize=marker_s_leg, linestyle='None', label='Model J')
    STN = mlines.Line2D([], [], color='#D03050', marker="s", markersize=marker_s_leg, linestyle='None', label='Model L')
    STN_pl = mlines.Line2D([], [], color='#53379B', marker="D", markersize=marker_s_leg, linestyle='None', label='Model I')
    STN_sw = mlines.Line2D([], [], color='#873770', marker="s", markersize=marker_s_leg, linestyle='None', label='Model K')
    ax.legend(handles=[Cb, RS_i, FS, RS_p, RS_i_Kv, Cb_pl, FS_Kv, RS_p_Kv, STN_pl, Cb_sw, STN_sw, STN], loc='center',
              bbox_to_anchor=pos, ncol=ncol, frameon=False)

def plot_AUC_alt(ax, model='FS', alteration='shift'):
    '''
    plot the AUC across an alteration for all models with one emphasized
    Parameters
    ----------
    ax : matplotlib axis
        axis to plot on
    model : string
        model to emphasize with thicker line
    alteration : string
        'shift', 'slope' or 'g'

    Returns
    -------
    ax : matplotlib axis
        updated axis with plot data
    '''
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    model_names = ['RS Pyramidal','RS Inhibitory','FS', 'RS Pyramidal +$K_V1.1$','RS Inhibitory +$K_V1.1$',
                   'FS +$K_V1.1$','Cb stellate','Cb stellate +$K_V1.1$', 'Cb stellate $\Delta$$K_V1.1$','STN',
                   'STN +$K_V1.1$', 'STN $\Delta$$K_V1.1$']

    model_name_dict = {'RS Pyramidal': 'RS Pyramidal',
                       'RS Inhibitory': 'RS Inhibitory',
                       'FS': 'FS',
                       'RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'RS Pyramidal +$K_V1.1$',
                       'RS Inhibitory +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'RS Inhibitory +$K_V1.1$',
                       'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'FS +$K_V1.1$',
                       'Cb stellate': 'Cb stellate',
                       'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'Cb stellate +$K_V1.1$',
                       'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'Cb stellate $\Delta$$K_V1.1$',
                       'STN': 'STN',
                       'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'STN +$K_V1.1$',
                       'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'STN $\Delta$$K_V1.1$'}
    colorslist = ['#007030',  # dark green
                  '#F0D730',  # yellow
                  '#C02717',  # red
                  '#478010',  # green
                  '#AAB71B',  # lightgreen
                  '#F78017',  # orange
                  '#40A787',  # cyan'#
                  '#008797',  # light blue
                  '#2060A7',  # blue
                  '#D03050',  # pink
                  '#53379B',  # purple
                  '#873770',  # magenta
                  ]

    import matplotlib.colors
    colors = [matplotlib.colors.to_rgb(c) for c in colorslist]
    clr_dict = {}
    for m in range(len(model_names)):
        clr_dict[model_names[m]] = colors[m]
    if alteration=='shift':
        df = pd.read_csv('./Figures/Data/AUC_shift_ex.csv')
        df = df.sort_values('alteration')
        ax.set_xlabel('$\Delta$$V_{1/2}$')
    elif alteration=='slope':
        df = pd.read_csv('./Figures/Data/AUC_slope_ex.csv')
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('$k$/$k_{WT}$')
    elif alteration=='g':
        df = pd.read_csv('./Figures/Data/AUC_g_ex.csv')
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('$g$/$g_{WT}$')
    for mod in model_names:
        if mod == model_name_dict[model]:
            ax.plot(df['alteration'], df[mod], color=clr_dict[mod], alpha=1, zorder=10, linewidth=2)
        else:
            ax.plot(df['alteration'], df[mod], color=clr_dict[mod],alpha=0.5, zorder=1, linewidth=1)

    if alteration=='shift':
        ax.set_ylabel('Normalized $\Delta$AUC', labelpad=4)
    else:
        ax.set_ylabel('Normalized $\Delta$AUC', labelpad=0)
    x = df['alteration']
    y = df[model_name_dict[model]]
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(df[model_names].min().min(), df[model_names].max().max())

    # x axis color gradient
    cvals = [-2., 2]
    colors = ['lightgrey', 'k']
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    (xstart, xend) = ax.get_xlim()
    (ystart, yend) = ax.get_ylim()
    print(ystart, yend)
    start = (xstart, ystart * 1.0)
    end = (xend, ystart * 1.0)
    ax = gradientaxis(ax, start, end, cmap, n=200, lw=4)
    ax.spines['bottom'].set_visible(False)
    return ax

def plot_fI(ax, model='RS Pyramidal', type='shift', alt='m', color1='red', color2='dodgerblue'):
    '''
    plot fI curves for a model across an alteration in current parameters
    Parameters
    ----------
    ax : matplotlib axis
        axis to plot on
    model : string
        model to plot
    type : string
        type of alteration: 'shift', 'slope', or 'g'
    alt : string
        model parameter that is altered
    color1 : string
        color at start of gradient
    color2 : string
        color at end of gradient

    Returns
    -------
    ax : matplotlib axis
        updated axis with plot data

    '''
    model_save_name = {'RS Pyramidal': 'RS_pyr_posp',
                       'RS Inhibitory': 'RS_inhib_posp',
                       'FS': 'FS_posp',
                       'RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'RS_pyr_Kv',
                       'RS Inhibitory +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'RS_inhib_Kv',
                       'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'FS_Kv',
                       'Cb stellate': 'Cb_stellate',
                       'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'Cb_stellate_Kv',
                       'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'Cb_stellate_Kv_only',
                       'STN': 'STN',
                       'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'STN_Kv',
                       'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': 'STN_Kv_only'}
    cvals = [-2., 2]
    colors = [color1, color2]
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    colors = cmap(np.linspace(0, 1, 22))
    df = pd.read_csv('./Figures/Data/Model_fI/{}_fI.csv'.format(model_save_name[model]))
    df.drop(['Unnamed: 0'], axis=1)
    newdf = df.loc[df.index[(df['alt'] == alt) & (df['type'] == type)], :]
    newdf['mag'] = newdf['mag'].astype('float')
    newdf = newdf.sort_values('mag').reset_index()
    c = 0
    for i in newdf.index:
        ax.plot(json.loads(newdf.loc[i, 'I']), json.loads(newdf.loc[i, 'F']), color=colors[c])
        c += 1
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Current [nA]')
    if model == 'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':
        ax.set_title("Model G", x=0.2, y=1.0)
    elif model == 'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':
        ax.set_title("Model I", x=0.2, y=1.0)
    else:
        ax.set_title("", x=0.2, y=1.0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    L = ax.get_ylim()
    ax.set_ylim([0, L[1]])
    return ax


#%%
corr_style()
color_dict = {'Cb stellate': '#40A787', # cyan'#
              'RS Inhibitory': '#F0D730',  # yellow
              'FS': '#C02717',  # red
              'RS Pyramidal': '#007030',  # dark green
              'RS Inhibitory +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': '#AAB71B',  # lightgreen
              'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':  '#008797',  # light blue
              'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': '#F78017',  # orange
              'RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': '#478010',  # green
              'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': '#53379B',  # purple
              'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': '#2060A7',  # blue
              'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': '#873770',  # magenta
              'STN':  '#D03050'  # pink
              }
model_letter = {
'Cb stellate': 'A',
'RS Inhibitory': 'B',
'FS': 'C',
'RS Pyramidal': 'D',
'RS Inhibitory': 'E',
'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':'F',
'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':'G',
'RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':'H',
'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':'I',
'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':'J',
'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$':'K',
'STN':'L',
}


# plot setup
marker_s_leg = 2
max_width = 20
pad_x = 0.85
pad_y= 0.4
pad_w = 1.1
pad_h = 0.7

fig = plt.figure()
gs = fig.add_gridspec(3, 7, wspace=1.2, hspace=1.)
ax0 = fig.add_subplot(gs[0,2:7])
ax0_ex = fig.add_subplot(gs[0,1])
ax0_fI = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,2:7])
ax1_ex = fig.add_subplot(gs[1,1])
ax1_fI = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[2,2:7])
ax2_ex = fig.add_subplot(gs[2,1])
ax2_fI = fig.add_subplot(gs[2,0])

line_width = 1
# plot fI examples
ax0_fI = plot_fI(ax0_fI, model='FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', type='shift', alt='s',  color1='lightgrey', color2='k')
rec = plt.Rectangle((-pad_x, -pad_y), 1 + pad_w, 1  + pad_h, fill=False, lw=line_width,transform=ax0_fI.transAxes,  color=color_dict['FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$'], alpha=1, zorder=-1)
rec = ax0_fI.add_patch(rec)
rec.set_clip_on(False)

ax1_fI = plot_fI(ax1_fI, model='FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', type='slope', alt='u',  color1='lightgrey', color2='k')
rec = plt.Rectangle((-pad_x, -pad_y), 1 + pad_w, 1  + pad_h, fill=False, lw=line_width,transform=ax1_fI.transAxes,  color=color_dict['FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$'], alpha=1, zorder=-1)
rec = ax1_fI.add_patch(rec)
rec.set_clip_on(False)

ax2_fI = plot_fI(ax2_fI, model='STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', type='g', alt='Leak',  color1='lightgrey', color2='k')
rec = plt.Rectangle((-pad_x, -pad_y), 1 + pad_w, 1  + pad_h, fill=False, lw=line_width,transform=ax2_fI.transAxes,  color=color_dict['STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$'], alpha=1, zorder=-1)
rec = ax2_fI.add_patch(rec)
rec.set_clip_on(False)

# plot boxplots
corr_with_markers(ax0,max_width, alteration='shift')
corr_with_markers(ax1,max_width, alteration='slope')
corr_with_markers(ax2,max_width, alteration='g')

# plot legend
pos = (0.225, -0.9)
ncol = 6
model_legend(ax2,  marker_s_leg, pos, ncol)

# plot examples
plot_AUC_alt(ax0_ex,model='FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',  alteration='shift')
plot_AUC_alt(ax1_ex,model='FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', alteration='slope')
plot_AUC_alt(ax2_ex, model='STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', alteration='g')

# label subplots with letters
ax0_fI.text(-0.875, 1.35, string.ascii_uppercase[0], transform=ax0_fI.transAxes, size=10, weight='bold')
ax0_ex.text(-0.8, 1.35, string.ascii_uppercase[1], transform=ax0_ex.transAxes, size=10, weight='bold')
ax0.text(-0.075, 1.35, string.ascii_uppercase[2], transform=ax0.transAxes, size=10, weight='bold')

ax1_fI.text(-0.875, 1.35, string.ascii_uppercase[3], transform=ax1_fI.transAxes,size=10, weight='bold')
ax1_ex.text(-0.8, 1.35, string.ascii_uppercase[4], transform=ax1_ex.transAxes, size=10, weight='bold')
ax1.text(-0.075, 1.35, string.ascii_uppercase[5], transform=ax1.transAxes, size=10, weight='bold')

ax2_fI.text(-0.875, 1.35, string.ascii_uppercase[6], transform=ax2_fI.transAxes,size=10, weight='bold')
ax2_ex.text(-0.8, 1.35, string.ascii_uppercase[7], transform=ax2_ex.transAxes, size=10, weight='bold')
ax2.text(-0.075, 1.35, string.ascii_uppercase[8], transform=ax2.transAxes, size=10, weight='bold')

#save
fig.set_size_inches(cm2inch(20.75,12))
fig.savefig('./Figures/AUC_correlation.tif', dpi=600)
# fig.savefig('./Figures/AUC_correlation.png', dpi=fig.dpi) #png
plt.show()
