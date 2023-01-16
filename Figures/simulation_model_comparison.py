# -*- coding: utf-8 -*-
"""
Script to plot KCNA1 mutations - Figure 5

"""
__author__ = "Nils A. Koch"
__copyright__ = "Copyright 2022, Nils A. Koch"
__license__ = "MIT"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import string
from Figures.plotstyle import mut_style
import seaborn as sns
import scipy.stats as stats
import matplotlib.lines as mlines

def cm2inch(*tupl):
    '''
    convert cm to inch for plots size tuple
    '''
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def Kendall_tau(df):
    '''
    Compute Kendall Tau correlation and corresponding p-values
    Parameters
    ----------
    df : pandas dataframe
        dataframe to do compute correlation on
    Returns
    -------
    tau : pandas dataframe
        Kendall Tau correlation
    p : pandas dataframe
        p-values

    '''
    tau = df.corr(method='kendall')
    p = pd.DataFrame(columns=df.columns, index=df.columns)
    for col in range((df.columns).shape[0]):
        for col2 in range((df.columns).shape[0]):
            if col != col2:
                _, p.loc[df.columns[col], df.columns[col2]] = stats.kendalltau(
                    df[df.columns[col]], df[df.columns[col2]], nan_policy='omit')
    return tau, p

def correlation_plot(ax, df='AUC', title='', cbar=False):
    '''
    Plot correlation matrix
    Parameters
    ----------
    ax : matplotlib axis
        axis to plot on
    df : string
        whether to plot correlations for 'AUC' or 'rheo'
    title : string
        title for axis

    cbar : Bool
        whether to plot a cbar or not

    '''
    cbar_ax = fig.add_axes([0.685, 0.48, .15, .01])
    cbar_ax.spines['left'].set_visible(False)
    cbar_ax.spines['bottom'].set_visible(False)
    cbar_ax.spines['right'].set_visible(False)
    cbar_ax.spines['top'].set_visible(False)
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([])
    if df == 'AUC':
        df = pd.read_csv(os.path.join('./Figures/Data/sim_mut_AUC.csv'), index_col='Unnamed: 0')
    elif df == 'rheo':
        df = pd.read_csv(os.path.join('./Figures/Data/sim_mut_rheo.csv'), index_col='Unnamed: 0')

    # array for names
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    models =['Cb_stellate','RS_inhib','Cb_stellate_Kv','FS', 'RS_pyramidal','STN_Kv', 'Cb_stellate_Kv_only','STN_Kv_only', 'STN']
    model_names = ['Cb stellate','RS inhibitory', 'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'FS',
                   'RS pyramidal', 'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
                   'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
                    'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$','STN']
    model_letter_names = ['A', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    col_dict = {}
    for m in range(len(models)):
        col_dict[model_names[m]] = model_letter_names[m]
    df.rename(columns=col_dict, inplace=True)
    df = df[model_letter_names]

    # calculate correlation matrix
    tau, p = Kendall_tau(df)

    # mask to hide upper triangle of matrix
    mask = np.zeros_like(tau, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    np.fill_diagonal(mask, False)

    # models and renaming of tau
    models = ['Cb_stellate', 'RS_inhib', 'Cb_stellate_Kv', 'FS', 'RS_pyramidal', 'STN_Kv', 'Cb_stellate_Kv_only',
              'STN_Kv_only', 'STN']
    model_names = ['Cb stellate', 'RS inhibitory', 'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'FS',
                   'RS pyramidal', 'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
                   'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
                   'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN']
    model_letter_names = ['A', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    col_dict = {}
    for m in range(len(models)):
        col_dict[model_names[m]] = model_letter_names[m]
    tau.rename(columns=col_dict, index=col_dict, inplace=True)
    tau = tau[model_letter_names]

    # plotting with or without colorbar
    if cbar==False:
        res = sns.heatmap(tau, annot=False, mask=mask, center=0, vmax=1, vmin=-1, linewidths=.5, square=True, ax=ax,
                          cbar=False, cmap=cmap, cbar_ax=cbar_ax, cbar_kws={"shrink": .52})
    else:
        res = sns.heatmap(tau, annot=False, mask=mask, center=0, vmax=1, vmin=-1, linewidths=.5, square=True, ax=ax,
                          cbar=True, cmap=cmap, cbar_ax=cbar_ax,
                          cbar_kws={"orientation": "horizontal",
                                    "ticks": [-1,-0.5, 0, 0.5, 1]} )
        cbar_ax.set_title(r'Kendall $\tau$', y=1.02, loc='center', fontsize=6)
        cbar_ax.tick_params(length=3)
        for tick in cbar_ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")

def mutation_plot(ax, model='RS_pyramidal'):
    '''
    Plot KCNA1 mutations for a given model
    Parameters
    ----------
    ax : matplotlib axis
        axis to plot on
    model : string
        model to plot

    Returns
    -------
    ax : matplotlib axis
        updated axis with KCNA1 mutations plotted

    '''
    models = ['RS_pyramidal', 'RS_inhib', 'FS', 'Cb_stellate', 'Cb_stellate_Kv', 'Cb_stellate_Kv_only', 'STN',
              'STN_Kv', 'STN_Kv_only']
    model_names = ['RS pyramidal', 'RS inhibitory', 'FS',
                   'Cb stellate', 'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
                   'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN', 'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$']
    model_display_names = ['RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'RS Inhibitory +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'Cb stellate',
                   'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
                   'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN',
                   'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$']
    model_letter_names = ['Model H',
                           'Model E',
                           'Model G', 'Model A',
                           'Model F',
                           'Model J', 'Model L',
                           'Model I',
                           'Model K']
    col_dict = {}
    for m in range(len(models)):
        col_dict[models[m]] = model_display_names[m]

    ax_dict = {}
    ax_dict['RS_pyramidal'] = (0, 0)
    ax_dict['RS_inhib'] = (0, 1)
    ax_dict['FS'] = (1, 0)
    ax_dict['Cb_stellate'] = (2, 0)
    ax_dict['Cb_stellate_Kv'] = (2, 1)
    ax_dict['Cb_stellate_Kv_only'] = (3, 0)
    ax_dict['STN'] = (3, 1)
    ax_dict['STN_Kv'] = (4, 0)
    ax_dict['STN_Kv_only'] = (4, 1)

    ylim_dict = {}
    ylim_dict['RS_pyramidal'] = (-0.1, 0.3)
    ylim_dict['RS_inhib'] = (-0.6, 0.6)
    ylim_dict['FS'] = (-0.06, 0.08)
    ylim_dict['Cb_stellate'] = (-0.1, 0.4)
    ylim_dict['Cb_stellate_Kv'] = (-0.1, 0.5)
    ylim_dict['Cb_stellate_Kv_only'] = (-1, 0.8)
    ylim_dict['STN'] = (-0.01, 0.015)
    ylim_dict['STN_Kv'] = (-0.4, 0.6)
    ylim_dict['STN_Kv_only'] = (-0.03, 0.3)
    Marker_dict = {'Cb stellate': 'o', 'RS Inhibitory': 'o', 'FS': 'o', 'RS Pyramidal': "^",
                   'RS Inhibitory +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': "^",
                   'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': "^",
                   'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': "D",
                   'RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': "D",
                   'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': "D",
                   'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': "s",
                   'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$': "s", 'STN': "s"}

    AUC = pd.read_csv(os.path.join('./Figures/Data/sim_mut_AUC.csv'), index_col='Unnamed: 0')
    rheo = pd.read_csv(os.path.join('./Figures/Data/sim_mut_rheo.csv'), index_col='Unnamed: 0')

    mod = models.index(model)
    mut_names = AUC.index
    ax.plot(rheo.loc[mut_names, model_names[mod]]*1000, AUC.loc[mut_names, model_names[mod]]*100, linestyle='',
            markeredgecolor='grey', markerfacecolor='grey', marker=Marker_dict[model_display_names[mod]],
            markersize=2)

    ax.plot(rheo.loc['wt', model_names[mod]], AUC.loc['wt', model_names[mod]]*100, 'sk')

    mut_col = sns.color_palette("pastel")
    ax.plot(rheo.loc['V174F', model_names[mod]]*1000, AUC.loc['V174F', model_names[mod]]*100, linestyle='',
            markeredgecolor=mut_col[0], markerfacecolor=mut_col[0], marker=Marker_dict[model_display_names[mod]],markersize=4)
    ax.plot(rheo.loc['F414C', model_names[mod]]*1000, AUC.loc['F414C', model_names[mod]]*100, linestyle='',
            markeredgecolor=mut_col[1], markerfacecolor=mut_col[1], marker=Marker_dict[model_display_names[mod]],markersize=4)
    ax.plot(rheo.loc['E283K', model_names[mod]]*1000, AUC.loc['E283K', model_names[mod]]*100, linestyle='',
            markeredgecolor=mut_col[2], markerfacecolor=mut_col[2], marker=Marker_dict[model_display_names[mod]],markersize=4)
    ax.plot(rheo.loc['V404I', model_names[mod]]*1000, AUC.loc['V404I', model_names[mod]]*100, linestyle='',
            markeredgecolor=mut_col[3], markerfacecolor=mut_col[5], marker=Marker_dict[model_display_names[mod]],markersize=4)
    ax.set_title(model_letter_names[mod], pad=14)
    ax.set_xlabel('$\Delta$Rheobase [pA]')
    ax.set_ylabel('Normalized $\Delta$AUC (%)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.hlines(0, xmin, xmax, colors='lightgrey', linestyles='--')
    ax.vlines(0, ymin,ymax, colors='lightgrey', linestyles='--')
    return ax

def mutation_legend(ax, marker_s_leg, pos, ncol):
    '''
    Plot legend for mutations
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
    colors = sns.color_palette("pastel")

    Markers = ["o", "o", "o", "o"]
    V174F = mlines.Line2D([], [], color=colors[0], marker=Markers[0], markersize=marker_s_leg, linestyle='None',
                         label='V174F')
    F414C = mlines.Line2D([], [], color=colors[1], marker=Markers[1], markersize=marker_s_leg, linestyle='None',
                         label='F414C')
    E283K = mlines.Line2D([], [], color=colors[2], marker=Markers[2], markersize=marker_s_leg, linestyle='None', label='E283K')
    V404I = mlines.Line2D([], [], color=colors[5], marker=Markers[3], markersize=marker_s_leg, linestyle='None',
                         label='V404I')
    WT = mlines.Line2D([], [], color='k', marker='s', markersize=marker_s_leg+2, linestyle='None', label='Wild type')

    ax.legend(handles=[WT, V174F, F414C, E283K, V404I], loc='center', bbox_to_anchor=pos, ncol=ncol, frameon=False)


mut_style()
# plot setup
fig = plt.figure()
gs0 = fig.add_gridspec(1, 6, wspace=-0.2)
gsl = gs0[0:3].subgridspec(3, 3, wspace=0.9, hspace=0.8)
gsr = gs0[4:6].subgridspec(7, 1, wspace=0.6, hspace=3.8)

ax00 = fig.add_subplot(gsl[1,1]) # model H
ax01 = fig.add_subplot(gsl[0,1]) # model E
ax02 = fig.add_subplot(gsl[1,0]) # model G
ax10 = fig.add_subplot(gsl[0,0]) # model A
ax11 = fig.add_subplot(gsl[0,2]) # model F
ax12 = fig.add_subplot(gsl[2,0]) # model J
ax20 = fig.add_subplot(gsl[2,2]) # model L
ax21 = fig.add_subplot(gsl[1,2]) # model I
ax22 = fig.add_subplot(gsl[2,1]) # model K
axr0 = fig.add_subplot(gsr[0:3,0])
axr1 = fig.add_subplot(gsr[4:,0])

# plot mutations in each model
ax00 = mutation_plot(ax00, model='RS_pyramidal')
ax01 = mutation_plot(ax01, model='RS_inhib')
ax02 = mutation_plot(ax02, model='FS')
ax10 = mutation_plot(ax10, model='Cb_stellate')
ax11 = mutation_plot(ax11, model='Cb_stellate_Kv')
ax12 = mutation_plot(ax12, model='Cb_stellate_Kv_only')
ax20 = mutation_plot(ax20, model='STN')
ax21 = mutation_plot(ax21, model='STN_Kv')
ax22 = mutation_plot(ax22, model='STN_Kv_only')

marker_s_leg = 4
pos = (0.425, -0.7)
ncol = 5
mutation_legend(ax22, marker_s_leg, pos, ncol)

# plot correlation matrices
correlation_plot(axr1,df = 'AUC', title='Normalized $\Delta$AUC', cbar=False)
correlation_plot(axr0,df = 'rheo', title='$\Delta$Rheobase', cbar=True)

# add subplot labels
axs = [ax10, ax01, ax11, ax02, ax00, ax21, ax12, ax22, ax20]
j=0
for i in range(0,9):
    axs[i].text(-0.625, 1.25, string.ascii_uppercase[i], transform=axs[i].transAxes, size=10, weight='bold')
    j +=1
axr0.text(-0.27, 1.075, string.ascii_uppercase[j], transform=axr0.transAxes, size=10, weight='bold')
axr1.text(-0.27, 1.075, string.ascii_uppercase[j+1], transform=axr1.transAxes, size=10, weight='bold')

# save
fig.set_size_inches(cm2inch(22.2,15))
fig.savefig('./Figures/simulation_model_comparison.pdf', dpi=fig.dpi)
# fig.savefig('./Figures/simulation_model_comparison.png', dpi=fig.dpi) #png
plt.show()

