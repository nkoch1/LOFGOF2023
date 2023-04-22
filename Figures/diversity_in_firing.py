# -*- coding: utf-8 -*-
"""
Script to plot model diversity - Figure 1

"""
__author__ = "Nils A. Koch"
__copyright__ = "Copyright 2022, Nils A. Koch"
__license__ = "MIT"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import Bbox
import string
from matplotlib.offsetbox import AnchoredOffsetbox


def cm2inch(*tupl):
    '''
    convert cm to inch for plots size tuple
    '''
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

#### from https://gist.github.com/dmeliza/3251476  #####################################################################
class AnchoredScaleBar(AnchoredOffsetbox):
    '''
        Modified from https://gist.github.com/dmeliza/3251476 (Dan Meliza)
        under the Python Software Foundation License (http://docs.python.org/license.html)
        '''
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, barcolor="black", barwidth=None,
                 **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0, 0), sizex, 0, ec=barcolor, lw=barwidth, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0, 0), 0, sizey, ec=barcolor, lw=barwidth, fc="none"))

        if sizex and labelx:
            self.xlabel = TextArea(labelx)
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
        if sizey and labely:
            self.ylabel = TextArea(labely)
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)

plt.rcParams.update({'font.size': 6})
def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    Modified from https://gist.github.com/dmeliza/3251476 (Dan Meliza)
    under the Python Software Foundation License (http://docs.python.org/license.html)


    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """

    def f(axis):
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex: ax.xaxis.set_visible(False)
    if hidey: ax.yaxis.set_visible(False)
    if hidex and hidey: ax.set_frame_on(False)

    return sb
########################################################################################################################

def plot_spike_train(ax, model='RS Pyramidal', stop=750):
    '''
    Plot spike train of a model

        Parameters
        ----------
        ax : matplotlib axis
            axis to plot spike train on
        model : string
            model to plot
        stop : int
            time step to stop plotting at
        '''
    model_spiking = pd.read_csv('./Figures/Data/model_spiking.csv')
    stop_ind = int(np.argmin(np.abs(model_spiking['t'] - stop)))
    ax.plot(model_spiking['t'][0:stop_ind], model_spiking[model][0:stop_ind], 'k', linewidth=1.5)
    ax.set_ylabel('V')
    ax.set_xlabel('Time [s]')
    ax.set_ylim(-85, 60)
    ax.axis('off')
    ax.set_title(model, fontsize=10)


def plot_fI(ax, model='RS Pyramidal'):
    '''
    Plot the fI curve for a model

        Parameters
        ----------
        ax : matplotlib axis
            axis to plot spike train on
        model : string
            model to plot
    '''
    firing_values = pd.read_csv('./Figures/Data/firing_values.csv', index_col='Unnamed: 0')
    model_F_inf = pd.read_csv('./Figures/Data/model_F_inf.csv')
    if model=='RS Inhibitory':
        ax.plot(model_F_inf['I_inhib'], model_F_inf[model], color='grey')
        ax.plot(firing_values.loc['spike_ind', model], model_F_inf[model][int(np.argmin(np.abs(model_F_inf['I_inhib'] - firing_values.loc['spike_ind', model])))],
                '.', color='k', markersize=3)
        ax.plot(firing_values.loc['ramp_up', model],
                model_F_inf[model][int(np.argmin(np.abs(model_F_inf['I_inhib'] - firing_values.loc['ramp_up', model])))],
                '.', color='g', markersize=3)
        ax.plot(firing_values.loc['ramp_down', model],
                model_F_inf[model][int(np.argmin(np.abs(model_F_inf['I_inhib'] - firing_values.loc['ramp_down', model])))],
                '.', color='r', markersize=3)
    else:
        ax.plot(model_F_inf['I'], model_F_inf[model], color='grey')
        ax.plot(firing_values.loc['spike_ind', model],
                model_F_inf[model][int(np.argmin(np.abs(model_F_inf['I'] - firing_values.loc['spike_ind', model])))],
                '.', color='k', markersize=3)
        ax.plot(firing_values.loc['ramp_up', model],
                model_F_inf[model][int(np.argmin(np.abs(model_F_inf['I'] - firing_values.loc['ramp_up', model])))],
                '.', color='g', markersize=3)
        ax.plot(firing_values.loc['ramp_down', model],
                model_F_inf[model][int(np.argmin(np.abs(model_F_inf['I'] - firing_values.loc['ramp_down', model])))],
                '.', color='r', markersize=3)
    f = 8
    ax.set_ylabel('Frequency [Hz]', fontsize=f)
    ax.set_xlabel('Current [nA]', fontsize=f)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#%%

# plot layout
fig = plt.figure()
gs0 = fig.add_gridspec(3, 3, wspace=0.4, hspace=0.2)

gs00 = gs0[:,0].subgridspec(5, 3, wspace=1.8, hspace=1.5)
gs01 = gs0[:,1].subgridspec(5, 3, wspace=1.8, hspace=1.5)
gs02 = gs0[:,2].subgridspec(5, 3, wspace=1.8, hspace=1.5)

ax_diag = fig.add_subplot(gs02[2:, :])
import matplotlib.image as mpimg
img = mpimg.imread('./Figures/model_diagram.png')
ax_diag.imshow(img)
ax_diag.spines['top'].set_visible(False)
ax_diag.spines['bottom'].set_visible(False)
ax_diag.spines['left'].set_visible(False)
ax_diag.spines['right'].set_visible(False)
ax_diag.set_yticks([])
ax_diag.set_xticks([])
ax_diag.text(-0.12, 1.075, string.ascii_uppercase[12], transform=ax_diag.transAxes, size=10, weight='bold')

ax1_spikes = fig.add_subplot(gs00[0,0:2])
ax1_fI = fig.add_subplot(gs00[0, 2])
ax2_spikes = fig.add_subplot(gs01[0,0:2])
ax2_fI = fig.add_subplot(gs01[0, 2])
ax3_spikes = fig.add_subplot(gs02[0,0:2])
ax3_fI = fig.add_subplot(gs02[0, 2])
ax4_spikes = fig.add_subplot(gs00[1,0:2])
ax4_fI = fig.add_subplot(gs00[1, 2])
ax5_spikes = fig.add_subplot(gs01[1, 0:2])
ax5_fI = fig.add_subplot(gs01[1,  2])
ax6_spikes = fig.add_subplot(gs02[1,0:2])
ax6_fI = fig.add_subplot(gs02[1, 2])
ax7_spikes = fig.add_subplot(gs00[2,0:2])
ax7_fI = fig.add_subplot(gs00[2, 2])
ax8_spikes = fig.add_subplot(gs01[2,0:2])
ax8_fI = fig.add_subplot(gs01[2, 2])
ax9_spikes = fig.add_subplot(gs00[3,0:2])
ax9_fI = fig.add_subplot(gs00[3, 2])
ax10_spikes = fig.add_subplot(gs01[3,0:2])
ax10_fI = fig.add_subplot(gs01[3, 2])
ax11_spikes = fig.add_subplot(gs00[4,0:2])
ax11_fI = fig.add_subplot(gs00[4, 2])
ax12_spikes = fig.add_subplot(gs01[4, 0:2])
ax12_fI = fig.add_subplot(gs01[4,  2])

spike_axs = [ax1_spikes, ax2_spikes, ax3_spikes, ax4_spikes, ax5_spikes,ax6_spikes, ax7_spikes, ax8_spikes,
              ax11_spikes,ax9_spikes,ax10_spikes,   ax12_spikes]#, ax13_spikes, ax14_spikes]
fI_axs = [ax1_fI, ax2_fI, ax3_fI, ax4_fI, ax5_fI,ax6_fI, ax7_fI,  ax8_fI, ax11_fI, ax9_fI,  ax10_fI,
              ax12_fI] #, ax13_fI, ax14_fI]

# model order
models = ['Cb stellate','RS Inhibitory','FS', 'RS Pyramidal','RS Inhibitory +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN']

# plot spike train and fI for each model
for i in range(len(models)):
    plot_spike_train(spike_axs[i], model=models[i])
    plot_fI(fI_axs[i], model=models[i])

# add scalebars
add_scalebar(ax6_spikes, matchx=False, matchy=False, hidex=True, hidey=True, sizex=100, sizey=50, labelx='100\u2009ms',
                 labely='50\u2009mV', loc=3, pad=-0.5, borderpad=-1.0, barwidth=2, bbox_to_anchor=Bbox.from_bounds(-0.275, -0.05, 1, 1),
                          bbox_transform=ax6_spikes.transAxes)
add_scalebar(ax11_spikes, matchx=False, matchy=False, hidex=True, hidey=True, sizex=100, sizey=50, labelx='100\u2009ms',
                 labely='50\u2009mV', loc=3, pad=-0.5, borderpad=-1.0, barwidth=2, bbox_to_anchor=Bbox.from_bounds(-0.275, -0.05, 1, 1),
                          bbox_transform=ax11_spikes.transAxes)
add_scalebar(ax12_spikes, matchx=False, matchy=False, hidex=True, hidey=True, sizex=100, sizey=50, labelx='100\u2009ms',
                 labely='50\u2009mV', loc=3, pad=-0.5, borderpad=-1.0, barwidth=2, bbox_to_anchor=Bbox.from_bounds(-0.275, -0.05, 1, 1),
                          bbox_transform=ax12_spikes.transAxes)
# add subplot labels
for i in range(0,len(models)):
   spike_axs[i].text(-0.22, 1.2, string.ascii_uppercase[i], transform=spike_axs[i].transAxes, size=10, weight='bold')

# save
fig.set_size_inches(cm2inch(21,15))
fig.savefig('./Figures/diversity_in_firing.jpg', dpi=300, bbox_inches='tight') #pdf # eps
plt.show()


