"""
Script to plot ramp protocol and responses of each model to ramp - Figure 2-1

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


def plot_ramp_V(ax, model='RS Pyramidal'):
    '''
    Plot response of a model to a ramp input current

    Parameters
    ----------
    ax : matplotlib axis
        axis to plot spike train on
    model : string
        model to plot

    '''
    model_ramp = pd.read_csv('./Figures/Data/model_ramp.csv')
    ax.plot(model_ramp['t'], model_ramp[model], 'k', linewidth=0.1)
    ax.set_ylabel('V')
    ax.set_xlabel('Time [s]')
    ax.set_ylim(-80, 60)
    ax.axis('off')
    ax.set_title(model, fontsize=8)

def plot_I_ramp(ax):
    '''
    Plot ramp input current

    Parameters
    ----------
    ax : matplotlib axis
        axis to plot spike train on
    '''
    dt = 0.01
    I_low = 0
    I_high = 0.001
    initial_period = 1000

    sec = 4
    ramp_len = int(4 * 1000 * 1 / dt)
    stim_time = ramp_len * 2
    I_amp = np.array([0])
    I_amp = np.reshape(I_amp, (1, I_amp.shape[0]))
    I_ramp = np.zeros((stim_time, 1)) @ I_amp
    I_ramp[:, :] = np.ones((stim_time, 1)) @ I_amp
    stim_num_step = I_ramp.shape[1]
    start=0
    I_ramp[start:int(start + ramp_len), 0] = np.linspace(0, I_high, ramp_len)
    I_ramp[int(start + ramp_len):int(start + ramp_len * 2), 0] = np.linspace(I_high, 0, ramp_len)

    t = np.arange(0, 4000 * 2, dt)
    ax.plot(t, I_ramp)
    ax.set_ylabel('I')
    ax.set_xlabel('Time [s]')
    ax.axis('off')
    ax.set_title('Ramp Current', fontsize=8, x=0.5, y=-0.5)
    return ax

#% plot setup
fig = plt.figure(figsize=cm2inch(17.6,25))

gs0 = fig.add_gridspec(2, 1, wspace=0.)
gs00 = gs0[:].subgridspec(13, 1, wspace=0.7, hspace=1.0)

ax1_ramp = fig.add_subplot(gs00[0])
ax2_ramp = fig.add_subplot(gs00[1])
ax3_ramp = fig.add_subplot(gs00[2])
ax4_ramp = fig.add_subplot(gs00[3])
ax5_ramp = fig.add_subplot(gs00[4])
ax6_ramp = fig.add_subplot(gs00[5])
ax7_ramp = fig.add_subplot(gs00[6])
ax8_ramp = fig.add_subplot(gs00[7])
ax9_ramp = fig.add_subplot(gs00[8])
ax10_ramp = fig.add_subplot(gs00[9])
ax11_ramp = fig.add_subplot(gs00[10])
ax12_ramp = fig.add_subplot(gs00[11])
ax13_I = fig.add_subplot(gs00[12])
ramp_axs = [ax1_ramp, ax2_ramp, ax3_ramp, ax4_ramp, ax5_ramp,ax6_ramp, ax7_ramp, ax8_ramp,
             ax9_ramp, ax10_ramp, ax11_ramp, ax12_ramp]

# order of models
models = ['Cb stellate','RS Inhibitory','FS', 'RS Pyramidal','RS Inhibitory +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'Cb stellate +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'FS +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'RS Pyramidal +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN +$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'Cb stellate $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$',
          'STN $\Delta$$\mathrm{K}_{\mathrm{V}}\mathrm{1.1}$', 'STN']

# plot ramps
for i in range(len(models)):
    plot_ramp_V(ramp_axs[i], model=models[i])

# add scalebar
plt.rcParams.update({'font.size': 6})

add_scalebar(ax12_ramp, matchx=False, matchy=False, hidex=True, hidey=True, sizex=1000, sizey=50, labelx='1 s',
                 labely='50 mV', loc=3, pad=-2, borderpad=0, barwidth=1, bbox_to_anchor=Bbox.from_bounds(0.01, 0.05, 1, 1),
                          bbox_transform=ax12_ramp.transAxes)

ax13_I = plot_I_ramp(ax13_I)
add_scalebar(ax13_I, matchx=False, matchy=False, hidex=True, hidey=True, sizex=1000, sizey=0.0005, labelx='1 s',
             labely='0.5 $I_{max}$', loc=3, pad=-2, borderpad=0, barwidth=1,
             bbox_to_anchor=Bbox.from_bounds(0.0, -0.01, 1, 1),  bbox_transform=ax13_I.transAxes)

# add subplot labels
for i in range(0,len(models)):
    ramp_axs[i].text(-0.01, 1.1, string.ascii_uppercase[i], transform=ramp_axs[i].transAxes, size=10, weight='bold')

#save
fig.set_size_inches(cm2inch(17.6,22))
fig.savefig('./Figures/ramp_firing.pdf', dpi=fig.dpi)
# fig.savefig('./Figures/ramp_firing.png', dpi=fig.dpi)
plt.show()

