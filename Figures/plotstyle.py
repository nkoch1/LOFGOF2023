# -*- coding: utf-8 -*-
"""
Script to define functions to change RC parameters of matplotlib

"""
__author__ = "Nils A. Koch"
__copyright__ = "Copyright 2022, Nils A. Koch"
__license__ = "MIT"


import matplotlib.pyplot as plt


def scheme_style():
    '''
    Update matplotlib RC parameters
    '''
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['figure.dpi'] = 300


def plot_style():
    '''
    Update matplotlib RC parameters for generic plot

    '''
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['figure.dpi'] = 300

def corr_style():
    '''
    Update matplotlib RC parameters for corr plot

    '''
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 0.5
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['legend.fontsize']= 6
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize']= 6
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['figure.dpi'] = 300



def mut_style():
    '''
    Update matplotlib RC parameters for KCNA1 mutation plot

    '''
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['axes.titlesize'] = 6
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams.update({'font.size': 6})
    plt.rcParams['figure.dpi'] = 300