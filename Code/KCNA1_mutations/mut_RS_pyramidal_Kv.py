# -*- coding: utf-8 -*-
"""
Script to simulate all KCNA1 mutations in RS Pyramidal +Kv1.1 model

"""
__author__ = "Nils A. Koch"
__copyright__ = "Copyright 2022, Nils A. Koch"
__license__ = "MIT"


import numpy as np
import os
import json
from numba import types
from numba.typed import Dict
from joblib import Parallel, delayed
from Code.Functions.Utility_fxns import capacitance, stimulus_init, init_dict
from Code.Functions.Pospischil_fxns import Pospischil_mut


# model parameters
dt = 0.01
sec = 2
low = 0
high = 0.001
number_steps = 200
initial_period = 1000
num_gating = 10
num_current = 7
C, surf_area = capacitance(61.4, 1)
stim_time, I_in, stim_num, V_m = stimulus_init(low, high, number_steps, initial_period, dt, sec)
shift, scale, slope_shift, E, currents_included, b_param, g = init_dict(
    np.array(['m', 'h', 'n', 'q', 'r', 'p', 's', 'u', 's_mut', 'u_mut']))
tau_max_p = 608
V_init = -70
V_T = -56.2

# initialize arrays
current = np.zeros((num_current, stim_num))
gating = np.zeros((num_gating, stim_num))

# initialize dictionary
ind_dict = Dict.empty(key_type=types.unicode_type, value_type=types.int64, )
i = 0
for var in np.array(['m', 'h', 'n', 'q', 'r', 'p', 's', 'u', 's_mut', 'u_mut']):
    ind_dict[var] = i
    i += 1
i = 0
for var in np.array(['Na', 'Kd', 'M', 'Kv', 'Kv_mut', 'L', 'Leak']):
    ind_dict[var] = i
    i += 1

# gating parameters
b_param = {}
b_param['m'] = np.array([-34.33054521, -8.21450277, 1.42295686])
b_param['h'] = np.array([-34.51951036, 4.04059373, 1., 0.05])
b_param['n'] = np.array([-63.76096946, -13.83488194, 7.35347425])
b_param['q'] = np.array([-39.03684525, -5.57756176, 2.25190197])
b_param['r'] = np.array([-57.37, 20.98, 1.0])
b_param['p'] = np.array([-45., -9.9998807337, 1.])
b_param['s'] = np.array([-14.16, -10.15, 1.0])
b_param['u'] = np.array([-31., 5.256, 1., 0.245])
b_param['s_mut'] = np.array([-14.16, -10.15, 1.])
b_param['u_mut'] = np.array([-31., 5.256, 1., 0.245])

mut_act_Vhalf_wt = -30.01851851851851
mut_act_k_wt = -7.7333333333333325
s_diff_Vhalf = mut_act_Vhalf_wt - b_param['s'][0]
s_diff_k = mut_act_k_wt - b_param['s'][1]
b_param['s'][1] = b_param['s'][1] + s_diff_k
b_param['u'][1] = b_param['u'][1] + s_diff_k
b_param['s'][0]  = b_param['s'][0] + s_diff_Vhalf
b_param['u'][0]  = b_param['u'][0] + s_diff_Vhalf
b_param['s_mut'][1] = b_param['s_mut'][1] + s_diff_k
b_param['u_mut'][1] = b_param['u_mut'][1] + s_diff_k
b_param['s_mut'][0]  = b_param['s_mut'][0] + s_diff_Vhalf
b_param['u_mut'][0]  = b_param['u_mut'][0] + s_diff_Vhalf

# reversal potentials
E["Na"] = 50.
E["K"] = -90.
E["Ca"] = 120.
E["Leak"] = -70.3

# model currents
currents_included["Na"] = True
currents_included["Kd"] = True
currents_included["Kv"] = True
currents_included["Kv_mut"] = True
currents_included["L"] = False
currents_included["M"] = True
currents_included["Leak"] = True

# model conductances
Kv_ratio = 0.1
g["Na"] = 56. * surf_area
g["Kd"] = 6. * (1 - Kv_ratio) * surf_area
g["M"] = 0.075 * surf_area
if currents_included["Kv_mut"] == True:
    g["Kv"] = 6. * Kv_ratio / 2 * surf_area
else:
    g["Kv"] = 6. * Kv_ratio / 2 * surf_area * 2
g["Kv_mut"] = 6. * Kv_ratio / 2 * surf_area
g["L"] = 0. * surf_area
g["Leak"] = 0.0205 * surf_area

# save folder
folder = './KCNA1_mutations/RS_pyramidal_Kv'
if not os.path.isdir(folder):
    os.makedirs(folder)

# mutation properties
mutations = json.load(open("./KCNA1_mutations/mutations_effects_dict.json"))


# # %%
prominence = 50
desired_AUC_width = high/5
Parallel(n_jobs=8, verbose=9)(
    delayed(Pospischil_mut)(V_init, V_T, g, E, I_in, dt, currents_included, stim_time, stim_num, C, tau_max_p, shift,
                            scale, b_param, slope_shift, gating, current, prominence, desired_AUC_width, mutations, mut,
                            folder, high, low, number_steps, initial_period, sec)
    for mut in list(mutations.keys()))
