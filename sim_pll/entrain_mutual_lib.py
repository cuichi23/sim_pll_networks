#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import sys, gc
import inspect

import networkx as nx
import numpy as np
import scipy
from scipy import signal
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
import itertools
from itertools import permutations as permu
from itertools import combinations as combi
import matplotlib
import os, pickle
if not os.environ.get('SGE_ROOT') is None:										# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from scipy.ndimage.filters import uniform_filter1d
import time
import datetime
import pandas as pd

from sim_pll import plot_lib
from sim_pll import evaluation_lib as eva
from sim_pll import coupling_fct_lib as coupfct

now = datetime.datetime.now()

''' Enable automatic carbage collector '''
gc.enable()

''' All plots in latex mode '''
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.rcParams['agg.path.chunksize'] = 10000

''' STYLEPACKS '''
titlefont = {
		'family' : 'serif',
		'color'  : 'black',
		'weight' : 'normal',
		'size'   : 9,
		}

labelfont = {
		'family' : 'sans-serif',
		'color'  : 'black',
		'weight' : 'normal',
		'size'   : 16,
		}

annotationfont = {
		'family' : 'monospace',
		'color'  : (0, 0.27, 0.08),
		'weight' : 'normal',
		'size'   : 14,
		}


def obtain_phase_config_entrainment_of_mutual_sync(dict_net: dict, dict_pll: dict, dict_algo: dict) -> None:
	"""
		Calculate the phase configurations of networks of mutually delay-coupled oscillators that are being entrained by a reference oscillator.
		The entrainment can be inserted into the network of mutually coupled oscillators at either one node, a subset of nodes or into all nodes.

		Args:
			dict_net: [dict] contains all the data of the simulations to be evaluated and the settings
			dict_pll: [dict] whether phases are wrapped into the interval 0) [0, 2*pi), 1) [-pi, pi), or 2) [-pi/2, 3*pi/2)
			dict_algo: [dict] the number of bins of the histogram of phases plotted for the final state of the simulation

		TODO:
			1) reorganize to a class
			2) structure into functions and simplify

		Returns:
			None, updates dictionaries directly
		"""

	if dict_net['topology'] == 'entrainOne-chain':
		phase_configuration_ref_to_one_for_chain_topology(dict_net, dict_pll)
	elif not dict_net['topology'] == 'entrainOne-chain' and 'entrain' in dict_net['topology']:
		numerically_calculate_phase_configuration_entrainment_of_mutual_sync(dict_net, dict_pll)

	return None


def phase_configuration_ref_to_one_for_chain_topology(dict_net: dict, dict_pll: dict) -> None:
	"""
		Calculates the phase configurations of networks of mutually delay-coupled oscillators in a chain topology that is entrained by a reference oscillator at one end.
		The oscillator with k=0 os considered to be the reference oscillator.

		Args:
			dict_net: [dict] contains all the data of the simulations to be evaluated and the settings
			dict_pll: [dict] whether phases are wrapped into the interval 0) [0, 2*pi), 1) [-pi, pi), or 2) [-pi/2, 3*pi/2)
			dict_algo: [dict] the number of bins of the histogram of phases plotted for the final state of the simulation

		TODO:
			1) reorganize to a class
			2) structure into functions and simplify

		Returns:
			None, updates dictionaries directly
		"""

	# start with the last oscillator in the chain
	asymptotic_phase_configuration_entrained_sync_state = []
	if dict_pll['responseVCO'] == 'linear':
		if isinstance(dict_pll['intrF'], list) or isinstance(dict_pll['intrF'], np.ndarray):
			frequency_or_voltage = dict_pll['intrF']
		elif isinstance(dict_pll['intrF'], np.float) or isinstance(dict_pll['intrF'], int):
			frequency_or_voltage = [dict_pll['intrF'] for i in range(dict_net['Nx'])]
	elif dict_pll['responseVCO'] == 'nonlinear_3rd_gen':
		# dict_pll['prebias_voltage_vco'] contains all the prebias voltages of all PLLs to realize the specific intrinsic VCO frequencies
		if dict_pll['coup_fct_sig'] == coupfct.triangular or dict_pll['coup_fct_sig'] == coupfct.square_wave:
			frequency_or_voltage = dict_pll['prebias_voltage_vco'] + g * (v_PD_peak2peak/2 + v_offset)
		elif dict_pll['coup_fct_sig'] == coupfct.sine or dict_pll['coup_fct_sig'] == coupfct.cosine:
			frequency_or_voltage = dict_pll['prebias_voltage_vco'] + g * v_offset
	else:
		print('Implement this case! So far only "linear" and "nonlinear_3rd_gen" exist.')
		sys.exit()

	asymptotic_phase_configuration_entrained_sync_state = np.empty(dict_net['Nx'])
	for i in range(dict_net['Nx']-1, 0, -1):
		if i == dict_net['Nx']-1:												# since the last oscillator in the chain only has one input, it depends only on one beta which can be calculated
			asymptotic_phase_configuration_entrained_sync_state[i] = -dict_pll['intrF'][i] * dict_pll['transmission_delay'] - dict_pll['div'] * dict_pll['inverse_coup_fct_sig'](
				2 * (dict_pll['inverse_fct_vco_response'](dict_pll['intrF'][i]) - frequency_or_voltage[i]))
		else:
			asymptotic_phase_configuration_entrained_sync_state[i] = -dict_pll['intrF'][i] * dict_pll['transmission_delay'] - dict_pll['div'] * dict_pll['inverse_coup_fct_sig'](
				2 * (dict_pll['inverse_fct_vco_response'](dict_pll['intrF'][i]) - frequency_or_voltage[i]) - dict_pll['coup_fct_sig'](
					-dict_pll['intrF'][i] * dict_pll['transmission_delay'] - asymptotic_phase_configuration_entrained_sync_state[i+1]))

	dict_net.update({'phiInitConfig': asymptotic_phase_configuration_entrained_sync_state})

	return None


def numerically_calculate_phase_configuration_entrainment_of_mutual_sync(dict_net: dict, dict_pll: dict) -> None:
	"""
		Calculates the phase configurations of networks of mutually delay-coupled oscillators in a chain topology that is entrained by a reference oscillator at one end.
		The oscillator with k=0 os considered to be the reference oscillator.

		Args:
			dict_net: [dict] contains all the data of the simulations to be evaluated and the settings
			dict_pll: [dict] whether phases are wrapped into the interval 0) [0, 2*pi), 1) [-pi, pi), or 2) [-pi/2, 3*pi/2)
			dict_algo: [dict] the number of bins of the histogram of phases plotted for the final state of the simulation

		TODO:
			1) reorganize to a class
			2) structure into functions and simplify

		Returns:
			None, updates dictionaries directly
		"""

	print('Include here Dimitris code to calculate phase configurations solving numerically the sets of coupled equations.')

	dict_net.update({'phiInitConfig': asymptotic_phase_configuration_entrained_sync_state})

	return None


