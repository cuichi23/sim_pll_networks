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


def phase_configuration_ref_to_one_for_chain_topology(dict_net: dict, dict_pll: dict, phase_wrap: int = 3) -> None:
	"""
		Calculates the phase configurations of networks of mutually delay-coupled oscillators in a chain topology that is entrained by a reference oscillator at one end.
		The oscillator with k=0 os considered to be the reference oscillator.

		Args:
			dict_net: [dict] contains all the data of the simulations to be evaluated and the settings
			dict_pll: [dict] whether phases are wrapped into the interval 0) [0, 2*pi), 1) [-pi, pi), or 2) [-pi/2, 3*pi/2)
			phase_wrap: determines the phase wrapping used: 1 = [-pi, pi], 2 = [-pi/2, 3*pi/2], 3 = [0, 2*pi]

		TODO:
			1) reorganize to a class
			2) structure into functions and simplify

		Returns:
			None, updates dictionaries directly
		"""

	if isinstance(dict_pll['transmission_delay'], list) or isinstance(dict_pll['transmission_delay'], np.ndarray):
		print('Implement the case of heterogeneous transmission delays in entrain_mutual_lib.py!')
		sys.exit()

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

	# start with the last oscillator in the chain
	asymptotic_phase_differences_entrained_sync_state = np.zeros(dict_net['Nx'])
	asymptotic_phase_configuration_entrained_sync_state = np.zeros(dict_net['Nx'])
	# calculate all phase-differences between neighboring oscillators
	for i in range(dict_net['Nx']-1, 0, -1):
		#print('i=', i)
		if i == dict_net['Nx']-1:			# since the last oscillator in the chain only has one input, it depends only on one beta which can be calculated
			#print('Starting with last osci!')
			print('Calculate beta%i%i' % (i, i - 1))
			# calculating beta_(N)(N-1) if counting from k = 1 to N
			asymptotic_phase_differences_entrained_sync_state[i] = -2*np.pi*dict_pll['intrF'][0]*dict_pll['transmission_delay'] - dict_pll['div']*dict_pll['inverse_coup_fct_sig'](
				(dict_pll['inverse_fct_vco_response'](dict_pll['intrF'][0]) - frequency_or_voltage[i]) / dict_pll['coupK'][i],
					branch=dict_pll['branch_of_inverse_coupling_fct_if_applies'], phase_wrap=phase_wrap)
		else:
			print('Calculate beta%i%i' % (i, i - 1))
			#print('Now processing the remaining!')
			asymptotic_phase_differences_entrained_sync_state[i] = -2*np.pi*dict_pll['intrF'][0]*dict_pll['transmission_delay'] - dict_pll['div']*dict_pll['inverse_coup_fct_sig'](
				(2*(dict_pll['inverse_fct_vco_response'](dict_pll['intrF'][0]) - frequency_or_voltage[i]))/dict_pll['coupK'][i] - dict_pll['coup_fct_sig'](
					(-2*np.pi*dict_pll['intrF'][0]*dict_pll['transmission_delay'] - asymptotic_phase_configuration_entrained_sync_state[i+1]) / dict_pll['div']),
						branch=dict_pll['branch_of_inverse_coupling_fct_if_applies'], phase_wrap=phase_wrap)

	# calculate the phase-offsets from the phase-differences for all oscillators, assuming that beta_0 = 0!
	print('CHECK: asymptotic_phase_differences_entrained_sync_state[0]=', asymptotic_phase_differences_entrained_sync_state[0])
	for i in range(1, dict_net['Nx']):
		# NOTE: asymptotic_phase_differences_entrained_sync_state has been filled from the end towards the beginning... the first entry is 0, representing beta_0=0
		# so we start \beta_

		asymptotic_phase_configuration_entrained_sync_state[i] = asymptotic_phase_configuration_entrained_sync_state[i-1] - asymptotic_phase_differences_entrained_sync_state[i]

	shift2piWin = 0.0
	if phase_wrap == 1:  # plot phase-differences in [-pi, pi] interval
		shift2piWin = np.pi
	elif phase_wrap == 2:  # plot phase-differences in [-pi/2, 3*pi/2] interval
		shift2piWin = 0.5 * np.pi
	elif phase_wrap == 3:  # plot phase-differences in [0, 2*pi] interval
		shift2piWin = 0.0

	asymptotic_phase_configuration_entrained_sync_state = [(asymptotic_phase_configuration_entrained_sync_state[i] + shift2piWin) % (2*np.pi) - shift2piWin for i in range(len(asymptotic_phase_configuration_entrained_sync_state))]

	if not any(np.isnan(asymptotic_phase_differences_entrained_sync_state)):
		print('\nComputed initial phase-differences for {tau=%0.2f, fR=%0.2f} to be dphi_kl=phi_k-phi_l' % (dict_pll['transmission_delay'],
																							dict_pll['intrF'][0]), asymptotic_phase_differences_entrained_sync_state)
		print('\nComputed initial phases for {tau=%0.2f, fR=%0.2f} to be phi_k=' % (dict_pll['transmission_delay'],
																							dict_pll['intrF'][0]), asymptotic_phase_configuration_entrained_sync_state)
		# print('dict_pll[*inverse_fct_vco_response*](dict_pll[*intrF*][0])', dict_pll['inverse_fct_vco_response'](dict_pll['intrF'][0]))
		# time.sleep(10)

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


