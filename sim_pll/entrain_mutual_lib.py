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
		phase_configuration_ref_to_one_for_chain_topology(dict_net, dict_pll, dict_algo)

	return None


def phase_configuration_ref_to_one_for_chain_topology(dict_net: dict, dict_pll: dict, dict_algo: dict) -> None:
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
	asymtotic_phase_configuration_entrained_sync_state = []

	-dict_pll['intrF'][0] * dict_pll['transmission_delay'] - dict_pll['intrF']*dict_pll['inverse_coup_fct_sig'](  )

	'responseVCO': 'linear',  # either string: 'linear' or a nonlinear function of omega, Kvco, e.g., lambda w, K, ...: expression


	asymtotic_phase_configuration_entrained_sync_state.append()
	for i in range(dict_pll['Nx']):
		asymtotic_phase_configuration_entrained_sync_state.append[]



	dict_net.update({'phiInitConfig': asymtotic_phase_configuration_entrained_sync_state})

	return None
