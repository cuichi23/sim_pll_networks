#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys, gc
import numpy as np
import random
#cimport numpy as np
#cimport cython
import networkx as nx
from scipy.signal import sawtooth
from scipy.signal import square
from scipy.stats import cauchy

#import matplotlib
#import matplotlib.pyplot as plt
import datetime
import time

from sim_pll import check_dicts_lib as chk_dicts
from sim_pll import coupling_fct_lib as coupfct

''' Enable automatic carbage collector '''
gc.enable();

#%%cython --annotate -c=-O3 -c=-march=native

''' THIS SIMULATES A 2ND ORDER KURAMOTO MODEL -- PROPERLY PREPROCESS ALL GAINS AND DETAILS WHEN COMPARING TO PLL CIRCIUTRY '''

def getDicts(Fsim=125):

	dictNet={
		'Nx': 4,																# oscillators in x-direction
		'Ny': 1,																# oscillators in y-direction
		'mx': 0,																# twist/chequerboard in x-direction (depends on closed or open boundary conditions)
		'my': -999,																# twist/chequerboard in y-direction
		'topology': 'global',													# 1d) ring, chain, 2d) square-open, square-periodic, hexagonal...
																				# 3) global, entrainOne, entrainAll, entrainPLLsHierarch, compareEntrVsMutual
		'Tsim': 50,															# simulation time in multiples of the period
		'computeFreqAndStab': False,											# compute linear stability and global frequency if possible: True or False (only identical oscis and Kuramoto 2nd order)
		'phi_array_mult_tau': 1,												# how many multiples of the delay is stored of the phi time series
		'phiInitConfig': [],#[0, np.pi, np.pi, np.pi],							# phase-configuration of sync state,  []: automatic, else provide list
		'freq_beacons': 0.25,													# frequency of external sender beacons, either a float or a list
		'special_case': 'timeDepInjectLockCoupStr',								# 'False', or 'test_case', 'timeDepInjectLockCoupStr', 'timeDepTransmissionDelay', 'timeDepChangeOfCoupStr'
		'typeOfTimeDependency': 'exponential',#'linear',						# 'exponential', 'linear', 'quadratic', 'triangle', 'cosine'
		'min_max_rate_timeDepPara': [0, 0.1, 0.1/(5/125)]						# provide a list with min, max and rate (per period) of the time-dependent parameter
	}

	dictNet.update({
		'phiPerturb': [np.random.uniform(-2,2) for i in range(dictNet['Nx']*dictNet['Ny'])], # delta-perturbation on initial state -- PROVIDE EITHER ONE OF THEM! if [] set to zero
		'phiPerturbRot': [],													# delta-perturbation on initial state -- in rotated space
	})

	dictPLL={
		'intrF': 1.0, #[1.0, 1.01, 0.99, 1.02],										# intrinsic frequency in Hz
		'syncF': 1.0,															# frequency of synchronized state in Hz
		'coupK': 0.25,														# [random.uniform(0.3, 0.4) for i in range(dictNet['Nx']*dictNet['Ny'])],# coupling strength (like phase model: K = Kvco/2 * G_all, NOTE: the /2 is for coupling functions that have peak2peal amplitude 2) in Hz float or [random.uniform(minK, maxK) for i in range(dictNet['Nx']*dictNet['Ny'])]
		'gPDin': 1,#np.array([[0,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]]), #np.random.randint(0,2,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]),# gains of the different inputs to PD k from input l -- G_kl, see PD, set to 1 and all G_kl=1 (so far only implemented for some cases, check!): np.random.uniform(0.95,1.05,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']])
		'gPDin_symmetric': False,												# set to True if G_kl == G_lk, False otherwise
		'cutFc': None,															# LF cut-off frequency in Hz, None for no LF, here N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05]
		'orderLF': 1,															# order of LF filter, either 1 or 2 at the moment
		'div': 1,																# divisor of divider (int)
		'friction_coefficient': 1,												# friction coefficient of 2nd order Kuramoto models
		'fric_coeff_PRE_vs_PRR': 'PRE',											# 'PRR': friction coefficient multiplied to instant. AND intrin. freq, 'PRE': friction coefficient multiplied only to instant. freq
		'noiseVarVCO': 1E-9,													# variance of VCO GWN
		'feedback_delay': 0,													# value of feedback delay in seconds
		'feedback_delay_var': None, 											# variance of feedback delay
		'transmission_delay': 0.0,	 											# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]), OR [np.random.uniform(min,max) for i in range(dictNet['Nx']*dictNet['Ny'])]
		'transmission_delay_var': None, 										# variance of transmission delays
		'distribution_for_delays': None,										# from what distribution are random delays drawn?
		# choose from coupfct.<ID>: sine, cosine, neg_sine, neg_cosine, triangular, deriv_triangular, square_wave, pfd, inverse_cosine, inverse_sine
		'coup_fct_sig': coupfct.sine,											# coupling function h(x) for PLLs with ideally filtered PD signals:
		'derivative_coup_fct': coupfct.cosine,									# derivative h'(x) of coupling function h(x)
		'includeCompHF': False,													# boolean True/False whether to simulate with HF components
		'vco_out_sig': coupfct.square_wave_symm_zero,							# for HF case, e.g.: coupfct.sine, coupfct.square_wave, coupfct.square_wave_symm_zero
		'typeVCOsig': 'analogHF',												# 'analogHF' or 'digitalHF'
		'responseVCO': 'linear',												# either string: 'linear' or a nonlinear function of omega, Kvco, e.g., lambda w, K, ...: expression
		'antenna': False,														# boolean True/False whether antenna present for PLLs
		'posX': 0,																# antenna position of PLL k -- x, y z coordinates, need to be set
		'posY': 0,
		'posZ': 0,
		'initAntennaState': 0,
		'antenna_sig': coupfct.sine,											# type of signal received by the antenna
		'extra_coup_sig': 'injection2ndHarm',									# choose from: 'injection2ndHarm', None
		'coupStr_2ndHarm': 1,												# the coupling constant for the injection of the 2nd harmonic: float, will be indepent of 'coupK'
		'typeOfHist': 'syncState',												# string, choose from: 'freeRunning', 'syncState'
		'sampleF': Fsim,														# sampling frequency
		'sampleFplot': 5,														# sampling frequency for reduced plotting (every sampleFplot time step)
		'treshold_maxT_to_plot': 50E3,											# maximum number of periods to plot for some plots
		'percentPeriodsAverage': 0.15,											# average of *percentPeriodsAverage* % of simulated periods
		'PSD_freq_resolution': 1E-4,											# frequency resolution aimed at with PSD: hence, T_analyze ~ 1/f
		'signal_propagation_speed': 0.0,										# speed of signal transmission when considering mobile oscillators --> mode: 'distanceDepTransmissionDelay'
		'space_dimensions_xyz': [10, 10, 10]									# dimension of the 3d space in which mobile oscillators can be simulated --> mode: 'distanceDepTransmissionDelay'
	}

	dictAlgo={
		'bruteForceBasinStabMethod': 'testNetworkMotifIsing',					# pick method for setting realizations 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'testNetworkMotifIsing'
		'paramDiscretization': 2,												# parameter discetization for brute force parameter space scans
		'min_max_range_parameter': [0.95, 1.05]									# specifies within which min and max value to linspace the detuning
	}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	dictPLL, dictNet, dictAlgo = chk_dicts.check_dicts_consistency(dictPLL, dictNet, dictAlgo)

	return dictPLL, dictNet, dictAlgo
