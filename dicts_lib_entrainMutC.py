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
gc.enable()

#%%cython --annotate -c=-O3 -c=-march=native

''' THIS SIMULATES A 2ND ORDER KURAMOTO MODEL -- PROPERLY PREPROCESS ALL GAINS AND DETAILS WHEN COMPARING TO PLL CIRCUITRY '''

def getDicts(Fsim=125):

	dictNet={
		'Nx': 3,																# oscillators in x-direction
		'Ny': 1,																# oscillators in y-direction
		'mx': 0	,																# twist/chequerboard in x-direction (depends on closed or open boundary conditions)
		'my': 0,																# twist/chequerboard in y-direction
		'topology': 'entrainOne-ring',#'square-open',							# 1d) ring, chain, 2d) square-open, square-periodic, hexagonal...
																				# 3) global, entrainOne-[ring,chain,square-open], entrainAll-[ring,chain,square-open], entrainPLLsHierarch, compareEntrVsMutual
		'Tsim': 500,															# simulation time in multiples of the period
		'computeFreqAndStab':  False,											# compute linear stability and global frequency if possible: True or False
		'phi_array_mult_tau': 1,												# how many multiples of the delay is stored of the phi time series
		'phiPerturb': [0 for i in range(3)],									# delta-perturbation on initial state -- PROVIDE EITHER ONE OF THEM! if [] set to zero
		'phiPerturbRot': [], #[0., 0.1],										# delta-perturbation on initial state -- in rotated space
		'phiInitConfig': [0, 0, 0],#[0 for i in range(3)],				# phase-configuration of sync state,  []: automatic, else provide list
		'freq_beacons': 0.1,													# frequency of external sender beacons, either a float or a list
		'special_case': 'False',#'timeDepTransmissionDelay',					# 'False', or 'test_case', 'timeDepInjectLockCoupStr', 'timeDepTransmissionDelay', 'timeDepChangeOfCoupStr', 'distanceDepTransmissionDelay'
		'typeOfTimeDependency': 'triangle',										# 'exponential', 'linear', 'quadratic', 'triangle', 'cosine'
		'min_max_rate_timeDepPara': [0.01, 1.51, 0.0025/100]					# provide a list with min, max and rate of change per unit time of the time-dependent parameter
	}

	dictPLL={
		'intrF': [2.4, 1.004, 0.996],											# intrinsic frequency in Hz
		'syncF': 2.4,															# frequency of synchronized state in Hz
		'coupK': [0., 0.408, 0.4225],											# [random.uniform(0.3, 0.4) for i in range(dictNet['Nx']*dictNet['Ny'])],# coupling strength (like phase model: K = Kvco/2 * G_all, NOTE: the /2 is for coupling functions that have peak2peal amplitude 2) in Hz float or [random.uniform(minK, maxK) for i in range(dictNet['Nx']*dictNet['Ny'])]
		'gPDin': 8.5,																# gains of the different inputs to PD k from input l -- G_kl, see PD, set to 1 and all G_kl=1 (so far only implemented for some cases, check!): np.random.uniform(0.95,1.05,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']])
		'gPDin_symmetric': True,												# set to True if G_kl == G_lk, False otherwise
		'cutFc': [0., 0.0156, 0.0156],											# LF cut-off frequency in Hz, None for no LF, or e.g., N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05], [0.0148, 0.0148, 0.0957, 0.0957, 0.0148, 0.0148, 0.0957, 0.0957, 0.0148, 0.0148, 0.0957, 0.0957, 0.0148, 0.0148, 0.0957, 0.0957] # [0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957], #
		'orderLF': 1,															# order of LF filter, either 1 or 2 at the moment (not compatible with synctools!)
		'div': 1,																# divisor of divider (int)
		'friction_coefficient': 1,												# friction coefficient of 2nd order Kuramoto models
		'fric_coeff_PRE_vs_PRR': 'PRE',											# 'PRR': friction coefficient multiplied to instant. AND intrin. freq, 'PRE': friction coefficient multiplied only to instant. freq
		'noiseVarVCO': [0, 5E-8, 1E-9],											# variance of VCO GWN
		'feedback_delay': 0,													# value of feedback delay in seconds
		'feedback_delay_var': None, 											# variance of feedback delay
		'transmission_delay': 0.256, #5.0, 										# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]), OR [np.random.uniform(min,max) for i in range(dictNet['Nx']*dictNet['Ny'])]
		'transmission_delay_var': None, 										# variance of transmission delays
		'distribution_for_delays': None,										# from what distribution are random delays drawn?
		# choose from coupfct.<ID>: sine, cosine, neg_sine, neg_cosine, triangular, deriv_triangular, square_wave, pfd, inverse_cosine, inverse_sine
		'coup_fct_sig': coupfct.triangular,	#coupfct.sine,						# coupling function h(x) for PLLs with ideally filtered PD signals:
		'derivative_coup_fct': coupfct.deriv_triangular,						# derivative h'(x) of coupling function h(x)
		'includeCompHF': False,													# boolean True/False whether to simulate with HF components
		'vco_out_sig': coupfct.square_wave_symm_zero,							# for HF case, e.g.: coupfct.sine, coupfct.square_wave, coupfct.square_wave_symm_zero
		'typeVCOsig': 'analogHF',												# 'analogHF' or 'digitalHF' - determines whether XOR PD or multiplier is used
		'responseVCO': 'linear',												# either string: 'linear' or a nonlinear function of omega, Kvco, e.g., lambda w, K, ...: expression
		'antenna': False,														# boolean True/False whether antenna present for PLLs
		'posX': 0,																# antenna position of PLL k -- x, y z coordinates, need to be set
		'posY': 0,
		'posZ': 0,
		'initAntennaState': 0,
		'antenna_sig': coupfct.sine,											# type of signal received by the antenna
		'coup_fct_phase_shift': 0*np.pi,										# phase shift in the coupling function in [0, 2pi], so far constant... may later dynamic
		'extra_coup_sig': None,													# choose from: 'injection2ndHarm', None
		'coupStr_2ndHarm': 0.6,													# the coupling constant for the injection of the 2nd harmonic: float, will be indepent of 'coupK'
		'typeOfHist': 'syncState',#'freeRunning',#								# string, choose from: 'freeRunning', 'syncState'
		'sampleF': Fsim,														# sampling frequency
		'sampleFplot': 5,														# sampling frequency for reduced plotting (every sampleFplot time step)
		'treshold_maxT_to_plot': 1E6,											# maximum number of periods to plot for some plots
		'percentPeriodsAverage': 0.15,											# average of *percentPeriodsAverage* % of simulated periods
		'PSD_freq_resolution': 1E-5,											# frequency resolution aimed at with PSD: hence, T_analyze ~ 1/f
		'PSD_from_signal': coupfct.square_wave,									# fore the PSD the following signals will be generated from the phases: coupfct.sine, coupfct.square_wave, coupfct.square_wave_symm_zero
		'signal_propagation_speed': 0.0,										# speed of signal transmission when considering mobile oscillators --> mode: 'distanceDepTransmissionDelay'
		'space_dimensions_xyz': [10, 10, 10]									# dimension of the 3d space in which mobile oscillators can be simulated --> mode: 'distanceDepTransmissionDelay'
	}

	dictAlgo={
		'bruteForceBasinStabMethod': 'listOfInitialPhaseConfigurations',		# pick method for setting realizations 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations'
		'paramDiscretization': [7, 1],#[15, 10],								# parameter discetization for brute force parameter space scans
		'param_id': 'None',														# parameter to be changed between different realizations, according to the min_max_range_parameter: 'None' or string of any other parameter
		'min_max_range_parameter': [0.95, 1.05],								# specifies within which min and max value to linspace the initial frequency difference (w.r.t. HF Frequency, not divided)
		'store_ctrl_and_clock': True,											# whether or not the control signals and clock signal is being computed (time and memory usage)
		'store_phases_tau_array': True											# whether or not the phases are saved when simulation on tau-array
	}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	dictPLL, dictNet, dictAlgo = chk_dicts.check_dicts_consistency(dictPLL, dictNet, dictAlgo)

	return dictPLL, dictNet, dictAlgo