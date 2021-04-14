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

''' Enable automatic carbage collector '''
gc.enable();

#%%cython --annotate -c=-O3 -c=-march=native

''' THIS SIMULATES A 2ND ORDER KURAMOTO MODEL -- PROPERLY PREPROCESS ALL GAINS AND DETAILS WHEN COMPARING TO PLL CIRCIUTRY '''

def getDicts(Fsim=125):

	dictNet={
		'Nx': 20,																# oscillators in x-direction
		'Ny': 1,																# oscillators in y-direction
		'mx': 0,																# twist/chequerboard in x-direction (depends on closed or open boundary conditions)
		'my': 0,																# twist/chequerboard in y-direction
		'topology': 'global',													# 1d) ring, chain, 2d) square-open, square-periodic, hexagonal...
																				# 3) global, entrainOne, entrainAll, entrainPLLsHierarch, compareEntrVsMutual
		'Tsim': 2500,															# simulation time in multiples of the period
		'phi_array_mult_tau': 1,												# how many multiples of the delay is stored of the phi time series
		'phiPerturb': [np.random.uniform(-1.01,1.01) for i in range(20)],		# delta-perturbation on initial state -- PROVIDE EITHER ONE OF THEM! if [] set to zero
		'phiPerturbRot': [],													# delta-perturbation on initial state -- in rotated space
		'phiInitConfig': [],													# phase-configuration of sync state,  []: automatic, else provide list
		'test_case': True														# True: run testcase sim, False: run other simulation mode
	}

	dictPLL={
		'intrF': 1.0,															# intrinsic frequency in Hz
		'syncF': 1.0,															# frequency of synchronized state in Hz
		'coupK': 0.3,#[random.uniform(0.3, 0.4) for i in range(dictNet['Nx']*dictNet['Ny'])],# coupling strength in Hz float or [random.uniform(minK, maxK) for i in range(dictNet['Nx']*dictNet['Ny'])]
		'gPDin': np.random.uniform(-10,10,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]),# gains of the different inputs to PD k from input l -- G_kl, see PD, set to 1 and all G_kl=1 (so far only implemented for some cases, check!): np.random.uniform(0.95,1.05,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']])
		'gPDin_symmetric': True,												# set to True if G_kl == G_lk, False otherwise
		'cutFc': None,															# LF cut-off frequency in Hz, here N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05]
		'div': 1,																# divisor of divider (int)
		'noiseVarVCO': 1E-9,													# variance of VCO GWN
		'feedback_delay': 0,													# value of feedback delay in seconds
		'feedback_delay_var': None, 											# variance of feedback delay
		'transmission_delay': 0,	 											# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]), OR [np.random.uniform(min,max) for i in range(dictNet['Nx']*dictNet['Ny'])]
		'transmission_delay_var': None, 										# variance of transmission delays
		'distribution_for_delays': None,										# from what distribution are random delays drawn?
		'posX': 0,																# antenna position of PLL k -- x, y z coordinates, need to be set
		'posY': 0,
		'posZ': 0,
		'coup_fct_sig': lambda x: np.sin(x),									# coupling function for PLLs with ideally filtered PD signals:
		# mixer+1sig shift: np.sin(x), mixer: np.cos(x), XOR: sawtooth(x,width=0.5), PSD: 0.5*(np.sign(x)*(1+sawtooth(1*x*np.sign(x), width=1)))
		'derivative_coup_fct': lambda x: np.cos(x),								# derivative of coupling function h
		'includeCompHF': False,													# boolean True/False whether to simulate with HF components
		'typeVCOsig': 'analogHF',												# 'analogHF' or 'digitalHF'
		'vco_out_sig': lambda x: 0.5*(1.0+square(x,duty=0.5)),					# for HF case, e.g.: np.sin(x) or 0.5*(1.0+square(x,duty=0.5))
		'antenna': False,														# boolean True/False whether antenna present for PLLs
		'initAntennaState': 0,
		'antenna_sig': lambda x: np.sin(x),										# type of signal received by the antenna
		'extra_coup_sig': 'injection2ndHarm',									# choose from: 'injection2ndHarm', None
		'responseVCO': 'linear',												# either string: 'linear' or a nonlinear function of omega, Kvco, e.g., lambda w, K, ...: expression
		'typeOfHist': 'syncState',												# string, choose from: 'freeRunning', 'syncState'
		'sampleF': Fsim,														# sampling frequency
		'sampleFplot': 5,														# sampling frequency for reduced plotting (every sampleFplot time step)
		'treshold_maxT_to_plot': 25E3,											# maximum number of periods to plot for some plots
		'percentPeriodsAverage': 0.7											# average of *percentPeriodsAverage* % of simulated periods
	}

	dictPLL.update({'dt': 1.0/dictPLL['sampleF']})
	if ( isinstance(dictPLL['gPDin'], np.ndarray) and dictPLL['gPDin_symmetric']):

		print('Generate symmetrical matrix for PD gains.')
		dictPLL.update({'gPDin': (dictPLL['gPDin']@dictPLL['gPDin'].T)/np.max(dictPLL['gPDin']@dictPLL['gPDin'].T)})

	if dictPLL['intrF'] > 1E-3:

		dictNet.update({'Tsim': dictNet['Tsim']*(1.0/dictPLL['intrF'])})		# simulation time in multiples of the period of the uncoupled oscillators
		dictPLL.update({'sim_time_steps': int(dictNet['Tsim']/dictPLL['dt'])})
		print('Total simulation time in multiples of the eigentfrequency:', int(dictNet['Tsim']*dictPLL['intrF']))
	else:

		print('Tsim not in multiples of T_omega, since F <= 1E-3')
		dictNet.update({'Tsim': dictNet['Tsim']*2})
		dictPLL.update({'sim_time_steps': int(dictNet['Tsim']/dictPLL['dt'])})
		print('Total simulation time in seconds:', int(dictNet['Tsim']))

	dictPLL.update({'timeSeriesAverTime': int(dictPLL['percentPeriodsAverage']*dictNet['Tsim']*dictPLL['syncF'])})

	if dictPLL['typeVCOsig'] == 'analogHF':
		dictPLL.update({})

	print('Setup (dictNet, dictPLL):', dictNet, dictPLL)

	return dictPLL, dictNet
