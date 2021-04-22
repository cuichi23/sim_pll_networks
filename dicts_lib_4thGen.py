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
		'Nx': 2,																# oscillators in x-direction
		'Ny': 1,																# oscillators in y-direction
		'mx': 0,																# twist/chequerboard in x-direction (depends on closed or open boundary conditions)
		'my': 0,																# twist/chequerboard in y-direction
		'topology': 'ring',														# 1d) ring, chain, 2d) square-open, square-periodic, hexagonal...
																				# 3) global, entrainOne, entrainAll, entrainPLLsHierarch, compareEntrVsMutual
		'Tsim': 50000,															# simulation time in multiples of the period
		'computeFreqAndStab': True,												# compute linear stability and global frequency if possible: True or False
		'phi_array_mult_tau': 1,												# how many multiples of the delay is stored of the phi time series
		'phiPerturb': [],														# delta-perturbation on initial state -- PROVIDE EITHER ONE OF THEM! if [] set to zero
		'phiPerturbRot': [],													# delta-perturbation on initial state -- in rotated space
		'phiInitConfig': [],													# phase-configuration of sync state,  []: automatic, else provide list
		'freq_beacons': 0.25,													# frequency of external sender beacons, either a float or a list
		'test_case': False														# True: run testcase sim, False: run other simulation mode
	}

	dictPLL={
		'intrF': 1.0,															# intrinsic frequency in Hz
		'syncF': 1.0,															# frequency of synchronized state in Hz
		'coupK': 27E-3,															#[random.uniform(0.3, 0.4) for i in range(dictNet['Nx']*dictNet['Ny'])],# coupling strength in Hz float or [random.uniform(minK, maxK) for i in range(dictNet['Nx']*dictNet['Ny'])]
		'gPDin': 1,																# gains of the different inputs to PD k from input l -- G_kl, see PD, set to 1 and all G_kl=1 (so far only implemented for some cases, check!): np.random.uniform(0.95,1.05,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']])
		'gPDin_symmetric': True,												# set to True if G_kl == G_lk, False otherwise
		'cutFc': 1.1E-4,															# LF cut-off frequency in Hz, here N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05]
		'div': 512,																# divisor of divider (int)
		'noiseVarVCO': 1E-9,													# variance of VCO GWN
		'feedback_delay': 0,													# value of feedback delay in seconds
		'feedback_delay_var': None, 											# variance of feedback delay
		'transmission_delay': 0.55, 												# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]), OR [np.random.uniform(min,max) for i in range(dictNet['Nx']*dictNet['Ny'])]
		'transmission_delay_var': None, 										# variance of transmission delays
		'distribution_for_delays': None,										# from what distribution are random delays drawn?
		'posX': 0,																# antenna position of PLL k -- x, y z coordinates, need to be set
		'posY': 0,
		'posZ': 0,
		'coup_fct_sig': lambda x: np.cos(x),									# coupling function for PLLs with ideally filtered PD signals:
		# mixer+1sig shift: np.sin(x), mixer: np.cos(x), XOR: sawtooth(x,width=0.5), PSD: 0.5*(np.sign(x)*(1+sawtooth(1*x*np.sign(x), width=1)))
		'derivative_coup_fct': lambda x: -np.sin(x),								# derivative of coupling function h
		'includeCompHF': False,													# boolean True/False whether to simulate with HF components
		'typeVCOsig': 'analogHF',												# 'analogHF' or 'digitalHF'
		'responseVCO': 'linear',												# either string: 'linear' or a nonlinear function of omega, Kvco, e.g., lambda w, K, ...: expression
		'vco_out_sig': lambda x: 0.5*(1.0+square(x,duty=0.5)),					# for HF case, e.g.: np.sin(x) or 0.5*(1.0+square(x,duty=0.5))
		'antenna': False,														# boolean True/False whether antenna present for PLLs
		'initAntennaState': 0,
		'antenna_sig': lambda x: np.sin(x),										# type of signal received by the antenna
		'extra_coup_sig': None,													# choose from: 'injection2ndHarm', None
		'coupStr_2ndHarm': 0.6,													# the coupling constant for the injection of the 2nd harmonic: float, will be indepent of 'coupK'
		'typeOfHist': 'syncState',												# string, choose from: 'freeRunning', 'syncState'
		'sampleF': Fsim,														# sampling frequency
		'sampleFplot': 5,														# sampling frequency for reduced plotting (every sampleFplot time step)
		'treshold_maxT_to_plot': 1E6,											# maximum number of periods to plot for some plots
		'percentPeriodsAverage': 0.7											# average of *percentPeriodsAverage* % of simulated periods
	}

	# calculate other parameters and test for incompatibilities
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

	if dictNet['computeFreqAndStab']:
		#try:
		isRadian = False														# set this False to get values returned in [Hz] instead of [rad * Hz]
		sf = synctools.SweepFactory(dictPLL, dictNet, isRadians=isRadian)
		fsl = sf.sweep()
		para_mat = fsl.get_parameter_matrix(isRadians=False)				    # extract variables from the sweep, this matrix contains all cases
		print('New parameter combinations with {intrF, coupK, cutFc, delay, Omega, ReLambda, ImLambda, TsimToPert1/e, Nx, Ny, mx, my, div}: \n', para_mat)
		choice = chooseSolution(para_mat)
		dictPLL.update({'syncF': para_mat[choice,4], 'ReLambda': para_mat[choice,5], 'ImLambda': para_mat[choice,6]})
		#except:
		#	print('Could not compute linear stability and global frequency! Check synctools and case!')

	print('Setup (dictNet, dictPLL):', dictNet, dictPLL)

	return dictPLL, dictNet

# ******************************************************************************

def chooseSolution(para_mat):													# ask user-input for which solution to simulate
	a_true = True
	while a_true:
		# get user input which of the possible cases to simulate
		choice = input('Choose which case to simulate [0,...,%i]:'%(len(para_mat[:,0])-1))
		if int(choice) >= 0 and int(choice) < len(para_mat[:,0]):
			break
		else:
			print('Please provide input as integer choice!')

	return int(choice)
