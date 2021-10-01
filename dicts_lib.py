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

import coupling_fct_lib as coupfct
import synctools_interface_lib as synctools

''' Enable automatic carbage collector '''
gc.enable();

#%%cython --annotate -c=-O3 -c=-march=native

''' THIS SIMULATES A 2ND ORDER KURAMOTO MODEL -- PROPERLY PREPROCESS ALL GAINS AND DETAILS WHEN COMPARING TO PLL CIRCUITRY '''

def getDicts(Fsim=125):

	dictNet={
		'Nx': 2,																# oscillators in x-direction
		'Ny': 1,																# oscillators in y-direction
		'mx': 0	,																# twist/chequerboard in x-direction (depends on closed or open boundary conditions)
		'my': -999,																# twist/chequerboard in y-direction
		'topology': 'ring',														# 1d) ring, chain, 2d) square-open, square-periodic, hexagonal...
																				# 3) global, entrainOne, entrainAll, entrainPLLsHierarch, compareEntrVsMutual
		'Tsim': 15000,															# simulation time in multiples of the period
		'computeFreqAndStab':  False,											# compute linear stability and global frequency if possible: True or False
		'phi_array_mult_tau': 1,												# how many multiples of the delay is stored of the phi time series
		'phiPerturb': [],		#[0, -0.001, 0, 0, 0.001, 0, 0, -0.001, 0],		# delta-perturbation on initial state -- PROVIDE EITHER ONE OF THEM! if [] set to zero
		'phiPerturbRot': [0., 0.1],													# delta-perturbation on initial state -- in rotated space
		'phiInitConfig': [],													# phase-configuration of sync state,  []: automatic, else provide list
		'freq_beacons': 0.1,													# frequency of external sender beacons, either a float or a list
		'special_case': 'False',#'timeDepTransmissionDelay',					# 'False', or 'test_case', 'timeDepInjectLockCoupStr', 'timeDepTransmissionDelay', 'timeDepChangeOfCoupStr'
		'typeOfTimeDependency': 'linear',										# 'exponential', 'linear', 'quadratic', 'triangle', 'cosine'
		'min_max_rate_timeDepPara': [0.15, 3.85, 0.4/100]						# provide a list with min, max and rate of the time-dependent parameter
	}

	dictPLL={
		'intrF': 1, #[0.95, 1.05],												# intrinsic frequency in Hz
		'syncF': 1.0,															# frequency of synchronized state in Hz
		'coupK': 0.2,															#[random.uniform(0.3, 0.4) for i in range(dictNet['Nx']*dictNet['Ny'])],# coupling strength in Hz float or [random.uniform(minK, maxK) for i in range(dictNet['Nx']*dictNet['Ny'])]
		'gPDin': 1,																# gains of the different inputs to PD k from input l -- G_kl, see PD, set to 1 and all G_kl=1 (so far only implemented for some cases, check!): np.random.uniform(0.95,1.05,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']])
		'gPDin_symmetric': True,												# set to True if G_kl == G_lk, False otherwise
		'cutFc': 0.014,	# LF cut-off frequency in Hz, None for no LF, or e.g., N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05], [0.0148, 0.0148, 0.0957, 0.0957, 0.0148, 0.0148, 0.0957, 0.0957, 0.0148, 0.0148, 0.0957, 0.0957, 0.0148, 0.0148, 0.0957, 0.0957] # [0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957, 0.0148, 0.0957], #
		'orderLF': 2,															# order of LF filter, either 1 or 2 at the moment (not compatible with synctools!)
		'div': 1,																# divisor of divider (int)
		'friction_coefficient': 1,												# friction coefficient of 2nd order Kuramoto models
		'fric_coeff_PRE_vs_PRR': 'PRE',											# 'PRR': friction coefficient multiplied to instant. AND intrin. freq, 'PRE': friction coefficient multiplied only to instant. freq
		'noiseVarVCO': 0,														# variance of VCO GWN
		'feedback_delay': 0,													# value of feedback delay in seconds
		'feedback_delay_var': None, 											# variance of feedback delay
		'transmission_delay': 0.25, #5.0, 												# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]), OR [np.random.uniform(min,max) for i in range(dictNet['Nx']*dictNet['Ny'])]
		'transmission_delay_var': None, 										# variance of transmission delays
		'distribution_for_delays': None,										# from what distribution are random delays drawn?
		# choose from coupfct.<ID>: sine, cosine, neg_sine, neg_cosine, triangular, deriv_triangular, square_wave, pfd, inverse_cosine, inverse_sine
		'coup_fct_sig': coupfct.sine, #coupfct.triangular,						# coupling function h(x) for PLLs with ideally filtered PD signals:
		'derivative_coup_fct': coupfct.cosine,									# derivative h'(x) of coupling function h(x)
		'includeCompHF': False,													# boolean True/False whether to simulate with HF components
		'vco_out_sig': coupfct.sine,										# for HF case, e.g.: coupfct.sine or coupfct.square_wave
		'typeVCOsig': 'analogHF',												# 'analogHF' or 'digitalHF'
		'responseVCO': 'linear',												# either string: 'linear' or a nonlinear function of omega, Kvco, e.g., lambda w, K, ...: expression
		'antenna': False,														# boolean True/False whether antenna present for PLLs
		'posX': 0,																# antenna position of PLL k -- x, y z coordinates, need to be set
		'posY': 0,
		'posZ': 0,
		'initAntennaState': 0,
		'antenna_sig': coupfct.sine,											# type of signal received by the antenna
		'extra_coup_sig': None,													# choose from: 'injection2ndHarm', None
		'coupStr_2ndHarm': 0.6,													# the coupling constant for the injection of the 2nd harmonic: float, will be indepent of 'coupK'
		'typeOfHist': 'syncState',#'freeRunning',#								# string, choose from: 'freeRunning', 'syncState'
		'sampleF': Fsim,														# sampling frequency
		'sampleFplot': 1,														# sampling frequency for reduced plotting (every sampleFplot time step)
		'treshold_maxT_to_plot': 1E6,											# maximum number of periods to plot for some plots
		'percentPeriodsAverage': 0.25											# average of *percentPeriodsAverage* % of simulated periods
	}

	dictAlgo={
		'bruteForceBasinStabMethod': 'listOfInitialPhaseConfigurations',		# pick method for setting realizations 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations'
		'paramDiscretization': [5, 3],#3										# parameter discetization for brute force parameter space scans
		'min_max_range_parameter': [0.95, 1.05]									# specifies within which min and max value to linspace the detuning
	}
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	if dictPLL['coup_fct_sig'] == coupfct.triangular or dictPLL['coup_fct_sig'] == coupfct.deriv_triangular or dictPLL['coup_fct_sig'] == coupfct.square_wave or dictPLL['coup_fct_sig'] == coupfct.pfd:
		if dictPLL['vco_out_sig'] == coupfct.sine:
			print('A coupling function associated to oscillators with DIGITAL output is chosen. Current PSD choice is only to analyze first harmonic contribution! Switch to full digital [y]?')
			choice = choose_yes_no()
			if choice == 'y':
				dictPLL.update({'vco_out_sig': coupfct.square_wave})
			elif choice == 'n':
				dictPLL.update({'vco_out_sig': coupfct.sine})
	elif dictPLL['coup_fct_sig'] == coupfct.sine or dictPLL['coup_fct_sig'] == coupfct.cosine:
		if dictPLL['vco_out_sig'] == coupfct.square_wave:
			print('A coupling function associated to oscillators with ANALOG output is chosen. Current PSD choice is to analyze a square wave signal of the phase! Switch to sine wave [y]?')
			choice = choose_yes_no()
			if choice == 'y':
				dictPLL.update({'vco_out_sig': coupfct.sine})
			elif choice == 'n':
				dictPLL.update({'vco_out_sig': coupfct.square_wave})
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# calculate other parameters and test for incompatibilities
	dictPLL.update({'dt': 1.0/dictPLL['sampleF']})
	if ( isinstance(dictPLL['gPDin'], np.ndarray) and dictPLL['gPDin_symmetric']):

		print('Generate symmetrical matrix for PD gains.')
		dictPLL.update({'gPDin': (dictPLL['gPDin']@dictPLL['gPDin'].T)/np.max(dictPLL['gPDin']@dictPLL['gPDin'].T)})
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	if np.all(dictPLL['intrF']) > 1E-3:

		dictNet.update({'Tsim': dictNet['Tsim']*(1.0/np.mean(dictPLL['intrF']))}) # simulation time in multiples of the period of the uncoupled oscillators
		dictPLL.update({'sim_time_steps': int(dictNet['Tsim']/dictPLL['dt'])})
		print('Total simulation time in multiples of the eigenfrequency:', int(dictNet['Tsim']*np.mean(dictPLL['intrF'])))
	else:

		print('Tsim not in multiples of T_omega, since F <= 1E-3')
		dictNet.update({'Tsim': dictNet['Tsim']*2})
		dictPLL.update({'sim_time_steps': int(dictNet['Tsim']/dictPLL['dt'])})
		print('Total simulation time in seconds:', int(dictNet['Tsim']))

	dictPLL.update({'timeSeriesAverTime': int(dictPLL['percentPeriodsAverage']*dictNet['Tsim']*dictPLL['syncF'])})

	#if dictPLL['typeVCOsig'] == 'analogHF' and inspect.getsourcelines(dictPLL['coup_fct_sig'])[0][0] == "dictPLL={'coup_fct_sig': lambda x: sawtooth(x,width=0.5)\n}"
	#	print('Recheck paramater combinations in dictPLL: set to analogHF while coupling function of digital PLL choosen.'); sys.exit()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	if ( isinstance(dictAlgo['paramDiscretization'], list) or isinstance(dictAlgo['paramDiscretization'], np.ndarray) ):
		if ( isinstance(dictAlgo['min_max_range_parameter'], np.float) or isinstance(dictAlgo['min_max_range_parameter'], np.int) ):
			print('NOTE: in multisim_lib, the case listOfInitialPhaseConfigurations needs a minimum and maximum intrinsic frequency, e.g., [wmin, wmax]! Please povide.'); sys.exit()

			# Implement that here, see below for perturbations!

			#dictAlgo.update({'min_max_range_parameter': [0.99*dictPLL['intrF'], 1.01*dictPLL['intrF']]})
		else:
 			print('NOTE: in multisim_lib, the minimum and maximum intrinsic frequency, [wmin, wmax] are used to calculate all initial detunings (including that with 0).')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	if dictPLL['extra_coup_sig'] != 'injection2ndHarm':
		dictPLL.update({'coupStr_2ndHarm': 0})
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	if dictPLL['typeVCOsig'] == 'analogHF':
		dictPLL.update({})
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	if dictNet['computeFreqAndStab'] and dictPLL['orderLF'] == 1:
		if( isinstance(dictPLL['intrF'], np.ndarray) or isinstance(dictPLL['intrF'], list) or isinstance(dictPLL['cutFc'], np.ndarray) or isinstance(dictPLL['cutFc'], list) or
			isinstance(dictPLL['coupK'], np.ndarray) or isinstance(dictPLL['coupK'], list) or isinstance(dictPLL['transmission_delay'], np.ndarray) or isinstance(dictPLL['transmission_delay'], list) ):
			print('USING SYNCTOOLS for heterogeneous parameters -- taking the mean value!')
			dictPLLsyncTool = dictPLL.copy()
			dictPLLsyncTool.update({'intrF': 				np.mean(dictPLL['intrF'])})
			dictPLLsyncTool.update({'cutFc': 				np.mean(dictPLL['cutFc'])})
			dictPLLsyncTool.update({'coupK': 				np.mean(dictPLL['coupK'])})
			dictPLLsyncTool.update({'transmission_delay': 	np.mean(dictPLL['transmission_delay'])})
			#try:
			isRadian = False													# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(dictPLLsyncTool, dictNet, isRadians=isRadian)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=isRadian)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {intrF, coupK, cutFc, delay, Omega, ReLambda, ImLambda, TsimToPert1/e, Nx, Ny, mx, my, div}: \n', [*para_mat])
			choice = chooseSolution(para_mat)
			dictPLL.update({'syncF': para_mat[choice,4], 'ReLambda': para_mat[choice,5], 'ImLambda': para_mat[choice,6]})
			print('Choice %i made. This state has global frequency Omega=%0.2f Hz, and ReLambda=%0.2f and ImLambda=%0.2f'%(choice, dictPLL['syncF'], dictPLL['ReLambda'], dictPLL['ImLambda']))
			#except:
			#	print('Could not compute linear stability and global frequency! Check synctools and case!')

		elif ( (isinstance(dictPLL['intrF'], np.float) or isinstance(dictPLL['intrF'], np.int)) and (isinstance(dictPLL['cutFc'], np.float) or isinstance(dictPLL['cutFc'], np.int)) and
			(isinstance(dictPLL['coupK'], np.float) or isinstance(dictPLL['coupK'], np.int)) and (isinstance(dictPLL['transmission_delay'], np.float) or isinstance(dictPLL['transmission_delay'], np.int)) ):
			#try:
			isRadian = False													# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(dictPLL, dictNet, isRadians=isRadian)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=isRadian)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {intrF, coupK, cutFc, delay, Omega, ReLambda, ImLambda, TsimToPert1/e, Nx, Ny, mx, my, div}: \n', [*para_mat])
			choice = chooseSolution(para_mat)
			dictPLL.update({'syncF': para_mat[choice,4], 'ReLambda': para_mat[choice,5], 'ImLambda': para_mat[choice,6]})
			print('Choice %i made. This state has global frequency Omega=%0.2f Hz, and ReLambda=%0.2f and ImLambda=%0.2f'%(choice, dictPLL['syncF'], dictPLL['ReLambda'], dictPLL['ImLambda']))
			#except:
			#	print('Could not compute linear stability and global frequency! Check synctools and case!')
	elif dictNet['computeFreqAndStab'] and dictPLL['orderLF'] > 1:
		print('In dicts_<NAME>: Synctools prediction not available for second order LFs!'); sys.exit();
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	dictNet = set_max_delay_for_time_dependentTau(dictPLL, dictNet)
	dictNet = check_consistency_initPert(dictNet)
	print('Setup (dictNet, dictPLL):', dictNet, dictPLL)

	return dictPLL, dictNet, dictAlgo

# ******************************************************************************

def set_max_delay_for_time_dependentTau(dictPLL, dictNet):
	if dictNet['special_case'] == 'timeDepTransmissionDelay':
		dictNet.update({'max_delay_steps': int(np.round(dictNet['min_max_rate_timeDepPara'][1]/dictPLL['dt']))})
	return dictNet
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def choose_yes_no():															# ask user-input
	a_true = True
	while a_true:
		# get user input which of the possible cases to simulate
		choice = input('Choose [y]es or [n]o: ')
		if isinstance(choice, str) and ( choice == 'y' or choice == 'n'):
			break
		else:
			print('Please provide answer as indicated by brackets, i.e., from {y, n}!')

	return str(choice)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def chooseSolution(para_mat):													# ask user-input for which solution to simulate
	a_true = True
	while a_true:
		# get user input which of the possible cases to simulate
		choice = input('Choose which case to simulate [0,...,%i]: '%(len(para_mat[:,0])-1))
		if int(choice) >= 0 and int(choice) < len(para_mat[:,0]):
			break
		else:
			print('Please provide input as integer choice!')

	return int(choice)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def check_consistency_initPert(dictNet):
	if len(dictNet['phiPerturb']) != dictNet['Nx']*dictNet['Ny'] and len(dictNet['phiPerturbRot']) != dictNet['Nx']*dictNet['Ny']:
		userIn = input('No initial perturbation defined, choose from [manual, allzero, iid]: ')
		if userIn == 'manual':
			a_true = True
			while a_true:
				# get user input for corrected set of perturbations
				choice = [float(item) for item in input('Initial perturbation vectors´ length does not match number of oscillators (%i), provide new list [only numbers without commas!]: '%(dictNet['Nx']*dictNet['Ny'])).split()]
				dictNet.update({'phiPerturb': choice})
				print('type(choice), len(choice)', type(choice), len(choice))
				if len(dictNet['phiPerturb']) == dictNet['Nx']*dictNet['Ny']:
					break
				elif dictNet['phiPerturb'][0] == -999:
					dictNet.update({'phiPerturb': np.zeros(dictNet['Nx']*dictNet['Ny']).tolist()})
				else:
					print('Please choose right number of perturbations or the value -999 for no perturbation!')
		elif userIn == 'allzero':
			dictNet.update({'phiPerturb': np.zeros(dictNet['Nx']*dictNet['Ny']).tolist()})
		elif userIn == 'iid':
			lowerPertBound = np.float(input('Lower bound [e.g., -3.14159]: '))
			upperPertBound = np.float(input('Upper bound [e.g., +3.14159]: '))
			dictNet.update({'phiPerturb': (np.random.rand(dictNet['Nx']*dictNet['Ny'])*np.abs((upperPertBound-lowerPertBound))-lowerPertBound).tolist()})
		else:
			return None
	return dictNet
