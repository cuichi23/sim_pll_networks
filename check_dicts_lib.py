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

def check_dicts_consistency(dictPLL, dictNet, dictAlgo):

	# check consistency of coupling functions and output signal type when calculating the PSD (sometimes it can be helpful to only consider the first harmonic contribution)
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

	# catch the case where the intrinsic frequency is set to zero
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

	# consistency check for basin stability plots
	if ( isinstance(dictAlgo['paramDiscretization'], list) or isinstance(dictAlgo['paramDiscretization'], np.ndarray) ):
		if ( isinstance(dictAlgo['min_max_range_parameter'], np.float) or isinstance(dictAlgo['min_max_range_parameter'], np.int) ):
			print('NOTE: in multisim_lib, the case listOfInitialPhaseConfigurations needs a minimum and maximum intrinsic frequency, e.g., [wmin, wmax]! Please povide.'); sys.exit()

			# Implement that here, see below for perturbations!

			#dictAlgo.update({'min_max_range_parameter': [0.99*dictPLL['intrF'], 1.01*dictPLL['intrF']]})
		else:
 			print('NOTE: in multisim_lib, the minimum and maximum intrinsic frequency, [wmin, wmax] are used to calculate all initial detunings (including that with 0).')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# correct the coupStr for second harmonic injection if none is chosen
	if dictPLL['extra_coup_sig'] != 'injection2ndHarm':
		dictPLL.update({'coupStr_2ndHarm': 0})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	if dictPLL['typeVCOsig'] == 'analogHF':
		dictPLL.update({})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# run synctools and chose from different (multistable) solutions
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
			sf  = synctools.SweepFactory(dictPLLsyncTool, dictNet, isRadians=isRadian)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=isRadian)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {intrF, coupK, cutFc, delay, Omega, ReLambda, ImLambda, TsimToPert1/e, Nx, Ny, mx, my, div}: \n', [*para_mat])
			choice = chooseSolution(para_mat)
			dictPLL.update({'syncF': para_mat[choice,4], 'ReLambda': para_mat[choice,5], 'ImLambda': para_mat[choice,6]})
			print('Choice %i made. This state has global frequency Omega=%0.9f Hz, and ReLambda=%0.9f and ImLambda=%0.9f'%(choice, dictPLL['syncF'], dictPLL['ReLambda'], dictPLL['ImLambda']))
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
			print('Choice %i made. This state has global frequency Omega=%0.9f Hz, and ReLambda=%0.9f and ImLambda=%0.9f'%(choice, dictPLL['syncF'], dictPLL['ReLambda'], dictPLL['ImLambda']))
			#except:
			#	print('Could not compute linear stability and global frequency! Check synctools and case!')
	elif dictNet['computeFreqAndStab'] and dictPLL['orderLF'] > 1:
		print('In dicts_<NAME>: Synctools prediction not available for second order LFs!'); sys.exit();

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	dictPLL = set_percent_of_Tsim(dictPLL, dictNet)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	dictNet = set_max_delay_for_time_dependentTau(dictPLL, dictNet)
	dictNet = check_consistency_initPert(dictNet)
	print('Setup (dictNet, dictPLL):', dictNet, dictPLL)

	return dictPLL, dictNet, dictAlgo






# helper functions
# ******************************************************************************

def set_percent_of_Tsim(dictPLL, dictNet):

	if 1.0/dictPLL['PSD_freq_resolution'] > 0.9*dictNet['Tsim']:
		print('0.9*Tsim=%0.1f is too small to achieve a PSD time resolution of %0.9f!'%(0.9*dictNet['Tsim'], dictPLL['PSD_freq_resolution']))
		print('Adjusting *PSD_freq_resolution* parameter to fit given Tsim to %0.9e'%(1.0/(0.9*dictNet['Tsim'])))
		dictPLL.update({'percent_of_Tsim': (0.9*dictNet['Tsim'])/dictNet['Tsim'], 'PSD_freq_resolution': 1.0/(0.9*dictNet['Tsim'])})
	else:
		print('Setting percent_of_Tsim for PSD analysis to %0.9f'%((1.0/dictPLL['PSD_freq_resolution'])/dictNet['Tsim']))
		dictPLL.update({'percent_of_Tsim': (1.0/dictPLL['PSD_freq_resolution'])/dictNet['Tsim']})

	return dictPLL

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
