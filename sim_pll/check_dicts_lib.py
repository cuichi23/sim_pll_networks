#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys
import gc
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

from sim_pll import coupling_fct_lib as coupfct
from sim_pll import synctools_interface_lib as synctools

''' Enable automatic carbage collector '''
gc.enable()

#%%cython --annotate -c=-O3 -c=-march=native

def check_dicts_consistency(dict_pll, dict_net, dict_algo):

	if ( dict_pll['includeCompHF'] and (dict_pll['vco_out_sig'] == coupfct.square_wave or dict_pll['vco_out_sig'] == coupfct.square_wave_symm_zero) ):
		print('NOTE: simulation with coupling function that has peak2peak amplitude 1 (NOT 2) --> calculate coupling strength K dict_pll[*coupK*] as K = G * A_pd * K_vco, where G are all gains in the feed-forward path, A_pd the output signal amplitude of the PD and K_vco the sensitivity of the VCO.')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# check consistency of mx and my
	if dict_net['Ny'] > 1 and dict_net['my'] == -999:
		print('NOTE: For 2D topology my should not be set to -999, setting to zero now!')
		dict_net.update({'my': 0})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# check consistency of coupling functions and output signal type when calculating the PSD (sometimes it can be helpful to only consider the first harmonic contribution)
	if dict_pll['coup_fct_sig'] == coupfct.triangular or dict_pll['coup_fct_sig'] == coupfct.deriv_triangular or dict_pll['coup_fct_sig'] == coupfct.square_wave or dict_pll['coup_fct_sig'] == coupfct.pfd:
		if dict_pll['vco_out_sig'] == coupfct.sine:
			print('A coupling function associated to oscillators with DIGITAL output is chosen. Current PSD choice is only to analyze first harmonic contribution! Switch to full digital [y]?')
			choice = choose_yes_no()
			if choice == 'y':
				dict_pll.update({'vco_out_sig': coupfct.square_wave})
			elif choice == 'n':
				dict_pll.update({'vco_out_sig': coupfct.sine})
	elif dict_pll['coup_fct_sig'] == coupfct.sine or dict_pll['coup_fct_sig'] == coupfct.cosine:
		if dict_pll['vco_out_sig'] == coupfct.square_wave:
			print('A coupling function associated to oscillators with ANALOG output is chosen. Current PSD choice is to analyze a square wave signal of the phase! Switch to sine wave [y]?')
			choice = choose_yes_no()
			if choice == 'y':
				dict_pll.update({'vco_out_sig': coupfct.sine})
			elif choice == 'n':
				dict_pll.update({'vco_out_sig': coupfct.square_wave})

	if dict_pll['coup_fct_sig'] == coupfct.triangular:
		dict_pll.update({'inverse_coup_fct_sig': coupfct.inverse_triangular})
	elif dict_pll['coup_fct_sig'] == coupfct.sine:
		dict_pll.update({'inverse_coup_fct_sig': coupfct.inverse_sine})
	elif dict_pll['coup_fct_sig'] == coupfct.cosine:
		dict_pll.update({'inverse_coup_fct_sig': coupfct.inverse_cosine})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# calculate other parameters and test for incompatibilities
	dict_pll.update({'dt': 1.0/dict_pll['sampleF']})
	if ( ( isinstance(dict_pll['gPDin'], np.ndarray) or isinstance(dict_pll['gPDin'], list) ) and dict_pll['gPDin_symmetric']):

		if not np.shape(dict_pll['gPDin']) == (dict_net['Nx']*dict_net['Ny'], dict_net['Nx']*dict_net['Ny']):
			print('Error! In order to set individual gains for all feed-foward paths, an [N, N] matrix of gains G_kl needs to provided! Please fix dict_pll[*gPDin*].')
			sys.exit()

		print('Generate symmetrical matrix for PD gains.')
		symm_matrix = (dict_pll['gPDin']@dict_pll['gPDin'].T)/np.max(dict_pll['gPDin']@dict_pll['gPDin'].T)
		if isinstance(symm_matrix, list):
			symm_matrix = np.array(symm_matrix)
		dict_pll.update({'gPDin': symm_matrix})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	if not (isinstance(dict_pll['transmission_delay'], np.ndarray) or isinstance(dict_pll['transmission_delay'], list)):
		if np.abs(dict_pll['dt']*np.int(np.round(dict_pll['transmission_delay']/dict_pll['dt'])) - dict_pll['transmission_delay']) > 0.01*dict_pll['transmission_delay']:
			print('NOTE: time step dt not small enough to resolve the time delay.\ntranmission time delay contineous time: %0.3f\ntransmission time delay after time discretization: %0.3f' % (dict_pll['transmission_delay'], dict_pll['dt']*np.int( np.round( dict_pll['transmission_delay']/dict_pll['dt'] ) ) ))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# catch the case where the intrinsic frequency is set to zero
	if np.all(dict_pll['intrF']) > 1E-3:

		dict_net.update({'Tsim': dict_net['Tsim']*(1.0/np.mean(dict_pll['intrF']))}) # simulation time in multiples of the period of the uncoupled oscillators
		dict_pll.update({'sim_time_steps': int(dict_net['Tsim']/dict_pll['dt'])})
		print('Total simulation time in multiples of the eigenfrequency:', int(dict_net['Tsim']*np.mean(dict_pll['intrF'])))
	else:

		print('Tsim not in multiples of T_omega, since F <= 1E-3')
		dict_net.update({'Tsim': dict_net['Tsim']*2})
		dict_pll.update({'sim_time_steps': int(dict_net['Tsim']/dict_pll['dt'])})
		print('Total simulation time in seconds:', int(dict_net['Tsim']))

	dict_pll.update({'timeSeriesAverTime': int(dict_pll['percentPeriodsAverage']*dict_net['Tsim']*dict_pll['syncF'])})

	#if dict_pll['typeVCOsig'] == 'analogHF' and inspect.getsourcelines(dict_pll['coup_fct_sig'])[0][0] == "dict_pll={'coup_fct_sig': lambda x: sawtooth(x,width=0.5)\n}"
	#	print('Recheck paramater combinations in dict_pll: set to analogHF while coupling function of digital PLL choosen.'); sys.exit()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# consistency check for basin stability plots
	#print('HERE type(dict_algo):', type(dict_algo))
	if ( isinstance(dict_algo['paramDiscretization'], list) or isinstance(dict_algo['paramDiscretization'], np.ndarray) ):
		if ( isinstance(dict_algo['min_max_range_parameter_0'], np.float) or isinstance(dict_algo['min_max_range_parameter_0'], np.int) ):
			print('NOTE: in multisim_lib, the case listOfInitialPhaseConfigurations needs a minimum and maximum intrinsic frequency, e.g., [wmin, wmax]! Please povide.'); sys.exit()

			# Implement that here, see below for perturbations!

			#dict_algo.update({'min_max_range_parameter_0': [0.99*dict_pll['intrF'], 1.01*dict_pll['intrF']]})
		else:
			print('NOTE: in multisim_lib, the minimum and maximum intrinsic frequency, [wmin, wmax] are used to calculate all initial detunings (including that with 0).')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# if 'entrain' in dict_net['topology'] and (dict_algo['param_id_0'] == 'intrF' or dict_algo['param_id_1'] == 'intrF'):
	if dict_net['topology'].find('entrain') != -1:
		print('Simulation of entrainment case: the first component of the list of intrinsic frequencies denotes that of the reference oscillator!')
		if isinstance(dict_pll['coupK'], list) or isinstance(dict_pll['coupK'], np.ndarray):
			if not dict_pll['coupK'][0] == 0:
				print('ABORT: simulating a topology with a reference oscillator. This needs to be the one indexed by k=0, hence dict_pll[*coupK*][0] needs to be zero!')
				sys.exit()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# correct the coupStr for second harmonic injection if none is chosen
	if dict_pll['extra_coup_sig'] != 'injection2ndHarm':
		dict_pll.update({'coupStr_2ndHarm': 0})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	if dict_pll['typeVCOsig'] == 'analogHF':
		dict_pll.update({})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# run synctools and chose from different (multistable) solutions
	if dict_net['computeFreqAndStab'] and dict_pll['orderLF'] == 1:
		if( isinstance(dict_pll['intrF'], np.ndarray) or isinstance(dict_pll['intrF'], list) or isinstance(dict_pll['cutFc'], np.ndarray) or isinstance(dict_pll['cutFc'], list) or
			isinstance(dict_pll['coupK'], np.ndarray) or isinstance(dict_pll['coupK'], list) or isinstance(dict_pll['transmission_delay'], np.ndarray) or isinstance(dict_pll['transmission_delay'], list) ):
			print('USING SYNCTOOLS for heterogeneous parameters -- taking the mean value!')
			dict_pllsyncTool = dict_pll.copy()
			dict_pllsyncTool.update({'intrF': 				np.mean(dict_pll['intrF'])})
			dict_pllsyncTool.update({'cutFc': 				np.mean(dict_pll['cutFc'])})
			dict_pllsyncTool.update({'coupK': 				np.mean(dict_pll['coupK'])})
			dict_pllsyncTool.update({'transmission_delay': 	np.mean(dict_pll['transmission_delay'])})
			#try:
			isRadian = False													# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf  = synctools.SweepFactory(dict_pllsyncTool, dict_net, isRadians=isRadian)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=isRadian)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {intrF, coupK, cutFc, delay, Omega, ReLambda, ImLambda, TsimToPert1/e, Nx, Ny, mx, my, div}: \n', [*para_mat])
			choice = chooseSolution(para_mat)
			dict_pll.update({'syncF': para_mat[choice,4], 'ReLambda': para_mat[choice,5], 'ImLambda': para_mat[choice,6]})
			print('Choice %i made. This state has global frequency Omega=%0.9f Hz, and ReLambda=%0.9f and ImLambda=%0.9f'%(choice, dict_pll['syncF'], dict_pll['ReLambda'], dict_pll['ImLambda']))
			#except:
			#	print('Could not compute linear stability and global frequency! Check synctools and case!')
			print('Calculate synchronized states and their stability using synctools to plot Omega and Re(lambda) vs tau and Omega*tau, [y]es or [n]o?')
			choice = choose_yes_no()
			if choice == 'y':
				synctools.generate_delay_plot(dict_pll, dict_net)

		elif ( (isinstance(dict_pll['intrF'], np.float) or isinstance(dict_pll['intrF'], np.int)) and (isinstance(dict_pll['cutFc'], np.float) or isinstance(dict_pll['cutFc'], np.int)) and
			(isinstance(dict_pll['coupK'], np.float) or isinstance(dict_pll['coupK'], np.int)) and (isinstance(dict_pll['transmission_delay'], np.float) or isinstance(dict_pll['transmission_delay'], np.int)) ):
			#try:
			isRadian = False													# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(dict_pll, dict_net, isRadians=isRadian)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=isRadian)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {intrF, coupK, cutFc, delay, Omega, ReLambda, ImLambda, TsimToPert1/e, Nx, Ny, mx, my, div}: \n', [*para_mat])
			choice = chooseSolution(para_mat)
			dict_pll.update({'syncF': para_mat[choice,4], 'ReLambda': para_mat[choice,5], 'ImLambda': para_mat[choice,6]})
			print('Choice %i made. This state has global frequency Omega=%0.9f Hz, and ReLambda=%0.9f and ImLambda=%0.9f'%(choice, dict_pll['syncF'], dict_pll['ReLambda'], dict_pll['ImLambda']))
			#except:
			#	print('Could not compute linear stability and global frequency! Check synctools and case!')
			print('Calculate synchronized states and their stability using synctools to plot Omega and Re(lambda) vs tau and Omega*tau, [y]es or [n]o?')
			choice = choose_yes_no()
			if choice == 'y':
				synctools.generate_delay_plot(dict_pll, dict_net)

	elif dict_net['computeFreqAndStab'] and dict_pll['orderLF'] > 1:
		print('In dicts_<NAME>: Synctools prediction not available for second order LFs!')
		sys.exit()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	dict_pll = set_percent_of_Tsim(dict_pll, dict_net)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	dict_pll = set_max_coupling_strength_for_time_dependent_coupling_strength(dict_pll, dict_net)
	dict_pll, dict_net = set_max_delay_for_time_dependentTau(dict_pll, dict_net)
	dict_net = check_consistency_initPert(dict_net)
	print('Setup (dict_net, dict_pll):', dict_net, dict_pll)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	if dict_pll['responseVCO'] == 'linear':
		dict_pll.update({'inverse_fct_vco_response': coupfct.inverse_linear})
	elif dict_pll['responseVCO'] == 'nonlinear_3rd_gen':
		dict_pll.update({'inverse_fct_vco_response': coupfct.inverse_nonlinear_response_vco_3rd_gen})
		# calculate the prebias voltage for all VCO's here and save it to the dict_pll
		if isinstance(dict_pll['intrF'], list) or isinstance(dict_pll['intrF'], np.ndarray):
			dict_pll.update({'prebias_voltage_vco': [coupfct.nonlinear_response_vco_3rd_gen_calculate_voltage_bias(dict_pll['intrF'][i]) for i in dict_pll['intrF'][:]]})
		else:
			dict_pll.update({'prebias_voltage_vco': [coupfct.nonlinear_response_vco_3rd_gen_calculate_voltage_bias(dict_pll['intrF']) for i in range(dict_net['Nx'])]})
		print('prebias_voltage_vco:', dict_pll['prebias_voltage_vco'])
		time.sleep(5)
	else:
		print('No inverse function for the nonlinear VCO response defined.')
		sys.exit()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	return dict_pll, dict_net, dict_algo

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# helper functions
# ******************************************************************************

def set_percent_of_Tsim(dict_pll, dict_net):

	if 1.0/dict_pll['PSD_freq_resolution'] > 0.9*dict_net['Tsim']:
		print('0.9*Tsim=%0.1f is too small to achieve a PSD time resolution of %0.9f!'%(0.9*dict_net['Tsim'], dict_pll['PSD_freq_resolution']))
		print('Adjusting *PSD_freq_resolution* parameter to fit given Tsim to %0.9e'%(1.0/(0.9*dict_net['Tsim'])))
		dict_pll.update({'percent_of_Tsim': (0.9*dict_net['Tsim'])/dict_net['Tsim'], 'PSD_freq_resolution': 1.0/(0.9*dict_net['Tsim'])})
	else:
		print('Setting percent_of_Tsim for PSD analysis to %0.9f'%((1.0/dict_pll['PSD_freq_resolution'])/dict_net['Tsim']))
		dict_pll.update({'percent_of_Tsim': (1.0/dict_pll['PSD_freq_resolution'])/dict_net['Tsim']})

	return dict_pll

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_max_delay_for_time_dependentTau(dict_pll, dict_net):
	if dict_net['special_case'] == 'timeDepTransmissionDelay':
		dict_net.update({'max_delay_steps': int(np.round(dict_net['min_max_rate_timeDepPara'][1]/dict_pll['dt']))})
		dict_pll.update({'transmission_delay': dict_net['min_max_rate_timeDepPara'][1]})
	return dict_pll, dict_net

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_max_coupling_strength_for_time_dependent_coupling_strength(dict_pll, dict_net):
	if dict_net['special_case'] == 'timeDepChangeOfCoupStr':
		dict_pll.update({'coupK': dict_net['min_max_rate_timeDepPara'][1]})
	return dict_pll

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

def check_consistency_initPert(dict_net):
	if len(dict_net['phiPerturb']) != dict_net['Nx']*dict_net['Ny'] and len(dict_net['phiPerturbRot']) != dict_net['Nx']*dict_net['Ny']:
		userIn = input('No initial perturbation defined, choose from [manual, allzero, iid]: ')
		if userIn == 'manual':
			a_true = True
			while a_true:
				# get user input for corrected set of perturbations
				choice = [float(item) for item in input('Initial perturbation vectors´ length does not match number of oscillators (%i), provide new list [only numbers without commas!]: '%(dict_net['Nx']*dict_net['Ny'])).split()]
				dict_net.update({'phiPerturb': choice})
				print('type(choice), len(choice)', type(choice), len(choice))
				if len(dict_net['phiPerturb']) == dict_net['Nx']*dict_net['Ny']:
					break
				elif dict_net['phiPerturb'][0] == -999:
					dict_net.update({'phiPerturb': np.zeros(dict_net['Nx']*dict_net['Ny']).tolist()})
				else:
					print('Please choose right number of perturbations or the value -999 for no perturbation!')
		elif userIn == 'allzero':
			dict_net.update({'phiPerturb': np.zeros(dict_net['Nx']*dict_net['Ny']).tolist()})
		elif userIn == 'iid':
			lowerPertBound = np.float(input('Lower bound [e.g., -3.14159]: '))
			upperPertBound = np.float(input('Upper bound [e.g., +3.14159]: '))
			dict_net.update({'phiPerturb': (np.random.rand(dict_net['Nx']*dict_net['Ny'])*np.abs((upperPertBound-lowerPertBound))-lowerPertBound).tolist()})
		else:
			return None

	return dict_net
