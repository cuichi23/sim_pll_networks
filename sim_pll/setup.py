#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys
import gc
import numpy as np
#cimport numpy as np
#cimport cython
import networkx as nx
from scipy.signal import sawtooth
from scipy.signal import square
from scipy.stats import cauchy

import itertools

from typing import Union

#import matplotlib
import matplotlib.pyplot as plt
import datetime
import time

from sim_pll import pll_lib as pll
from sim_pll import evaluation_lib as eva
from sim_pll import entrain_mutual_lib as ent_mut

''' Enable automatic carbage collector '''
gc.enable()

''' CREATE PLL LIST '''
def generate_plls(dict_pll: dict, dict_net: dict, dict_data: dict) -> list:
	"""
	Creates a list with PLL objects as defined by the pll_lib.py classes.

	Args:
		dict_pll:  [dict] contains the setup information for the PLL objects
		dict_net:  [dict] contains the setup information for the network and simulation
		dict_data: [dict] contains data to prepare and results from the simulation

	Returns:
		list of PLL objects that belong to a network with a specified topology

	"""

	dict_pll.update({'G': setup_topology(dict_net)})

	pll_list = [ pll.PhaseLockedLoop(idx_pll,											# setup PLLs and store in a list as PLL class objects
					pll.Delayer(idx_pll, dict_pll, dict_net, dict_data),				# setup delayer object of PLL k it organizes the delayed communications
					pll.PhaseDetectorCombiner(idx_pll, dict_pll, dict_net),				# setup PDadder object of PLL k
					pll.LowPassFilter(idx_pll, dict_pll, dict_net),						# setup LF(1st) object of PLL k
					pll.SignalControlledOscillator(idx_pll, dict_pll, dict_net),		# setup VCO object of PLL k
					pll.InjectionLockingSignal(idx_pll, dict_pll, dict_net),			# setup an injection locking signal generator object for each oscillator k
					pll.Counter()														# setup Counter object of PLL k
					) for idx_pll in range(dict_net['Nx']*dict_net['Ny']) ]

	if isinstance(dict_pll['intrF'], list) or isinstance(dict_pll['coupK'], list) or isinstance(dict_pll['transmission_delay'], list) or isinstance(dict_pll['cutFc'], list):
		output_pll_specs(pll_list, dict_data, dict_pll)

	return pll_list

################################################################################

def output_pll_specs(pll_list: list, dict_data: dict, dict_pll: dict) -> dict:
	"""Function that collects the properties of all PLLs and stores them in a list.

		Args:
			pll_list: a list that contains all the PLL objects that have been setup for the network from the PLL library
			dict_data: stores the parameters of all PLLs, later in the simulation also stores the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged

		Returns:
			dict_data
		"""
	dict_data.update({'list_of_intrinsic_frequencies_Hz': [pll.signal_controlled_oscillator.intr_freq_rad / (2 * np.pi) for pll in pll_list]})
	dict_data.update({'list_of_coupling_strengths_Hz': [pll.signal_controlled_oscillator.K_rad / (2 * np.pi) for pll in pll_list]})
	dict_data.update({'list_of_cutoff_frequencies_Hz': [pll.low_pass_filter.cutoff_freq_Hz for pll in pll_list]})
	dict_data.update({'list_of_time_delays': [np.array(pll.delayer.transmit_delay_steps) * dict_pll['dt'] for pll in pll_list]})

	return dict_data

################################################################################

def generate_space(dict_net: dict, dict_pll: dict, dict_data: dict):
	"""
	Creates a space object that provides a 3d space.

	Args:
		dict_pll:  [dict] contains the setup information for the PLL objects
		dict_net:  [dict] contains the setup information for the network and simulation
		dict_data: [dict] contains data to prepare and results from the simulation

	Returns:
		space object
	"""
	space = pll.Space(dict_pll['signal_propagation_speed'], dict_pll['space_dimensions_xyz'])

	return space

################################################################################

def generate_phi0(dict_net: dict, dict_pll: dict, dict_algo: dict) -> None:
	"""
	Set the initial phase perturbation of each oscillator in the network depending on the type of solution to be investigated.

	Args:
		dict_net:  [dict] contains the setup information for the network and simulation
		dict_pll:  [dict] contains the setup information for the PLL objects

	Returns:
		None
	"""

	if 'entrainOne' in dict_net['topology'] or 'entrainAll' in dict_net['topology']:
		print('Provide phase-configuration for these cases in physical coordinates!')
		#phiM  = eva.rotate_phases(phiSr.flatten(), isInverse=False)
		# use phase_configuration_ref_to_one_for_chain_topology function from entrain_mutual_lib.py to calculate the phase configuration from the analytic expressions
		if dict_net['topology'] == 'entrainOne-chain':
			ent_mut.obtain_phase_config_entrainment_of_mutual_sync(dict_net, dict_pll, dict_algo)
		elif not dict_net['topology'] == 'entrainOne-chain':
			dict_net.update({'phiInitConfig': [0 for i in range(dict_net['Nx']*dict_net['Ny'])]})
			print('Initial phase configuration of entrained state WAS NOT calculated automatically: tbi!')

		phiM = dict_net['phiInitConfig']  # phiConfig: user specified configuration of initial phi states

		temporary_switch = 0
		if temporary_switch == 1:
			#phiS = np.array([2., 2., 2.])
			#dict_net.update({'phiSr': eva.rotate_phases(phiS.flatten(), isInverse=True)})
			dict_net.update({'phiPerturb': np.array([2., 2., 2.])})
			dict_net.update({'phiPerturbRot': eva.rotate_phases(dict_net['phiPerturb'].flatten(), isInverse=True)})
		else:
			#dict_net.update({'phiS': eva.rotate_phases(dict_net['phiSr'].flatten(), isInverse=False)})
			#print('Calculated phiS=',phiS,' from phiSr=',phiSr,'.\n')
			if len(dict_net['phiPerturb']) == 0 and len(dict_net['phiPerturbRot']) == 0:
				print('Provide intitial phase perturbations in phase or rotated phase space!')
			elif len(dict_net['phiPerturbRot']) == 0:
				dict_net.update({'phiPerturbRot': eva.rotate_phases(np.array(dict_net['phiPerturb']).flatten(), isInverse=True)})
			elif len(dict_net['phiPerturb']) == 0:
				dict_net.update({'phiPerturb': eva.rotate_phases(np.array(dict_net['phiPerturbRot']).flatten(), isInverse=False)})

		print('entrainOne or entrainAll topology, {tau, f_R}={', dict_pll['transmission_delay'], ', ', dict_pll['intrF'][0],
			  '}: assumed initial phase-configuration of entrained synced state (physical coordinates):', dict_net['phiInitConfig'],
			  ' and on top a perturbation of (original coordinates):', dict_net['phiPerturb'], ' and in, (rotated coordinates):', dict_net['phiPerturbRot'], '\n')

	elif dict_net['topology'] == 'compareEntrVsMutual':
		print('REWORK THIS!')
		sys.exit()
		phiM = dict_net['phiConfig']
		if len(dict_net['phiPerturbRot']) == 0:
			phiS = np.zeros(dict_net['Nx']*dict_net['Ny'])
		dict_net.update({'phiPerturb': eva.rotate_phases(phiSr.flatten(), isInverse=False)})
	else:
		print('Run single time-series and plot phase and frequency time series!')
		initPhiPrime0 = 0.0
		#print('dict_net[*phiPerturbRot*]', dict_net['phiPerturbRot'], 'dict_net[*phiPerturb*]', dict_net['phiPerturb'])
		if len(dict_net['phiPerturbRot']) > 0 and len(dict_net['phiPerturb']) == 0:			#dict_net['phiPerturbRot'] and not dict_net['phiPerturb']:
			print('Parameters set, perturbations provided manually in rotated phase space of phases.')
			#if len(dict_net['phiPerturbRot'].shape)==1:
			if len(dict_net['phiPerturbRot']) == dict_net['Nx']*dict_net['Ny']:
				print('Shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
				dict_net['phiPerturbRot'][0] = initPhiPrime0			# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
				dict_net.update({'phiPerturb': eva.rotate_phases(dict_net['phiPerturbRot'], isInverse=False)}) 		# rotate back into physical phase space for simulation
				print('\nPerturbations in ROTATED phase space:', dict_net['phiPerturbRot'])
				print('Dirac delta phase perturbation in ORIGINAL phase space:', dict_net['phiPerturb'])
			else:
				dict_net.update({'phiPerturb': np.zeros(dict_net['Nx']*dict_net['Ny'])})
				dict_net.update({'phiPerturbRot': eva.rotate_phases(dict_net['phiPerturb'], isInverse=True)})
				dict_net['phiPerturbRot'][0] = initPhiPrime0
				print('No perturbations defined, work it out! So far no perturbations are set, i.e., all zero!')
				#sys.exit()

			# elif len(dict_net['phiPerturbRot'].shape)==2:
			# 	if len(dict_net['phiPerturbRot'][0,:])==dict_net['Nx']*dict_net['Ny']:
			# 		print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
			# 		dict_net['phiPerturbRot'][:,0] = initPhiPrime0				# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
			# 		print('\nvalues of the perturbations in ROTATED phase space, i.e., last time-step of history set as initial condition:', dict_net['phiPerturbRot'])
			# 		dict_net.update({'phiPerturb': eva.rotate_phases(dict_net['phiPerturbRot'].flatten(), isInverse=False)}) # rotate back into physical phase space for simulation
			# 		print('dirac delta phase perturbation in ORIGINAL phase space:', dict_net['phiPerturb'], '\n')

		elif len(dict_net['phiPerturbRot']) == 0 and len(dict_net['phiPerturb']) > 0: 					#dict_net['phiPerturb'] and not dict_net['phiPerturbRot']:
			print('Parameters set, perturbations provided manually in original phase space of phases.')
			#if len(dict_net['phiPerturb'].shape)==1:
			if len(dict_net['phiPerturb']) == dict_net['Nx']*dict_net['Ny']:
				dict_net.update({'phiPerturbRot': eva.rotate_phases(dict_net['phiPerturb'], isInverse=True)})
				print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
				dict_net['phiPerturbRot'][0] = initPhiPrime0		# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
				print('\nPerturbations in ROTATED phase space:', dict_net['phiPerturbRot'])
				print('Dirac delta phase perturbation in ORIGINAL phase space:', dict_net['phiPerturb'])
			else:
				dict_net.update({'phiPerturb': np.zeros(dict_net['Nx']*dict_net['Ny'])})
				dict_net.update({'phiPerturbRot': eva.rotate_phases(dict_net['phiPerturb'], isInverse=True)})
				dict_net['phiPerturbRot'][0] = initPhiPrime0
				print('No perturbations defined, work it out! So far no perturbations are set, i.e., all zero!')
				# sys.exit()

			# elif len(dict_net['phiPerturb'].shape)==2:
			# 	if len(dict_net['phiPerturb'][0,:])==dict_net['Nx']*dict_net['Ny']:
			# 		dict_net.update({'phiPerturbRot': eva.rotate_phases(dict_net['phiPerturb'].flatten(), isInverse=True)})
			# 		print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
			# 		dict_net['phiPerturbRot'][:,0] = initPhiPrime0				# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
			# 		print('\nvalues of the perturbations in ROTATED phase space, i.e., last time-step of history set as initial condition:', dict_net['phiPerturbRot'])
			# 		print('dirac delta phase perturbation in ORIGINAL phase space:', dict_net['phiPerturb'], '\n')
			#
			# else:
			# 	print('Either no initial perturbations given, or Error in parameters - supply:\ncase_[sim_mode].py [topology] [#osci] [K] [F_c] [delay] [F_Omeg] [k] [Tsim] [c] [Nsim] [Nx] [Ny] [mx] [my] [cPD] [N entries for the value of the perturbation to oscis]')
			# 	print('\nNo perturbation set, hence all perturbations have the default value zero (in original phase space of phases)!')
			# 	dict_net.update({'phiPerturb': np.zeros(dict_net['Nx']*dict_net['Ny'])})
			# 	dict_net.update({'phiPerturbRot': np.zeros(dict_net['Nx']*dict_net['Ny'])})

	twistdelta = 0
	cheqdelta = 0
	twistdelta_x = 0
	twistdelta_y = 0
	if not (dict_net['topology'] == 'ring' or dict_net['topology'] == 'chain'):
		if dict_net['topology'] == 'square-open' or dict_net['topology'] == 'hexagon' or dict_net['topology'] == 'octagon':
			cheqdelta_x = np.pi 																				# phase difference between neighboring oscillators in a stable chequerboard state
			cheqdelta_y = np.pi 																				# phase difference between neighboring oscillators in a stable chequerboard state
			# print('phase differences of',k,'-twist:', twistdelta, '\n')
			if (dict_net['mx'] == 0 and dict_net['my'] == 0):
				dict_net.update( {'phiInitConfig': np.zeros(dict_net['Nx']*dict_net['Ny'])} )					# phiInitConfig denotes the unperturbed initial phases according to the m-twist state under investigation

			elif dict_net['mx'] != 0 and dict_net['my'] != 0:
				for rows in range(dict_net['Ny']):																# set the mx-my-twist state's initial condition (history of "perfect" configuration)
					phiMtemp = np.arange(cheqdelta_y*rows, dict_net['Nx']*cheqdelta_x+cheqdelta_y*rows, cheqdelta_x)
					dict_net['phiInitConfig'].append(phiMtemp)
				dict_net.update( {'phiInitConfig': np.array(dict_net['phiInitConfig'])%(2.0*np.pi)} )
				#phiM = phiM.flatten();  # print('phiM: ', phiM, ' phiM.ndim: ', phiM.ndim)
				dict_net.update( {'phiInitConfig': np.concatenate( dict_net['phiInitConfig'], axis=0 )} )

			elif dict_net['mx'] == 0 and dict_net['my'] != 0:													# prepare chequerboard only in y-direction
				for rows in range(dict_net['Ny']):																# set the chequerboard state's initial condition (history of "perfect" configuration)
					phiMtemp = np.arange(0.0, (dict_net['Nx'])*cheqdelta_x, cheqdelta_x)
					dict_net['phiInitConfig'].append(phiMtemp)
					#print('rows:', rows, 'dict_net[*phiInitConfig*]',dict_net['phiInitConfig'])
				dict_net.update( {'phiInitConfig': np.array(dict_net['phiInitConfig'])%(2.0*np.pi)} )
				# phiM = phiM.flatten(); # print('phiM: ', phiM)
				dict_net.update( {'phiInitConfig': np.concatenate( dict_net['phiInitConfig'], axis=0 )} )
				#print('dict_net[*phiInitConfig*]',dict_net['phiInitConfig']); sys.exit()

			elif dict_net['mx'] != 0 and dict_net['my'] == 0:													# prepare chequerboard only in x-direction
				for columns in range(dict_net['Nx']):															# set the chequerboard state's initial condition (history of "perfect" configuration)
					phiMtemp = np.arange(0.0, (dict_net['Ny'])*cheqdelta_y, cheqdelta_y)
					dict_net['phiInitConfig'].append(phiMtemp)
				dict_net.update({'phiInitConfig': np.array(phiM)%(2.0*np.pi)})
				# phiM = phiM.flatten(); # print('phiM: ', phiM)
				dict_net.update({'phiInitConfig': np.concatenate( dict_net['phiInitConfig'], axis=0)})

		elif dict_net['topology'] == 'hexagon-periodic' or dict_net['topology'] == 'octagon-periodic' or dict_net['topology'] == 'square-periodic':
			twistdelta_x = ( 2.0 * np.pi * dict_net['my'] / ( float( dict_net['Nx'] ) ) )						# phase difference between neighboring oscillators in a stable m-twist state
			twistdelta_y = ( 2.0 * np.pi * dict_net['my'] / ( float( dict_net['Ny'] ) ) )						# phase difference between neighboring oscillators in a stable m-twist state
			# print('phase differences of',k,'-twist:', twistdelta, '\n')
			# print('N =', N, '    Nx =', Nx, '    Ny =', Ny, '    k =', k, '    kx =', kx, '    ky =', ky)
			if dict_net['mx'] == 0 and dict_net['my'] == 0:
				dict_net.update( {'phiInitConfig': np.zeros(dict_net['Nx']*dict_net['Ny'])} ) # phiM denotes the unperturbed initial phases according to the m-twist state under investigation
				print('Length, type and shape of phiM:', len(dict_net['phiInitConfig']), type(dict_net['phiInitConfig']), dict_net['phiInitConfig'].shape)
			else:
				# print('type phiM at initialization', type(phiM))
				# print('Entering loop over Ny to set initial phiM.')
				for rows in range(dict_net['Ny']):																# set the mx-my-twist state's initial condition (history of "perfect" configuration)
					# print('loop #', rows)
					#phiMtemp = np.arange(twistdelta_y*rows, Nx*twistdelta_x+twistdelta_y*rows, twistdelta_x)
					phiMtemp = twistdelta_x * np.arange(dict_net['Nx']) + twistdelta_y * rows
					# print('phiMtemp=', phiMtemp, '    of type ', type(phiMtemp), '    and length ', len(phiMtemp))
					dict_net['phiInitConfig'].append(phiMtemp)
					# print('phiM(list)=', phiMt, '    of type ', type(phiMt))

				dict_net.update({'phiInitConfig': np.array(dict_net['phiInitConfig'])} )
				# print('phiM[1,]', phiM[1,])
				# print('phiM(array)=', phiM, '    of type ', type(phiM), '    and shape ', phiM.shape)

				phiMreorder=np.zeros(dict_net['Nx']*dict_net['Ny'])					 							# could be replaced by phiM = np.concatenate( phiM, axis=0 )
				counter = 0
				for i in range(dict_net['Nx']):
					for j in range(dict_net['Ny']):
						# print('counter:', counter)
						phiMreorder[counter] = dict_net['phiInitConfig'][i][j]
						counter += 1
				dict_net.update({'phiInitConfig': phiMreorder % (2.0*np.pi)})
				# print('phiMreorderd: ', phiM, '    of type ', type(phiM), '    and shape ', phiM.shape)

				# NOPE phiM = np.reshape(phiM, (np.product(phiM.shape),))
				# phiM = phiM.flatten();
				# phiM = phiM[:][:].flatten();
				# print('phiMflattened: ', phiM, '    of type ', type(phiM), '    and shape ', phiM.shape)
				# print('Length, type and shape of phiMflattened that was generated:', len(phiM), type(phiM), phiM.shape)
	if dict_net['topology'] == 'ring' or dict_net['topology'] == 'chain':
		if dict_net['topology'] == 'chain':
			cheqdelta = np.pi																					# phase difference between neighboring oscillators in a stable chequerboard state
			if dict_net['mx'] == 0:
				dict_net.update({'phiInitConfig': np.zeros(dict_net['Nx']*dict_net['Ny'])})						# phiM denotes the unperturbed initial phases according to the chequerboard state under investigation
			else:
				dict_net.update({'phiInitConfig': np.arange(0.0, dict_net['Nx']*dict_net['Ny']*cheqdelta, cheqdelta)}) # vector mit N entries from 0 increasing by twistdelta for every element, i.e., the phase-configuration
				# print('phiM: ', phiM)																			# in the original phase space of an chequerboard solution
		else:
			twistdelta = (2.0 * np.pi * dict_net['mx'] / (float( dict_net['Nx']*dict_net['Ny'])))				# phase difference between neighboring oscillators in a stable m-twist state
			# print('phase differences of',k,'-twist:', twistdelta, '\n')
			if dict_net['mx'] == 0:
				dict_net.update( {'phiInitConfig': np.zeros(dict_net['Nx']*dict_net['Ny'])} )					# phiM denotes the unperturbed initial phases according to the m-twist state under investigation
			else:
				dict_net.update({'phiInitConfig': np.arange(0.0, dict_net['Nx']*twistdelta, twistdelta)})		# vector mit N entries from 0 increasing by twistdelta for every element, i.e., the phase-configuration
				dict_net.update({'phiInitConfig': np.array(dict_net['phiInitConfig'])%(2.0*np.pi)})				# bring back into interval [0 2pi]
		print('dict_net[*phiInitConfig*]:\t', dict_net['phiInitConfig'])										# in the original phase space of an m-twist solution
	if dict_net['topology'] == 'global' or dict_net['topology'] == 'entrainPLLsHierarch':
		dict_net.update({'phiInitConfig': np.zeros(dict_net['Nx']*dict_net['Ny'])})								# for all-to-all coupling we assume no twist states with m > 0


################################################################################
def setup_time_dependent_parameter(dict_net: dict, dict_pll: dict, dict_data: dict, parameter: str = 'coupK', zero_initially: bool = False, start_time_dependency_after_percent_of_tsim: float = 0.15, for_all_plls_different_time_dependence: bool = False) -> np.ndarray:
	"""
	Set time dependent parameters to be used in the simulation. Different functional forms are available: triangle, linear, exponential

	Args:
		dict_pll:  [dict] contains the setup information for the PLL objects
		dict_net:  [dict] contains the setup information for the network and simulation
		dict_data: [dict] contains data to prepare and results from the simulation
		parameter: [str] the name of the parameter that shall be time-dependent
		zero_initially: [bool] if True, the initial value until start_time_dependency_after_percent_of_tsim is reached is set to zero, if False it will be set to the first value set in
		start_time_dependency_after_percent_of_tsim: [float] how long after the start of the simulation before the parameter starts changing in percent of Tsim, the simulation time dict_net['min_max_rate_timeDepPara']
		for_all_plls_different_time_dependence: [bool] whether that time-dependence will be different or the same for all oscillators in the network

	Returns:
		time series as np.ndarray that contains the values of the parameter over time
	"""

	delta_max = np.abs(dict_net['min_max_rate_timeDepPara'][1] - dict_net['min_max_rate_timeDepPara'][0])

	if dict_net['typeOfTimeDependency'] == 'triangle' or dict_net['typeOfTimeDependency'] == 'linear':
		if dict_net['typeOfTimeDependency'] == 'triangle':
			length_modifier = 2
		else:
			length_modifier = 1
		dict_net.update({'Tsim': (length_modifier * delta_max / dict_net['min_max_rate_timeDepPara'][2]) * (1.0 + start_time_dependency_after_percent_of_tsim)})
	elif dict_net['typeOfTimeDependency'] == 'exponential':
		dict_net.update({'Tsim': (1 / dict_net['min_max_rate_timeDepPara'][2]) * np.log(dict_net['min_max_rate_timeDepPara'][1]/dict_net['min_max_rate_timeDepPara'][0]) * (1.0 + start_time_dependency_after_percent_of_tsim)})
	else:
		print('Unknown functional form for time-dependent parameter generation. Introduce! °)°')
		sys.exit()

	dict_pll.update({'sim_time_steps': int(dict_net['Tsim'] / dict_pll['dt'])})

	if for_all_plls_different_time_dependence:
		time_series = np.zeros([ dict_net['Nx']*dict_net['Ny'], dict_net['max_delay_steps']+dict_pll['sim_time_steps'] ])
	else:
		time_series = np.zeros([ 1, dict_net['max_delay_steps']+dict_pll['sim_time_steps'] ])
	#print('dict_net[*min_max_rate_timeDepPara*][1]:', dict_net['min_max_rate_timeDepPara'][1])

	tstep_annealing_start = dict_net['max_delay_steps'] + int(start_time_dependency_after_percent_of_tsim * dict_pll['sim_time_steps'])
	print('Adjusted and updated Tsim to fit time-dependent parameter properties:', dict_net['Tsim'], 'sim_time_steps:', dict_pll['sim_time_steps'], '\nTime-dependent change starts at t_c=', tstep_annealing_start * dict_pll['dt'])

	# starting after 'max_delay_steps', we write a list/array of the time-dependent parameter given the functional form chosen
	sign = -1
	if dict_net['typeOfTimeDependency'] == 'linear':
		if (dict_net['special_case'] == 'timeDepInjectLockCoupStr'):
			print('NOTE: coupStr_2ndHarm overridden by min_max_rate_timeDepPara!')
		for i in range(len(time_series[:, 0])):
			sign *= -1
			if zero_initially:
				time_series[i, 0:tstep_annealing_start - 2] = 0
				time_series[i, tstep_annealing_start - 2:tstep_annealing_start] = dict_net['min_max_rate_timeDepPara'][0]
			else:
				time_series[i, 0:tstep_annealing_start] = dict_net['min_max_rate_timeDepPara'][0]
			for j in range(tstep_annealing_start-1, dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1):
				if np.abs( time_series[i, j] - dict_net['min_max_rate_timeDepPara'][0] ) <= delta_max:
					time_series[i, j+1] = time_series[i, j] + sign * dict_pll['dt'] * dict_net['min_max_rate_timeDepPara'][2]
					#print('time_series', time_series)
				else:
					time_series[i, j+1] = time_series[i, j]

	elif dict_net['typeOfTimeDependency'] == 'exponential':
		if (dict_net['special_case'] == 'timeDepInjectLockCoupStr'):
			print('NOTE: coupStr_2ndHarm overridden by min_max_rate_timeDepPara!')
		for i in range(len(time_series[:, 0])):
			print('Annealing starts after: ', tstep_annealing_start, ' steps.')
			if zero_initially:
				time_series[i, 0:tstep_annealing_start - 2] = 0
				time_series[i, tstep_annealing_start - 2:tstep_annealing_start] = dict_net['min_max_rate_timeDepPara'][0]
			else:
				time_series[i, 0:tstep_annealing_start] = dict_net['min_max_rate_timeDepPara'][0]
			for j in range(tstep_annealing_start-1, dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1):
				if np.abs(time_series[i, j] - dict_net['min_max_rate_timeDepPara'][0]) <= delta_max:
					#print('j:', j, '\ttime = (j-j0)*dt=', (j-tstep_annealing_start+1)*dict_pll['dt'])
					time_series[i, j+1] = time_series[i, j] + dict_pll['dt'] * dict_net['min_max_rate_timeDepPara'][0] * np.exp((j-tstep_annealing_start+1)*dict_pll['dt'] * dict_net['min_max_rate_timeDepPara'][2]) * dict_net['min_max_rate_timeDepPara'][2]
					# print('time_series[%i,%i+1]' % (i, j), time_series[i, j+1], '\tincrement:', dict_pll['dt'] * dict_net['min_max_rate_timeDepPara'][0] * np.exp((j-tstep_annealing_start+1)*dict_pll['dt'] * ( dict_net['min_max_rate_timeDepPara'][2]/1)) * dict_net['min_max_rate_timeDepPara'][2])
					# time.sleep(0.2)
				else:
					time_series[i, j+1] = time_series[i, j]

	elif dict_net['typeOfTimeDependency'] == 'triangle':
		index_max_value_time_dependent_parameter = np.zeros(len(time_series[:, 0]))
		if (dict_net['special_case'] == 'timeDepInjectLockCoupStr'):
			print('NOTE: coupStr_2ndHarm overridden by min_max_rate_timeDepPara!')
		for i in range(len(time_series[:, 0])):
			sign *= -1
			if zero_initially:
				time_series[i, 0:tstep_annealing_start - 2] = 0
				time_series[i, tstep_annealing_start - 2:tstep_annealing_start] = dict_net['min_max_rate_timeDepPara'][0]
			else:
				time_series[i, 0:tstep_annealing_start] = dict_net['min_max_rate_timeDepPara'][0]
			start_t_steps = tstep_annealing_start-1
			final_growth_t_steps = start_t_steps + np.int(np.floor((1-start_time_dependency_after_percent_of_tsim)*dict_pll['sim_time_steps']))
			if final_growth_t_steps <= start_t_steps:
				print('In setup_time_dependent_parameter(), the final time step value is smaller or equal w.r.t its initial! Change either start_time_dependency_after_percent_of_tsim or Tsim!')
				sys.exit()
			#print( 'Set rising interval! Starting at t=%0.5f'%(start_t_steps*dict_pll['dt']) )
			#print( 'Start raising value if %0.5f <= %0.5f'%(np.abs( time_series[i,start_t_steps] - dict_net['min_max_rate_timeDepPara'][0] ), delta_max) )

			for j in range(start_t_steps, final_growth_t_steps):
				time_series[i, j+1] = time_series[i, j] + sign * dict_pll['dt'] * dict_net['min_max_rate_timeDepPara'][2]

				if np.abs(time_series[i, j+1] - dict_net['min_max_rate_timeDepPara'][0]) >= delta_max:
					sign *= -1
			#print( 'Finished raising value at t=%0.5f and current value %0.5f'%( final_growth_t_steps*dict_pll['dt'], time_series[i,final_growth_t_steps] ) )
			#print('np.where(time_series[i, :] == np.max(time_series[i, :])):', np.where(time_series[i, :] == np.max(time_series[i, :])))
			index_max_value_time_dependent_parameter[i] = np.where(time_series[i, :] == np.max(time_series[i, :]))[0][0]
			print('Maximum of time-dependent parameter reached after t=%0.2f of the simulation time for PLL %i.' % (index_max_value_time_dependent_parameter * dict_pll['dt'], i))
		dict_net.update({'index_max_value_time_dependent_parameter': index_max_value_time_dependent_parameter})
		print('Hence, index_max_value_time_dependent_parameter=', index_max_value_time_dependent_parameter, ' while max_index_tsim=', len(time_series[0, :]))
		# sys.exit()
	else:
		print('Unknown functional form for time-dependent parameter generation. Introduce! °)°')
		sys.exit()

	dict_data.update({'timeDependentParameter': time_series})
	dict_data.update({'tstep_annealing_start': tstep_annealing_start})

	print('time-series time-dependent parameter: ', [*time_series])

	return time_series


################################################################################
def setup_topology(dict_net: dict):
	"""
	Generate a graph object of the network with the specified topology to compute the interactions between the oscillators.
	Will be used to assign the coupling partners of each oscillator.

	Args:
		dict_net:  [dict] contains the setup information for the network and simulation

	Returns:
		list of indices for each oscillator that contains the indexes of all neighbors of an oscillator k
	"""

	# see also: https://networkx.org/documentation/stable/reference/generators.html
	if dict_net['topology'] == 'global':
		G = nx.complete_graph(dict_net['Nx']*dict_net['Ny'])
		# print('G and G.neighbors(PLL0):', G, G.neighbors(0)); sys.exit(1)

	elif ( dict_net['topology'] == 'compareEntrVsMutual' and dict_net['Nx'] == 6):
		G = nx.DiGraph()
		G.add_nodes_from([i for i in range(dict_net['Nx']*dict_net['Ny'])])
		G.add_edges_from([(0, 1), (1, 0), (3, 2)])			  						# bidirectional coupling between 0 and 1 and 3 receives from 2, i.e., 2 entrains 3
		for i in range(dict_net['Nx']*dict_net['Ny']):
			print('For comparison entrainment vs mutual sync: neighbors of oscillator ', i, ':', list(G.neighbors(i)), ' and egdes of', i, ':', list(G.edges(i)))

	elif ( dict_net['topology'] == 'entrainPLLsHierarch'):
		G = nx.DiGraph()
		if dict_net['hierarchy_level'] > dict_net['Nx']*dict_net['Ny']:
			sys.exit('Special dict_net[*topology*] does not work like that... decrease hierarchy level - cannot exceed the number of PLLs in the system!')
		G.add_nodes_from([i for i in range(dict_net['Nx']*dict_net['Ny'])])
		for i in range(0, dict_net['hierarchy_level']):
			G.add_edge(i+1 ,i)														# add unidirectional edge from osci 0 to 1, 1 to 2, and so on until level_hierarch is reached

		for i in range(dict_net['hierarchy_level']+1, dict_net['Nx']*dict_net['Ny']):
			G.add_edge(i, dict_net['hierarchy_level']) 								# add unidirectional edge from highest hierarchy level to all remaining PLLS

	elif dict_net['topology'] == 'ring' or dict_net['topology'] == 'entrainAll-ring' or dict_net['topology'] == 'entrainOne-ring' or dict_net['topology'] == 'entrainAll':
		G = nx.cycle_graph(dict_net['Nx']*dict_net['Ny'])

		if dict_net['topology'] == 'entrainOne-ring':
			G = G.to_directed()  # convert to directed graph with directed edges
			G.remove_edge(0, 1)  # remove the unidirectional edge from oscillator 1 to oscillator 0
			G.remove_edge(0, dict_net['Nx'] - 1)  # remove the unidirectional edge from oscillator N to oscillator 0
			G.remove_edge(dict_net['Nx'] - 1, 0)  # remove the unidirectional edge from oscillator 0 to oscillator N
			G.add_edge(dict_net['Nx'] - 1, 1)  # add an edge to close the ring between oscillator N-1 and 1, while oscillator 0 becomes the reference
			G.add_edge(1, dict_net['Nx'] - 1)  # add an edge to close the ring between oscillator 1 and N-1, while oscillator 0 becomes the reference

		if dict_net['topology'] == 'entrainAll-ring':
			G = G.to_directed()  # convert to directed graph with directed edges
			G.remove_edge(0, 1)  # remove the unidirectional edge from oscillator 1 to oscillator 0
			G.remove_edge(0, dict_net['Nx'] - 1)  # remove the unidirectional edge from oscillator N to oscillator 0
			G.add_edge(dict_net['Nx'] - 1, 1)  # add an edge to close the ring between oscillator N-1 and 1, while oscillator 0 becomes the reference
			G.add_edge(1, dict_net['Nx'] - 1)  # add an edge to close the ring between oscillator 1 and N-1, while oscillator 0 becomes the reference
			for i in range(2, dict_net['Nx'] -1):
				G.add_edge(i, 0)

		if dict_net['topology'] == 'entrainAll':
			G = G.to_directed()  # convert to directed graph with directed edges
			G.remove_edge(0, 1)  # remove the unidirectional edge from oscillator 1 to oscillator 0
			G.remove_edge(0, dict_net['Nx'] - 1)  # remove the unidirectional edge from oscillator N to oscillator 0
			for i in range(1, dict_net['Nx'] - 1): # remove all edges so that all are entrained by oscillator zero
				G.remove_edge(i, i+1)
				G.remove_edge(i+1, i)
			for i in range(2, dict_net['Nx'] - 1):
				G.add_edge(i, 0)

	elif dict_net['topology'] == 'chain' or dict_net['topology'] == 'entrainOne-chain' or dict_net['topology'] == 'entrainAll-chain':
		G = nx.path_graph(dict_net['Nx']*dict_net['Ny'])

		if dict_net['topology'] == 'entrainOne-chain':
			G = G.to_directed()  # convert to directed graph with directed edges
			G.remove_edge(0, 1)  # remove the unidirectional edge from oscillator 1 to oscillator 0

		if dict_net['topology'] == 'entrainAll-chain':
			G = G.to_directed()  # convert to directed graph with directed edges
			G.remove_edge(0, 1)  # remove the unidirectional edge from oscillator 1 to oscillator 0
			for i in range(2, dict_net['Nx']):
				G.add_edge(i, 0)

	else:
		if dict_net['topology'] == 'square-open' or dict_net['topology'] == 'entrainOne-square-open' or dict_net['topology'] == 'entrainAll-square-open':
			G = nx.grid_2d_graph(dict_net['Nx'], dict_net['Ny'], periodic=False)
			if dict_net['topology'] == 'entrainOne-square-open':
				G = G.to_directed()  # convert to directed graph with directed edges
				G.remove_edge(0, 1)  # remove the unidirectional edge from oscillator 1 in row 0 to oscillator 0 in row 0
				G.remove_edge(0, dict_net['Nx'])  # remove the unidirectional edge from oscillator 0 (index Nx) in row 1 to oscillator 0 in row 0
				G.remove_edge(dict_net['Nx'], 0)  # remove the unidirectional edge from oscillator 0 (index Nx) in row 1 to oscillator 0 in row 0

			if dict_net['topology'] == 'entrainAll-square-open':
				G = G.to_directed()  # convert to directed graph with directed edges
				G.remove_edge(0, 1)  # remove the unidirectional edge from oscillator 1 in row 0 to oscillator 0 in row 0
				G.remove_edge(0, dict_net['Nx'])  # remove the unidirectional edge from oscillator 0 (index Nx) in row 1 to oscillator 0 in row 0
				for i in range(2, dict_net['Nx']*dict_net['Ny']-1):
					G.add_edge(0, i)

		elif dict_net['topology'] == 'square-periodic':
			G = nx.grid_2d_graph(dict_net['Nx'],dict_net['Ny'], periodic=True)		# for periodic boundary conditions:

		elif dict_net['topology'] == 'hexagon':
			print('\nIf Nx =! Ny, then check the graph that is generated again!')
			G = nx.grid_2d_graph(dict_net['Nx'],dict_net['Ny'])							# why not ..._graph(Nx,Ny) ? NOTE the for n in G: loop has to be changed, loop over nx and ny respectively, etc....
			for n in G:
				x, y = n
				if x > 0 and y > 0:
					G.add_edge(n, (x-1, y-1))
				if x < Nx-1 and y < Ny-1:
					G.add_edge(n, (x+1, y+1))

		elif dict_net['topology'] == 'hexagon-periodic':
			G = nx.grid_2d_graph(dict_net['Nx'],dict_net['Ny'], periodic=True)
			for n in G:
				x, y = n
				G.add_edge(n, ((x-1) % dict_net['Nx'], (y-1) % dict_net['Ny']))

		elif dict_net['topology'] == 'octagon':									# why not ..._graph(Nx,Ny) ? NOTE the for n in G: loop has to be changed, loop over nx and ny respectively, etc....
			print('\nIf Nx =! Ny, then check the graph that is generated again!')
			G = nx.grid_2d_graph(dict_net['Nx'],dict_net['Ny'])
			for n in G:
				x, y = n
				if x > 0 and y > 0:
					G.add_edge(n, (x-1, y-1))
				if x < Nx-1 and y < Ny-1:
					G.add_edge(n, (x+1, y+1))
				if x < Nx-1 and y > 0:
					G.add_edge(n, (x+1, y-1))
				if x < Nx-1 and y > 0:
					G.add_edge(n, (x+1, y-1))
				if x > 0 and y < Ny-1:
					G.add_edge(n, (x-1, y+1))

		elif dict_net['topology'] == 'octagon-periodic':
			G = nx.grid_2d_graph(dict_net['Nx'], dict_net['Ny'], periodic=True)
			for n in G:
				x, y = n
				G.add_edge(n, ((x-1) % dict_net['Nx'], (y-1) % dict_net['Ny']))
				G.add_edge(n, ((x-1) % dict_net['Nx'], (y+1) % dict_net['Ny']))

		# G = nx.convert_node_labels_to_integers(G)
		G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted') 						# converts 2d coordinates to 1d index of integers, e.g., k=0,...,N-1

	if dict_net['Nx']*dict_net['Ny'] < 36 and not (dict_net['special_case'] == 'timeDepChangeOfCoupStr' or dict_net['special_case'] == 'timeDepInjectLockCoupStr'):
		plt.figure(99999)
		nx.draw(G)
		F = nx.adjacency_matrix(G)
		print('nx.adjacency_matrix(G)\n', F.todense())
		print('nx.adjacency_spectrum(G)/max(nx.adjacency_spectrum(G))', nx.adjacency_spectrum(G)/max(nx.adjacency_spectrum(G)))

	return G


################################################################################
def all_initial_phase_combinations(dict_pll: dict, dict_net: dict, dict_algo: dict, paramDiscretization: int=10):
	"""
	Generates all initial phase combinations for the different simulation realizations to be computed.

	Args:
		dict_pll:  [dict] contains the setup information for the PLL objects
		dict_net:  [dict] contains the setup information for the network and simulation
		dict_data: [dict] contains data to prepare and results from the simulation
		paramDiscretization: [integer] specifies the discretization of the parameters

	Returns:
		a list of lists and a list of all combinations of this
	"""

	if isinstance(paramDiscretization, np.int):
		paramDiscr = [paramDiscretization, paramDiscretization]
	elif isinstance(paramDiscretization, list):
		paramDiscr = paramDiscretization
		print('List of paramDiscretizations was provided individually for the x- and y-axis.')
	else:
		print('Variable paramDiscretization needs to be integer or list of integers!')
		sys.exit()

	if dict_net['Nx']*dict_net['Ny'] == 2:
		if dict_algo['parameter_space_sweeps'] == 'listOfInitialPhaseConfigurations' and ( isinstance(dict_algo['min_max_range_parameter_0'], list) or isinstance(dict_algo['min_max_range_parameter_0'], np.ndarray) ):
			tempDetune = ( dict_algo['min_max_range_parameter_0'][1] - dict_algo['min_max_range_parameter_0'][0] ) / dict_pll['div']
			scanValueslist1 = list( np.linspace(-dict_pll['div']*(np.pi), +dict_pll['div']*(np.pi), paramDiscr[0]) ) 		# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
			scanValueslist2 = list( np.linspace(-tempDetune/2.0, tempDetune/2.0, paramDiscr[1]) )	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
			#print('scanValueslist2', scanValueslist2)
			scanValues = np.array([scanValueslist1, scanValueslist2], dtype=object)
			_allPoints 		= itertools.product(scanValues[0], scanValues[1])
			allPoints 		= list(_allPoints)									# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
			allPoints 		= np.array(allPoints)								# convert the list to an array
		else:
			scanValues = np.zeros((dict_net['Nx']*dict_net['Ny'], paramDiscr[0]), dtype=np.float) # create container for all points in the discretized rotated phase space, +/- pi around each dimension (unit area)
			# scanValues[0,:] = np.linspace(phiMr[0]-(np.pi), phiMr[0]+(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space
			# scanValues[1,:] = np.linspace(phiMr[1]-(np.pi), phiMr[1]+(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space
			scanValues[0, :] = np.linspace(-(np.pi), +(np.pi), paramDiscr[0]) 	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
			scanValues[1, :] = np.linspace(-(np.pi), +(np.pi), paramDiscr[0]) 	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
			#print('row', i,'of matrix with all intervals of the rotated phase space:\n', scanValues[i,:], '\n')

			_allPoints 		= itertools.product(*scanValues)
			allPoints 		= list(_allPoints)									# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
			allPoints 		= np.array(allPoints)								# convert the list to an array
		# allPoints_unitCell  = []
		# for point in allPoints:
		# 	if unit_cell.is_inside(point, isRotated=True):
		# 		allPoints_unitCell.append(point)
		# allPoints			= np.array(allPoints_unitCell)

	elif dict_net['Nx'] * dict_net['Ny'] == 3:
		# setup a matrix for all N variables/dimensions and create a cube around the origin with side lengths 2pi
		scanValues = np.zeros((dict_net['Nx'] * dict_net['Ny'] - 1, paramDiscr[0]), dtype=np.float)		# create container for all points in the discretized rotated phase space, +/- pi around each dimension (unit area)
		for i in range(0, dict_net['Nx'] * dict_net['Ny'] - 1):											# the different coordinates of the solution, discretize an interval plus/minus pi around each variable
			# scanValues[i,:] = np.linspace(phiMr[i+1]-np.pi, phiMr[i+1]+np.pi, paramDiscretization) # all entries are in rotated, and reduced phase space
			if i == 0:															# theta2 (x-axis)
				#scanValues[i,:] = np.linspace(-(np.pi), +(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
				#scanValues[i,:] = np.linspace(-0.25*np.pi, 0.25*np.pi, paramDiscretization)
				scanValues[i, :] = np.linspace(-1.0*np.pi, 1.0*np.pi, paramDiscr[0])
			else:																# theta3 (y-axis)
				#scanValues[i,:] = np.linspace(-(1.35*np.pi), +(1.35*np.pi), paramDiscretization)
				#scanValues[i,:] = np.linspace(-0.35*np.pi, 0.35*np.pi, paramDiscretization)
				scanValues[i, :] = np.linspace(-1.35*np.pi, 1.35*np.pi, paramDiscr[0])

			print('row', i, 'of matrix with all intervals of the rotated phase space:\n', scanValues[i, :], '\n')

		_allPoints 			= itertools.product(*scanValues)
		allPoints 			= list(_allPoints)									# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
		allPoints 			= np.array(allPoints) 								# convert the list to an array

	elif dict_net['Nx'] * dict_net['Ny'] > 3:
		# setup a matrix for all N variables/dimensions and create a cube around the origin with side lengths 2pi
		scanValues = np.zeros((dict_net['Nx'] * dict_net['Ny'] - 1, paramDiscr[0]), dtype=np.float)		# create container for all points in the discretized rotated phase space, +/- pi around each dimension (unit area)
		for i in range(0, dict_net['Nx'] * dict_net['Ny'] - 1):											# the different coordinates of the solution, discretize an interval plus/minus pi around each variable
			# scanValues[i,:] = np.linspace(phiMr[i+1]-np.pi, phiMr[i+1]+np.pi, paramDiscretization) # all entries are in rotated, and reduced phase space
			if i == 0:															# theta2 (x-axis)
				#scanValues[i,:] = np.linspace(-(np.pi), +(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
				#scanValues[i,:] = np.linspace(-0.25*np.pi, 0.25*np.pi, paramDiscretization)
				scanValues[i, :] = np.linspace(-1.0*np.pi, 1.0*np.pi, paramDiscr[0])
			else:																# theta3 (y-axis)
				#scanValues[i,:] = np.linspace(-(1.35*np.pi), +(1.35*np.pi), paramDiscretization)
				#scanValues[i,:] = np.linspace(-0.35*np.pi, 0.35*np.pi, paramDiscretization)
				scanValues[i, :] = np.linspace(-1.35*np.pi, 1.35*np.pi, paramDiscr[0])

			print('row', i,'of matrix with all intervals of the rotated phase space:\n', scanValues[i, :], '\n')

		_allPoints 			= itertools.product(*scanValues)
		allPoints 			= list(_allPoints)									# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
		allPoints 			= np.array(allPoints) 								# convert the list to an array
	#print('scanValues:', scanValues, '\tallPoints:', allPoints)
	# sys.exit()

	return scanValues, allPoints

################################################################################
def all_parameter_combinations_2d(dict_pll: dict, dict_net: dict, dict_algo: dict):
	"""
	Generates all parameter combinations for the different simulation realizations to be computed.

	Args:
		dict_pll:  [dict] contains the setup information for the PLL objects
		dict_net:  [dict] contains the setup information for the network and simulation
		dict_algo: [dict] contains the information which parameters are being changed, with which discretization

	Returns:
		list of lists with parameters to be scanned and array
	"""

	parameter_sweep_0 = np.linspace(dict_algo['min_max_range_parameter_0'][0], dict_algo['min_max_range_parameter_0'][1], dict_algo['paramDiscretization'][0])
	parameter_sweep_1 = np.linspace(dict_algo['min_max_range_parameter_1'][0], dict_algo['min_max_range_parameter_1'][1], dict_algo['paramDiscretization'][1])

	# in the case of entrainment only the frequency of the reference oscillator is to be changed
	if 'entrain' in dict_net['topology'] and dict_algo['param_id_0'] == 'intrF':
		print('In the case of entrainment only the frequency of the reference oscillator is to be changed!')
		temp = dict_pll['intrF'][1:]
		parameter_sweep_0 = [[i] + temp for i in parameter_sweep_0]
	elif 'entrain' in dict_net['topology'] and dict_algo['param_id_1'] == 'intrF':
		print('In the case of entrainment only the frequency of the reference oscillator is to be changed!')
		temp = dict_pll['intrF'][1:]
		parameter_sweep_1 = [[i] + temp for i in parameter_sweep_1]

	scanValues = np.array([parameter_sweep_0, parameter_sweep_1], dtype=object)

	if 'entrain' in dict_net['topology'] and (isinstance(dict_pll['intrF'], list) or isinstance(dict_pll['intrF'], np.ndarray)):
		_allPoints = itertools.product([i[0] for i in scanValues[0]], scanValues[1])
	else:
		_allPoints = itertools.product(scanValues[0], scanValues[1])
	allPoints = list(_allPoints)  								# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
	allPoints = np.array(allPoints)  							# convert the list to an array

	return scanValues, allPoints
