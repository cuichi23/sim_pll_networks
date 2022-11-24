#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys, gc
import numpy as np
#cimport numpy as np
#cimport cython
import networkx as nx
from scipy.signal import sawtooth
from scipy.signal import square
from scipy.stats import cauchy
from operator import add, sub
import matplotlib.cm as cm

from multiprocessing import Pool, freeze_support
import itertools

import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import pickle

from sim_pll.sim_lib import simulateSystem
from sim_pll import setup
from sim_pll import evaluation_lib as eva
from sim_pll import eval_ising_lib as eva_ising
from sim_pll import plot_lib as plot

now = datetime.datetime.now()

''' Enable automatic carbage collector '''
gc.enable()

def distributeProcesses(dict_net: dict, dict_pll: dict, dict_algo=None) -> object:
	"""Function that

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'

		Returns:
			pool_data: object with the simulation results of all simulations
	"""

	t0 = time.time()
	scanValues = prepare_multiple_simulations(dict_net, dict_pll, dict_algo)

	global number_period_dyn
	number_period_dyn = 20.5


	np.random.seed()
	# carry out all scheduled simulations
	pool_data = perform_multiple_simulations(dict_net, dict_pll, dict_algo, scanValues)

	print('time needed for execution of simulations in multiproc mode: ', (time.time()-t0), ' seconds')
	#sys.exit()

	eva.saveDictionaries(dict_pll, 'dict_pll', dict_pll['coupK'], dict_pll['transmission_delay'], dict_pll['cutFc'], dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'],
						 dict_net['topology'])  # save the dicts
	eva.saveDictionaries(dict_net, 'dict_net', dict_pll['coupK'], dict_pll['transmission_delay'], dict_pll['cutFc'], dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'],
						 dict_net['topology'])  # save the dicts
	eva.saveDictionaries(dict_algo, 'dict_algo', dict_pll['coupK'], dict_pll['transmission_delay'], dict_pll['cutFc'], dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'],
						 dict_net['topology'])  # save the dicts
	try:
		eva.saveDictionaries(pool_data, 'pool_data', dict_pll['coupK'], dict_pll['transmission_delay'], dict_pll['cutFc'], dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'],
							 dict_net['topology'])  # save the dicts
	except:
		print('Could not save the data, check memory and whether target directory exists.')


	# evaluate results of all simulations and parameter sweeps
	evaluate_pool_data(dict_net, dict_pll, dict_algo, pool_data)

	return pool_data


def prepare_multiple_simulations(dict_net: dict, dict_pll: dict, dict_algo: dict):
	if dict_algo['parameter_space_sweeps'] == 'classicBruteForceMethodRotatedSpace':  # classic approach with LP-adaptation developed with J. Asmus, D. Platz
		scanValues, allPoints = setup.all_initial_phase_combinations(dict_pll, dict_net, dict_algo,
																	 paramDiscretization=dict_algo['paramDiscretization'])  # set paramDiscretization for the number of points to be simulated
		print('allPoints:', [allPoints], '\nscanValues', scanValues)
		Nsim = allPoints.shape[0]
		dict_algo.update({'scanValues': scanValues, 'allPoints': allPoints})
		print('multiprocessing', Nsim, 'realizations')

	elif dict_algo['parameter_space_sweeps'] == 'listOfInitialPhaseConfigurations':  									# so far for N=2, work it out for N>2
		if (isinstance(dict_algo['paramDiscretization'], np.float) or isinstance(dict_algo['paramDiscretization'], np.int)) and (
				isinstance(dict_algo['min_max_range_parameter_0'], np.float) or isinstance(dict_algo['min_max_range_parameter_0'], np.int)):
			scanValues = np.linspace(-np.pi, np.pi, dict_algo['paramDiscretization'])
			print('scanValues', scanValues)
			Nsim = len(scanValues)
			print('multiprocessing', Nsim, 'realizations')
		elif isinstance(dict_algo['min_max_range_parameter_0'], np.ndarray) or isinstance(dict_algo['min_max_range_parameter_0'], list):
			scanValues, allPoints = setup.all_initial_phase_combinations(dict_pll, dict_net, dict_algo, paramDiscretization=dict_algo['paramDiscretization'])
			print('allPoints:', [allPoints], '\nscanValues', scanValues)
			Nsim = allPoints.shape[0]
			print('multiprocessing', Nsim, 'realizations')
		else:
			print(
				'2 modes: iterate for no detuning over phase-differences, or detuning and phase-differences! Choose one. HINT: if dict_algo[*paramDiscretization*] == an instance of list or ndarray, there needs to be a list of intrinsic frequencies!')
			sys.exit()

	elif dict_algo['parameter_space_sweeps'] == 'single':
		if dict_algo['param_id_0'] == 'None':
			dict_algo.update({'min_max_range_parameter_0': [1, 1], 'paramDiscretization': [1, 1]})
			print('No parameter to be changed, simulate only one realization!')
		else:
			print('Implement this!')
			sys.exit()
		scanValues, allPoints = setup.all_initial_phase_combinations(dict_pll, dict_net, dict_algo, paramDiscretization=dict_algo['paramDiscretization'])
		print('allPoints:', [allPoints], '\nscanValues', scanValues)
		Nsim = allPoints.shape[0]
		print('multiprocessing', Nsim, 'realizations')

	elif dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing':
		if dict_net['Nx'] * dict_net['Ny'] <= 3:
			scanValues, allPoints = setup.all_initial_phase_combinations(dict_pll, dict_net, dict_algo,
																		 paramDiscretization=dict_algo['paramDiscretization'])  # set paramDiscretization for the number of points to be simulated
			print('allPoints:', [allPoints], '\nscanValues', scanValues)
			Nsim = allPoints.shape[0]
			print('multiprocessing', Nsim, 'realizations')
		elif dict_net['Nx'] * dict_net['Ny'] > 3:
			if isinstance(dict_algo['paramDiscretization'], list):
				scanValues = np.random.uniform(low=-np.pi, high=np.pi, size=[dict_algo['paramDiscretization'][0], dict_net['Nx'] * dict_net['Ny']])
				if not dict_algo['param_id_0'] == None:
					print('Implement here that a parameter == changed and then generate all combinations with the above realizations?!')

			elif isinstance(dict_algo['paramDiscretization'], float) or isinstance(dict_algo['paramDiscretization'], int):
				scanValues = np.random.uniform(low=-np.pi, high=np.pi, size=[dict_algo['paramDiscretization'], dict_net['Nx'] * dict_net['Ny']])
			print('scanValues:', scanValues)
			# sys.exit()

	elif dict_algo['parameter_space_sweeps'] == 'two_parameter_sweep':  # organize after the data has been collected
		scanValues, allPoints = setup.all_parameter_combinations_2d(dict_pll, dict_net, dict_algo)  # set paramDiscretization for the number of points to be simulated
		print('scanning 2d parameter regime {', dict_algo['param_id_0'], ',', dict_algo['param_id_1'], '} allPoints:', [allPoints], '\nscanValues', scanValues)
		# print('\nscanValues[0, 1][0]', scanValues[0, 4][0])
		# print('\nscanValues[1, 1]', scanValues[1, 1])
		# sys.exit()
		Nsim = allPoints.shape[0]
		dict_algo.update({'scanValues': scanValues, 'allPoints': allPoints})
		print('multiprocessing', Nsim, 'realizations')

	elif dict_algo['parameter_space_sweeps'] == 'one_parameter_sweep':  # organize after the data has been collected
		scanValues, allPoints = setup.all_parameter_combinations_1d(dict_pll, dict_net, dict_algo)  # set paramDiscretization for the number of points to be simulated
		print('scanning 1d parameter regime', dict_algo['param_id_0'], ' allPoints:', [allPoints], '\nscanValues', scanValues)
		# print('\nscanValues[0, 1][0]', scanValues[0, 4][0])
		# print('\nscanValues[1, 1]', scanValues[1, 1])
		# sys.exit()
		Nsim = allPoints.shape[0]
		dict_algo.update({'scanValues': scanValues, 'allPoints': allPoints})
		print('multiprocessing', Nsim, 'realizations')

	elif dict_algo['parameter_space_sweeps'] == 'statistics':  # organize after the data has been collected
		print('Not yet tested, not yet implemented! Needs function that evaluates the data from the many realizations.')
		sys.exit()
	return scanValues


def perform_multiple_simulations(dict_net, dict_pll, dict_algo, scanValues):
	pool = Pool(processes=dict_algo['number_of_processes_in_multisim'])  # create a Pool object, pick number of processes
	freeze_support()
	initPhiPrime0 = 0
	pool_data = []
	# def multihelper(phiSr, initPhiPrime0, dict_net, dict_pll, phi, clock_counter, pll_list):
	if dict_algo['parameter_space_sweeps'] == 'classicBruteForceMethodRotatedSpace':
		pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
			itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	elif dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing':
		pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
			itertools.product(scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	elif dict_algo['parameter_space_sweeps'] == 'listOfInitialPhaseConfigurations':
		if isinstance(dict_algo['min_max_range_parameter_0'], np.float) or isinstance(dict_algo['min_max_range_parameter_0'], np.int):
			pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
				itertools.product(scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
		elif isinstance(dict_algo['min_max_range_parameter_0'], np.ndarray) or isinstance(dict_algo['min_max_range_parameter_0'], list):
			pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
				itertools.product(scanValues[0], scanValues[1]), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	elif dict_algo['parameter_space_sweeps'] == 'single':
		pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
			itertools.product(scanValues[0], scanValues[1]), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	elif dict_algo['parameter_space_sweeps'] == 'two_parameter_sweep':
		pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
			itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	elif dict_algo['parameter_space_sweeps'] == 'one_parameter_sweep':
		pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
			itertools.product(scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	else:
		print('This case is unknown in perform_multiple_simulations(...) in multisim_lib.py!')
		sys.exit()
	## def multihelper(phiSr, initPhiPrime0, dict_net, dict_pll, phi, clock_counter, pll_list):
	# if dict_algo['parameter_space_sweeps'] == 'classicBruteForceMethodRotatedSpace' or 'two_parameter_sweep':
	# 	pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
	# 		itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	# elif dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing' or dict_algo['parameter_space_sweeps'] == 'one_parameter_sweep':
	# 	pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
	# 		itertools.product(scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	# elif dict_algo['parameter_space_sweeps'] == 'listOfInitialPhaseConfigurations':
	# 	if isinstance(dict_algo['min_max_range_parameter_0'], np.float) or isinstance(dict_algo['min_max_range_parameter_0'], np.int):
	# 		pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
	# 			itertools.product(scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	# 	elif isinstance(dict_algo['min_max_range_parameter_0'], np.ndarray) or isinstance(dict_algo['min_max_range_parameter_0'], list):
	# 		pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
	# 			itertools.product(scanValues[0], scanValues[1]), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	# elif dict_algo['parameter_space_sweeps'] == 'single':
	# 	pool_data.append(pool.map(multihelper_star, zip(  # this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
	# 		itertools.product(scanValues[0], scanValues[1]), itertools.repeat(initPhiPrime0), itertools.repeat(dict_net), itertools.repeat(dict_pll), itertools.repeat(dict_algo))))
	# else:
	# 	print('This case is unknown in perform_multiple_simulations(...) in multisim_lib.py!')
	# 	sys.exit()
	return pool_data


def evaluate_pool_data(dict_net, dict_pll, dict_algo, pool_data):
	if dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing':
		eva_ising.evaluateSimulationIsing(pool_data, phase_wrap=2)

	elif dict_algo['parameter_space_sweeps'] == 'listOfInitialPhaseConfigurations':
		eva.evaluateSimulationsChrisHoyer(pool_data)

	elif dict_algo['parameter_space_sweeps'] == 'single':
		if dict_net['special_case'] != 'False':
			print('Plotting frequency vs time-dependent parameter!')
			plot.plot_instantaneous_freqs_vs_time_dependent_parameter(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'])
		if 'ctrl' in pool_data[0][0]['dict_data']:
			plot.plot_control_signal_dynamics(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'])
		if 'clock_counter' in pool_data[0][0]['dict_data']:
			plot.plot_clock_time_in_period_fractions(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'])
		plot.plot_phases_unwrapped(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'])
		plot.plot_phases_two_pi_periodic(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'])
		plot.plot_inst_frequency(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'])
		plot.plot_order_parameter(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'])
		plot.plot_phase_difference_wrt_to_osci_kzero(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'])
		plot.plot_phase_difference(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'])
		plot.plot_periodic_output_signal_from_phase(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'])
		plot.plot_inst_frequency_and_phase_difference(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_algo'], pool_data[0][0]['dict_data'])
		plot.plot_power_spectral_density(pool_data[0][0]['dict_pll'], pool_data[0][0]['dict_net'], pool_data[0][0]['dict_data'], pool_data[0][0]['dict_algo'], [], saveData=False)
		plt.draw()
		plt.show()

	elif dict_algo['parameter_space_sweeps'] == 'classicBruteForceMethodRotatedSpace':
		print('Implement evaluation as in the old version! Copy plots, etc...')
		sys.exit()

	elif dict_algo['parameter_space_sweeps'] == 'two_parameter_sweep':
		average_time_order_parameter_in_periods = 1.5
		#if (dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1] < 10
		#		and dict_pll['intrF'][0] == dict_algo['scanValues'][0, 4][0] and dict_pll['transmission_delay'] == dict_algo['scanValues'][1, 1]):
		if (dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1] < 10
				and dict_pll['intrF'][0] == dict_algo['scanValues'][0, 1][0] and dict_pll['transmission_delay'] == dict_algo['scanValues'][1, 1]):
			for i in range(dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1]):
				plot.plot_phases_unwrapped(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
				#plot.plot_phases_two_pi_periodic(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
				#plot.plot_inst_frequency(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
				plot.plot_order_parameter(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
				#plot.plot_phase_difference_wrt_to_osci_kzero(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
				#plot.plot_phase_difference(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
				#plot.plot_periodic_output_signal_from_phase(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
				plot.plot_inst_frequency_and_phase_difference(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_algo'], pool_data[0][i]['dict_data'])
				plt.draw()
				plt.show()
		plot.plot_order_param_vs_parameter_space(pool_data, average_time_order_parameter_in_periods=average_time_order_parameter_in_periods, colormap=cm.hsv)
		plot.plot_final_phase_configuration_vs_parameter_space(pool_data, average_time_phase_difference_in_periods=average_time_order_parameter_in_periods,
															phase_wrap=1, std_treshold_determine_time_dependency=0.15*np.pi,
															std_treshold_order_param_determine_time_dependency=0.075, colormap=cm.hsv)
		plt.draw()
		plt.show()

	elif dict_algo['parameter_space_sweeps'] == 'one_parameter_sweep':
		if 'entrain' in dict_net['topology'] and dict_algo['param_id_0'] == 'intrF':
			eva.evaluate_entrainment_of_mutual_sync(pool_data)
			if dict_algo['store_ctrl_and_clock']:
				for i in range(len(pool_data[0][:])):
					plot.plot_control_signal_dynamics(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'], plot_id=i)
		else:
			for i in range(len(pool_data[0][:])):
				if i == int(len(pool_data[0][:])/2):
					if dict_algo['store_ctrl_and_clock']:
						plot.plot_control_signal_dynamics(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'], plot_id=i)
					# plot.plot_phases_unwrapped(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
					plot.plot_phases_two_pi_periodic(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
					plot.plot_inst_frequency(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
					plot.plot_order_parameter(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
					plot.plot_phase_difference_wrt_to_osci_kzero(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
					plot.plot_phase_difference(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
					plot.plot_periodic_output_signal_from_phase(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
					plot.plot_inst_frequency_and_phase_difference(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_algo'], pool_data[0][i]['dict_data'])
					plot.plot_power_spectral_density(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'], pool_data[0][0]['dict_algo'], [], saveData=False)
				if i != len(pool_data[0][:])-1:
					plt.close('all')


def multihelper(iterConfig, initPhiPrime0, dict_net, dict_pll, dict_algo, param_id='None'):
	"""Function that

		Args:
			iterConfig
			initPhiPrime0
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'

		Returns:
	"""

	# make also copies of all other dictionaries so that later changes to not interfere with other realizations
	dict_pll_rea = dict_pll.copy()
	dict_algo_rea = dict_algo.copy()
	dict_net_rea = dict_net.copy()
	global number_period_dyn
	number_period_dyn = 20.5

	if dict_algo['parameter_space_sweeps'] == 'classicBruteForceMethodRotatedSpace':				# classic approach with LP-adaptation developed with J. Asmus, D. Platz
		phiSr = list(iterConfig)
		if dict_net['Nx']*dict_net['Ny'] > 2:
			phiSr = np.insert(phiSr, 0, initPhiPrime0)								# insert the first variable in the rotated space, constant initPhiPrime0
		phiS = eva.rotate_phases(phiSr, isInverse=False)							# rotate back into physical phase space
		print('TEST len(phiS):', len(phiS))
		# print('TEST in multihelper, phiS:', phiS, ' and phiSr:', phiSr)
		unit_cell = eva.PhaseDifferenceCell(dict_net['Nx']*dict_net['Ny'])
		# SO anpassen, dass auch gegen verschobene Einheitszelle geprueft werden kann (e.g. if not k==0...)
		# ODER reicht schon:
		# if not unit_cell.is_inside(( phiS ), isRotated=False):   ???
		# if not unit_cell.is_inside((phiS-phiM), isRotated=False):					# and not N == 2:	# +phiM

		dict_net_rea.update({'phiPerturb': phiS}) 			# 'phiPerturbRot': phiSr})
		#print('dict_net[*phiPerturb*]', dict_net['phiPerturb'])

		#print('Check whether perturbation is inside unit-cell! phiS:', dict_net_rea['phiPerturb'], '\tInside? True/False:', unit_cell.is_inside((dict_net_rea['phiPerturb']), isRotated=False)); time.sleep(2)
		if not unit_cell.is_inside((dict_net_rea['phiPerturb']), isRotated=False):				# NOTE this case is for scanValues set only in -pi to pi
			print('Set dummy solution! Detected case outside of unit-cell.')
			dict_data = {'mean_order': -1., 'last_orderP': -1., 'stdev_orderP': np.zeros(1), 'phases': dict_net['phiInitConfig'],
					 		'intrinfreq': np.zeros(1), 'coupling_strength': np.zeros(1), 'transdelays': dict_pll['transmission_delay'], 'orderP_t': np.zeros(int(number_period_dyn/(dict_pll['intrF']*dict_pll['dt'])))-1.0}
			realizationDict = {'dict_net': dict_net_rea, 'dict_pll': dict_pll_rea, 'dict_algo': dict_algo_rea, 'dict_data': dict_data}
			return realizationDict
		else:
			return simulateSystem(dict_net_rea, dict_pll_rea, dict_algo_rea, multi_sim=True)

	elif dict_algo['parameter_space_sweeps'] == 'listOfInitialPhaseConfigurations':		# so far for N=2, work it out for N>2
		initFreqDetune_vs_intrFreqDetune_equal = False
		change_param = list(iterConfig)
		#print('iterConfig:', iterConfig, '\tchange_param[0]:', change_param[0], '\tchange_param[1]:', change_param[1])
		if initFreqDetune_vs_intrFreqDetune_equal:
			if isinstance(dict_pll['intrF'], list):								# change_param[1] represents half the frequency difference to be achieved in the uncoupled state
				meanIntF = np.mean(dict_pll['intrF'])
				dict_pll_rea.update({'intrF': [meanIntF-change_param[1], meanIntF+change_param[1]]})
				#print('Intrinsic frequencies:', dict_pll_rea['intrF'], '\tfor detuning', 2*change_param[1]); time.sleep(2)
			else:
				dict_pll_rea.update({'intrF': [dict_pll['intrF']-change_param[1], dict_pll['intrF']+change_param[1]]})
		else:																	# here: oscillators have intrinsic frequencies as given in dict_pll['intrF'], however initially they evolve
																				# with different frequencies given by syncF +/- half_the_freq_difference given by change_param[1]
			dict_pll_rea.update({'syncF': [dict_pll['syncF']-change_param[1], dict_pll['syncF']+change_param[1]]})
			dict_pll_rea.update({'typeOfHist': 'syncState'})						# makes sure this mode is active
			print('WATCH OUT: dirty trick to achieve different frequency differences at the end of the history!!! Discuss with Chris Hoyer and address issue.')

		config = [0, change_param[0]]
		dict_net_rea.update({'phiInitConfig': config, 'phiPerturb': np.zeros(dict_net['Nx']*dict_net['Ny']), 'phiPerturbRot': np.zeros(dict_net['Nx']*dict_net['Ny'])})

		return simulateSystem(dict_net_rea, dict_pll_rea, dict_algo_rea, multi_sim=True)

	elif dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing':
		change_param = list(iterConfig)
		print('iterConfig:', iterConfig, '\tchange_param[0]:', change_param[0])
		# sys.exit()
		# if dict_net['Nx']*dict_net['Ny']
		config = []
		[config.append(entry) for entry in change_param[0]]
		dict_net_rea.update({'phiInitConfig': config, 'phiPerturb': np.zeros(dict_net['Nx']*dict_net['Ny']), 'phiPerturbRot': np.zeros(dict_net['Nx']*dict_net['Ny'])})
		#print('NEW REALIZATION WITH INITIAL PHASES: dict_net_rea[*phiInitConfig*]:', dict_net_rea['phiInitConfig'])

		return simulateSystem(dict_net_rea, dict_pll_rea, dict_algo_rea, multi_sim=True)

	elif dict_algo['parameter_space_sweeps'] == 'single':
		change_param = list(iterConfig)
		if not dict_algo['param_id_0'] == 'None':
			dict_pll_rea.update({param_id: change_param})							# update the parameter chosen in change_param with a value of all scanvalues
		else:
			print('No parameters for sweep specified -- hence simulating the same parameter set for all realizations!')

		return simulateSystem(dict_net_rea, dict_pll_rea, dict_algo_rea, multi_sim=True)

	elif dict_algo['parameter_space_sweeps'] == 'one_parameter_sweep':
		change_param = list(iterConfig)

		print('######### realization for %s #########\n' % dict_algo['param_id_0'], change_param[0])

		if not dict_algo['param_id_0'] == 'None':
			dict_pll_rea.update({dict_algo['param_id_0']: change_param[0]})
		else:
			print('No parameters for sweep specified -- hence simulating the same parameter set for all realizations!')

		return simulateSystem(dict_net_rea, dict_pll_rea, dict_algo_rea, multi_sim=True)

	elif dict_algo['parameter_space_sweeps'] == 'two_parameter_sweep':
		change_param = list(iterConfig)

		# print('######### realization: {delay, intrinsic freqs.} #########\n', change_param[0], change_param[1])

		if not dict_algo['param_id_0'] == 'None' and not dict_algo['param_id_1'] == 'None':
			dict_pll_rea.update({dict_algo['param_id_0']: change_param[0]})
			dict_pll_rea.update({dict_algo['param_id_1']: change_param[1]})
			# print('dict_net[*phiPerturb*]', dict_net['phiPerturb'])
			# print('\n\nCHANGED dict for realization: ', dict_pll_rea[dict_algo['param_id_0']], dict_pll_rea[dict_algo['param_id_1']])
		else:
			print('No parameters for sweep specified -- hence simulating the same parameter set for all realizations!')

		if not dict_net['phiInitConfig'] and 'entrain' in dict_net_rea['topology']:  # needs to be done here to catch this case below if there are np.nans!
			print('\nPhase configuration of synchronized state will be set according to supplied topology and twist state information!')
			setup.generate_phi0(dict_net_rea, dict_pll_rea, dict_algo_rea)
		if np.isnan(dict_net_rea['phiInitConfig']).any() and 'entrain' in dict_net_rea['topology']:
			print('Set dummy solution! Detected case has no valid solution to the inverse coupling function, hence no solution exists.')
			dict_data = {'order_parameter': np.zeros(int(number_period_dyn / (dict_pll['syncF'] * dict_pll['dt']))) - dict_pll_rea['div'], 'phi': np.zeros([1, int(dict_net_rea['Nx'] * dict_net_rea['Ny'])]),
							  'order_parameter_divided_phases': np.zeros(int(number_period_dyn / (dict_pll['syncF'] * dict_pll['dt']))) - 1, 'F1': 1}
			realizationDict = {'dict_net': dict_net_rea, 'dict_pll': dict_pll_rea, 'dict_algo': dict_algo_rea, 'dict_data': dict_data}
			return realizationDict
		else:
			return simulateSystem(dict_net_rea, dict_pll_rea, dict_algo_rea, multi_sim=True)

	elif dict_algo['parameter_space_sweeps'] == 'statistics':
		change_param = list(iterConfig)
		if not dict_algo['param_id_0'] == 'None':
			dict_pll_rea.update({param_id: change_param})							# update the parameter chosen in change_param with a value of all scanvalues

		return simulateSystem(dict_net_rea, dict_pll_rea, dict_algo_rea, multi_sim=True)

	else:
		print('No case fulfilled in multihelper in multisim_lib!')
		sys.exit()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def multihelper_star(dynparam_fixparam):
	return multihelper(*dynparam_fixparam)
