#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys, gc, os
import numpy as np
#cimport numpy as np
#cimport cython
import networkx as nx
from scipy.signal import sawtooth
from scipy.signal import square
from scipy.stats import cauchy
from operator import add, sub
from tempfile import TemporaryFile

import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import pickle

from sim_pll import setup
from sim_pll import evaluation_lib as eva
from sim_pll import plot_lib as plot

now = datetime.datetime.now()

''' Enable automatic carbage collector '''
gc.enable()


''' SIMULATE NETWORK '''
def simulateSystem(dict_net: dict, dict_pll: dict, dict_algo: dict, multi_sim=False):
	"""Function that organizes the simulation. a) sets up a list of PLLs and a container for the results (dict_data). Writes the initial histories as specified. b) carries out the
		evolution of the PLL's phases. c) save results and returns results or plots them.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'

		Returns:
			dict_net
			dict_pll
			dict_algo
			dict_data
	"""
	#mode,div,Nplls,F,F_Omeg,K,Fc,delay,feedback_delay,dt,c,Nsteps,topology,couplingfct,histtype,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx=0,Ny=0,Trelax=0,Kadiab_value_r=0):

	t0 = time.time()
	# restart pseudo random-number generator
	np.random.seed()

	# prepare simulation, write histories of phases, set initial conditions, etc.
	dict_data, pll_list = prepare_simulation(dict_net, dict_pll, dict_algo)

	# now simulate the system after history is set
	perform_simulation_case(dict_net, dict_pll, dict_algo, dict_data, pll_list)

	# save the data before performing any additional evaluation or plotting
	if not multi_sim:
		print('Time needed for execution of simulation: ', (time.time()-t0), ' seconds, now save result dicts.')
		save_results_simulation(dict_net, dict_pll, dict_algo, dict_data)

	# run evaluations
	perform_evaluation(dict_net, dict_pll, dict_data)

	# if this simulation is one of many, the results are returned to the container that hold the results of all simulations, otherwise the results are plotted
	if multi_sim:
		return {'dict_net': dict_net, 'dict_pll': dict_pll, 'dict_algo': dict_algo, 'dict_data': dict_data}
	else:
		plot_results_simulation(dict_net, dict_pll, dict_algo, dict_data)
		print('Time needed for simulation, evaluation and plotting: ', (time.time() - t0), ' seconds')
		plt.show()
		return dict_net, dict_pll, dict_algo, dict_data

###########################################################################################################################################
###########################################################################################################################################

def prepare_simulation(dict_net: dict, dict_pll: dict, dict_algo: dict):
	"""Function that prepares an individual simulation. It sets up the network of oscillators with their properties, including the neighbor relations as specified by the topology.
		Creates the data structures to store the histories depending on the time delay (memory). It sets the initial histories of all oscillators and stores the result.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged

		Returns:
			pll_list
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
	"""
	dict_data = {}  # setup dictionary that hold all the data
	space = setup.generate_space(dict_net, dict_pll, dict_data)  # generates a space object used to handle pll distributed in a continuous space
	if len(dict_net['phiInitConfig']) == 0:  							# if no custom phase configuration is provided, generate it
		print('\nPhase configuration of synchronized state will be set according to supplied topology and twist state information!')
		setup.generate_phi0(dict_net, dict_pll, dict_algo)  # generate the initial phase configuration for twist, chequerboard, in- and anti-phase states
	pll_list = setup.generate_plls(dict_pll, dict_net, dict_data)  # generate a list that contains all the PLLs of the network
	all_transmit_delay = [np.max(n.delayer.transmit_delay_steps) for n in pll_list]  # obtain all the transmission delays for all PLLs
	all_feedback_delay = [n.delayer.feedback_delay_steps for n in pll_list]  # obtain all the feedback delays for all PLLs
	max_feedback_delay = np.max(all_feedback_delay)
	max_transmit_delay = np.max(np.array([np.max(i) for i in all_transmit_delay]))
	print('\nMaximum cross-coupling time delay: max_transmit_delay_steps:', max_transmit_delay, '\tmax_feedback_delay_steps:', max_feedback_delay)
	max_delay_steps = np.max([max_transmit_delay, max_feedback_delay])  # pick largest time delay to setup container for phases
	# prepare container for the phases
	if max_delay_steps == 0:
		print('No delay case, not yet tested, see sim_lib.py! Setting container length to that of Tsim/dt!')
		# sys.exit()
		phi_array_len = dict_pll['sim_time_steps']  # length of phi contrainer in case the delay is zero; the +int(dict_pll['orderLF']) is necessary to cover filter up to order dict_pll['orderLF']
	else:
		phi_array_len = 1 + int(dict_net['phi_array_mult_tau']) * max_delay_steps  # length of phi contrainer, must be at least 1 delay length if delay > 0
	phi = np.zeros([phi_array_len, dict_net['Nx'] * dict_net['Ny']])  # prepare container for phase time series of all PLLs
	clock_counter = np.empty([phi_array_len, dict_net['Nx'] * dict_net['Ny']])  # list for counter, i.e., the clock derived from the PLL
	for i in range(len(pll_list)):  # set max_delay in delayer: CHECK AGAIN!
		pll_list[i].delayer.phi_array_len = phi_array_len
	dict_net.update({'max_delay_steps': max_delay_steps, 'phi_array_len': phi_array_len})
	## TESTS!
	# print('Test coupling %s(pi)='%dict_pll['coup_fct_sig'], dict_pll['coup_fct_sig'](1.0*np.pi)); sys.exit()
	# set initial phase configuration and history -- performed such that any configuration can be obtained when simulations starts
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SET INITIAL HISTORY AND PERTURBATION '''
	# start by setting last entries to initial phase configuration, i.e., phiInitConfig + phiPerturb
	print('The perturbation will be set to dict_net[*phiPerturb*]=', dict_net['phiPerturb'], ', and dict_net[*phiInitConfig*]=', dict_net['phiInitConfig'])
	if not (isinstance(dict_net['phiPerturb'], list) or isinstance(dict_net['phiPerturb'], np.ndarray)):
		print('All initial perturbations set to zero as none were supplied!')
		dict_net['phiPerturb'] = [0 for i in range(len(phi[0, :]))]
		phi[dict_net['max_delay_steps'], :] = list(map(add, dict_net['phiInitConfig'], dict_net['phiPerturb']))  # set phase-configuration phiInitConfig at t=0 + the perturbation phiPerturb
	elif (len(phi[0, :]) == len(dict_net['phiInitConfig']) and len(phi[0, :]) == len(dict_net['phiPerturb'])):
		phi[dict_net['max_delay_steps'], :] = list(map(add, dict_net['phiInitConfig'], dict_net['phiPerturb']))  # set phase-configuration phiInitConfig at t=0 + the perturbation phiPerturb
	else:
		print('len(phi[0,:])', len(phi[0, :]), '\t len(dict_net[*phiInitConfig*])', len(dict_net['phiInitConfig']))
		print('Provide initial phase-configuration of length %i to setup simulation!' % len(phi[dict_net['max_delay_steps'], :]))
		sys.exit()
	## TESTS!
	# plt.plot(phi[:,0], 'o'); plt.plot(phi[:,1], 'd'); plt.draw(); plt.show();
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SET THE INTERNAL PHI VARS OF THE VCO TO THEIR INITIAL VALUE '''
	for i in range(len(pll_list)):  # set initial phases at time equivalent to the time-delay, then setup the history from there
		pll_list[i].signal_controlled_oscillator.phi = phi[max_delay_steps, i]
	# print('VCOs internal phis are set to:', [pll.vco.phi for pll in pll_list]); sys.exit()
	# if uncoupled history, just evolve backwards in time until the beginning of the phi container is reached
	if dict_pll['typeOfHist'] == 'freeRunning':  # in the 'uncoupled' case the oscillators evolve to the perturbed state during the history
		for i in range(max_delay_steps + 1, 0, -1):
			# print('i-1',i-1)
			phi[i - 1, :] = [pll.setup_hist_reverse() for pll in pll_list]
	elif dict_pll['typeOfHist'] == 'syncState':  # in the 'syncstate' case the oscillators evolve as if synced and then receive a delta perturbation -- syncF is the frequency
		# since we want a delta perturbation, the perturbation is removed towards the prior step
		phi[max_delay_steps - 1, :] = list(map(sub, [pll.setup_hist_reverse() for pll in pll_list], dict_net['phiPerturb']))  # local container to help the setup

		for i in range(len(pll_list)):
			pll_list[i].signal_controlled_oscillator.phi = phi[max_delay_steps - 1, i]  # set this step as initial for reverse history setup
		# print('VCOs internal phis are set to:', [pll.vco.phi for pll in pll_list]); sys.exit()
		for i in range(max_delay_steps - 1, 0, -1):
			# print('i-1',i-1)
			phi[i - 1, :] = [pll.setup_hist_reverse() for pll in pll_list]
		for i in range(len(pll_list)):
			pll_list[i].signal_controlled_oscillator.phi = phi[max_delay_steps, i]
	else:
		print('Specify the type of history, syncState or freeRunning supported!')
		sys.exit()
	## TESTS!
	# plt.plot(phi[:,0], 'o'); plt.plot(phi[:,1], 'd'); plt.draw(); plt.show();
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SHIFT ALL PHASES UP SUCH THAT AT t=-tau ALL ARE ABOVE ZERO '''
	phi[0:max_delay_steps + 1, :] = phi[0:max_delay_steps + 1, :] - np.min(phi[0, :])  # shift up/down all phases by the smallest phase of any PLL
	t = np.arange(0, len(phi[:, 0])) * dict_pll['dt']
	params = {'x': t, 'y': phi, 'label': 'phi', 'xlabel': 't', 'ylabel': 'phi', 'delay_steps': max_delay_steps, 'len_phi': phi_array_len - 1, 'dt': dict_pll['dt']}
	# eva.plotTest(params)
	for i in range(len(pll_list)):  # update all internal VCO phi variables
		pll_list[i].signal_controlled_oscillator.phi = phi[max_delay_steps, i]
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SET INITIAL CONTROL SIGNAL, ACCORDING AND CONSISTENT TO HISTORY WRITTEN, CORRECT INTERNAL PHASES OF VCO AND CLOCK '''
	for i in range(len(pll_list)):
		if max_delay_steps >= int(dict_pll['orderLF']):
			# print('last freqs:', ( phi[max_delay_steps-0,i]-phi[max_delay_steps-1,i]-dict_net['phiPerturb'][i] ) / (2.0*np.pi*dict_pll['dt']), ( phi[max_delay_steps-1,i]-phi[max_delay_steps-2,i] ) / (2.0*np.pi*dict_pll['dt']));
			pll_list[i].low_pass_filter.set_initial_control_signal(
				(phi[max_delay_steps - 0, i] - phi[max_delay_steps - 1, i] - dict_net['phiPerturb'][i]) / (
							2.0 * np.pi * dict_pll['dt']),
				(phi[max_delay_steps - 1, i] - phi[max_delay_steps - 2, i]) / (2.0 * np.pi * dict_pll['dt']))
		# NOTE: very important, the delta-like phase perturbation needs to be accounted for when calculating the instantaneous frequency of the last time step before simulation,
		#		here first argument of lf.set_initial_control_signal()

		elif max_delay_steps < int(dict_pll['orderLF']) and (int(dict_pll['orderLF']) == 2 or int(dict_pll['orderLF']) == 1):
			if dict_pll['typeOfHist'] == 'freeRunning':  # set the frequencyies in the past to determine the LF filter state for no delay
				inst_freq_lastStep = dict_pll['intrF'] + np.random.normal(loc=0.0, scale=np.sqrt(dict_pll['noiseVarVCO'] * dict_pll['dt']))
				inst_freq_prior_to_lastStep = dict_pll['intrF'] + np.random.normal(loc=0.0, scale=np.sqrt(dict_pll['noiseVarVCO'] * dict_pll['dt']))
			elif dict_pll['typeOfHist'] == 'syncState':
				inst_freq_lastStep = dict_pll['syncF'] + np.random.normal(loc=0.0, scale=np.sqrt(dict_pll['noiseVarVCO'] * dict_pll['dt']))
				inst_freq_prior_to_lastStep = dict_pll['syncF'] + np.random.normal(loc=0.0, scale=np.sqrt(dict_pll['noiseVarVCO'] * dict_pll['dt']))
			pll_list[i].low_pass_filter.set_initial_control_signal(inst_freq_lastStep, inst_freq_prior_to_lastStep)
		else:
			print('in simPLL.lib: Higher order LFs are net yet supported!')
		# print('Set internal initial VCO phi at t-dt for PLL %i:'%i, phi[max_delay_steps,i])
		# pll_list[i].vco.phi = phi[max_delay_steps,i]
		pll_list[i].counter.reset(phi[max_delay_steps, i])
	## TESTS!
	# print('dict_net[*max_delay_steps*]', dict_net['max_delay_steps'], '\tmax_delay_steps:', max_delay_steps, '\tlen(phi): ', len(phi))
	# print('history: ', phi[0:dict_net['max_delay_steps'],:])
	# plt.plot(phi[:,0], 'o'); plt.plot(phi[:,1], 'd'); plt.draw(); plt.show();
	# sys.exit()
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	dict_data.update({'clock_counter': clock_counter, 'phi': phi, 'all_transmit_delay': all_transmit_delay,
					 'all_feedback_delay': all_feedback_delay})
	return dict_data, pll_list


def perform_simulation_case(dict_net: dict, dict_pll: dict, dict_algo: dict, dict_data: dict, pll_list: list) -> None:
	"""Function that initiates the simulation of the network of oscillators.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""

	treshold_operate_on_tau_array = 1.5E6
	if dict_pll['sim_time_steps'] * dict_pll['dt'] <= treshold_operate_on_tau_array and dict_net['phi_array_mult_tau'] == 1 and dict_net['special_case'] == 'False':  # container to flush data
		evolve_system_on_tsim_array(dict_net, dict_pll, pll_list, dict_data, dict_algo)
	elif dict_pll['sim_time_steps'] * dict_pll['dt'] > treshold_operate_on_tau_array and dict_net['phi_array_mult_tau'] == 1 and dict_net['special_case'] == 'False':
		#print('Simulation on tau-array.')
		evolve_system_on_tau_array(dict_net, dict_pll, pll_list, dict_data, dict_algo)
	elif dict_net['special_case'] == 'test_case':
		print('Simulating testcase scenario!')
		evolve_system_test_cases(dict_net, dict_pll, pll_list, dict_data, dict_algo)
	elif dict_net['special_case'] == 'timeDepTransmissionDelay':
		if dict_algo['store_ctrl_and_clock'] == False:
			dict_algo.update({'store_ctrl_and_clock': True})
		evolve_system_on_tsim_array_time_dependent_change_of_delay_save_ctrl_signal(dict_net, dict_pll, pll_list, dict_data, dict_algo)
		plot.plot_control_signal_dynamics(dict_pll, dict_net, dict_data)
	elif dict_net['special_case'] == 'timeDepInjectLockCoupStr':
		evolve_system_on_tsim_array_time_dependent_change_of_inject_locking_coupling_strength_shil(dict_net, dict_pll, pll_list, dict_data, dict_algo)
	elif dict_net['special_case'] == 'timeDepChangeOfCoupStr':
		if dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing':
			if dict_pll['shil_generation_through_filter'] is False:
				evolve_system_on_tsim_array_time_dependent_change_of_coupling_strength_shil(dict_net, dict_pll, pll_list, dict_data, dict_algo)
			else:
				evolve_system_on_tsim_array_time_dependent_change_of_coupling_strength_shil_explicit_shil_filtering(dict_net, dict_pll, pll_list, dict_data, dict_algo)
		else:
			evolve_system_on_tsim_array_time_dependent_change_of_coupling_strength(dict_net, dict_pll, pll_list, dict_data, dict_algo)
	elif dict_net['special_case'] == 'timeDepChangeOfIntrFreq':
		evolve_system_on_tsim_array_time_dependent_change_of_intrinsic_frequency_save_ctrl_signal(dict_net, dict_pll, pll_list, dict_data, dict_algo)
	# run evaluations - necessary here?
	perform_evaluation(dict_net, dict_pll, dict_data)
	if 'timeDep' in dict_net['special_case'] and (dict_algo['parameter_space_sweeps'] is None or dict_algo['parameter_space_sweeps'] == 'single'):
		plot.plot_order_parameter_vs_time_dependent_parameter_div_and_undiv(dict_pll, dict_net, dict_data, dict_algo)


def perform_evaluation(dict_net: dict, dict_pll: dict, dict_data: dict) -> None:
	"""Function that

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_data  contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.

		Returns:
			nothing, operates on existing dicts
	"""
	order_parameter, order_parameter_divided_phases, F1 = eva.compute_order_parameter(dict_pll, dict_net, dict_data)
	dict_data.update({'order_parameter': order_parameter, 'order_parameter_divided_phases': order_parameter_divided_phases, 'F1': F1})

	# print('\n\ntype(order_parameter[0])', type(order_parameter[0]))
	# print('\n\ntype(order_parameter)', type(order_parameter))
	# print('\n\norder_parameter[0]', order_parameter[0])
	# sys.exit()
	# dynFreq, phaseDiff        = calculateFreqPhaseDiff(dict_data)
	# dict_data.update({'dynFreq': dynFreq, 'phaseDiff': phaseDiff})


def save_results_simulation(dict_net: dict, dict_pll: dict, dict_algo: dict, dict_data: dict) -> None:
	"""Function that saves the data contained in the dictionaries to a file.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.

		Returns:
			nothing, operates on existing dicts
	"""
	eva.saveDictionaries(dict_pll, 'dict_pll', dict_pll['coupK'], dict_pll['transmission_delay'], dict_pll['cutFc'], dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'], dict_net['topology'])
	eva.saveDictionaries(dict_net, 'dict_net', dict_pll['coupK'], dict_pll['transmission_delay'], dict_pll['cutFc'], dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'], dict_net['topology'])
	eva.saveDictionaries(dict_data, 'dict_data', dict_pll['coupK'], dict_pll['transmission_delay'], dict_pll['cutFc'], dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'], dict_net['topology'])
	eva.saveDictionaries(dict_algo, 'dict_algo', dict_pll['coupK'], dict_pll['transmission_delay'], dict_pll['cutFc'], dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'], dict_net['topology'])


def plot_results_simulation(dict_net: dict, dict_pll: dict, dict_algo: dict, dict_data: dict) -> None:
	"""Function that plots the results of the simulation.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.

		Returns:
			nothing, operates on existing dicts
	"""
	if dict_net['special_case'] != 'False':
		print('Plotting frequency vs time-dependent parameter!')
		plot.plot_instantaneous_freqs_vs_time_dependent_parameter(dict_pll, dict_net, dict_data)
		plot.plot_order_parameter_vs_time_dependent_parameter_div_and_undiv(dict_pll, dict_net, dict_data)
		plot.plot_phase_differences_vs_time_dependent_parameter_divided_or_undivided(dict_pll, dict_net, dict_data, plotlist=[], phase_diff_wrap_to_interval=2, phases_of_divided_signals=True)
		plot.plot_phase_differences_vs_time_dependent_parameter_divided_or_undivided(dict_pll, dict_net, dict_data, plotlist=[], phase_diff_wrap_to_interval=2, phases_of_divided_signals=False)
		plot.plot_inst_frequency_and_phase_difference_vs_time_dependent_parameter_divided_or_undivided(dict_pll, dict_net, dict_data, phases_of_divided_signals=True,
																										frequency_of_divided_signals=False, plotlist=[], phase_diff_wrap_to_interval=2)

	# plot.plot_phases_unwrapped(dict_pll, dict_net, dict_data)
	# plot.plot_phases_two_pi_periodic(dict_pll, dict_net, dict_data)
	# plot.plot_inst_frequency(dict_pll, dict_net, dict_data)
	# plot.plot_order_parameter(dict_pll, dict_net, dict_data)
	# plot.plot_phase_difference_wrt_to_osci_kzero(dict_pll, dict_net, dict_data)
	# plot.plot_phase_difference(dict_pll, dict_net, dict_data)
	# plot.plot_clock_time_in_period_fractions(dict_pll, dict_net, dict_data)
	# plot.plot_periodic_output_signal_from_phase(dict_pll, dict_net, dict_data)
	if dict_net['Nx']*dict_net['Ny'] == 2:
		plot.plot_inst_frequency_and_phase_difference(dict_pll, dict_net, dict_algo, dict_data)
		plot.plot_inst_frequency_and_order_parameter(dict_pll, dict_net, dict_data)
		plot.plot_power_spectral_density(dict_pll, dict_net, dict_data, dict_algo, [0, 1], saveData=False)
		plot.plot_allan_variance(dict_pll, dict_net, dict_data, 0.4 * dict_net['Tsim'], [0, 1], 'overlapping_adev', 'frequency', 0.5 * dict_net['Tsim'])
	elif dict_net['Nx']*dict_net['Ny'] == 3:
		plot.plot_order_parameter(dict_pll, dict_net, dict_data)
		plot.plot_inst_frequency_and_phase_difference(dict_pll, dict_net, dict_algo, dict_data, True, [], 2)
		plot.plot_inst_frequency_and_order_parameter(dict_pll, dict_net, dict_data, [], True)
		plot.plot_phase_relations_of_divided_signal(dict_pll, dict_net, dict_data, [], 2)
		plot.plot_control_signal_dynamics(dict_pll, dict_net, dict_data)
		if dict_net['special_case'] == 'False':
			plot.plot_power_spectral_density(dict_pll, dict_net, dict_data, dict_algo, [0, 1, 2], saveData=False)
		# try:
		# 	plot.plot_allan_variance(dict_pll, dict_net, dict_data, 0.4 * dict_net['Tsim'], [0, 1, 2], 'overlapping_adev', 'frequency', 0.5 * dict_net['Tsim'])
		# except:
		# 	print('Failed to caluclate Allan variance!')
	elif dict_net['Nx'] * dict_net['Ny'] == 4:
		plot.plot_order_parameter(dict_pll, dict_net, dict_data)
		plot.plot_inst_frequency_and_phase_difference(dict_pll, dict_net, dict_algo, dict_data, True, [], 2)
		plot.plot_inst_frequency_and_order_parameter(dict_pll, dict_net, dict_data, [], True)
		plot.plot_phase_relations_of_divided_signal(dict_pll, dict_net, dict_data, [], 2)
		plot.plot_control_signal_dynamics(dict_pll, dict_net, dict_data)
	elif dict_net['Nx']*dict_net['Ny'] == 64:
		plot.plot_inst_frequency_and_phase_difference(dict_pll, dict_net, dict_algo, dict_data)
		plot.plot_inst_frequency_and_order_parameter(dict_pll, dict_net, dict_data)
		plot.plot_power_spectral_density(dict_pll, dict_net, dict_data, dict_algo, [0, 1, 7, 28, 29, 35, 36, 56, 63], saveData=False)			# [0, 1]
		plot.plot_allan_variance(dict_pll, dict_net, dict_data, 0.4 * dict_net['Tsim'], [0, 1, 7, 28, 29, 35, 36, 56, 63], 'overlapping_adev', 'frequency', 0.5 * dict_net['Tsim'])
	elif 256 <= dict_net['Nx']*dict_net['Ny'] <= 1024:
		plot.plot_inst_frequency_and_phase_difference(dict_pll, dict_net, dict_algo, dict_data, [i for i in range(0, dict_net['Nx']*dict_net['Ny'], int(dict_net['Nx']*dict_net['Ny']/33))])
		plot.plot_inst_frequency_and_order_parameter(dict_pll, dict_net, dict_data, [i for i in range(0, dict_net['Nx']*dict_net['Ny'], int(dict_net['Nx']*dict_net['Ny']/33))])
		psd_list = [0, int(dict_net['Nx']*dict_net['Ny']/4), int(dict_net['Nx']*dict_net['Ny']/2), int(2*dict_net['Nx']*dict_net['Ny']/3), int(3*dict_net['Nx']*dict_net['Ny']/4), int(dict_net['Nx']*dict_net['Ny'])-1]
		plot.plot_power_spectral_density(dict_pll, dict_net, dict_data, dict_algo, psd_list, saveData=False)
		plot.plot_histogram(dict_pll, dict_net, dict_data, -1, 'phase-difference', 2, [], True, 15, 0.9)
		plot.plot_histogram(dict_pll, dict_net, dict_data, -1, 'phase', 2, [], True, 15, 0.9)
		plot.plot_histogram(dict_pll, dict_net, dict_data, int(dict_pll['transmission_delay'] / dict_pll['dt']), 'phase-difference', 2, [], True, 15, 0.9)
		plot.plot_histogram(dict_pll, dict_net, dict_data, int(dict_pll['transmission_delay'] / dict_pll['dt']), 'phase', 2, [], True, 15, 0.9)
		plot.plot_histogram(dict_pll, dict_net, dict_data, 0, 'phase-difference', 2, [], True, 15, 0.9)
		plot.plot_histogram(dict_pll, dict_net, dict_data, 0, 'phase', 2, [], True, 15, 0.9)
		plot.plot_histogram(dict_pll, dict_net, dict_data, -1, 'frequency', 0, [], True, 15, 0.9)
		plot.plot_histogram(dict_pll, dict_net, dict_data, int(dict_pll['transmission_delay'] / dict_pll['dt']), 'frequency', 0, [], True, 15, 0.9)
		plot.plot_histogram(dict_pll, dict_net, dict_data, 0, 'frequency', 0, [], True, 15, 0.9)
		if dict_net['special_case'] == 'False':
			plot.plot_allan_variance(dict_pll, dict_net, dict_data, 0.4 * dict_net['Tsim'], [0, 1, 7, 28, 29, 35, 36, 56, 63, int(dict_net['Nx']*dict_net['Ny']/2), dict_net['Nx']*dict_net['Ny']-1], 'overlapping_adev', 'frequency', 0.5 * dict_net['Tsim'])
	elif dict_net['Nx']*dict_net['Ny'] <= 36:
		plot.plot_inst_frequency_and_phase_difference(dict_pll, dict_net, dict_algo, dict_data)
		plot.plot_inst_frequency_and_order_parameter(dict_pll, dict_net, dict_data)
		if dict_net['special_case'] == 'False':
			plot.plot_allan_variance(dict_pll, dict_net, dict_data, 0.4 * dict_net['Tsim'], [0, 16, 35], 'overlapping_adev', 'frequency', 0.5 * dict_net['Tsim'])
		#plot.plot_power_spectral_density(dict_pll, dict_net, dict_data, dict_algo, [i for i in range(dict_net['Nx']*dict_net['Ny'])], saveData=False)
		if dict_pll['extra_coup_sig'] == 'injection2ndHarm':
			plot.plot_histogram(dict_pll, dict_net, dict_data, -1, 'phase-difference', 2, [], True, 15, 0.9)
			plot.plot_histogram(dict_pll, dict_net, dict_data, 0, 'phase-difference', 2, [], True, 15, 0.9)
	else:
		plot.plot_inst_frequency_and_phase_difference(dict_pll, dict_net, dict_algo, dict_data, [0, 1])
		plot.plot_inst_frequency_and_order_parameter(dict_pll, dict_net, dict_data, [0, 1])
		plot.plot_power_spectral_density(dict_pll, dict_net, dict_data, dict_algo, [0, 1], saveData=False)
		plot.plot_allan_variance(dict_pll, dict_net, dict_data, 0.4 * dict_net['Tsim'], [0, int(dict_net['Nx']*dict_net['Ny']/2), dict_net['Nx']*dict_net['Ny']-1], 'overlapping_adev', 'frequency', 0.5 * dict_net['Tsim'])
	plt.draw()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolve_system_on_tau_array(dict_net: dict, dict_pll: dict, pll_list: list, dict_data: dict, dict_algo: dict):
	"""Function that evolves the dynamics of a systems of oscillators in time. This implementation is the most memory efficient for the simulation of delay systems
		as only the history of each oscillator is kept. It has the option to write out the data of the entire time-series during the simulation.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list: list of PLL objects according to dict_net and dict_pll

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']
	store_phases_tau_array = dict_algo['store_phases_tau_array']

	print('Phi container only of length tau or multiple, no write-out of phases so far.')
	phi = dict_data['phi']
	phi_array_len = dict_net['phi_array_len']
	sample_every_dt = dict_pll['sampleFplot']

	if store_ctrl_and_clock and store_phases_tau_array:
		try:
			os.remove("results/phases_clock_ctrlsig")
		except OSError:
			pass
		with open("results/phases_clock_ctrlsig", "a") as file:

			clock_counter = dict_data['clock_counter']
			for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):
				if idx_time % sample_every_dt == 0:
					file.write(str(phi[idx_time % phi_array_len, :])+'\t'+str(clock_counter[idx_time % phi_array_len, :])+'\t'+str([pll.low_pass_filter.get_control_signal() for pll in pll_list])+'\n')
				# print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
				# print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
				# print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dict_pll['dt']); #time.sleep(0.5)
				phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
				clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]
				# print('clock count for all:', clock_counter[-1])
		file.close()
	elif store_phases_tau_array and not store_ctrl_and_clock:
		try:
			os.remove("results/phases")
		except OSError:
			pass
		with open("results/phases", "a") as file:

			for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):
				if idx_time % sample_every_dt == 0:
					# np.savez(file, phases=phi[idx_time%phi_array_len,:])
					file.write(str(phi[idx_time % phi_array_len, :])+'\n')

				phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] 	# now the network is iterated, starting at t=0 with the history as prepared above

		file.close()
	else:
		for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):
			phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] 			# now the network is iterated, starting at t=0 with the history as prepared above

	t = np.arange(0, len(phi[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'], 0]))*dict_pll['dt']
	if store_ctrl_and_clock:
		dict_data.update({'t': t, 'phi': phi})
		print('Full time-series of phases, clock and control-signal results saved in file phase_clock_ctrlsig in results folder!')
	else:
		dict_data.pop('clock_counter', None)
		dict_data.update({'t': t, 'phi': phi})


#############################################################################################################################################################################
def evolve_system_on_tsim_array(dict_net: dict, dict_pll: dict, pll_list: list, dict_data: dict, dict_algo: dict):
	"""Function that evolves the dynamics of a systems of oscillators in time. This implementation stores the phases of all oscillators for the entire simulation time Tsim.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list: list of PLL objects according to dict_net and dict_pll

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	print('Phi container has length of Tsim.')
	clock_counter = dict_data['clock_counter']
	phi = dict_data['phi']

	phi_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store[0:dict_net['max_delay_steps']+1, :] = phi[0:dict_net['max_delay_steps']+1, :]
	phi_array_len = dict_net['phi_array_len']

	# line = []; tlive = np.arange(0,phi_array_len-1)*dict_pll['dt']
	if store_ctrl_and_clock:
		clk_store = np.empty([dict_net['max_delay_steps'] + dict_pll['sim_time_steps'], dict_net['Nx'] * dict_net['Ny']])
		ctl_store = np.empty([dict_net['max_delay_steps'] + dict_pll['sim_time_steps'], dict_net['Nx'] * dict_net['Ny']])
		ctl_store[0:dict_net['max_delay_steps'], :] = 0
		ctl_store[dict_net['max_delay_steps']+1, :] = [pll.low_pass_filter.control_signal for pll in pll_list]
		for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

			# print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
			# print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
			# print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dict_pll['dt']); #time.sleep(0.5)
			phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] 	# now the network is iterated, starting at t=0 with the history as prepared above

			clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]
			# print('clock count for all:', clock_counter[-1])

			clk_store[idx_time+1, :] = clock_counter[(idx_time+1) % phi_array_len, :]
			phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]
			ctl_store[idx_time+1, :] = [pll.low_pass_filter.get_control_signal() for pll in pll_list]
			# phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dict_pll['dt'])
			# line = livplt.live_plotter(tlive, phidot, line)
	else:
		for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

			phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] 	# now the network is iterated, starting at t=0 with the history as prepared above
			phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]

	t = np.arange(0, len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'], 0]))*dict_pll['dt']
	if store_ctrl_and_clock:
		dict_data.update({'t': t, 'phi': phi_store, 'clock_counter': clk_store, 'ctrl': ctl_store})
	else:
		dict_data.pop('clock_counter', None)
		dict_data.update({'t': t, 'phi': phi_store})


#############################################################################################################################################################################
def evolve_system_on_tsim_array_time_dependent_change_of_coupling_strength(dict_net: dict, dict_pll: dict, pll_list: list, dict_data: dict, dict_algo: dict):
	"""Function that evolves the dynamics of a systems of oscillators in time. The coupling strength is time-dependent.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	clock_counter = dict_data['clock_counter']
	phi = dict_data['phi']

	couplingStrVal_vs_time = setup.setup_time_dependent_parameter(dict_net, dict_pll, dict_data, parameter='coupK', zero_initially=False, start_time_dependency_after_percent_of_tsim=0.1, for_all_plls_different_time_dependence=False)[0]

	# if not dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing':
	plot.plot_time_dependent_parameter(dict_pll, dict_net, dict_data, couplingStrVal_vs_time, 'K')

	t_first_pert = 450
	phi_array_len = dict_net['phi_array_len']

	clk_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store[0:dict_net['max_delay_steps']+1, :] = phi[0:dict_net['max_delay_steps']+1, :]
	#line = []; tlive = np.arange(0,phi_array_len-1)*dict_pll['dt']
	for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

		# print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
		# print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		# print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dict_pll['dt']); #time.sleep(0.5)
		phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
		# print('injectionLock:', [pll.pdc.compute(np.zeros(dict_net['Nx']*dict_net['Ny']-1), 0, np.zeros(dict_net['Nx']*dict_net['Ny']-1), idx_time) for pll in pll_list])
		[pll.signal_controlled_oscillator.evolve_coupling_strength(couplingStrVal_vs_time[idx_time], dict_net) for pll in pll_list]
		# [print('at time t=', idx_time*dict_pll['dt'] , 'K_inject2ndHarm=', couplingStrVal_vs_time[idx_time]) for pll in pll_list]
		clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]
		# print('clock count for all:', clock_counter[-1])

		clk_store[idx_time+1, :] = clock_counter[(idx_time+1) % phi_array_len, :]
		phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]
		# phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dict_pll['dt'])
		# line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0, len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'], 0]))*dict_pll['dt']
	dict_data.update({'t': t, 'phi': phi_store, 'clock_counter': clk_store, 'timeDepPara': couplingStrVal_vs_time})


#############################################################################################################################################################################
def evolve_system_on_tsim_array_time_dependent_change_of_inject_locking_coupling_strength_shil(dict_net: dict, dict_pll: dict, pll_list: list, dict_data: dict, dict_algo: dict):
	"""Function that

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	clock_counter = dict_data['clock_counter']
	phi = dict_data['phi']

	injectLockCoupStrVal_vs_time = setup.setup_time_dependent_parameter(dict_net, dict_pll, dict_data, parameter='coupStr_2ndHarm', zero_initially=False, start_time_dependency_after_percent_of_tsim=0.1, for_all_plls_different_time_dependence=False)[0]

	t = np.arange( 0, dict_net['max_delay_steps']+dict_pll['sim_time_steps'] ) * dict_pll['dt']
	# if not dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing':
	plot.plot_time_dependent_parameter(dict_pll, dict_net, dict_data, injectLockCoupStrVal_vs_time, 'K2ndHarm')

	# t_first_pert = 150
	phi_array_len = dict_net['phi_array_len']
	pll_dt = dict_pll['dt']

	clk_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store[0:dict_net['max_delay_steps']+1, :] = phi[0:dict_net['max_delay_steps']+1, :]
	#line = []; tlive = np.arange(0,phi_array_len-1)*pll_dt
	for idx_time in range(dict_net['max_delay_steps'],dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*pll_dt); #time.sleep(0.5)
		phi[(idx_time+1) % phi_array_len, :] = [pll.next_ising_shil(idx_time, phi_array_len, phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
		#print('injectionLock:', [pll.pdc.compute(np.zeros(dict_net['Nx']*dict_net['Ny']-1), 0, np.zeros(dict_net['Nx']*dict_net['Ny']-1), idx_time) for pll in pll_list])
		[pll.inject_second_harm_signal.evolve_coupling_strength_inject_lock(injectLockCoupStrVal_vs_time[idx_time], dict_net) for pll in pll_list]
		#[print('at time t=', idx_time*pll_dt , 'K_inject2ndHarm=', injectLockCoupStrVal_vs_time[idx_time]) for pll in pll_list]
		clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])
		# if idx_time * pll_dt > t_first_pert and idx_time * pll_dt < t_first_pert + 2 * pll_dt:
		# 	print('Perturbation added at t=', idx_time*pll_dt, '!')
		# 	[pll.signal_controlled_oscillator.add_perturbation(np.random.uniform(-np.pi, np.pi)) for pll in pll_list]
		# 	t_first_pert = t_first_pert + 500

		clk_store[idx_time+1, :] = clock_counter[(idx_time+1) % phi_array_len, :]
		phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*pll_dt)
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'],0]))*pll_dt
	dict_data.update({'t': t, 'phi': phi_store, 'clock_counter': clk_store, 'timeDepPara': injectLockCoupStrVal_vs_time})


#############################################################################################################################################################################
def evolve_system_on_tsim_array_time_dependent_change_of_coupling_strength_shil(dict_net: dict, dict_pll: dict, pll_list: list, dict_data: dict, dict_algo: dict):
	"""Function that evolves the dynamics of a systems of oscillators in time. The coupling strength is time-dependent and there is a second harmonic injection locking
		signal injected to each oscillator (shil). This is used when simulating Ising Machines.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	clock_counter = dict_data['clock_counter']
	phi = dict_data['phi']

	couplingStrVal_vs_time = setup.setup_time_dependent_parameter(dict_net, dict_pll, dict_data, parameter='coupK', zero_initially=False, start_time_dependency_after_percent_of_tsim=0.1, for_all_plls_different_time_dependence=False)[0]

	# if not dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing':
	try:
		plot.plot_time_dependent_parameter(dict_pll, dict_net, dict_data, couplingStrVal_vs_time, 'K')
	except:
		print('Could not print time-dependent parameter!')

	t_first_pert = 450
	phi_array_len = dict_net['phi_array_len']

	clk_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store[0:dict_net['max_delay_steps']+1, :] = phi[0:dict_net['max_delay_steps']+1, :]
	#line = []; tlive = np.arange(0,phi_array_len-1)*dict_pll['dt']
	for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

		# print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
		# print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		# print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dict_pll['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%phi_array_len, :] = [pll.next_ising_shil(idx_time, phi_array_len, phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
		# print('injectionLock:', [pll.pdc.compute(np.zeros(dict_net['Nx']*dict_net['Ny']-1), 0, np.zeros(dict_net['Nx']*dict_net['Ny']-1), idx_time) for pll in pll_list])
		[pll.signal_controlled_oscillator.evolve_coupling_strength(couplingStrVal_vs_time[idx_time], dict_net) for pll in pll_list]
		# [print('at time t=', idx_time*dict_pll['dt'] , 'K_inject2ndHarm=', couplingStrVal_vs_time[idx_time]) for pll in pll_list]
		clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]
		# print('clock count for all:', clock_counter[-1])

		clk_store[idx_time+1, :] = clock_counter[(idx_time+1) % phi_array_len, :]
		phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]
		# phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dict_pll['dt'])
		# line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0, len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'], 0]))*dict_pll['dt']
	dict_data.update({'t': t, 'phi': phi_store, 'clock_counter': clk_store, 'timeDepPara': couplingStrVal_vs_time})


#############################################################################################################################################################################
def evolve_system_on_tsim_array_time_dependent_change_of_coupling_strength_shil_explicit_shil_filtering(dict_net: dict, dict_pll: dict, pll_list: list, dict_data: dict, dict_algo: dict):
	"""Function that evolves the dynamics of a systems of oscillators in time. The coupling strength is time-dependent and there is a second harmonic injection locking
		signal injected to each oscillator (shil=SHIL). This is used when simulating Ising Machines.
		Here, the SHIL signal is generated by filtering a mixed signal with a bandpass filter.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""

	#TODO: check whether band-pass has been implemented already!

	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	clock_counter = dict_data['clock_counter']
	phi = dict_data['phi']

	couplingStrVal_vs_time = setup.setup_time_dependent_parameter(dict_net, dict_pll, dict_data, parameter='coupK', zero_initially=False, start_time_dependency_after_percent_of_tsim=0.1, for_all_plls_different_time_dependence=False)[0]

	# if not dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing':
	plot.plot_time_dependent_parameter(dict_pll, dict_net, dict_data, couplingStrVal_vs_time, 'K')

	t_first_pert = 450
	phi_array_len = dict_net['phi_array_len']

	clk_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store[0:dict_net['max_delay_steps']+1, :] = phi[0:dict_net['max_delay_steps']+1, :]
	#line = []; tlive = np.arange(0,phi_array_len-1)*dict_pll['dt']
	for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

		# print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
		# print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		# print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dict_pll['dt']); #time.sleep(0.5)
		phi[(idx_time+1) % phi_array_len, :] = [pll.next_ising_shil(idx_time, phi_array_len, phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
		# print('injectionLock:', [pll.pdc.compute(np.zeros(dict_net['Nx']*dict_net['Ny']-1), 0, np.zeros(dict_net['Nx']*dict_net['Ny']-1), idx_time) for pll in pll_list])
		[pll.signal_controlled_oscillator.evolve_coupling_strength(couplingStrVal_vs_time[idx_time], dict_net) for pll in pll_list]
		# [print('at time t=', idx_time*dict_pll['dt'] , 'K_inject2ndHarm=', couplingStrVal_vs_time[idx_time]) for pll in pll_list]
		clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]
		# print('clock count for all:', clock_counter[-1])

		clk_store[idx_time+1, :] = clock_counter[(idx_time+1) % phi_array_len, :]
		phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]
		# phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dict_pll['dt'])
		# line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'],0]))*dict_pll['dt']
	dict_data.update({'t': t, 'phi': phi_store, 'clock_counter': clk_store, 'timeDepPara': couplingStrVal_vs_time})


#############################################################################################################################################################################
def evolve_system_on_tsim_array_time_dependent_change_of_delay_save_ctrl_signal(dict_net: dict, dict_pll: dict, pll_list: list, dict_data: dict, dict_algo: dict):
	"""Function that evolves the dynamics of a systems of oscillators in time. The time delay is time-dependent and the time evolution of the control signal is saved.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	clock_counter = dict_data['clock_counter']
	phi = dict_data['phi']

	phi_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	if store_ctrl_and_clock:
		clk_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
		ctl_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
		ctl_store[0:dict_net['max_delay_steps'], :] = 0
		ctl_store[dict_net['max_delay_steps']+1, :] = [pll.low_pass_filter.control_signal for pll in pll_list]

	phi_store[0:dict_net['max_delay_steps']+1, :] = phi[0:dict_net['max_delay_steps']+1, :]
	phi_array_len = dict_net['phi_array_len']

	#line = []; tlive = np.arange(0,phi_array_len-1)*dict_pll['dt']
	if store_ctrl_and_clock:
		for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

			#print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
			#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
			#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dict_pll['dt']); #time.sleep(0.5)
			phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] 	# now the network is iterated, starting at t=0 with the history as prepared above

			clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]
			#print('clock count for all:', clock_counter[-1])

			clk_store[idx_time+1, :] = clock_counter[(idx_time+1) % phi_array_len, :]
			phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]
			ctl_store[idx_time+1, :] = [pll.low_pass_filter.get_control_signal() for pll in pll_list]
			#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dict_pll['dt'])
			#line = livplt.live_plotter(tlive, phidot, line)
	else:
		for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

			phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] 	# now the network is iterated, starting at t=0 with the history as prepared above
			phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]

	t = np.arange(0, len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'], 0]))*dict_pll['dt']
	if store_ctrl_and_clock:
		dict_data.update({'t': t, 'phi': phi_store, 'clock_counter': clk_store, 'ctrl': ctl_store})
	else:
		dict_data.pop('clock_counter', None)
		dict_data.update({'t': t, 'phi': phi_store})

#############################################################################################################################################################################
def evolve_system_on_tsim_array_time_dependent_change_of_intrinsic_frequency_save_ctrl_signal(dict_net: dict, dict_pll: dict, pll_list: list, dict_data: dict, dict_algo: dict):
	"""Function that evolves the dynamics of a systems of oscillators in time. The intrinsic frequency of one or more oscillators is time-dependent and the time evolution of the control signal is saved.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	clock_counter = dict_data['clock_counter']
	phi = dict_data['phi']

	only_change_freq_of_reference = True
	one_time_switch = True

	if only_change_freq_of_reference:
		#	print('\n\nAlso changing coupling strength here in this case!')
		# coup_strength_vs_time = setup.setup_time_dependent_parameter(dict_net, dict_pll, dict_data, parameter='coupK', zero_initially=True, start_time_dependency_after_percent_of_tsim=0.1,
		#															 for_all_plls_different_time_dependence=False)[0]

		if dict_net['topology'] == 'entrainOne-ring':
			pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2, dict_net['Nx'] * dict_net['Ny'] - 1], dict_pll=dict_pll, idx_time=dict_net['max_delay_steps']) 		# cut the first PLL of the mutually coupled PLLs off from the reference
		elif dict_net['topology'] == 'entrainOne-chain':
			pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2], dict_pll=dict_pll, idx_time=dict_net['max_delay_steps'])
		elif dict_net['topology'] == 'entrainAll-ring':
			pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2, dict_net['Nx'] * dict_net['Ny'] - 1], dict_pll=dict_pll, idx_time=dict_net['max_delay_steps'])
			pll_list[dict_net['Nx'] * dict_net['Ny'] - 1].update_list_of_neighbors_of_pll(new_neighbor_list=[dict_net['Nx'] * dict_net['Ny'] - 2, 1], dict_pll=dict_pll, idx_time=dict_net['max_delay_steps'])
			for k in range(2, dict_net['Nx'] * dict_net['Ny']-1):
				pll_list[k].update_list_of_neighbors_of_pll(new_neighbor_list=[k-1, k+1], dict_pll=dict_pll, idx_time=dict_net['max_delay_steps'])
		elif dict_net['topology'] == 'entrainAll-chain':
			pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2], dict_pll=dict_pll, idx_time=dict_net['max_delay_steps'])
			pll_list[dict_net['Nx'] * dict_net['Ny'] - 1].update_list_of_neighbors_of_pll(new_neighbor_list=[dict_net['Nx'] * dict_net['Ny'] - 2], dict_pll=dict_pll, idx_time=dict_net['max_delay_steps'])
			for k in range(2, dict_net['Nx'] * dict_net['Ny'] - 1):
				pll_list[k].update_list_of_neighbors_of_pll(new_neighbor_list=[k - 1, k + 1], dict_pll=dict_pll, idx_time=dict_net['max_delay_steps'])
		else:
			print('Please recheck the function evolve_system_on_tsim_array_time_dependent_change_of_intrinsic_frequency_save_ctrl_signal() in sim_lib.py before usage!')
			sys.exit()

		if isinstance(dict_pll['intrF'], np.int) or isinstance(dict_pll['intrF'], np.float):
			dict_pll.update({'intrF': np.zeros(dict_net['Nx'] * dict_net['Ny']) + dict_pll['intrF']})
		intr_freq_vs_time = setup.setup_time_dependent_parameter(dict_net, dict_pll, dict_data, parameter='intrF', zero_initially=False, start_time_dependency_after_percent_of_tsim=0.20,
																for_all_plls_different_time_dependence=False)[0]

	else:
		intr_freq_vs_time = setup.setup_time_dependent_parameter(dict_net, dict_pll, dict_data, parameter='intrF', zero_initially=False, start_time_dependency_after_percent_of_tsim=0.35,
																for_all_plls_different_time_dependence=False)[0]

	# if not dict_algo['parameter_space_sweeps'] == 'testNetworkMotifIsing':
	plot.plot_time_dependent_parameter(dict_pll, dict_net, dict_data, intr_freq_vs_time, r'f(t)')

	phi_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	if store_ctrl_and_clock:
		clk_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
		ctl_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
		ctl_store[0:dict_net['max_delay_steps'], :] = 0
		ctl_store[dict_net['max_delay_steps']+1, :] = [pll.low_pass_filter.control_signal for pll in pll_list]

	phi_store[0:dict_net['max_delay_steps']+1, :] = phi[0:dict_net['max_delay_steps']+1, :]
	phi_array_len = dict_net['phi_array_len']

	#line = []; tlive = np.arange(0,phi_array_len-1)*dict_pll['dt']
	if store_ctrl_and_clock:
		for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

			#print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
			#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
			#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dict_pll['dt']); #time.sleep(0.5)
			phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] 	# now the network is iterated, starting at t=0 with the history as prepared above
			clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]
			#print('clock count for all:', clock_counter[-1])
			[pll.signal_controlled_oscillator.evolve_intrinsic_freq(intr_freq_vs_time[idx_time], dict_net, only_change_freq_of_reference=only_change_freq_of_reference) for pll in pll_list]
			# print('neighbors of PLLs at time=%0.2f:' % (idx_time * dict_pll['dt']), [pll.phase_detector_combiner.idx_neighbors for pll in pll_list])
			# if not one_time_switch:
			# 	print('neighbors of PLLs at time=%0.2f:' % (idx_time * dict_pll['dt']), [pll.phase_detector_combiner.idx_neighbors for pll in pll_list])
			# 	print('intrinsic frequencies at time=%0.2f:' % (idx_time*dict_pll['dt']), [pll.signal_controlled_oscillator.intr_freq_rad for pll in pll_list])
			if idx_time == dict_data['tstep_annealing_start'] and one_time_switch:
				print('Adding back the reference to the mutually coupled network!')
				if dict_net['topology'] == 'entrainOne-ring':
					pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2, dict_net['Nx'] * dict_net['Ny'] - 1, 0], dict_pll=dict_pll, idx_time=idx_time)  #
				elif dict_net['topology'] == 'entrainOne-chain':
					pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2, 0], dict_pll=dict_pll, idx_time=idx_time)
				elif dict_net['topology'] == 'entrainAll-ring':
					pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2, dict_net['Nx'] * dict_net['Ny'] - 1, 0], dict_pll=dict_pll, idx_time=idx_time)
					pll_list[dict_net['Nx'] * dict_net['Ny'] - 1].update_list_of_neighbors_of_pll(new_neighbor_list=[dict_net['Nx'] * dict_net['Ny'] - 2, 1, 0], dict_pll=dict_pll,
																										idx_time=idx_time)
					for k in range(2, dict_net['Nx'] * dict_net['Ny'] - 1):
						pll_list[k].update_list_of_neighbors_of_pll(new_neighbor_list=[k - 1, k + 1, 0], dict_pll=dict_pll, idx_time=idx_time)
				elif dict_net['topology'] == 'entrainAll-chain':
					pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2, 0], dict_pll=dict_pll, idx_time=idx_time)
					pll_list[dict_net['Nx'] * dict_net['Ny'] - 1].update_list_of_neighbors_of_pll(new_neighbor_list=[dict_net['Nx'] * dict_net['Ny'] - 2, 0], dict_pll=dict_pll,
																										idx_time=idx_time)
					for k in range(2, dict_net['Nx'] * dict_net['Ny'] - 1):
						pll_list[k].update_list_of_neighbors_of_pll(new_neighbor_list=[k - 1, k + 1, 0], dict_pll=dict_pll, idx_time=idx_time)
				# pll_list[2].update_list_of_neighbors_of_pll(new_neighbor_list=[0, 1], dict_pll=dict_pll, idx_time=idx_time)
				one_time_switch = False 	# prevents from calling this all the time!

			clk_store[idx_time+1, :] = clock_counter[(idx_time+1) % phi_array_len, :]
			phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]
			ctl_store[idx_time+1, :] = [pll.low_pass_filter.get_control_signal() for pll in pll_list]
			#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dict_pll['dt'])
			#line = livplt.live_plotter(tlive, phidot, line)
	else:
		for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

			phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] 	# now the network is iterated, starting at t=0 with the history as prepared above
			phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]
			# print('intrinsic frequencies to be set:', intr_freq_vs_time[idx_time])
			[pll.signal_controlled_oscillator.evolve_intrinsic_freq(intr_freq_vs_time[idx_time], dict_net, only_change_freq_of_reference=only_change_freq_of_reference) for pll in pll_list]
			if idx_time == dict_data['tstep_annealing_start'] and one_time_switch:
				print('Adding back the reference to the mutually coupled network!')
				if dict_net['topology'] == 'entrainOne-ring':
					pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2, dict_net['Nx'] * dict_net['Ny'] - 1, 0], dict_pll=dict_pll, idx_time=idx_time)  #
				elif dict_net['topology'] == 'entrainOne-chain':
					pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2, 0], dict_pll=dict_pll, idx_time=idx_time)
				elif dict_net['topology'] == 'entrainAll-ring':
					pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2, dict_net['Nx'] * dict_net['Ny'] - 1, 0], dict_pll=dict_pll, idx_time=idx_time)
					pll_list[dict_net['Nx'] * dict_net['Ny'] - 1].update_list_of_neighbors_of_pll(new_neighbor_list=[dict_net['Nx'] * dict_net['Ny'] - 2, 1, 0], dict_pll=dict_pll,
																										idx_time=idx_time)
					for k in range(2, dict_net['Nx'] * dict_net['Ny'] - 1):
						pll_list[k].update_list_of_neighbors_of_pll(new_neighbor_list=[k - 1, k + 1, 0], dict_pll=dict_pll, idx_time=idx_time)
				elif dict_net['topology'] == 'entrainAll-chain':
					pll_list[1].update_list_of_neighbors_of_pll(new_neighbor_list=[2, 0], dict_pll=dict_pll, idx_time=idx_time)
					pll_list[dict_net['Nx'] * dict_net['Ny'] - 1].update_list_of_neighbors_of_pll(new_neighbor_list=[dict_net['Nx'] * dict_net['Ny'] - 2, 0], dict_pll=dict_pll,
																										idx_time=idx_time)
					for k in range(2, dict_net['Nx'] * dict_net['Ny'] - 1):
						pll_list[k].update_list_of_neighbors_of_pll(new_neighbor_list=[k - 1, k + 1, 0], dict_pll=dict_pll, idx_time=idx_time)
				one_time_switch = False  # prevents from calling this all the time!
			# print('intrinsic frequencies:', [pll.signal_controlled_oscillator.intr_freq_rad for pll in pll_list])
			# time.sleep(1)

	t = np.arange(0, len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'], 0]))*dict_pll['dt']
	if store_ctrl_and_clock:
		dict_data.update({'t': t, 'phi': phi_store, 'clock_counter': clk_store, 'ctrl': ctl_store, 'timeDepPara': intr_freq_vs_time, 'only_change_freq_of_reference': only_change_freq_of_reference})
	else:
		dict_data.pop('clock_counter', None)
		dict_data.update({'t': t, 'phi': phi_store, 'timeDepPara': intr_freq_vs_time, 'only_change_freq_of_reference': only_change_freq_of_reference})

#############################################################################################################################################################################
def distributed_pll_in_3d_mobile(dict_net, dict_pll, phi, pos, coup_matrix, clock_counter, pll_list, dict_data, dict_algo):
	"""Function that evolves the dynamics of a systems of oscillators in time. The oscillators can move in a 3d space and interact with each other according to their interaction radii.
	Pairs of oscillators that may interact will constantly have to check their past positions and relative distance to determine the time delay with which they receive the other oscillators signal.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	print('Phi container has length of Tsim.')
	# create data container
	clock_signal_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']]) # stores clock signal
	phases_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']]) # stores phases
	positions_3d_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']]) # stores positions, from this the signaling delays can be calculated
	adjacency_matrix_over_time_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']]) # stores the coupling matrix at any point in time
	# copy history to data container
	phases_store[0:dict_net['max_delay_steps']+1,:] = phi[0:dict_net['max_delay_steps']+1,:]
	positions_3d_store[0:dict_net['max_delay_steps']+1,:] = pos[0:dict_net['max_delay_steps']+1,:]
	adjacency_matrix_over_time_store[0:dict_net['max_delay_steps']+1,:] = coup_matrix[0:dict_net['max_delay_steps']+1,:]
	# obtain the length of the temporary phi array
	phi_array_len = dict_net['phi_array_len']

	for idx_time in range(dict_net['max_delay_steps'],dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1,1):

		# update the positions and store them
		positions_3d_store[(idx_time+1)%phi_array_len, :] = [pll.evolve_position_in_3d() for pll in pll_list]

		# use all current positions to calculate the current adjacency matrix and identify potential coupling pairs, i.e., oscillators which enter each others receptions zone
		# we only need to compute one side of the matrix of neighbor interactions (since it is symmetric about the main diagonal)
		adjacency_matrix_over_time_store[(idx_time+1)%phi_array_len,:] = space.update_adjacency_matrix_for_all_plls_potentially_receiving_a_signal(all_pll_positions,
																				distance_treshold, geometry_of_treshold)

		# for all identified potential signal exchanges, starting at a time t_e,kl (emission time of PLL k that may reach a neighbor l), the position and velocity vector of the
		# emitting PLL will be of interest: TODO save time t_e,kl and the position pos_k(t_e,kl), as long as k and l are within each others range, all signal emission events at
		# all time steps need to be followed and checked, this can only be stopped once two PLLs have been out of each others range long enough such that the signals cannot reach
		# the other since the have travelled their maximum distance trough space and damped such that they cannot be received anymore: TODO follow each signal emission event that
		# happens within the reception regime of another PLL until it has either arrived or decayed since travelling more than the distance treshold


		#adjacency_matrix_store -> current_adjacency_matrix[time_at_which_a_neighbor_moved_into_reception_range + time_for_signal_from_boundary_of_reception_range_to_oscillator]
		#position_3d_store -> current_transmit_delay_steps[time_at_which_a_neighbor_moved_into_reception_range + time_for_signal_from_boundary_of_reception_range_to_oscillator]

		# set the current time-delays and coupling parterns is range for all oscillators receiving signals from past or current neighbors:
		# NOTE that there is a time shift by the maximum delay until an oscillator receives the signal of an oscillator that enters his reception zone
		[pll.delayer.update_list_of_neighbors_for_delayer(current_adjacency_matrix[:, pll.pll_id].tolist()) for pll in pll_list]
		[pll.delayer.set_current_transmit_delay_steps(current_transmit_delay_steps[:, pll.pll_id].tolist()) for pll in pll_list]
		# advance the phases of all oscillators by one time-increment
		phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
		# calculate the clocks' state from the current phases
		clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]

		# write out current states to data containers
		clock_signal_store[idx_time+1, :] = clock_counter[(idx_time+1) % phi_array_len, :]
		phases_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]


	t = np.arange(0,len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'],0]))*dict_pll['dt']
	dict_data.update({'t': t, 'phi': phases_store, 'clock_counter': clock_signal_store, 'positions': position_3d_store, 'adjacency_matrix_store': adjacency_matrix_store})


#############################################################################################################################################################################
def evolve_system_interface_live_ctrl(dict_net, dict_pll, phi, clock_counter, pll_list, dict_data, dict_algo):
	"""Function that evolves the dynamics of a systems of oscillators in time. This implements an interface for real-time/in-simulation modification of parameters via a graphical input layer.

	NEEDS to be implemented!

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	phi_array_len = dict_net['phi_array_len']

	for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

		phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] 	# now the network is iterated, starting at t=0 with the history as prepared above
		clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]

	t = np.arange(0, len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'], 0]))*dict_pll['dt']
	dict_data.update({'t': t, 'phi': phi_store, 'clock_counter': clock_counter})


#############################################################################################################################################################################
def evolve_system_test_cases(dict_net: dict, dict_pll: dict, pll_list: list, dict_data: dict, dict_algo: dict):
	"""Function that evolves the dynamics of a systems of oscillators in time. This case implements test cases to experiment and extend the code.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	clock_sync_scheduled = 75
	clock_counter = dict_data['clock_counter']
	phi = dict_data['phi']
	clk_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store[0:dict_net['max_delay_steps']+1, :] = phi[0:dict_net['max_delay_steps']+1, :]
	#line = []; tlive = np.arange(0,dict_net['phi_array_len']-1)*dict_pll['dt']
	phi_array_len = dict_net['phi_array_len']

	for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

		#print('[pll.next(idx_time,dict_net['phi_array_len'],phi) for pll in pll_list]:', [pll.next(idx_time,dict_net['phi_array_len'],phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dict_pll['dt']); #time.sleep(0.5)
		phi[(idx_time+1) % phi_array_len, :] = [pll.next_antenna(idx_time, phi_array_len, phi, ext_field) for pll in pll_list] 		# now the network is iterated, starting at t=0 with the history as prepared above

		clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])

		if clock_counter[(idx_time+1) % phi_array_len][0] == clock_sync_scheduled:
			clock_sync_scheduled = -23
			print('Assume transient dynamics decayed, reset (synchronize) all clocks!')
			[pll.clock_reset(phi[(idx_time+1) % phi_array_len, pll.pll_id].copy() - 2.0 * np.pi) for pll in pll_list]

		clk_store[idx_time+1, :] = clock_counter[(idx_time+1) % phi_array_len, :]
		phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dict_pll['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0, len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'], 0]))*dict_pll['dt']
	dict_data.update({'t': t, 'phi': phi_store, 'clock_counter': clk_store})


#############################################################################################################################################################################
def evolve_system_test_perturbations(dict_net, dict_pll, phi, clock_counter, pll_list, dict_data, dict_algo):
	"""Function that evolves the dynamics of a systems of oscillators in time. In this case, the system is perturbed at specific times.

		Args:
			dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
			dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
			dict_algo: contains as parameters the information how the simulation has to be carried out and what type is run: e.g., 'single', 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations', 'single', 'statistics'
			dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
			pll_list

		Returns:
			nothing, operates on existing dicts
	"""
	store_ctrl_and_clock = dict_algo['store_ctrl_and_clock']

	clock_sync_scheduled = 75
	clk_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store = np.empty([dict_net['max_delay_steps']+dict_pll['sim_time_steps'], dict_net['Nx']*dict_net['Ny']])
	phi_store[0:dict_net['max_delay_steps']+1, :] = phi[0:dict_net['max_delay_steps']+1, :]
	phi_array_len = dict_net['phi_array_len']

	#line = []; tlive = np.arange(0,dict_net['phi_array_len']-1)*dict_pll['dt']
	for idx_time in range(dict_net['max_delay_steps'], dict_net['max_delay_steps']+dict_pll['sim_time_steps']-1, 1):

		#print('[pll.next(idx_time,dict_net['phi_array_len'],phi) for pll in pll_list]:', [pll.next(idx_time,dict_net['phi_array_len'],phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dict_pll['dt']); #time.sleep(0.5)
		phi[(idx_time+1) % phi_array_len, :] = [pll.next(idx_time, phi_array_len, phi) for pll in pll_list] 		# now the network is iterated, starting at t=0 with the history as prepared above

		clock_counter[(idx_time+1) % phi_array_len, :] = [pll.clock_halfperiods_count(phi[(idx_time+1) % phi_array_len, pll.pll_id]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])

		if clock_counter[(idx_time+1) % phi_array_len][0] == clock_sync_scheduled:
			clock_sync_scheduled = -23
			print('Assume transient dynamics decayed, reset (synchronize) all clocks!')
			[pll.clock_reset(phi[(idx_time+1) % phi_array_len,pll.pll_id].copy() - 2.0 * np.pi) for pll in pll_list]

		if clock_counter[(idx_time+1) % phi_array_len][0] % 750 == 0:
			tempMon = [pll.low_pass_filter.get_control_signal() for pll in pll_list]
			print('Monitor control signal and then add perturbation. xc(%0.2f)~' %(clock_counter[(idx_time+1) % phi_array_len][0]/(2*dict_pll['syncF'])), tempMon)
			phi[(idx_time+1) % phi_array_len, :] = [pll.signal_controlled_oscillator.add_perturbation((-0.5) ** pll.pll_id) for pll in pll_list]
			clock_sync_scheduled = clock_counter[(idx_time+1)%phi_array_len][0] + 50

		clk_store[idx_time+1, :] = clock_counter[(idx_time+1) % phi_array_len, :]
		phi_store[idx_time+1, :] = phi[(idx_time+1) % phi_array_len, :]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dict_pll['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0, len(phi_store[0:dict_net['max_delay_steps']+dict_pll['sim_time_steps'], 0]))*dict_pll['dt']
	dict_data.update({'t': t, 'phi': phi_store, 'clock_counter': clk_store})
