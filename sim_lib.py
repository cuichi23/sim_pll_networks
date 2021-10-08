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

import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import pickle

import setup
import evaluation_lib as eva
import plot_lib as plot

now = datetime.datetime.now()

''' Enable automatic carbage collector '''
gc.enable()

''' SIMULATE NETWORK '''
def simulateSystem(dictNet, dictPLL, dictAlgo=None):
#mode,div,Nplls,F,F_Omeg,K,Fc,delay,feedback_delay,dt,c,Nsteps,topology,couplingfct,histtype,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx=0,Ny=0,Trelax=0,Kadiab_value_r=0):

	t0 = time.time()

	np.random.seed()															# restart pseudo random-number generator
	dictData = {}																# setup dictionary that hold all the data
	if not dictNet['phiInitConfig']:											# if no custom phase configuration is provided, generate it
		print('\nPhase configuration of synchronized state set according to supplied topology and twist state information!')
		dictNet			= setup.generatePhi0(dictNet)							# generate the initial phase configuration for twist, chequerboard, in- and anti-phase states
	pll_list			= setup.generatePLLs(dictPLL, dictNet, dictData)		# generate a list that contains all the PLLs of the network
	all_transmit_delay 	= [np.max(n.delayer.transmit_delay_steps) for n in pll_list] 	# obtain all the transmission delays for all PLLs
	all_feedback_delay 	= [n.delayer.feedback_delay_steps for n in pll_list] 	# obtain all the feedback delays for all PLLs
	max_feedback_delay 	= np.max(all_feedback_delay)
	max_transmit_delay 	= np.max(np.array([np.max(i) for i in all_transmit_delay]))
	print('\nMaximum cross-coupling time delay: max_transmit_delay_steps:', max_transmit_delay, '\tmax_feedback_delay_steps:', max_feedback_delay)
	max_delay_steps		= np.max([max_transmit_delay, max_feedback_delay])		# pick largest time delay to setup container for phases
	# prepare container for the phases
	if max_delay_steps == 0:
		print('No delay case, not yet tested, see sim_lib.py! Setting container length to that of Tsim/dt!'); #sys.exit()
		phi_array_len = dictPLL['sim_time_steps']								# length of phi contrainer in case the delay is zero; the +int(dictPLL['orderLF']) is necessary to cover filter up to order dictPLL['orderLF']
	else:
		phi_array_len = 1+int(dictNet['phi_array_mult_tau'])*max_delay_steps	# length of phi contrainer, must be at least 1 delay length if delay > 0
	phi 		  = np.zeros([phi_array_len, dictNet['Nx']*dictNet['Ny']])		# prepare container for phase time series of all PLLs
	clock_counter = np.empty([phi_array_len, dictNet['Nx']*dictNet['Ny']])		# list for counter, i.e., the clock derived from the PLL
	for i in range(len(pll_list)):												# set max_delay in delayer: CHECK AGAIN!
		pll_list[i].delayer.phi_array_len = phi_array_len

	dictNet.update({'max_delay_steps': max_delay_steps, 'phi_array_len': phi_array_len})

	## TESTS!
	#print('Test coupling %s(pi)='%dictPLL['coup_fct_sig'], dictPLL['coup_fct_sig'](1.0*np.pi)); sys.exit()

	# set initial phase configuration and history -- performed such that any configuration can be obtained when simulations starts
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SET INITIAL HISTORY AND PERTURBATION '''
	# start by setting last entries to initial phase configuration, i.e., phiInitConfig + phiPerturb
	print('The perturbation will be set to dictNet[*phiPerturb*]=', dictNet['phiPerturb'], ', and dictNet[*phiInitConfig*]=', dictNet['phiInitConfig'])

	if not ( isinstance(dictNet['phiPerturb'], list) or isinstance(dictNet['phiPerturb'], np.ndarray) ):
		print('All initial perturbations set to zero as none were supplied!')
		dictNet['phiPerturb']				= [0 for i in range(len(phi[0,:]))]
		phi[dictNet['max_delay_steps'],:] 	= list( map(add, dictNet['phiInitConfig'], dictNet['phiPerturb']) ) # set phase-configuration phiInitConfig at t=0 + the perturbation phiPerturb
	elif ( len(phi[0,:]) == len(dictNet['phiInitConfig']) and len(phi[0,:]) == len(dictNet['phiPerturb']) ):
		phi[dictNet['max_delay_steps'],:] 	= list( map(add, dictNet['phiInitConfig'], dictNet['phiPerturb']) ) # set phase-configuration phiInitConfig at t=0 + the perturbation phiPerturb
	else:
		print('len(phi[0,:])', len(phi[0,:]), '\t len(dictNet[*phiInitConfig*])', len(dictNet['phiInitConfig']))
		print('Provide initial phase-configuration of length %i to setup simulation!' %len(phi[dictNet['max_delay_steps'],:])); sys.exit()

	## TESTS!
	# plt.plot(phi[:,0], 'o'); plt.plot(phi[:,1], 'd'); plt.draw(); plt.show();

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SET THE INTERNAL PHI VARS OF THE VCO TO THEIR INITIAL VALUE '''
	for i in range(len(pll_list)):												# set initial phases at time equivalent to the time-delay, then setup the history from there
		pll_list[i].signal_controlled_oscillator.phi = phi[max_delay_steps, i]
	# print('VCOs internal phis are set to:', [pll.vco.phi for pll in pll_list]); sys.exit()

	# if uncoupled history, just evolve backwards in time until the beginning of the phi container is reached
	if dictPLL['typeOfHist'] == 'freeRunning':									# in the 'uncoupled' case the oscillators evolve to the perturbed state during the history
		for i in range(max_delay_steps+1,0,-1):
			#print('i-1',i-1)
			phi[i-1,:] = [pll.setup_hist_reverse() for pll in pll_list]
	elif dictPLL['typeOfHist'] == 'syncState':									# in the 'syncstate' case the oscillators evolve as if synced and then receive a delta perturbation
		# since we want a delta perturbation, the perturbation is removed towards the prior step
		phi[max_delay_steps-1,:] = list( map(sub, [pll.setup_hist_reverse() for pll in pll_list], dictNet['phiPerturb']) ) # local container to help the setup

		for i in range(len(pll_list)):
			pll_list[i].signal_controlled_oscillator.phi = phi[max_delay_steps - 1, i]						# set this step as initial for reverse history setup
		# print('VCOs internal phis are set to:', [pll.vco.phi for pll in pll_list]); sys.exit()
		for i in range(max_delay_steps-1,0,-1):
			#print('i-1',i-1)
			phi[i-1,:] = [pll.setup_hist_reverse() for pll in pll_list]
		for i in range(len(pll_list)):
			pll_list[i].signal_controlled_oscillator.phi = phi[max_delay_steps, i]
	else:
		print('Specify the type of history, syncState or freeRunning supported!'); sys.exit()

	## TESTS!
	#plt.plot(phi[:,0], 'o'); plt.plot(phi[:,1], 'd'); plt.draw(); plt.show();

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SHIFT ALL PHASES UP SUCH THAT AT t=-tau ALL ARE ABOVE ZERO '''
	phi[0:max_delay_steps+1,:] = phi[0:max_delay_steps+1,:] - np.min(phi[0,:])	# shift up/down all phases by the smallest phase of any PLL

	t = np.arange(0,len(phi[:,0]))*dictPLL['dt']
	params={'x': t, 'y': phi, 'label': 'phi', 'xlabel': 't', 'ylabel': 'phi', 'delay_steps': max_delay_steps, 'len_phi': phi_array_len-1, 'dt': dictPLL['dt']}
	#eva.plotTest(params)
	for i in range(len(pll_list)):												# update all internal VCO phi variables
		pll_list[i].signal_controlled_oscillator.phi = phi[max_delay_steps, i]

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SET INITIAL CONTROL SIGNAL, ACCORDING AND CONSISTENT TO HISTORY WRITTEN, CORRECT INTERNAL PHASES OF VCO AND CLOCK '''
	for i in range(len(pll_list)):
		if max_delay_steps >= int(dictPLL['orderLF']):
			#print('last freqs:', ( phi[max_delay_steps-0,i]-phi[max_delay_steps-1,i]-dictNet['phiPerturb'][i] ) / (2.0*np.pi*dictPLL['dt']), ( phi[max_delay_steps-1,i]-phi[max_delay_steps-2,i] ) / (2.0*np.pi*dictPLL['dt']));
			pll_list[i].low_pass_filter.set_initial_control_signal((phi[max_delay_steps - 0, i] - phi[max_delay_steps - 1, i] - dictNet['phiPerturb'][i]) / (2.0 * np.pi * dictPLL['dt']),
																   ( phi[max_delay_steps-1,i]-phi[max_delay_steps-2,i] ) / (2.0*np.pi*dictPLL['dt']))
		# NOTE: very important, the delta-like phase perturbation needs to be accounted for when calculating the instantaneous frequency of the last time step before simulation,
 		#		here first argument of lf.set_initial_control_signal()

		elif max_delay_steps < int(dictPLL['orderLF']) and ( int(dictPLL['orderLF']) == 2 or int(dictPLL['orderLF']) == 1 ):
			if dictPLL['typeOfHist'] == 'freeRunning':							# set the frequencyies in the past to determine the LF filter state for no delay
				inst_freq_lastStep			= dictPLL['intrF'] + np.random.normal(loc=0.0, scale=np.sqrt( dictPLL['noiseVarVCO'] * dictPLL['dt'] ))
				inst_freq_prior_to_lastStep = dictPLL['intrF'] + np.random.normal(loc=0.0, scale=np.sqrt( dictPLL['noiseVarVCO'] * dictPLL['dt'] ))
			elif dictPLL['typeOfHist'] == 'syncState':
				inst_freq_lastStep			= dictPLL['syncF'] + np.random.normal(loc=0.0, scale=np.sqrt( dictPLL['noiseVarVCO'] * dictPLL['dt'] ))
				inst_freq_prior_to_lastStep = dictPLL['syncF'] + np.random.normal(loc=0.0, scale=np.sqrt( dictPLL['noiseVarVCO'] * dictPLL['dt'] ))
			pll_list[i].low_pass_filter.set_initial_control_signal(inst_freq_lastStep, inst_freq_prior_to_lastStep)
		else:
			print('in simPLL.lib: Higher order LFs are net yet supported!')
		# print('Set internal initial VCO phi at t-dt for PLL %i:'%i, phi[max_delay_steps,i])
		#pll_list[i].vco.phi = phi[max_delay_steps,i]
		pll_list[i].counter.reset(phi[max_delay_steps, i])

	## TESTS!
	# print('dictNet[*max_delay_steps*]', dictNet['max_delay_steps'], '\tmax_delay_steps:', max_delay_steps, '\tlen(phi): ', len(phi))
	# print('history: ', phi[0:dictNet['max_delay_steps'],:])
	# plt.plot(phi[:,0], 'o'); plt.plot(phi[:,1], 'd'); plt.draw(); plt.show();
	#sys.exit()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' NOW SIMULATE THE SYSTEM AFTER HISTORY IS SET '''
	dictData.update({'clock_counter': clock_counter, 'phi': phi, 'all_transmit_delay': all_transmit_delay, 'all_feedback_delay': all_feedback_delay})

	if dictPLL['sim_time_steps']*dictPLL['dt'] <= 1E6 and dictNet['phi_array_mult_tau'] == 1 and dictNet['special_case'] == 'False': # container to flush data
		dictData = evolveSystemOnTsimArray(dictNet, dictPLL, phi, clock_counter, pll_list, dictData)
	elif dictPLL['sim_time_steps']*dictPLL['dt'] > 1E6 and dictNet['phi_array_mult_tau'] == 1 and dictNet['special_case'] == 'False':
		dictData = evolveSystemOnTauArray(dictNet, dictPLL, phi, clock_counter, pll_list, dictData)
	elif dictNet['special_case'] == 'test_case':
		print('Simulating testcase scenario!'); time.sleep(2)
		dictData = evolveSystemTestCases(dictNet, dictPLL, phi, clock_counter, pll_list, dictData)
	elif dictNet['special_case'] == 'timeDepTransmissionDelay':
		dictData = evolveSystemOnTsimArray_varDelaySaveCtrlSig(dictNet, dictPLL, phi, clock_counter, pll_list, dictData)
		plot.plotCtrlSigDny(dictPLL, dictNet, dictData)
	elif dictNet['special_case'] == 'timeDepInjectLockCoupStr':
		dictData = evolveSystemOnTsimArray_varInjectLockCoupStrength(dictNet, dictPLL, phi, clock_counter, pll_list, dictData)
	elif dictNet['special_case'] == 'timeDepChangeOfCoupStr':
		dictData = evolveSystemOnTsimArray_timeDepChangeOfCoupStrength(dictNet, dictPLL, phi, clock_counter, pll_list, dictData)
		plot.plotOrderPvsTimeDepPara(dictPLL, dictNet, dictData)

	print('Time needed for execution of simulation: ', (time.time()-t0), ' seconds')

	eva.saveDictionaries(dictPLL, 'dictPLL',   dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts
	eva.saveDictionaries(dictNet, 'dictNet',   dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts
	eva.saveDictionaries(dictData, 'dictData', dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts
	eva.saveDictionaries(dictAlgo, 'dictAlgo', dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts

	plot.plotPhasesInf(dictPLL, dictNet, dictData)
	plot.plotPhases2pi(dictPLL, dictNet, dictData)
	plot.plotFrequency(dictPLL, dictNet, dictData)
	plot.plotOrderPara(dictPLL, dictNet, dictData)
	plot.plotPhaseRela(dictPLL, dictNet, dictData)
	plot.plotPhaseDiff(dictPLL, dictNet, dictData)
	plot.plotClockTime(dictPLL, dictNet, dictData)
	plot.plotOscSignal(dictPLL, dictNet, dictData)
	plot.plotFreqAndPhaseDiff(dictPLL, dictNet, dictData)
	plot.plotPSD(dictPLL, dictNet, dictData, [], saveData=False)

	print('Time needed for simulation, evaluation and plotting: ', (time.time()-t0), ' seconds')

	plt.draw()
	plt.show()

	return dictNet, dictPLL, dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemOnTauArray(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	print('Phi container only of length tau or multiple, no write-out so far of phases.')
	phi_array_len = dictNet['phi_array_len']
	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%phi_array_len,:] = [pll.next(idx_time,phi_array_len,phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above

		#clock_counter[(idx_time+1)%phi_array_len,:] = [pll.clock_halfperiods_count(phi[(idx_time+1)%phi_array_len,pll.idx_self]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])
	phiStore = phi
	clkStore = clock_counter

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore})

	return dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemOnTsimArray(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	print('Phi container has length of Tsim.')
	clkStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore[0:dictNet['max_delay_steps']+1,:] = phi[0:dictNet['max_delay_steps']+1,:]
	phi_array_len = dictNet['phi_array_len']

	## TESTS!
	# plt.plot(phiStore[:,0], 'o'); plt.plot(phiStore[:,1], 'd'); plt.title('1'); plt.xlim([0, 40]); plt.draw(); plt.show();
	# plt.plot(phi[:,0], 'o'); plt.plot(phi[:,1], 'd'); plt.title('2'); plt.xlim([0, 40]); plt.draw(); plt.show();
	# print('phiStore[0:31,:]', phiStore[0:31,:])

	#line = []; tlive = np.arange(0,phi_array_len-1)*dictPLL['dt']
	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%phi_array_len,:] = [pll.next(idx_time,phi_array_len,phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above

		clock_counter[(idx_time+1)%phi_array_len,:] = [pll.clock_halfperiods_count(phi[(idx_time+1)%phi_array_len,pll.pll_id]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])

		clkStore[idx_time+1,:] = clock_counter[(idx_time+1)%phi_array_len,:]
		phiStore[idx_time+1,:] = phi[(idx_time+1)%phi_array_len,:]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dictPLL['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore})

	return dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemOnTsimArray_timeDepChangeOfCoupStrength(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	couplingStrVal_vs_time = setup.setupTimeDependentParameter(dictNet, dictPLL, dictData, parameter='coupK', afterTsimPercent=0.2, forAllPLLsDifferent=False)[0]

	plt.figure(1234);
	t = np.arange( 0, dictNet['max_delay_steps']+dictPLL['sim_time_steps'] ) * dictPLL['dt']
	plt.plot(t, couplingStrVal_vs_time)
	plt.draw(); plt.show()

	t_first_pert = 450;
	phi_array_len = dictNet['phi_array_len']

	clkStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore[0:dictNet['max_delay_steps']+1,:] = phi[0:dictNet['max_delay_steps']+1,:]
	#line = []; tlive = np.arange(0,phi_array_len-1)*dictPLL['dt']
	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%phi_array_len,:] = [pll.next(idx_time,phi_array_len,phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
		#print('injectionLock:', [pll.pdc.compute(np.zeros(dictNet['Nx']*dictNet['Ny']-1), 0, np.zeros(dictNet['Nx']*dictNet['Ny']-1), idx_time) for pll in pll_list])
		[pll.signal_controlled_oscillator.evolveCouplingStrength(couplingStrVal_vs_time[idx_time], dictNet) for pll in pll_list]
		#[print('at time t=', idx_time*dictPLL['dt'] , 'K_inject2ndHarm=', couplingStrVal_vs_time[idx_time]) for pll in pll_list]
		clock_counter[(idx_time+1)%phi_array_len,:] = [pll.clock_halfperiods_count(phi[(idx_time+1)%phi_array_len,pll.pll_id]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])

		clkStore[idx_time+1,:] = clock_counter[(idx_time+1)%phi_array_len,:]
		phiStore[idx_time+1,:] = phi[(idx_time+1)%phi_array_len,:]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dictPLL['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore, 'timeDepPara': couplingStrVal_vs_time})

	return dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemOnTsimArray_varInjectLockCoupStrength(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	injectLockCoupStrVal_vs_time = setup.setupTimeDependentParameter(dictNet, dictPLL, dictData, parameter='coupStr_2ndHarm', afterTsimPercent=0.0, forAllPLLsDifferent=False)[0]
	phi_array_len = dictNet['phi_array_len']

	plt.figure(1234);
	t = np.arange( 0, dictNet['max_delay_steps']+dictPLL['sim_time_steps'] ) * dictPLL['dt']
	plt.plot(t, injectLockCoupStrVal_vs_time)
	plt.draw(); plt.show()

	t_first_pert = 450;
	phi_array_len = dictNet['phi_array_len']

	clkStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore[0:dictNet['max_delay_steps']+1,:] = phi[0:dictNet['max_delay_steps']+1,:]
	#line = []; tlive = np.arange(0,dictNet['phi_array_len']-1)*dictPLL['dt']
	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%phi_array_len,:] = [pll.next(idx_time,phi_array_len,phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
		#print('injectionLock:', [pll.pdc.compute(np.zeros(dictNet['Nx']*dictNet['Ny']-1), 0, np.zeros(dictNet['Nx']*dictNet['Ny']-1), idx_time) for pll in pll_list])
		[pll.phase_detector_combiner.evolve_coupling_strength_inject_lock(injectLockCoupStrVal_vs_time[idx_time], dictNet) for pll in pll_list]
		#[print('at time t=', idx_time*dictPLL['dt'] , 'K_inject2ndHarm=', injectLockCoupStrVal_vs_time[idx_time]) for pll in pll_list]
		clock_counter[(idx_time+1)%phi_array_len,:] = [pll.clock_halfperiods_count(phi[(idx_time+1)%phi_array_len,pll.pll_id]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])
		if idx_time*dictPLL['dt'] > t_first_pert and idx_time*dictPLL['dt'] < t_first_pert+2*dictPLL['dt']:
			print('Perturbation added at t=', idx_time*dictPLL['dt'], '!')
			[pll.signal_controlled_oscillator.add_perturbation(np.random.uniform(-np.pi, np.pi)) for pll in pll_list]
			t_first_pert = t_first_pert + 500;

		clkStore[idx_time+1,:] = clock_counter[(idx_time+1)%phi_array_len,:]
		phiStore[idx_time+1,:] = phi[(idx_time+1)%phi_array_len,:]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dictPLL['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore, 'timeDepPara': injectLockCoupStrVal_vs_time})

	return dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemOnTsimArray_varDelaySaveCtrlSig(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	clkStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	ctlStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore[0:dictNet['max_delay_steps']+1,:] = phi[0:dictNet['max_delay_steps']+1,:]
	ctlStore[0:dictNet['max_delay_steps'],:] = 0; ctlStore[dictNet['max_delay_steps']+1,:] = [pll.low_pass_filter.y for pll in pll_list];
	#line = []; tlive = np.arange(0,dictNet['phi_array_len']-1)*dictPLL['dt']
	phi_array_len = dictNet['phi_array_len']

	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time,phi_array_len,phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%phi_array_len,:] = [pll.next(idx_time,phi_array_len,phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above

		clock_counter[(idx_time+1)%phi_array_len,:] = [pll.clock_halfperiods_count(phi[(idx_time+1)%phi_array_len,pll.pll_id]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])

		clkStore[idx_time+1,:] = clock_counter[(idx_time+1)%phi_array_len,:]
		phiStore[idx_time+1,:] = phi[(idx_time+1)%phi_array_len,:]
		ctlStore[idx_time+1,:] = [pll.low_pass_filter.monitor_ctrl() for pll in pll_list]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dictPLL['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore, 'ctrl': ctlStore})

	return dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemInterface(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	phi_array_len = dictNet['phi_array_len']

	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		phi[(idx_time+1)%phi_array_len,:] = [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
		clock_counter[(idx_time+1)%phi_array_len,:] = [pll.clock_halfperiods_count(phi[(idx_time+1)%phi_array_len,pll.pll_id]) for pll in pll_list]

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clock_counter})

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemTestCases(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	clock_sync_scheduled = 75
	clkStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore[0:dictNet['max_delay_steps']+1,:] = phi[0:dictNet['max_delay_steps']+1,:]
	#line = []; tlive = np.arange(0,dictNet['phi_array_len']-1)*dictPLL['dt']
	phi_array_len = dictNet['phi_array_len']

	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list]:', [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%phi_array_len,:] = [pll.next_antenna(idx_time,dictNet['phi_array_len'],phi,ext_field) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above

		clock_counter[(idx_time+1)%phi_array_len,:] = [pll.clock_halfperiods_count(phi[(idx_time+1)%phi_array_len,pll.pll_id]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])

		if clock_counter[(idx_time+1)%phi_array_len][0] == clock_sync_scheduled:
			clock_sync_scheduled = -23
			print('Assume transient dynamics decayed, reset (synchronize) all clocks!')
			[pll.clock_reset(phi[(idx_time+1)%phi_array_len,pll.pll_id].copy() - 2.0 * np.pi) for pll in pll_list]

		clkStore[idx_time+1,:] = clock_counter[(idx_time+1)%phi_array_len,:]
		phiStore[idx_time+1,:] = phi[(idx_time+1)%phi_array_len,:]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dictPLL['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore})

	return dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemTestPerturbations(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	clock_sync_scheduled = 75
	clkStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore[0:dictNet['max_delay_steps']+1,:] = phi[0:dictNet['max_delay_steps']+1,:]
	phi_array_len = dictNet['phi_array_len']

	#line = []; tlive = np.arange(0,dictNet['phi_array_len']-1)*dictPLL['dt']
	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list]:', [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%phi_array_len,:] = [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above

		clock_counter[(idx_time+1)%phi_array_len,:] = [pll.clock_halfperiods_count(phi[(idx_time+1)%phi_array_len,pll.pll_id]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])

		if clock_counter[(idx_time+1)%phi_array_len][0] == clock_sync_scheduled:
			clock_sync_scheduled = -23
			print('Assume transient dynamics decayed, reset (synchronize) all clocks!')
			[pll.clock_reset(phi[(idx_time+1)%phi_array_len,pll.pll_id].copy() - 2.0 * np.pi) for pll in pll_list]

		if clock_counter[(idx_time+1)%phi_array_len][0]%750 == 0:
			tempMon = [pll.low_pass_filter.monitor_ctrl() for pll in pll_list]
			print('Monitor control signal and then add perturbation. xc(%0.2f)~' %(clock_counter[(idx_time+1)%phi_array_len][0]/(2*dictPLL['syncF'])), tempMon)
			phi[(idx_time+1)%phi_array_len,:] = [pll.signal_controlled_oscillator.add_perturbation((-0.5) ** pll.pll_id) for pll in pll_list]
			clock_sync_scheduled = clock_counter[(idx_time+1)%phi_array_len][0] + 50

		clkStore[idx_time+1,:] = clock_counter[(idx_time+1)%phi_array_len,:]
		phiStore[idx_time+1,:] = phi[(idx_time+1)%phi_array_len,:]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dictPLL['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore})

	return dictData



# ''' NOW SIMULATE THE SYSTEM AFTER HISTORY IS SET '''
# if dictPLL['sim_time_steps']*dictPLL['dt'] < 1E6 and dictNet['phi_array_mult_tau'] == 1: # container to flush data
#
# 	phiStore = np.empty([max_delay_steps+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
# 	phiStore[0:max_delay_steps+1,:] = phi[0:max_delay_steps+1,:]
# 	#line = []; tlive = np.arange(0,phi_array_len-1)*dictPLL['dt']
# 	for idx_time in range(max_delay_steps,max_delay_steps+dictPLL['sim_time_steps']-1,1):
# 		#print('[pll.next(idx_time%phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time%phi_array_len,phi) for pll in pll_list])
# 		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
# 		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dictPLL['dt']); #time.sleep(0.5)
# 		phi[(idx_time+1)%phi_array_len,:] = [pll.next(idx_time%phi_array_len,phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
# 		clock_counter[(idx_time+1)%phi_array_len] = [pll.clock_periods_count(phi[(idx_time+1)%phi_array_len,pll.idx_self]) for pll in pll_list]
# 		#print('clock count for all:', clock_counter[-1])
# 		phiStore[idx_time+1,:] = phi[(idx_time+1)%phi_array_len,:]
# 		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dictPLL['dt'])
# 		#line = livplt.live_plotter(tlive, phidot, line)
# else:
# 	print('Phi container only of length tau or multiple, no write-out so far of phases.')
# 	for idx_time in range(max_delay_steps,max_delay_steps+dictPLL['sim_time_steps']-1,1):
# 		#print('[pll.next(idx_time%phi_array_len,phi) for pll in pll_list]:', [pll.next(idx_time%phi_array_len,phi) for pll in pll_list])
# 		#print('Current state: phi[(idx_time)%phi_array_len,:]', phi[(idx_time)%phi_array_len,:], '\t(idx_time)%phi_array_len',(idx_time)%phi_array_len); sys.exit()
# 		#print('(idx_time+1)%phi_array_len', ((idx_time+1)%phi_array_len)*dictPLL['dt']); #time.sleep(0.5)
# 		phi[(idx_time+1)%phi_array_len,:] = [pll.next(idx_time%phi_array_len,phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
# 		clock_counter[(idx_time+1)%phi_array_len] = [pll.clock_periods_count(phi[(idx_time+1)%phi_array_len,pll.idx_self]) for pll in pll_list]
# 		#print('clock count for all:', clock_counter[-1])
# 	phiStore = phi
