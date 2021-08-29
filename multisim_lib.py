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

from multiprocessing import Pool, freeze_support
import itertools

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
gc.enable();

def distributeProcesses(dictNet, dictPLL, dictAlgo=None):

	t0 = time.time()
	if dictAlgo['bruteForceBasinStabMethod'] == 'classicBruteForceMethodRotatedSpace' or dictAlgo['bruteForceBasinStabMethod'] == 'testNetworkMotifIsing': # classic approach with LP-adaptation developed with J. Asmus, D. Platz
		scanValues, allPoints = setup.allInitPhaseCombinations(dictPLL, dictNet, dictAlgo, paramDiscretization=dictAlgo['paramDiscretization']) # set paramDiscretization for the number of points to be simulated
		print('allPoints:', [allPoints], '\nscanValues', scanValues); Nsim = allPoints.shape[0]; print('multiprocessing', Nsim, 'realizations')

	elif dictAlgo['bruteForceBasinStabMethod'] == 'listOfInitialPhaseConfigurations':	# so far for N=2, work it out for N>2
		if isinstance(dictPLL['intrF'], np.float) or isinstance(dictPLL['intrF'], np.int):
			scanValues = np.linspace(-np.pi, np.pi, dictAlgo['paramDiscretization'])
			print('scanValues', scanValues); Nsim = len(scanValues); print('multiprocessing', Nsim, 'realizations')
		elif isinstance(dictPLL['intrF'], np.ndarray) or isinstance(dictPLL['intrF'], list):
			scanValues, allPoints = setup.allInitPhaseCombinations(dictPLL, dictNet, dictAlgo, paramDiscretization=dictAlgo['paramDiscretization'])
			print('allPoints:', [allPoints], '\nscanValues', scanValues); Nsim = allPoints.shape[0]; print('multiprocessing', Nsim, 'realizations')
		else:
			print('2 modes: iterate for no detuning over phase-differences, or detuning and phase-differences! Choose one.'); sys.exit()

	global number_period_dyn;
	number_period_dyn 	= 20.5;
	initPhiPrime0		= 0;

	np.random.seed()
	poolData = [];																# should this be recasted to be an np.array?
	freeze_support()
	pool = Pool(processes=7)													# create a Pool object, pick number of processes

	#def multihelper(phiSr, initPhiPrime0, dictNet, dictPLL, phi, clock_counter, pll_list):
	if dictAlgo['bruteForceBasinStabMethod'] == 'classicBruteForceMethodRotatedSpace':
		poolData.append( pool.map(multihelper_star, zip( 							# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
						itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dictNet), itertools.repeat(dictPLL), itertools.repeat(dictAlgo) ) ) )
	elif dictAlgo['bruteForceBasinStabMethod'] == 'listOfInitialPhaseConfigurations' or dictAlgo['bruteForceBasinStabMethod'] == 'testNetworkMotifIsing':
		if isinstance(dictPLL['intrF'], np.float) or isinstance(dictPLL['intrF'], np.int):
			poolData.append( pool.map(multihelper_star, zip( 							# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
							itertools.product(scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dictNet), itertools.repeat(dictPLL), itertools.repeat(dictAlgo) ) ) )
		elif isinstance(dictPLL['intrF'], np.ndarray) or isinstance(dictPLL['intrF'], list):
			poolData.append( pool.map(multihelper_star, zip( 							# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
							itertools.product(scanValues[0], scanValues[1]), itertools.repeat(initPhiPrime0), itertools.repeat(dictNet), itertools.repeat(dictPLL), itertools.repeat(dictAlgo) ) ) )


	print('time needed for execution of simulations in multiproc mode: ', (time.time()-t0), ' seconds'); #sys.exit()

	eva.saveDictionaries(poolData, 'poolData', dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts

	if dictAlgo['bruteForceBasinStabMethod']   == 'testNetworkMotifIsing':
		eva.evaluateSimulationIsing(poolData)
	elif dictAlgo['bruteForceBasinStabMethod'] == 'listOfInitialPhaseConfigurations':
		eva.evaluateSimulationsChrisHoyer(poolData)
	elif dictAlgo['bruteForceBasinStabMethod'] == 'classicBruteForceMethodRotatedSpace':
		print('Implement evaluation as in the old version! Copy plots, etc...'); sys.exit()

	return poolData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

''' SIMULATE NETWORK '''
def simulateSystem(dictNet, dictPLL, dictAlgo=None):
#mode,div,Nplls,F,F_Omeg,K,Fc,delay,feedback_delay,dt,c,Nsteps,topology,couplingfct,histtype,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx=0,Ny=0,Trelax=0,Kadiab_value_r=0):

	np.random.seed()															# restart pseudo random-number generator
	dictData = {}																# setup dictionary that hold all the data
	if not dictNet['phiInitConfig']:											# if no custom phase configuration is provided, generate it
		print('Phase configuation of synchronized state set according to supplied topology and twist state information!')
		dictNet			= setup.generatePhi0(dictNet)							# generate the initial phase configuration for twist, chequerboard, in- and anti-phase states
	pll_list			= setup.generatePLLs(dictPLL, dictNet, dictData)		# generate a list that contains all the PLLs of the network
	all_transmit_delay 	= [np.max(n.delayer.transmit_delay_steps) for n in pll_list] 	# obtain all the transmission delays for all PLLs
	all_feedback_delay 	= [n.delayer.feedback_delay_steps for n in pll_list] 	# obtain all the feedback delays for all PLLs
	max_feedback_delay 	= np.max(all_feedback_delay)
	max_transmit_delay 	= np.max(np.array([np.max(i) for i in all_transmit_delay]))
	print('max_transmit_delay_steps:', max_transmit_delay, '\tmax_feedback_delay_steps:', max_feedback_delay)
	max_delay_steps		= np.max([max_transmit_delay, max_feedback_delay])		# pick largest time delay to setup container for phases
	# prepare container for the phases
	if max_delay_steps == 0:
		print('No delay case, not yet tested, see sim_lib.py! Setting container length to that of Tsim/dt!'); #sys.exit()
		phi_array_len = dictPLL['sim_time_steps']								# length of phi contrainer in case the delay is zero; the +int(dictPLL['orderLF']) is necessary to cover filter up to order dictPLL['orderLF']
	else:
		phi_array_len = 1+int(dictNet['phi_array_mult_tau'])*max_delay_steps	# length of phi contrainer, must be at least 1 delay length if delay > 0
	phi 		  = np.empty([phi_array_len, dictNet['Nx']*dictNet['Ny']])		# prepare container for phase time series of all PLLs and with length tau_max
	clock_counter = np.empty([phi_array_len, dictNet['Nx']*dictNet['Ny']])		# list for counter, i.e., the clock derived from the PLL
	for i in range(len(pll_list)):												# set max_delay in delayer
		pll_list[i].delayer.phi_array_len = phi_array_len

	dictNet.update({'max_delay_steps': max_delay_steps, 'phi_array_len': phi_array_len})
	# set initial phase configuration and history -- performed such that any configuration can be obtained when simulations starts
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SET INITIAL HISTORY AND PERTURBATION '''
	# start by setting last entries to initial phase configuration, i.e., phiInitConfig + phiPerturb
	if not dictNet['phiPerturb']:
		print('All initial perturbations set to zero as none were supplied!')
		dictNet['phiPerturb']  = [0 for i in range(len(phi[0,:]))]				# updates the phiPerturb list in dictNet
		phi[max_delay_steps,:] = list( map(add, dictNet['phiInitConfig'], dictNet['phiPerturb']) ) # set phase-configuration phiInitConfig at t=0 + the perturbation phiPerturb
	elif ( len(phi[0,:]) == len(dictNet['phiInitConfig']) and len(phi[0,:]) == len(dictNet['phiPerturb']) ):
		phi[max_delay_steps,:] = list( map(add, dictNet['phiInitConfig'], dictNet['phiPerturb']) ) # set phase-configuration phiInitConfig at t=0 + the perturbation phiPerturb
			# print('SET INITIAL HISTORY AND PERTURBATION: phi[max_delay_steps+int(dictPLL[*orderLF*]),:]=', phi[max_delay_steps+int(dictPLL['orderLF']),:])
	else:
		print('Provide initial phase-configuration of length %i to setup simulation!' %len(phi[max_delay_steps,:])); sys.exit()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SET THE INTERNAL PHI VARS OF THE VCO TO THEIR INITIAL VALUE '''
	for i in range(len(pll_list)):
		pll_list[i].vco.phi = phi[dictNet['max_delay_steps'],i]
	#print('VCOs internal phis are set to:', [pll.vco.phi for pll in pll_list])

	# if uncoupled history, just evolve backwards in time until the beginning of the phi container is reached
	if dictPLL['typeOfHist'] == 'freeRunning':									# in the 'uncoupled' case the oscillators evolve to the perturbed state during the history
		for i in range(dictNet['max_delay_steps']+1,0,-1):
			#print('i-1',i-1)
			phi[i-1,:] = [pll.setup_hist_reverse() for pll in pll_list]
	elif dictPLL['typeOfHist'] == 'syncState':									# in the 'syncstate' case the oscillators evolve as if synced and then receive a delta perturbation
		phi[dictNet['max_delay_steps']-1,:] = list( map(sub, [pll.setup_hist_reverse() for pll in pll_list], dictNet['phiPerturb']) )  # since we want a delta perturbation, the perturbation is removed towards the prior step
		for i in range(len(pll_list)):
			pll_list[i].vco.phi = phi[dictNet['max_delay_steps']-1,i]						# set this step as initial for reverse history setup
		for i in range(dictNet['max_delay_steps']-1,0,-1):
			#print('i-1',i-1)
			phi[i-1,:] = [pll.setup_hist_reverse() for pll in pll_list]
	else:
		print('Specify the type of history, so far syncState or freeRunning supported!'); sys.exit()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SHIFT ALL PHASES UP SUCH THAT AT t=-tau ALL ARE ABOVE ZERO '''
	phi[0:dictNet['max_delay_steps']+1,:] = phi[0:dictNet['max_delay_steps']+1,:] - np.min(phi[0,:])	# shift up/down all phases by the smallest phase of any PLL

	t = np.arange(0,len(phi[:,0]))*dictPLL['dt']
	params={'x': t, 'y': phi, 'label': 'phi', 'xlabel': 't', 'ylabel': 'phi', 'delay_steps': dictNet['max_delay_steps'], 'len_phi': phi_array_len-1, 'dt': dictPLL['dt']}
	#eva.plotTest(params)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SET INITIAL CONTROL SIGNAL, ACCORDING AND CONSISTENT TO HISTORY WRITTEN, CORRECT INTERNAL PHASES OF VCO AND CLOCK '''
	for i in range(len(pll_list)):
		if max_delay_steps >= int(dictPLL['orderLF']):
			pll_list[i].lf.set_initial_control_signal( ( phi[max_delay_steps-0,i]-phi[max_delay_steps-1,i] ) / (2.0*np.pi*dictPLL['dt']),
													   ( phi[max_delay_steps-1,i]-phi[max_delay_steps-2,i] ) / (2.0*np.pi*dictPLL['dt']) )
		elif max_delay_steps < int(dictPLL['orderLF']) and ( int(dictPLL['orderLF']) == 2 or int(dictPLL['orderLF']) == 1 ):
			if dictPLL['typeOfHist'] == 'freeRunning':							# set the frequencyies in the past to determine the LF filter state for no delay
				inst_freq_lastStep			= dictPLL['intrF'] + np.random.normal(loc=0.0, scale=np.sqrt( dictPLL['noiseVarVCO'] * dictPLL['dt'] ))
				inst_freq_prior_to_lastStep = dictPLL['intrF'] + np.random.normal(loc=0.0, scale=np.sqrt( dictPLL['noiseVarVCO'] * dictPLL['dt'] ))
			elif dictPLL['typeOfHist'] == 'syncState':
				inst_freq_lastStep			 = dictPLL['syncF'] + np.random.normal(loc=0.0, scale=np.sqrt( dictPLL['noiseVarVCO'] * dictPLL['dt'] ))
				inst_freq_prior_to_lastStep = dictPLL['syncF'] + np.random.normal(loc=0.0, scale=np.sqrt( dictPLL['noiseVarVCO'] * dictPLL['dt'] ))
			pll_list[i].lf.set_initial_control_signal( inst_freq_lastStep, inst_freq_prior_to_lastStep )
		else:
			print('in simPLL.lib: Higher order LFs are net yet supported!')
		# print('Set internal initial VCO phi at t-dt for PLL %i:'%i, phi[max_delay_steps,i])
		pll_list[i].vco.phi = phi[max_delay_steps,i]
		pll_list[i].counter.phase_init = phi[max_delay_steps,i]

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' NOW SIMULATE THE SYSTEM AFTER HISTORY IS SET '''
	dictData.update({'clock_counter': clock_counter, 'phi': phi, 'all_transmit_delay': all_transmit_delay, 'all_feedback_delay': all_feedback_delay})

	if dictPLL['sim_time_steps']*dictPLL['dt'] <= 1E6 and dictNet['phi_array_mult_tau'] == 1 and dictNet['special_case'] == 'False': # container to flush data
		dictData = evolveSystemOnTsimArray(dictNet, dictPLL, phi, clock_counter, pll_list, dictData)
	elif dictPLL['sim_time_steps']*dictPLL['dt'] > 1E6 and dictNet['phi_array_mult_tau'] == 1 and dictNet['special_case'] == 'False':
		dictData = evolveSystemOnTauArray(dictNet, dictPLL, phi, clock_counter, pll_list, dictData)
	#elif dictNet['special_case'] == 'test_case':
	#	print('Simulating testcase scenario!'); time.sleep(2)
	#	dictData = evolveSystemTestCases(dictNet, dictPLL, phi, clock_counter, pll_list, dictData)
	#elif dictNet['special_case'] == 'timeDepTransmissionDelay':
	#	dictData = evolveSystemOnTsimArray_varDelaySaveCtrlSig(dictNet, dictPLL, phi, clock_counter, pll_list, dictData)
	#	plot.plotCtrlSigDny(dictPLL, dictNet, dictData)
	elif dictNet['special_case'] == 'timeDepInjectLockCoupStr':
		dictData = evolveSystemOnTsimArray_varInjectLockCoupStrength(dictNet, dictPLL, phi, clock_counter, pll_list, dictData, dictAlgo)

	# run evaluations
	r, orderParam, F1 	= eva.obtainOrderParam(dictPLL, dictNet, dictData)
	dictData.update({'orderParam': orderParam, 'R': r})
	#dynFreq, phaseDiff	= calculateFreqPhaseDiff(dictData)
	#dictData.update({'dynFreq': dynFreq, 'phaseDiff': phaseDiff})

	# plot.plotPhasesInf(dictPLL, dictNet, dictData)
	# plot.plotFrequency(dictPLL, dictNet, dictData)
	# plot.plotOrderPara(dictPLL, dictNet, dictData)
	# plot.plotPhaseDiff(dictPLL, dictNet, dictData)

	plt.draw()
	#plt.show()

	realizationDict = {'dictNet': dictNet, 'dictPLL': dictPLL, 'dictData': dictData}

	return realizationDict

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemOnTauArray(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	print('Phi container only of length tau or multiple, no write-out so far of phases.')
	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list]:', [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%dictNet['phi_array_len'],:]', phi[(idx_time)%dictNet['phi_array_len'],:], '\t(idx_time)%dictNet['phi_array_len']',(idx_time)%dictNet['phi_array_len']); sys.exit()
		#print('(idx_time+1)%dictNet['phi_array_len']', ((idx_time+1)%dictNet['phi_array_len'])*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%dictNet['phi_array_len'],:] = [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above

		#clock_counter[(idx_time+1)%dictNet['phi_array_len'],:] = [pll.clock_halfperiods_count(idx_time%dictNet['phi_array_len'],phi[(idx_time+1)%dictNet['phi_array_len'],pll.idx_self]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])
	phiStore = phi
	clkStore = clock_counter

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore})

	return dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemOnTsimArray(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	clkStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore[0:dictNet['max_delay_steps']+1,:] = phi[0:dictNet['max_delay_steps']+1,:]
	#line = []; tlive = np.arange(0,dictNet['phi_array_len']-1)*dictPLL['dt']
	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list]:', [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%dictNet['phi_array_len'],:]', phi[(idx_time)%dictNet['phi_array_len'],:], '\t(idx_time)%dictNet['phi_array_len']',(idx_time)%dictNet['phi_array_len']); sys.exit()
		#print('(idx_time+1)%dictNet['phi_array_len']', ((idx_time+1)%dictNet['phi_array_len'])*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%dictNet['phi_array_len'],:] = [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above

		clock_counter[(idx_time+1)%dictNet['phi_array_len'],:] = [pll.clock_halfperiods_count(idx_time%dictNet['phi_array_len'],phi[(idx_time+1)%dictNet['phi_array_len'],pll.idx_self]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])

		clkStore[idx_time+1,:] = clock_counter[(idx_time+1)%dictNet['phi_array_len'],:]
		phiStore[idx_time+1,:] = phi[(idx_time+1)%dictNet['phi_array_len'],:]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dictPLL['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore})

	return dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemOnTsimArray_varInjectLockCoupStrength(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	injectLockCoupStrVal_vs_time = setup.setupTimeDependentParameter(dictNet, dictPLL, dictData, parameter='coupStr_2ndHarm', afterTsimPercent=0.0, forAllPLLsDifferent=False)[0]

	t = np.arange( 0, dictNet['max_delay_steps']+dictPLL['sim_time_steps'] ) * dictPLL['dt']
	if not dictAlgo['bruteForceBasinStabMethod'] == 'testNetworkMotifIsing':
		plt.figure(1234);
		plt.plot(t, injectLockCoupStrVal_vs_time)
		plt.draw(); plt.show()

	t_first_pert = 150;

	clkStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore[0:dictNet['max_delay_steps']+1,:] = phi[0:dictNet['max_delay_steps']+1,:]
	#line = []; tlive = np.arange(0,dictNet['phi_array_len']-1)*dictPLL['dt']
	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list]:', [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%dictNet['phi_array_len'],:]', phi[(idx_time)%dictNet['phi_array_len'],:], '\t(idx_time)%dictNet['phi_array_len']',(idx_time)%dictNet['phi_array_len']); sys.exit()
		#print('(idx_time+1)%dictNet['phi_array_len']', ((idx_time+1)%dictNet['phi_array_len'])*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%dictNet['phi_array_len'],:] = [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above
		#print('injectionLock:', [pll.pdc.compute(np.zeros(dictNet['Nx']*dictNet['Ny']-1), 0, np.zeros(dictNet['Nx']*dictNet['Ny']-1), idx_time) for pll in pll_list])
		[pll.pdc.evolveCouplingStrengthInjectLock(injectLockCoupStrVal_vs_time[idx_time],dictNet) for pll in pll_list]
		#[print('at time t=', idx_time*dictPLL['dt'] , 'K_inject2ndHarm=', injectLockCoupStrVal_vs_time[idx_time]) for pll in pll_list]
		clock_counter[(idx_time+1)%dictNet['phi_array_len'],:] = [pll.clock_halfperiods_count(idx_time%dictNet['phi_array_len'],phi[(idx_time+1)%dictNet['phi_array_len'],pll.idx_self]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])
		if idx_time*dictPLL['dt'] > t_first_pert and idx_time*dictPLL['dt'] < t_first_pert+2*dictPLL['dt']:
			print('Perturbation added at t=', idx_time*dictPLL['dt'], '!')
			[pll.vco.add_perturbation( np.random.uniform(-np.pi, np.pi) ) for pll in pll_list]
			t_first_pert = t_first_pert + 500;

		clkStore[idx_time+1,:] = clock_counter[(idx_time+1)%dictNet['phi_array_len'],:]
		phiStore[idx_time+1,:] = phi[(idx_time+1)%dictNet['phi_array_len'],:]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dictPLL['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore})

	return dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evolveSystemOnTsimArray_varDelaySaveCtrlSig(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	clkStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	ctlStore = np.empty([dictNet['max_delay_steps']+dictPLL['sim_time_steps'], dictNet['Nx']*dictNet['Ny']])
	phiStore[0:dictNet['max_delay_steps']+1,:] = phi[0:dictNet['max_delay_steps']+1,:]
	ctlStore[0:dictNet['max_delay_steps'],:] = 0; ctlStore[dictNet['max_delay_steps']+1,:] = [pll.lf.y for pll in pll_list];
	#line = []; tlive = np.arange(0,dictNet['phi_array_len']-1)*dictPLL['dt']
	for idx_time in range(dictNet['max_delay_steps'],dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1,1):

		#print('[pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list]:', [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list])
		#print('Current state: phi[(idx_time)%dictNet['phi_array_len'],:]', phi[(idx_time)%dictNet['phi_array_len'],:], '\t(idx_time)%dictNet['phi_array_len']',(idx_time)%dictNet['phi_array_len']); sys.exit()
		#print('(idx_time+1)%dictNet['phi_array_len']', ((idx_time+1)%dictNet['phi_array_len'])*dictPLL['dt']); #time.sleep(0.5)
		phi[(idx_time+1)%dictNet['phi_array_len'],:] = [pll.next(idx_time,dictNet['phi_array_len'],phi) for pll in pll_list] # now the network is iterated, starting at t=0 with the history as prepared above

		clock_counter[(idx_time+1)%dictNet['phi_array_len'],:] = [pll.clock_halfperiods_count(idx_time%dictNet['phi_array_len'],phi[(idx_time+1)%dictNet['phi_array_len'],pll.idx_self]) for pll in pll_list]
		#print('clock count for all:', clock_counter[-1])

		clkStore[idx_time+1,:] = clock_counter[(idx_time+1)%dictNet['phi_array_len'],:]
		phiStore[idx_time+1,:] = phi[(idx_time+1)%dictNet['phi_array_len'],:]
		ctlStore[idx_time+1,:] = [pll.lf.monitor_ctrl() for pll in pll_list]
		#phidot = (phi[1:,0]-phi[:-1,0])/(2*np.pi*dictPLL['dt'])
		#line = livplt.live_plotter(tlive, phidot, line)

	t = np.arange(0,len(phiStore[0:dictNet['max_delay_steps']+dictPLL['sim_time_steps'],0]))*dictPLL['dt']
	dictData.update({'t': t, 'phi': phiStore, 'clock': clkStore, 'ctrl': ctlStore})

	return dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def multihelper(iterConfig, initPhiPrime0, dictNet, dictPLL, dictAlgo):

	if dictAlgo['bruteForceBasinStabMethod'] == 'classicBruteForceMethodRotatedSpace':	# classic approach with LP-adaptation developed with J. Asmus, D. Platz
		phiSr = list(iterConfig)
		global number_period_dyn; number_period_dyn = 20.5;
		if dictNet['Nx']*dictNet['Ny'] > 2:
			phiSr = np.insert(phiSr, 0, initPhiPrime0)								# insert the first variable in the rotated space, constant initPhiPrime0
		phiS = eva.rotate_phases(phiSr, isInverse=False)							# rotate back into physical phase space
		# print('TEST in multihelper, phiS:', phiS, ' and phiSr:', phiSr)
		unit_cell = eva.PhaseDifferenceCell(dictNet['Nx']*dictNet['Ny'])
		# SO anpassen, dass auch gegen verschobene Einheitszelle geprueft werden kann (e.g. if not k==0...)
		# ODER reicht schon:
		# if not unit_cell.is_inside(( phiS ), isRotated=False):   ???
		# if not unit_cell.is_inside((phiS-phiM), isRotated=False):					# and not N == 2:	# +phiM

		dictNetRea = dictNet.copy()
		dictNetRea.update({'phiPerturb': phiS}) # 'phiPerturbRot': phiSr})
		#print('dictNet[*phiPerturb*]', dictNet['phiPerturb'])

		#print('Check whether perturbation is inside unit-cell! phiS:', dictNetRea['phiPerturb'], '\tInside? True/False:', unit_cell.is_inside((dictNetRea['phiPerturb']), isRotated=False)); time.sleep(2)
		if not unit_cell.is_inside((dictNetRea['phiPerturb']), isRotated=False):	# NOTE this case is for scanValues set only in -pi to pi
			print('Set dummy solution! Detected case outside of unit-cell.')
			dictData = {'mean_order': -1., 'last_orderP': -1., 'stdev_orderP': np.zeros(1), 'phases': dictNet['phiInitConfig'],
					 		'intrinfreq': np.zeros(1), 'coupling_strength': np.zeros(1), 'transdelays': dictPLL['transmission_delay'], 'orderP_t': np.zeros(int(number_period_dyn/(dictPLL['intrF']*dictPLL['dt'])))-1.0}
			realizationDict = {'dictNet': dictNetRea, 'dictPLL': dictPLL, 'dictData': dictData}
			return realizationDict
		else:
			return simulateSystem(dictNetRea, dictPLL, dictAlgo)

	elif dictAlgo['bruteForceBasinStabMethod'] == 'listOfInitialPhaseConfigurations':					 # so far for N=2, work it out for N>2
		temp   =  list(iterConfig)
		#print('iterConfig:', iterConfig, '\ttemp[0]:', temp[0], '\ttemp[1]:', temp[1])
		dictPLLRea = dictPLL.copy()
		if isinstance(dictPLL['intrF'], list):
			meanIntF = np.mean(dictPLL['intrF'])
			dictPLLRea.update({'intrF': [meanIntF-temp[1], meanIntF+temp[1]]})
			#print('Intrinsic frequencies:', dictPLLRea['intrF'], '\tfor detuning', 2*temp[1]); time.sleep(2)

		config = [0, temp[0]]
		dictNetRea = dictNet.copy()
		dictNetRea.update({'phiInitConfig': config, 'phiPerturb': np.zeros(dictNet['Nx']*dictNet['Ny']), 'phiPerturbRot': np.zeros(dictNet['Nx']*dictNet['Ny'])})

		return simulateSystem(dictNetRea, dictPLLRea, dictAlgo)

	elif dictAlgo['bruteForceBasinStabMethod'] == 'testNetworkMotifIsing':
		temp   =  list(iterConfig)
		#print('iterConfig:', iterConfig, '\ttemp[0]:', temp[0])
		config = [0]
		[config.append(entry) for entry in temp[0]]
		dictNetRea = dictNet.copy()
		dictNetRea.update({'phiInitConfig': config, 'phiPerturb': np.zeros(dictNet['Nx']*dictNet['Ny']), 'phiPerturbRot': np.zeros(dictNet['Nx']*dictNet['Ny'])})
		print('dictNetRea[*phiInitConfig*]:', dictNetRea['phiInitConfig'])

		return simulateSystem(dictNetRea, dictPLL, dictAlgo)

	else:
		print('No case fullfilled in multihelper in multisim_lib!'); sys.exit()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def multihelper_star(dynparam_fixparam):
	return multihelper(*dynparam_fixparam)
