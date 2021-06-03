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
gc.enable();

''' SIMULATE NETWORK '''
def simulateSystem(dictNet, dictPLL, dictAlgo=None):
#mode,div,Nplls,F,F_Omeg,K,Fc,delay,feedback_delay,dt,c,Nsteps,topology,couplingfct,histtype,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx=0,Ny=0,Trelax=0,Kadiab_value_r=0):

	t0 = time.time()

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
		phi_array_len = dictPLL['sim_time_steps']								# length of phi contrainer in case the delay is zero
	else:
		phi_array_len = 1+int(dictNet['phi_array_mult_tau'])*max_delay_steps	# length of phi contrainer, must be at least 1 delay length if delay > 0
	phi 		  = np.empty([phi_array_len, dictNet['Nx']*dictNet['Ny']])		# prepare container for phase time series of all PLLs and with length tau_max
	clock_counter = np.empty([phi_array_len, dictNet['Nx']*dictNet['Ny']])		# list for counter, i.e., the clock derived from the PLL
	for i in range(len(pll_list)):												# set max_delay in delayer
		pll_list[i].delayer.phi_array_len = phi_array_len

	dictNet.update({'max_delay_steps': max_delay_steps, 'phi_array_len': phi_array_len})

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	scanValues, allPoints = setup.allInitPhaseCombinations(dictPLL, dictNet, paramDiscretization=10)
	global number_period_dyn; number_period_dyn = 20.5;	initPhiPrime0 = 0;

	Nsim = allPoints.shape[0]; print('multiprocessing', Nsim, 'realizations')
	pool_data=[];																# should this be recated to be an np.array?
	freeze_support()
	pool = Pool(processes=7)													# create a Pool object

	#def multihelper(phiSr, initPhiPrime0, dictNet, dictPLL, phi, clock_counter, pll_list):
	pool_data.append( pool.map(multihelper_star, zip( 							# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
						itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dictNet), itertools.repeat(dictPLL), itertools.repeat(phi),
						itertools.repeat(clock_counter), itertools.repeat(pll_list) ) ) )

	results=[]; results_t=[]; phi=[]; omega_0=[]; K_0=[]; delays_0=[];
	for i in range(Nsim):
		''' evaluate dictionaries '''
		results.append( [ pool_data[0][i]['mean_order'],  pool_data[0][i]['last_orderP'], pool_data[0][i]['stdev_orderP'] ] )
		#results = np.concatenate(results, pool_data[0][i]['orderP_t'] )
		results_t.append( [ pool_data[0][i]['orderP_t'] ] )
		#print('type(results_t)', type(results_t), ' len(results_t)', len(results_t), ' len(results_t[0][0])', len(results_t[0][0]))
		# phi.append( pool_data[0][i]['phases'] )
		omega_0.append( pool_data[0][i]['intrinfreq'] )
		K_0.append( pool_data[0][i]['coupling_strength'] )
		delays_0.append( pool_data[0][i]['transdelays'] )
		# cPD_t.append( pool_data[0][i]['cPD'] )

	now = datetime.datetime.now()
	# print('np.shape(results_t)', np.shape(results_t), 'np.shape(np.stack(results_t))', np.shape(np.stack(results_t)))
	''' SAVE RESULTS '''
	np.savez('results/orderparam_K%.2f_Fc%.2f_FOm%.2f_div%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, div, delay, now.year, now.month, now.day), results=results)
	np.savez('results/orderparam_t_K%.2f_Fc%.2f_FOm%.2f_div%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, div, delay, now.year, now.month, now.day), results=results_t)
	np.savez('results/allInitPerturbPoints_K%.2f_Fc%.2f_FOm%.2f_div%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, div, delay, now.year, now.month, now.day), allPoints=allPoints)
	del pool_data; del _allPoints;											# emtpy pool data, allPoints variables to free memory

	print( 'size {omega_0, K_0, delays_0, results}:', sys.getsizeof(omega_0), '\t', sys.getsizeof(K_0), '\t', sys.getsizeof(delays_0), '\t', sys.getsizeof(results), '\n' )
	omega_0=np.array(omega_0); K_0=np.array(K_0); results=np.array(results);
	# np.savez('results/phases_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), phi=phi) # save phases of trajectories
	# phi=np.array(phi);
	# delays_0=np.array(delays_0);

	#print( list( pool.map(multihelper_star, itertools.izip( 				# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
	#					itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(topology), itertools.repeat(F), itertools.repeat(Nsteps), itertools.repeat(dt),
	#					itertools.repeat(c),itertools.repeat(Fc), itertools.repeat(F_Omeg), itertools.repeat(K), itertools.repeat(N), itertools.repeat(k), itertools.repeat(delay), itertools.repeat(feedback_delay),
	#					itertools.repeat(phiM), itertools.repeat(plot_Phases_Freq) ) ) ) )
	#print('results:', results)
	print('time needed for execution of simulations in multiproc mode: ', (time.time()-t0), ' seconds')

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prepareSimRealization(init_phi, initPhiPrime0, dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None):

	# set initial phase configuration and history -- performed such that any configuration can be obtained when simulations starts
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	''' SET INITIAL HISTORY AND PERTURBATION '''
	# start by setting last entries to initial phase configuration, i.e., phiInitConfig + phiPerturb
	if not dictNet['phiPerturb']:
		print('All initial perturbations set to zero as none were supplied!')
		dictNet['phiPerturb']		= [0 for i in range(len(phi[0,:]))]
		phi[dictNet['max_delay_steps'],:] = list( map(add, dictNet['phiInitConfig'], dictNet['phiPerturb']) ) # set phase-configuration phiInitConfig at t=0 + the perturbation phiPerturb
	elif ( len(phi[0,:]) == len(dictNet['phiInitConfig']) and len(phi[0,:]) == len(dictNet['phiPerturb']) ):
		phi[dictNet['max_delay_steps'],:] = list( map(add, dictNet['phiInitConfig'], dictNet['phiPerturb']) ) # set phase-configuration phiInitConfig at t=0 + the perturbation phiPerturb
	else:
		print('Provide initial phase-configuration of length %i to setup simulation!' %len(phi[dictNet['max_delay_steps'],:])); sys.exit()

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
		pll_list[i].lf.set_initial_control_signal( ( phi[dictNet['max_delay_steps']-0,i]-phi[dictNet['max_delay_steps']-1,i] ) / (2.0*np.pi*dictPLL['dt']) )
		# print('Set internal initial VCO phi at t-dt for PLL %i:'%i, phi[dictNet['max_delay_steps'],i])
		pll_list[i].vco.phi = phi[dictNet['max_delay_steps'],i]
		pll_list[i].counter.phase_init = phi[dictNet['max_delay_steps'],i]

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

	# eva.saveDictionaries(dictPLL, 'dictPLL',   dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts
	# eva.saveDictionaries(dictNet, 'dictNet',   dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts
	# eva.saveDictionaries(dictData, 'dictData', dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts

	# plot.plotPhasesInf(dictPLL, dictNet, dictData)
	# plot.plotPhases2pi(dictPLL, dictNet, dictData)
	# plot.plotFrequency(dictPLL, dictNet, dictData)
	# plot.plotOrderPara(dictPLL, dictNet, dictData)
	# plot.plotPhaseRela(dictPLL, dictNet, dictData)
	# plot.plotPhaseDiff(dictPLL, dictNet, dictData)
	# plot.plotClockTime(dictPLL, dictNet, dictData)
	# plot.plotOscSignal(dictPLL, dictNet, dictData)
	#plot.plotPSD(dictPLL, dictNet, dictData, [], saveData=False)

	# plt.draw()
	# plt.show()

	return dictNet, dictPLL, phi, clock_counter, pll_list, dictData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

def multihelper(phiSr, initPhiPrime0, dictNet, dictPLL, phi, clock_counter, pll_list):

	global number_period_dyn; number_period_dyn = 20.5;
	if N > 2:
		phiSr = np.insert(phiSr, 0, initPhiPrime0)												# insert the first variable in the rotated space, constant initPhiPrime0
	phiS = eva.rotate_phases(phiSr, isInverse=False)											# rotate back into physical phase space
	# print('TEST in multihelper, phiS:', phiS, ' and phiSr:', phiSr)
	unit_cell = eva.PhaseDifferenceCell(N)
	# SO anpassen, dass auch gegen verschobene Einheitszelle geprueft werden kann (e.g. if not k==0...)
	# ODER reicht schon:
	# if not unit_cell.is_inside(( phiS ), isRotated=False):   ???
	# if not unit_cell.is_inside((phiS-phiM), isRotated=False):					# and not N == 2:	# +phiM
	if not unit_cell.is_inside((phiS), isRotated=False):						# NOTE this case is for scanValues set only in -pi to pi
		return {'mean_order': -1., 'last_orderP': -1., 'stdev_orderP': np.zeros(1), 'phases': dictNet['phiInitConfig'],
		 		'intrinfreq': np.zeros(1), 'coupling_strength': np.zeros(1), 'transdelays': dictPLL['transmission_delay'], 'orderP_t': np.zeros(int(number_period_dyn/(dictPLL['F']*dictPLL['dt'])))-1.0}
	else:
		np.random.seed()
		return evolveSystemOnTauArray(dictNet, dictPLL, phi, clock_counter, pll_list, dictData=None, dictAlgo=None)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def multihelper_star(dynparam_fixparam):
	return multihelper(*dynparam_fixparam)
