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

from sim_pll.sim_lib import evolveSystemOnTsimArray_varInjectLockCoupStrength, evolveSystemOnTsimArray, evolveSystemOnTauArray, \
	simulateSystem
from sim_pll import setup
from sim_pll import evaluation_lib as eva
from sim_pll import plot_lib as plot

now = datetime.datetime.now()

''' Enable automatic carbage collector '''
gc.enable()

def distributeProcesses(dictNet, dictPLL, dictAlgo=None):

	t0 = time.time()
	if dictAlgo['bruteForceBasinStabMethod'] == 'classicBruteForceMethodRotatedSpace' or dictAlgo['bruteForceBasinStabMethod'] == 'testNetworkMotifIsing': # classic approach with LP-adaptation developed with J. Asmus, D. Platz
		scanValues, allPoints = setup.allInitPhaseCombinations(dictPLL, dictNet, dictAlgo, paramDiscretization=dictAlgo['paramDiscretization']) # set paramDiscretization for the number of points to be simulated
		print('allPoints:', [allPoints], '\nscanValues', scanValues); Nsim = allPoints.shape[0]; print('multiprocessing', Nsim, 'realizations')

	elif dictAlgo['bruteForceBasinStabMethod'] == 'listOfInitialPhaseConfigurations':	# so far for N=2, work it out for N>2
		if ( isinstance(dictAlgo['paramDiscretization'], np.float) or isinstance(dictAlgo['paramDiscretization'], np.int) ) and ( isinstance(dictAlgo['min_max_range_parameter'], np.float) or isinstance(dictAlgo['min_max_range_parameter'], np.int) ):
			scanValues = np.linspace(-np.pi, np.pi, dictAlgo['paramDiscretization'])
			print('scanValues', scanValues); Nsim = len(scanValues); print('multiprocessing', Nsim, 'realizations')
		elif isinstance(dictAlgo['min_max_range_parameter'], np.ndarray) or isinstance(dictAlgo['min_max_range_parameter'], list):
			scanValues, allPoints = setup.allInitPhaseCombinations(dictPLL, dictNet, dictAlgo, paramDiscretization=dictAlgo['paramDiscretization'])
			print('allPoints:', [allPoints], '\nscanValues', scanValues); Nsim = allPoints.shape[0]; print('multiprocessing', Nsim, 'realizations')
		else:
			print('2 modes: iterate for no detuning over phase-differences, or detuning and phase-differences! Choose one. HINT: if dictAlgo[*paramDiscretization*] is an instance of list or ndarray, there needs to be a list of intrinsic frequencies!'); sys.exit()
	elif dictAlgo['bruteForceBasinStabMethod'] == 'single':
		if dictAlgo['param_id'] == 'None':
			dictAlgo.update({'min_max_range_parameter': [1, 1], 'paramDiscretization': [1, 1]})
			print('No parameter to be changed, simulate only one realization!')
		scanValues, allPoints = setup.allInitPhaseCombinations(dictPLL, dictNet, dictAlgo, paramDiscretization=dictAlgo['paramDiscretization'])
		print('allPoints:', [allPoints], '\nscanValues', scanValues); Nsim = allPoints.shape[0]; print('multiprocessing', Nsim, 'realizations')
	elif dictAlgo['bruteForceBasinStabMethod'] == 'statistics':
		print('Not yet tested, not yet implemented! Needs function that evaluates the data from the many realizations.'); sys.exit()

	global number_period_dyn;
	number_period_dyn 	= 20.5;
	initPhiPrime0		= 0;

	np.random.seed()
	poolData = [];																# should this be recasted to be an np.array?
	freeze_support()
	pool = Pool(processes=7)													# create a Pool object, pick number of processes


	#def multihelper(phiSr, initPhiPrime0, dictNet, dictPLL, phi, clock_counter, pll_list):
	if dictAlgo['bruteForceBasinStabMethod'] == 'classicBruteForceMethodRotatedSpace':
		poolData.append( pool.map(multihelper_star, zip( 						# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
						itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dictNet), itertools.repeat(dictPLL), itertools.repeat(dictAlgo) ) ) )
	elif dictAlgo['bruteForceBasinStabMethod'] == 'listOfInitialPhaseConfigurations' or dictAlgo['bruteForceBasinStabMethod'] == 'testNetworkMotifIsing':
		if isinstance(dictAlgo['min_max_range_parameter'], np.float) or isinstance(dictAlgo['min_max_range_parameter'], np.int):
			poolData.append( pool.map(multihelper_star, zip( 					# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
							itertools.product(scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(dictNet), itertools.repeat(dictPLL), itertools.repeat(dictAlgo) ) ) )
		elif isinstance(dictAlgo['min_max_range_parameter'], np.ndarray) or isinstance(dictAlgo['min_max_range_parameter'], list):
			poolData.append( pool.map(multihelper_star, zip( 					# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
							itertools.product(scanValues[0], scanValues[1]), itertools.repeat(initPhiPrime0), itertools.repeat(dictNet), itertools.repeat(dictPLL), itertools.repeat(dictAlgo) ) ) )
	elif dictAlgo['bruteForceBasinStabMethod'] == 'single':
		poolData.append( pool.map(multihelper_star, zip( 						# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
						itertools.product(scanValues[0], scanValues[1]), itertools.repeat(initPhiPrime0), itertools.repeat(dictNet), itertools.repeat(dictPLL), itertools.repeat(dictAlgo) ) ) )

	print('time needed for execution of simulations in multiproc mode: ', (time.time()-t0), ' seconds'); #sys.exit()


	eva.saveDictionaries(poolData, 'poolData', dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts
	eva.saveDictionaries(dictPLL, 'dictPLL',   dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts
	eva.saveDictionaries(dictNet, 'dictNet',   dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts
	eva.saveDictionaries(dictAlgo, 'dictAlgo', dictPLL['coupK'], dictPLL['transmission_delay'], dictPLL['cutFc'], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'], dictNet['topology'])	   # save the dicts

	if dictAlgo['bruteForceBasinStabMethod']   == 'testNetworkMotifIsing':
		eva.evaluateSimulationIsing(poolData)
	elif dictAlgo['bruteForceBasinStabMethod'] == 'listOfInitialPhaseConfigurations':
		eva.evaluateSimulationsChrisHoyer(poolData)
	elif dictAlgo['bruteForceBasinStabMethod'] == 'single':
		plot.plotCtrlSigDny(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'])
		plot.plotPhasesInf(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'])
		plot.plotPhases2pi(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'])
		plot.plotFrequency(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'])
		plot.plotOrderPara(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'])
		plot.plotPhaseRela(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'])
		plot.plotPhaseDiff(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'])
		plot.plotClockTime(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'])
		plot.plotOscSignal(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'])
		plot.plotFreqAndPhaseDiff(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'])
		plot.plotPSD(poolData[0][0]['dictPLL'], poolData[0][0]['dictNet'], poolData[0][0]['dictData'], [], saveData=False)
		plt.draw(); plt.show();
	elif dictAlgo['bruteForceBasinStabMethod'] == 'classicBruteForceMethodRotatedSpace':
		print('Implement evaluation as in the old version! Copy plots, etc...'); sys.exit()

	return poolData

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def multihelper(iterConfig, initPhiPrime0, dictNet, dictPLL, dictAlgo, param_id='None'):

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
			return simulateSystem(dictNetRea, dictPLL, dictAlgo, multi_sim=True)

	elif dictAlgo['bruteForceBasinStabMethod'] == 'listOfInitialPhaseConfigurations':	# so far for N=2, work it out for N>2
		initFreqDetune_vs_intrFreqDetune_equal = False
		temp   =  list(iterConfig)
		#print('iterConfig:', iterConfig, '\ttemp[0]:', temp[0], '\ttemp[1]:', temp[1])
		dictPLLRea = dictPLL.copy()
		if initFreqDetune_vs_intrFreqDetune_equal:
			if isinstance(dictPLL['intrF'], list):								# temp[1] represents half the frequency difference to be achieved in the uncoupled state
				meanIntF = np.mean(dictPLL['intrF'])
				dictPLLRea.update({'intrF': [meanIntF-temp[1], meanIntF+temp[1]]})
				#print('Intrinsic frequencies:', dictPLLRea['intrF'], '\tfor detuning', 2*temp[1]); time.sleep(2)
			else:
				dictPLLRea.update({'intrF': [dictPLL['intrF']-temp[1], dictPLL['intrF']+temp[1]]})
		else:																	# here: oscillators have intrinsic frequencies as given in dictPLL['intrF'], however initially they evolve
																				# with different frequencies given by syncF +/- half_the_freq_difference given by temp[1]
			dictPLLRea.update({'syncF': [dictPLL['syncF']-temp[1], dictPLL['syncF']+temp[1]]})
			dictPLLRea.update({'typeOfHist': 'syncState'})						# makes sure this mode is active
			print('WATCH OUT: dirty trick to achieve different frequency differences at the end of the history!!! Discuss with Chris Hoyer and address issue.')


		config = [0, temp[0]]
		dictNetRea = dictNet.copy()
		dictNetRea.update({'phiInitConfig': config, 'phiPerturb': np.zeros(dictNet['Nx']*dictNet['Ny']), 'phiPerturbRot': np.zeros(dictNet['Nx']*dictNet['Ny'])})

		return simulateSystem(dictNetRea, dictPLLRea, dictAlgo, multi_sim=True)

	elif dictAlgo['bruteForceBasinStabMethod'] == 'testNetworkMotifIsing':
		temp   =  list(iterConfig)
		#print('iterConfig:', iterConfig, '\ttemp[0]:', temp[0])
		config = [0]
		[config.append(entry) for entry in temp[0]]
		dictNetRea = dictNet.copy()
		dictNetRea.update({'phiInitConfig': config, 'phiPerturb': np.zeros(dictNet['Nx']*dictNet['Ny']), 'phiPerturbRot': np.zeros(dictNet['Nx']*dictNet['Ny'])})
		print('dictNetRea[*phiInitConfig*]:', dictNetRea['phiInitConfig'])

		return simulateSystem(dictNetRea, dictPLL, dictAlgo, multi_sim=True)

	elif dictAlgo['bruteForceBasinStabMethod'] == 'single':
		change_param = list(iterConfig)
		dictPLLRea = dictPLL.copy()
		if not dictAlgo['param_id'] == 'None':
			dictPLLRea.update({param_id: change_param})							# update the parameter chosen in change_param with a value of all scanvalues

		return simulateSystem(dictNet, dictPLLRea, dictAlgo, multi_sim=True)

	elif dictAlgo['bruteForceBasinStabMethod'] == 'statistics':
		change_param = list(iterConfig)
		dictPLLRea = dictPLL.copy()
		if not dictAlgo['param_id'] == 'None':
			dictPLLRea.update({param_id: change_param})							# update the parameter chosen in change_param with a value of all scanvalues

		return simulateSystem(dictNet, dictPLLRea, dictAlgo, multi_sim=True)

	else:
		print('No case fullfilled in multihelper in multisim_lib!'); sys.exit()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def multihelper_star(dynparam_fixparam):
	return multihelper(*dynparam_fixparam)
