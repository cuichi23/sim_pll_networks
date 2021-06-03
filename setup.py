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

#import matplotlib
#import matplotlib.pyplot as plt
import datetime
import time

import pll_lib as pll
import evaluation_lib as eva

''' Enable automatic carbage collector '''
gc.enable();

''' CREATE PLL LIST '''
def generatePLLs(dictPLL, dictNet, dictData):									#mode,div,params['topology'],couplingfct,histtype,Nplls,dt,c,delay,feedback_delay,F,F_Omeg,K,Fc,y0,phiM,domega):

	dictPLL.update({'G': setupTopology(dictNet)})

	pll_list = [ pll.PhaseLockedLoop(idx_pll,									# setup PLLs and store in a list as PLL class objects
					pll.Delayer(idx_pll, dictPLL, dictNet, dictData),			# setup delayer object of PLL k; it organizes the delayed communications
					pll.PhaseDetectorCombiner(idx_pll, dictPLL, dictNet),		# setup PDadder object of PLL k;
					pll.LowPass(idx_pll, dictPLL, dictNet),						# setup LF(1st) object of PLL k;
					pll.VoltageControlledOscillator(idx_pll,dictPLL, dictNet),	# setup VCO object of PLL k;
					pll.Antenna(idx_pll, dictPLL, dictNet),						# setup Antenna object of PLL k;
					pll.Counter(idx_pll, dictPLL)								# setup Counter object of PLL k;
					)  for idx_pll in range(dictNet['Nx']*dictNet['Ny']) ]

	return pll_list

################################################################################

def generatePhi0(dictNet):

	if ( dictNet['topology'] == 'entrainOne' or dictNet['topology'] == 'entrainAll' ):
		print('Provide phase-configuration for these cases in physical coordinates!')
		#phiM  = eva.rotate_phases(phiSr.flatten(), isInverse=False);
		phiM  = dictNet['phiConfig'];											# phiConfig: user specified configuration of initial phi states
		special_case = 0;
		if special_case == 1:
			phiS  = np.array([2., 2., 2.]);
			dictNet.update({'phiSr': eva.rotate_phases(phiS.flatten(), isInverse=True)})
		else:
			dictNet.update({'phiS': eva.rotate_phases(dictNet['phiSr'].flatten(), isInverse=False)})
			#print('Calculated phiS=',phiS,' from phiSr=',phiSr,'.\n')
		print('For entrainOne and entrainAll assumed initial phase-configuration of entrained synced state (physical coordinates):', phiM,
				' and on top a perturbation of (rotated coordinates):', phiSr, '  and in (original coordinates):', phiS, '\n')
		#phiS  = phiSr;
		#phiSr =	eva.rotate_phases(phiS.flatten(), isInverse=True)		  	# rotate back into rotated phase space for simulation
		#print('For entrainOne and entrainAll assumed initial phase-configuration of entrained synced state (physical coordinates):', phiS, ' and (rotated coordinates):', phiSr, '\n')
	elif dictNet['topology'] == 'compareEntrVsMutual':
		print('REWORK THIS!'); sys.exit()
		phiM  = dictNet['phiConfig'];
		if not dictNet['phiPerturbRot']:
			phiS = np.zeros(dictNet['Nx']*dictNet['Ny']);
		dictNet.update({'phiPerturb': eva.rotate_phases(phiSr.flatten(), isInverse=False)})
	else:
		print('Run single time-series and plot phase and frequency time series!')
		initPhiPrime0 = 0.0
		if dictNet['phiPerturbRot'] and not dictNet['phiPerturb']:
			print('Parameters set, perturbations provided manually in rotated phase space of phases.')
			#if len(dictNet['phiPerturbRot'].shape)==1:
			if len(dictNet['phiPerturbRot'])==dictNet['Nx']*dictNet['Ny']:
				print('Shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
				dictNet['phiPerturbRot'][0] = initPhiPrime0					# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
				dictNet.update({'phiPerturb': eva.rotate_phases(dictNet['phiPerturbRot'], isInverse=False)}) # rotate back into physical phase space for simulation
				print('\nPerturbations in ROTATED phase space:', dictNet['phiPerturbRot'])
				print('Dirac delta phase perturbation in ORIGINAL phase space:', dictNet['phiPerturb'])
			else:
				dictNet.update({'phiPerturb': np.zeros(dictNet['Nx']*dictNet['Ny'])})
				dictNet.update({'phiPerturbRot': eva.rotate_phases(dictNet['phiPerturb'], isInverse=True)})
				dictNet['phiPerturbRot'][0] = initPhiPrime0
				print('No perturbations defined, work it out! So far no perturbations are set, i.e., all zero!'); #sys.exit()

			# elif len(dictNet['phiPerturbRot'].shape)==2:
			# 	if len(dictNet['phiPerturbRot'][0,:])==dictNet['Nx']*dictNet['Ny']:
			# 		print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
			# 		dictNet['phiPerturbRot'][:,0] = initPhiPrime0				# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
			# 		print('\nvalues of the perturbations in ROTATED phase space, i.e., last time-step of history set as initial condition:', dictNet['phiPerturbRot'])
			# 		dictNet.update({'phiPerturb': eva.rotate_phases(dictNet['phiPerturbRot'].flatten(), isInverse=False)}) # rotate back into physical phase space for simulation
			# 		print('dirac delta phase perturbation in ORIGINAL phase space:', dictNet['phiPerturb'], '\n')

		elif dictNet['phiPerturb'] and not dictNet['phiPerturbRot']:
			print('Parameters set, perturbations provided manually in original phase space of phases.')
			#if len(dictNet['phiPerturb'].shape)==1:
			if len(dictNet['phiPerturb'])==dictNet['Nx']*dictNet['Ny']:
				dictNet.update({'phiPerturbRot': eva.rotate_phases(dictNet['phiPerturb'], isInverse=True)})
				print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
				dictNet['phiPerturbRot'][0] = initPhiPrime0						# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
				print('\nPerturbations in ROTATED phase space:', dictNet['phiPerturbRot'])
				print('Dirac delta phase perturbation in ORIGINAL phase space:', dictNet['phiPerturb'])
			else:
				dictNet.update({'phiPerturb': np.zeros(dictNet['Nx']*dictNet['Ny'])})
				dictNet.update({'phiPerturbRot': eva.rotate_phases(dictNet['phiPerturb'], isInverse=True)})
				dictNet['phiPerturbRot'][0] = initPhiPrime0
				print('No perturbations defined, work it out! So far no perturbations are set, i.e., all zero!'); #sys.exit()

			# elif len(dictNet['phiPerturb'].shape)==2:
			# 	if len(dictNet['phiPerturb'][0,:])==dictNet['Nx']*dictNet['Ny']:
			# 		dictNet.update({'phiPerturbRot': eva.rotate_phases(dictNet['phiPerturb'].flatten(), isInverse=True)})
			# 		print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
			# 		dictNet['phiPerturbRot'][:,0] = initPhiPrime0				# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
			# 		print('\nvalues of the perturbations in ROTATED phase space, i.e., last time-step of history set as initial condition:', dictNet['phiPerturbRot'])
			# 		print('dirac delta phase perturbation in ORIGINAL phase space:', dictNet['phiPerturb'], '\n')
			#
			# else:
			# 	print('Either no initial perturbations given, or Error in parameters - supply:\ncase_[sim_mode].py [topology] [#osci] [K] [F_c] [delay] [F_Omeg] [k] [Tsim] [c] [Nsim] [Nx] [Ny] [mx] [my] [cPD] [N entries for the value of the perturbation to oscis]')
			# 	print('\nNo perturbation set, hence all perturbations have the default value zero (in original phase space of phases)!')
			# 	dictNet.update({'phiPerturb': np.zeros(dictNet['Nx']*dictNet['Ny'])})
			# 	dictNet.update({'phiPerturbRot': np.zeros(dictNet['Nx']*dictNet['Ny'])})

	twistdelta=0; cheqdelta=0; twistdelta_x=0; twistdelta_y=0;
	if not ( dictNet['topology'] == 'ring' or dictNet['topology'] == 'chain' ):
		if dictNet['topology'] == 'square-open' or dictNet['topology'] == 'hexagon' or dictNet['topology'] == 'octagon':
			cheqdelta_x = np.pi 												# phase difference between neighboring oscillators in a stable chequerboard state
			cheqdelta_y = np.pi 												# phase difference between neighboring oscillators in a stable chequerboard state
			# print('phase differences of',k,'-twist:', twistdelta, '\n')
			if (dictNet['mx'] == 0 and dictNet['my'] == 0):
				dictNet.update( {'phiInitConfig': np.zeros(dictNet['Nx']*dictNet['Ny'])} )	# phiInitConfig denotes the unperturbed initial phases according to the m-twist state under investigation

			elif (dictNet['mx'] != 0 and dictNet['my'] != 0):
				for rows in range(dictNet['Ny']):								# set the mx-my-twist state's initial condition (history of "perfect" configuration)
					phiMtemp = np.arange(cheqdelta_y*rows, dictNet['Nx']*cheqdelta_x+cheqdelta_y*rows, cheqdelta_x)
					dictNet['phiInitConfig'].append(phiMtemp)
				dictNet.update( {'phiInitConfig': np.array(dictNet['phiInitConfig'])%(2.0*np.pi)} )
				#phiM = phiM.flatten();  # print('phiM: ', phiM, ' phiM.ndim: ', phiM.ndim)
				dictNet.update( {'phiInitConfig': np.concatenate( dictNet['phiInitConfig'], axis=0 )} )

			elif (dictNet['mx'] == 0 and dictNet['my'] != 0):					# prepare chequerboard only in y-direction
				for rows in range(dictNet['Ny']):								# set the chequerboard state's initial condition (history of "perfect" configuration)
					phiMtemp = np.arange(0.0, (dictNet['Nx'])*cheqdelta_x, cheqdelta_x)
					dictNet['phiInitConfig'].append(phiMtemp)
					#print('rows:', rows, 'dictNet[*phiInitConfig*]',dictNet['phiInitConfig'])
				dictNet.update( {'phiInitConfig': np.array(dictNet['phiInitConfig'])%(2.0*np.pi)} )
				# phiM = phiM.flatten(); # print('phiM: ', phiM)
				dictNet.update( {'phiInitConfig': np.concatenate( dictNet['phiInitConfig'], axis=0 )} )
				#print('dictNet[*phiInitConfig*]',dictNet['phiInitConfig']); sys.exit()

			elif (dictNet['mx'] != 0 and dictNet['my'] == 0):					# prepare chequerboard only in x-direction
				for columns in range(dictNet['Nx']):								# set the chequerboard state's initial condition (history of "perfect" configuration)
					phiMtemp = np.arange(0.0, (dictNet['Ny'])*cheqdelta_y, cheqdelta_y)
					dictNet['phiInitConfig'].append(phiMtemp)
				dictNet.update( {'phiInitConfig': np.array(phiM)%(2.0*np.pi)} )
				# phiM = phiM.flatten(); # print('phiM: ', phiM)
				dictNet.update( {'phiInitConfig': np.concatenate( dictNet['phiInitConfig'], axis=0 )} )

		elif (dictNet['topology'] == 'hexagon-periodic' or dictNet['topology'] == 'octagon-periodic' or dictNet['topology'] == 'square-periodic'):
			twistdelta_x = ( 2.0 * np.pi * dictNet['my'] / ( float( dictNet['Nx'] ) ) )	# phase difference between neighboring oscillators in a stable m-twist state
			twistdelta_y = ( 2.0 * np.pi * dictNet['my'] / ( float( dictNet['Ny'] ) ) )	# phase difference between neighboring oscillators in a stable m-twist state
			# print('phase differences of',k,'-twist:', twistdelta, '\n')
			# print('N =', N, '    Nx =', Nx, '    Ny =', Ny, '    k =', k, '    kx =', kx, '    ky =', ky)
			if (dictNet['mx'] == 0 and dictNet['my'] == 0):
				dictNet.update( {'phiInitConfig': np.zeros(dictNet['Nx']*dictNet['Ny'])} ) # phiM denotes the unperturbed initial phases according to the m-twist state under investigation
				print('Length, type and shape of phiM:', len(dictNet['phiInitConfig']), type(dictNet['phiInitConfig']), dictNet['phiInitConfig'].shape)
			else:
				# print('type phiM at initialization', type(phiM))
				# print('Entering loop over Ny to set initial phiM.')
				for rows in range(dictNet['Ny']):								# set the mx-my-twist state's initial condition (history of "perfect" configuration)
					# print('loop #', rows)
					#phiMtemp = np.arange(twistdelta_y*rows, Nx*twistdelta_x+twistdelta_y*rows, twistdelta_x)
					phiMtemp = twistdelta_x * np.arange(dictNet['Nx']) + twistdelta_y * rows
					# print('phiMtemp=', phiMtemp, '    of type ', type(phiMtemp), '    and length ', len(phiMtemp))
					dictNet['phiInitConfig'].append(phiMtemp)
					# print('phiM(list)=', phiMt, '    of type ', type(phiMt))

				dictNet.update({'phiInitConfig': np.array(dictNet['phiInitConfig'])} )
				# print('phiM[1,]', phiM[1,])
				# print('phiM(array)=', phiM, '    of type ', type(phiM), '    and shape ', phiM.shape)

				phiMreorder=np.zeros(dictNet['Nx']*dictNet['Ny']); counter=0;		 # could be replaced by phiM = np.concatenate( phiM, axis=0 )
				for i in range(dictNet['Nx']):
					for j in range(dictNet['Ny']):
						# print('counter:', counter)
						phiMreorder[counter]=dictNet['phiInitConfig'][i][j]; counter=counter+1;
				dictNet.update({'phiInitConfig': phiMreorder%(2.0*np.pi)} )
				# print('phiMreorderd: ', phiM, '    of type ', type(phiM), '    and shape ', phiM.shape)

				# NOPE phiM = np.reshape(phiM, (np.product(phiM.shape),))
				# phiM = phiM.flatten();
				# phiM = phiM[:][:].flatten();
				# print('phiMflattened: ', phiM, '    of type ', type(phiM), '    and shape ', phiM.shape)
				# print('Length, type and shape of phiMflattened that was generated:', len(phiM), type(phiM), phiM.shape)
	if ( dictNet['topology'] == 'ring' or dictNet['topology'] == 'chain' ):
		if dictNet['topology'] == 'chain':
			cheqdelta = np.pi													# phase difference between neighboring oscillators in a stable chequerboard state
			if dictNet['mx'] == 0:
				dictNet.update( {'phiInitConfig': np.zeros(dictNet['Nx']*dictNet['Ny'])} )	# phiM denotes the unperturbed initial phases according to the chequerboard state under investigation
			else:
				dictNet.update( {'phiInitConfig': np.arange(0.0, dictNet['Nx']*dictNet['Ny']*cheqdelta, cheqdelta)} ) # vector mit N entries from 0 increasing by twistdelta for every element, i.e., the phase-configuration
				# print('phiM: ', phiM)											# in the original phase space of an chequerboard solution
		else:
			twistdelta = ( 2.0 * np.pi * dictNet['mx'] / ( float( dictNet['Nx']*dictNet['Ny'] ) ) )					# phase difference between neighboring oscillators in a stable m-twist state
			# print('phase differences of',k,'-twist:', twistdelta, '\n')
			if dictNet['mx'] == 0:
				dictNet.update( {'phiInitConfig': np.zeros(dictNet['Nx']*dictNet['Ny'])} )	# phiM denotes the unperturbed initial phases according to the m-twist state under investigation
			else:
				dictNet.update({'phiInitConfig': np.arange(0.0, dictNet['Nx']*twistdelta, twistdelta)})	# vector mit N entries from 0 increasing by twistdelta for every element, i.e., the phase-configuration
				dictNet.update({'phiInitConfig': np.array(dictNet['phiInitConfig'])%(2.0*np.pi)})			# bring back into interval [0 2pi]
		print('dictNet[*phiInitConfig*]:\t', dictNet['phiInitConfig'])									# in the original phase space of an m-twist solution
	if dictNet['topology'] == 'global' or dictNet['topology'] == 'entrainPLLsHierarch':
		dictNet.update( {'phiInitConfig': np.zeros(dictNet['Nx']*dictNet['Ny'])} )	# for all-to-all coupling we assume no twist states with m > 0

	return dictNet

################################################################################

def setupTimeDependentParameter(dictNet, dictPLL, dictData, parameter='coupStr_2ndHarm', afterTsimPercent=0.5, forAllPLLsDifferent=False):

	if forAllPLLsDifferent:
		time_series = np.zeros([ dictNet['Nx']*dictNet['Ny'], dictNet['max_delay_steps']+dictPLL['sim_time_steps'] ])
	else:
		time_series = np.zeros([ 1, dictNet['max_delay_steps']+dictPLL['sim_time_steps'] ])
	#print('dictNet[*min_max_rate_timeDepPara*][1]:', dictNet['min_max_rate_timeDepPara'][1])

	sign = -1
	if dictNet['typeOfTimeDependency'] == 'linear':
		for i in range(len(time_series[:,0])):
			sign = sign * (-1)
			time_series[i,0:dictNet['max_delay_steps']+int(afterTsimPercent*dictPLL['sim_time_steps'])] = dictNet['min_max_rate_timeDepPara'][0];
			for j in range(dictNet['max_delay_steps']+int(afterTsimPercent*dictPLL['sim_time_steps'])-1, dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1):
				if np.abs( time_series[i,j] - dictNet['min_max_rate_timeDepPara'][0] ) <= np.abs( dictNet['min_max_rate_timeDepPara'][1] - dictNet['min_max_rate_timeDepPara'][0] ):
					time_series[i,j+1] = time_series[i,j] + sign * dictPLL['dt'] * dictNet['min_max_rate_timeDepPara'][2]
					#print('time_series', time_series)
				else:
					time_series[i,j+1] = time_series[i,j]

	elif dictNet['typeOfTimeDependency'] == 'exponential':
		for i in range(len(time_series[:,0])):
			tstep_annealing_start = dictNet['max_delay_steps']+int(afterTsimPercent*dictPLL['sim_time_steps'])
			time_series[i,0:tstep_annealing_start] = dictNet['min_max_rate_timeDepPara'][0];
			for j in range(tstep_annealing_start-1, dictNet['max_delay_steps']+dictPLL['sim_time_steps']-1):
				if np.abs( time_series[i,j] - dictNet['min_max_rate_timeDepPara'][0] ) <= np.abs( dictNet['min_max_rate_timeDepPara'][1] - dictNet['min_max_rate_timeDepPara'][0] ):
					print('j:', j, '\ttime = (j-j0)*dt=', (j-tstep_annealing_start+1)*dictPLL['dt'])
					time_series[i,j+1] = time_series[i,j] + dictPLL['dt'] * dictNet['min_max_rate_timeDepPara'][1] * np.exp( -(j-tstep_annealing_start+1)*dictPLL['dt'] / dictNet['min_max_rate_timeDepPara'][2] ) / dictNet['min_max_rate_timeDepPara'][2]
					print('time_series[%i,%i+1]'%(i,j), time_series[i,j+1], '\tincrement:', dictPLL['dt'] * dictNet['min_max_rate_timeDepPara'][1] * np.exp( -(j-tstep_annealing_start+1)*dictPLL['dt'] / ( dictNet['min_max_rate_timeDepPara'][2]/1 ) ) / dictNet['min_max_rate_timeDepPara'][2])
					#time.sleep(0.2)
				else:
					time_series[i,j+1] = time_series[i,j]

	else:
		print('Unknown functional form. Introduce! °)°'); sys.exit()

	dictData.update({'timeDependentParameter': time_series})

	print('time-series: ', [*time_series])

	return time_series

################################################################################

def setupTopology(dictNet):

	# see also: https://networkx.org/documentation/stable/reference/generators.html
	if dictNet['topology'] == 'global':
		G = nx.complete_graph(dictNet['Nx']*dictNet['Ny'])
		# print('G and G.neighbors(PLL0):', G, G.neighbors(0)); sys.exit(1)

	elif ( dictNet['topology'] == 'compareEntrVsMutual' and dictNet['Nx'] == 6):
		G = nx.DiGraph();
		G.add_nodes_from([i for i in range(dictNet['Nx']*dictNet['Ny'])]);
		G.add_edges_from([(0,1),(1,0),(3,2)]);				  						# bidirectional coupling between 0 and 1 and 3 receives from 2, i.e., 2 entrains 3
		for i in range(dictNet['Nx']*dictNet['Ny']):
			print('For comparison entrainment vs mutual sync: neighbors of oscillator ',i,':', list(G.neighbors(i)) , ' and egdes of',i,':', list(G.edges(i)))

	elif ( dictNet['topology'] == 'entrainPLLsHierarch'):
		G = nx.DiGraph();
		if dictNet['hierarchy_level'] > dictNet['Nx']*dictNet['Ny']:
			sys.exit('Special dictNet[*topology*] does not work like that... decrease hierarchy level - cannot exceed the number of PLLs in the system!')
		G.add_nodes_from([i for i in range(dictNet['Nx']*dictNet['Ny'])]);
		for i in range(0,dictNet['hierarchy_level']):
			G.add_edge(i+1 ,i);														# add unidirectional edge from osci 0 to 1, 1 to 2, and so on until level_hierarch is reached

		for i in range(dictNet['hierarchy_level']+1, dictNet['Nx']*dictNet['Ny']):
			G.add_edge(i, dictNet['hierarchy_level']); 								# add unidirectional edge from highest hierarchy level to all remaining PLLS

	elif ( dictNet['topology'] == 'ring' or dictNet['topology'] == 'entrainAll' ):
		G = nx.cycle_graph(dictNet['Nx']*dictNet['Ny'])
		# if dictNet['topology'] == 'entrainAll':
		# 	G.remove_edge(0,2);
		# 	G.remove_edge(1,2);

	elif ( dictNet['topology'] == 'chain' or dictNet['topology'] == 'entrainOne' ):
		G = nx.path_graph(dictNet['Nx']*dictNet['Ny'])
		# if dictNet['topology'] == 'entrainOne':
		# 	G.remove_edge(0,2);
		# 	G.remove_edge(1,2);
		# 	G.remove_edge(2,1);

	# elif dictNet['topology'] == 'entrainOne':
	# 	print('STOP, not working yet with G.neighbors(idx)! dictNet['topology'] of entrainment of synchronized state -- reference feeds into one of the oscillators.')
	# 	if not dictNet['Nx']*dictNet['Ny'] == 3:
	# 		print('Not yet configured for N != 3! Check.')
	# 	G = nx.MultiDiGraph();
	# 	G.add_edges_from([(0,1),(1,2),(2,1)]);
	#
	# elif dictNet['topology'] == 'entrainAll':
	# 	print('STOP, not working yet with G.neighbors(idx)! dictNet['topology'] of entrainment of synchronized state -- reference feeds into all of the oscillators.')
	# 	if not dictNet['Nx']*dictNet['Ny'] == 3:
	# 		print('Not yet configured for N != 3! Check.')
	# 	G = nx.MultiDiGraph();
	# 	G.add_edges_from([(0,1),(0,2),(1,2),(2,1)]);

	else:
		if dictNet['topology'] == 'square-open':
			G = nx.grid_2d_graph(dictNet['Nx'],dictNet['Ny'], periodic=False)

		elif dictNet['topology'] == 'square-periodic':
			G = nx.grid_2d_graph(dictNet['Nx'],dictNet['Ny'], periodic=True)		# for periodic boundary conditions:

		elif dictNet['topology'] == 'hexagon':
			print('\nIf Nx =! Ny, then check the graph that is generated again!')
			G=nx.grid_2d_graph(dictNet['Nx'],dictNet['Ny'])							# why not ..._graph(Nx,Ny) ? NOTE the for n in G: loop has to be changed, loop over nx and ny respectively, etc....
			for n in G:
				x,y=n
				if x>0 and y>0:
					G.add_edge(n,(x-1,y-1))
				if x<Nx-1 and y<Ny-1:
					G.add_edge(n,(x+1,y+1))

		elif dictNet['topology'] == 'hexagon-periodic':
			G=nx.grid_2d_graph(dictNet['Nx'],dictNet['Ny'], periodic=True)
			for n in G:
				x,y=n
				G.add_edge(n, ((x-1)%dictNet['Nx'], (y-1)%dictNet['Ny']))

		elif dictNet['topology'] == 'octagon':									# why not ..._graph(Nx,Ny) ? NOTE the for n in G: loop has to be changed, loop over nx and ny respectively, etc....
			print('\nIf Nx =! Ny, then check the graph that is generated again!')
			G=nx.grid_2d_graph(dictNet['Nx'],dictNet['Ny'])
			for n in G:
				x,y=n
				if x>0 and y>0:
					G.add_edge(n,(x-1,y-1))
				if x<Nx-1 and y<Ny-1:
					G.add_edge(n,(x+1,y+1))
				if x<Nx-1 and y>0:
					G.add_edge(n,(x+1,y-1))
				if x<Nx-1 and y>0:
					G.add_edge(n,(x+1,y-1))
				if x>0 and y<Ny-1:
					G.add_edge(n,(x-1,y+1))

		elif dictNet['topology'] == 'octagon-periodic':
			G=nx.grid_2d_graph(dictNet['Nx'],dictNet['Ny'], periodic=True)
			for n in G:
				x,y=n
				G.add_edge(n, ((x-1)%dictNet['Nx'], (y-1)%dictNet['Ny']))
				G.add_edge(n, ((x-1)%dictNet['Nx'], (y+1)%dictNet['Ny']))

		# G = nx.convert_node_labels_to_integers(G)
		G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted') # converts 2d coordinates to 1d index of integers, e.g., k=0,...,N-1

	if dictNet['Nx']*dictNet['Ny'] < 36:
		F=nx.adjacency_matrix(G)
		print('nx.adjacency_matrix(G)', F.todense())
		print('nx.adjacency_spectrum(G)/max(nx.adjacency_spectrum(G))', nx.adjacency_spectrum(G)/max(nx.adjacency_spectrum(G)))

	return G

def allInitPhaseCombinations(dictPLL, dictNet, paramDiscretization=10):

	if dictNet['Nx']*dictNet['Ny'] == 2:
		scanValues = np.zeros((dictNet['Nx']*dictNet['Ny'], paramDiscretization), dtype=np.float)			# create container for all points in the discretized rotated phase space, +/- pi around each dimension (unit area)
		# scanValues[0,:] = np.linspace(phiMr[0]-(np.pi), phiMr[0]+(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space
		# scanValues[1,:] = np.linspace(phiMr[1]-(np.pi), phiMr[1]+(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space
		scanValues[0,:] = np.linspace(-(np.pi), +(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
		scanValues[1,:] = np.linspace(-(np.pi), +(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
		#print('row', i,'of matrix with all intervals of the rotated phase space:\n', scanValues[i,:], '\n')

		_allPoints 		= itertools.product(*scanValues)
		allPoints 		= list(_allPoints)										# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
		allPoints 		= np.array(allPoints)									# convert the list to an array
		# allPoints_unitCell  = []
		# for point in allPoints:
		# 	if unit_cell.is_inside(point, isRotated=True):
		# 		allPoints_unitCell.append(point)
		# allPoints			= np.array(allPoints_unitCell)
	elif dictNet['Nx']*dictNet['Ny'] > 2:
		# setup a matrix for all N variables/dimensions and create a cube around the origin with side lengths 2pi
		scanValues = np.zeros((dictNet['Nx']*dictNet['Ny']-1,paramDiscretization), dtype=np.float)		# create container for all points in the discretized rotated phase space, +/- pi around each dimension (unit area)
		for i in range (0, dictNet['Nx']*dictNet['Ny']-1):												# the different coordinates of the solution, discretize an interval plus/minus pi around each variable
			# scanValues[i,:] = np.linspace(phiMr[i+1]-np.pi, phiMr[i+1]+np.pi, paramDiscretization) # all entries are in rotated, and reduced phase space
			if i==0:															# theta2 (x-axis)
				#scanValues[i,:] = np.linspace(-(np.pi), +(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
				#scanValues[i,:] = np.linspace(-0.25*np.pi, 0.25*np.pi, paramDiscretization)
				scanValues[i,:] = np.linspace(-1.0*np.pi, 1.0*np.pi, paramDiscretization)
			else:																# theta3 (y-axis)
				#scanValues[i,:] = np.linspace(-(1.35*np.pi), +(1.35*np.pi), paramDiscretization)
				#scanValues[i,:] = np.linspace(-0.35*np.pi, 0.35*np.pi, paramDiscretization)
				scanValues[i,:] = np.linspace(-1.35*np.pi, 1.35*np.pi, paramDiscretization)

			#print('row', i,'of matrix with all intervals of the rotated phase space:\n', scanValues[i,:], '\n')

		_allPoints 			= itertools.product(*scanValues)
		allPoints 			= list(_allPoints)									# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
		allPoints 			= np.array(allPoints) 								# convert the list to an array

	return scanValues, allPoints
