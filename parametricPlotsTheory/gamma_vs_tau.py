#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import numpy as np
import datetime
import os, gc
now = datetime.datetime.now()

import parametricPlots as paraPlot
import synctools_interface_lib as synctools
import coupling_fct_lib as coupfct

dict_net={
	'Nx': 2,																	# oscillators in x-direction
	'Ny': 1,																	# oscillators in y-direction
	'mx': 0	,																	# twist/chequerboard in x-direction (depends on closed or open boundary conditions)
	'my': -999,																	# twist/chequerboard in y-direction
	'Tsim': 100,
	'topology': 'ring',															# 1d) ring, chain, 2d) square-open, square-periodic, hexagonal...
																				# 3) global, entrainOne, entrainAll, entrainPLLsHierarch, compareEntrVsMutual
	'zeta': -1, 																# real part of eigenvalue of slowest decaying perturbation mode for the set of parameters, also a fct. of tau!
	'psi': np.pi,																# real part of eigenvalue of slowest decaying perturbation
	'computeFreqAndStab': True													# compute linear stability and global frequency if possible: True or False
}

dict_pll={
	'analyzeFreq': 'max',														# choose from 'max', 'min', 'middle' --> which of up to three multistable Omega to analyze
	'intrF': 1.0,																# intrinsic frequency in Hz
	'syncF': 1.0,																# frequency of synchronized state in Hz
	'coupK': 0.4,																# [random.uniform(0.3, 0.4) for i in range(dict_net['Nx']*dict_net['Ny'])],# coupling strength in Hz float or [random.uniform(minK, maxK) for i in range(dict_net['Nx']*dict_net['Ny'])]
	'cutFc': np.array([0.014]),													# LF cut-off frequency in Hz, None for no LF, or e.g., N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05]
	'div': 1,																	# divisor of divider (int)
	'friction_coefficient': 1,													# friction coefficient of 2nd order Kuramoto models
	'fric_coeff_PRE_vs_PRR': 'PRE',												# 'PRR': friction coefficient multiplied to instant. AND intrin. freq, 'PRE': friction coefficient multiplied only to instant. freq
	'feedback_delay': 0,														# value of feedback delay in seconds
	'feedback_delay_var': None, 												# variance of feedback delay
	'transmission_delay': 0.65, 												# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dict_net['Nx']*dict_net['Ny'],dict_net['Nx']*dict_net['Ny']]), OR [np.random.uniform(min,max) for i in range(dict_net['Nx']*dict_net['Ny'])]
	# choose from coupfct.<ID>: sine, cosine, neg_sine, neg_cosine, triangular, deriv_triangular, square_wave, pfd, inverse_cosine, inverse_sine
	'coup_fct_sig': coupfct.neg_cosine,											# coupling function h(x) for PLLs with ideally filtered PD signals:
	'derivative_coup_fct': coupfct.sine											# derivative h'(x) of coupling function h(x)
}

w 		= 2.0*np.pi*dict_pll['intrF']
K		= 2.0*np.pi*dict_pll['coupK']
wc		= 2.0*np.pi*dict_pll['cutFc']#, 0.14, 0.9])
z 		= dict_net['zeta']														# eigenvalue of the perturbation mode
psi		= dict_net['psi']														# imaginary part of complex representation of zeta in polar coordinates

h  		= dict_pll['coup_fct_sig']
hp 		= dict_pll['derivative_coup_fct']

beta 	= 0#np.pi																# choose according to choice of mx, my and the topology!

tau 	= np.arange(0, 4, 0.01)

#OmegInTauVsGamma = []; alpha = []; ReLambda = []; ImLambda = [];
OmegInTauVsGamma = np.zeros([len(wc), len(tau)]); alpha = np.zeros([len(wc), len(tau)]); ReLambda = np.zeros([len(wc), len(tau)]); ImLambda = np.zeros([len(wc), len(tau)]);
for i in range(len(wc)):
	dict_pll.update({'cutFc': wc[i]/(2*np.pi)})									# set this temporarly to one value -- in Hz
	for j in range(len(tau)):
		dict_pll.update({'transmission_delay': tau[j]})							# set this temporarly to one value -- in seconds
		isRadian 	= False														# set this False to get values returned in [Hz] instead of [rad * Hz]
		sf 			= synctools.SweepFactory(dict_pll, dict_net, isRadians=isRadian)
		fsl 		= sf.sweep()
		para_mat 	= fsl.get_parameter_matrix(isRadians=isRadian)
		if len(para_mat[:,4]) > 1:
			if len(para_mat[:,4]) > 3:
				print('NOTE: more than 3 existing solutions found -- adjust min, max, middle solution!')
			#print('Found multistability of synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dict_pll['coupK'], dict_pll['transmission_delay'], beta,')\nPick state with largest frequency!')
			if dict_pll['analyzeFreq'] == 'max':
				index = np.argmax(para_mat[:,4], axis=0)
			elif dict_pll['analyzeFreq'] == 'min':
				index = np.argmin(para_mat[:,4], axis=0)
			elif dict_pll['analyzeFreq'] == 'middle':
				index = np.where(para_mat[:,4]==sorted(para_mat[:,4])[1])[0][0]
			OmegInTauVsGamma[i,j] = 2.0*np.pi*para_mat[index,4];
			alpha[i,j] = ((2.0*np.pi*para_mat[index,1]/para_mat[index,12])*dict_pll['derivative_coup_fct']( (-2.0*np.pi*para_mat[index,4]*para_mat[index,3]+beta)/para_mat[index,12] ))
			ReLambda[i,j] = para_mat[index,5]
			ImLambda[i,j] = para_mat[index,6]
		else:
			#print('Found one synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dict_pll['coupK'], dict_pll['transmission_delay'], beta,').')
			OmegInTauVsGamma[i,j] = 2.0*np.pi*para_mat[:,4][0];
			alpha[i,j] = ((2.0*np.pi*para_mat[:,1]/para_mat[:,12])*dict_pll['derivative_coup_fct']( (-2.0*np.pi*para_mat[:,4]*para_mat[:,3]+beta)/para_mat[:,12] ))[0]
			ReLambda[i,j] = para_mat[:,5][0]
			ImLambda[i,j] = para_mat[:,6][0]

dict_pll.update({'transmission_delay': tau})										# set coupling strength key in dict_pll back to the array
dict_pll.update({'cutFc': wc})													# set coupling strength key in dict_pll back to the array

# provide the parameter that is NOT the x-axis as the outer loop, i.e., loopP1
loopP1	= 'wc'																	# discrete parameter
loopP2 	= 'tau'																	# x-axis
discrP	= 'wc'																	# plot for different values of this parameter if given

######################################################################################################################################
# no user input below (unless you know what you are doing!)

paramsDict  = {'h': h, 'hp': hp, 'w': w, 'K': K, 'wc': wc, 'Omeg': OmegInTauVsGamma, 'alpha': alpha,
			'tau': tau, 'zeta': z, 'psi': psi, 'beta': beta, 'loopP1': loopP1, 'loopP2': loopP2, 'discrP': discrP}

if ( isinstance(paramsDict[paramsDict['discrP']], int) or isinstance(paramsDict[paramsDict['discrP']], float) ):
	#print('type(params[params[*discrP*]])', type(params[params['discrP']]))
	paramsDict[paramsDict['discrP']] = [paramsDict[paramsDict['discrP']]]
	#print('type(params[params[*discrP*]])', type(params[params['discrP']]))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot Omega as parameter plot in the tau - wc plot

#		 makePlotsFromSynctoolsResults(figID, x, y,  z, rescale_x, rescale_y, rescale_z, x_label, y_label, z_label, x_identifier, y_identifier, z_identifier)
# paraPlot.makePlotsFromSynctoolsResults(100, tau, wc, OmegInTauVsGamma, w/(2.0*np.pi), 1.0/w, 1.0,
# 				r'$\frac{\omega\tau}{2\pi}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\Omega$', 'tau', 'K', 'Omeg', None, cm.coolwarm)
# paraPlot.makePlotsFromSynctoolsResults(101, tau, wc, ReLambda, w/(2.0*np.pi), 1.0/w, w/(2.0*np.pi),
# 				r'$\frac{\omega\tau}{2\pi}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'tau', 'wc', 'ReLambda', None, cm.PuOr)
# paraPlot.makePlotsFromSynctoolsResults(102, tau, wc, ImLambda, w/(2.0*np.pi), 1.0/w, 1.0/w,
# 				r'$\frac{\omega\tau}{2\pi}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'tau', 'wc', 'ImLambda', None, cm.PuOr)
# plt.draw(); #plt.show();

paraPlot.plot2D(paramsDict)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
