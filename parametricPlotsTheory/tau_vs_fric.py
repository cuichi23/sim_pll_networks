#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import datetime, time
import os, gc, sys, pickle
now = datetime.datetime.now()

# plot parameter
axisLabel = 50;
titleLabel= 10;
dpi_val   = 150;
figwidth  = 10;
figheight = 5;

sys.path.append('..') 															# adds higher directory to python modules path
import parametricPlots as paraPlot
import synctools_interface_lib as synctools
import coupling_fct_lib as coupfct

''' Set system and network parameters here, be careful as there are no sanity checks! This can be improved in the future. '''

dictNet={
	'Nx': 3,																	# oscillators in x-direction
	'Ny': 3,																	# oscillators in y-direction
	'mx': 0	,																	# twist/chequerboard in x-direction (depends on closed or open boundary conditions)
	'my': 0,																	# twist/chequerboard in y-direction
	'Tsim': 100,
	'topology': 'square-periodic',															# 1d) ring, chain, 2d) square-open, square-periodic, hexagonal...
																				# 3) global, entrainOne, entrainAll, entrainPLLsHierarch, compareEntrVsMutual
	'zeta': [-0.5, 0.25], #[-1], 												# real part of eigenvalue of slowest decaying perturbation mode for the set of parameters, also a fct. of tau!
	'psi': [np.pi, 0],															# real part of eigenvalue of slowest decaying perturbation
	'computeFreqAndStab': True,													# compute linear stability and global frequency if possible: True or False
	'calcSynctoolsStab': False													# also calclate stability from synctools - time consuming! to validate conditions
}

dictPLL={
	'analyzeFreq': 'max',														# choose from 'max', 'min', 'middle' --> which of up to three multistable Omega to analyze
	'intrF': 1.0,																# intrinsic frequency in Hz
	'syncF': 1.0,																# frequency of synchronized state in Hz
	'coupK': 0.65,																# [random.uniform(0.3, 0.4) for i in range(dictNet['Nx']*dictNet['Ny'])],# coupling strength in Hz float or [random.uniform(minK, maxK) for i in range(dictNet['Nx']*dictNet['Ny'])]
	'cutFc': 0.20,																# LF cut-off frequency in Hz, None for no LF, or e.g., N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05]
	'div': 1,																	# divisor of divider (int)
	'friction_coefficient': 1,													# friction coefficient of 2nd order Kuramoto models
	'feedback_delay': 0,														# value of feedback delay in seconds
	'feedback_delay_var': None, 												# variance of feedback delay
	'transmission_delay': 2.95, 												# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]), OR [np.random.uniform(min,max) for i in range(dictNet['Nx']*dictNet['Ny'])]
	# choose from coupfct.<ID>: sine, cosine, neg_sine, neg_cosine, triangular, deriv_triangular, square_wave, pfd, inverse_cosine, inverse_sine
	'coup_fct_sig': coupfct.sine,												# coupling function h(x) for PLLs with ideally filtered PD signals:
	'derivative_coup_fct': coupfct.cosine,										# derivative h'(x) of coupling function h(x)
	'inve_deriv_coup_fct': coupfct.inverse_cosine								# inverse of derivative of coupling function
}

w 		= 2.0*np.pi*dictPLL['intrF']
wc		= 2.0*np.pi*dictPLL['cutFc']
K		= 2.0*np.pi*dictPLL['coupK']
z 		= dictNet['zeta']														# eigenvalue of the perturbation mode
psi		= dictNet['psi']														# imaginary part of complex representation of zeta in polar coordinates

h  		= dictPLL['coup_fct_sig']
hp 		= dictPLL['derivative_coup_fct']

beta 	= 0																		# choose according to choice of mx, my and the topology!

#tau 	= np.arange(0, 16, 0.1)
#fric  	= np.arange( 0.25, 2, 0.1 )
tau		= np.arange(0, 16, 0.0005)
fric  	= np.arange( 0.25, 2, 0.0005 )

fzeta = 1+np.sqrt(1-np.max(np.abs(z))**2)
#OmegInTauvsFric = []; alpha = []; ReLambda = []; ImLambda = [];
OmegInTauvsFric = np.zeros([len(tau), len(fric)]); alpha = np.zeros([len(tau), len(fric)]); ReLambda = np.zeros([len(tau), len(fric)]); ImLambda = np.zeros([len(tau), len(fric)]);
CondStab = np.zeros([len(tau), len(fric)]);
t0 = time.time()
for i in range(len(tau)):
	dictPLL.update({'transmission_delay': tau[i]})								# set this temporarly to one value -- in seconds
	for j in range(len(fric)):
		isRadian 	= False														# set this False to get values returned in [Hz] instead of [rad * Hz]
		dictPLL.update({'friction_coefficient': fric[j]})						# set this temporarly to one value -- in Hz
		sf 			= synctools.SweepFactory(dictPLL, dictNet, isRadians=isRadian)
		fsl 		= sf.sweep()
		if dictNet['calcSynctoolsStab']:
			para_mat = fsl.get_parameter_matrix(isRadians=isRadian)
		else:
			para_mat = fsl.get_parameter_matrix_nostab(isRadians=isRadian)
		if len(para_mat[:,4]) > 1:
			#print('Found multistability of synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dictPLL['coupK'], dictPLL['transmission_delay'], beta,')\nPick state with largest frequency!')
			if dictPLL['analyzeFreq'] == 'max':
				index = np.argmax(para_mat[:,4], axis=0)
			elif dictPLL['analyzeFreq'] == 'min':
				index = np.argmin(para_mat[:,4], axis=0)
			elif dictPLL['analyzeFreq'] == 'middle':
				index = np.where(para_mat[:,4]==sorted(para_mat[:,4])[1])[0][0]
			OmegInTauvsFric[i,j] = 2.0*np.pi*para_mat[index,4];
			alpha[i,j] = ((2.0*np.pi*para_mat[index,1]/para_mat[index,12])*dictPLL['derivative_coup_fct']( (-2.0*np.pi*para_mat[index,4]*para_mat[index,3]+beta)/para_mat[index,12] ))
			ReLambda[i,j] = para_mat[index,5]
			ImLambda[i,j] = para_mat[index,6]
		else:
			#print('Found one synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dictPLL['coupK'], dictPLL['transmission_delay'], beta,').')
			OmegInTauvsFric[i,j] = 2.0*np.pi*para_mat[:,4][0];
			alpha[i,j] = ((2.0*np.pi*para_mat[:,1]/para_mat[:,12])*dictPLL['derivative_coup_fct']( (-2.0*np.pi*para_mat[:,4]*para_mat[:,3]+beta)/para_mat[:,12] ))[0]
			ReLambda[i,j] = para_mat[:,5][0]
			ImLambda[i,j] = para_mat[:,6][0]
		# diff1zeta = wc*fric[j]**2/(2*alpha[i,j]) - ( 1 - np.sqrt(1 - np.abs(np.array(z))**2) )
		# diff2	  = wc*fric[j]**2/(2*alpha[i,j]) - 1
		# if np.all(diff1zeta > 0):
		# 	CondStab[i,j] = 0
		# elif diff2 > 0:
		# 	CondStab[i,j] = 1
		# else:
		# 	CondStab[i,j] = None
		if wc*fric[j]**2/(2*alpha[i,j]) > fzeta[k]:
			CondStab[i,j] = 0
		elif wc*fric[j]**2/(2*alpha[i,j]) > 1:
			CondStab[i,j] = 1
		else:
			CondStab[i,j] = None


print('Time computation in sweep_factory: ', (time.time()-t0), ' seconds');

dictPLL.update({'transmission_delay': tau})										# set coupling strength key in dictPLL back to the array
dictPLL.update({'friction_coefficient': fric})									# set coupling strength key in dictPLL back to the array

loopP1	= 'tau'																	# x-axis -- NOTE: this needs to have the same order as the loops above!
loopP2 	= 'fric'																# y-axis	otherwise, the Omega sorting will be INCORRECT!
discrP	= None																	# does not apply to parametric plots
rescale = 'K_to_2alpha'															# set this in case you want to plot against a rescaled loopP variable

paramsDict = {'h': h, 'hp': hp, 'w': w, 'K': K, 'wc': wc, 'fric': fric, 'Omeg': OmegInTauvsFric, 'alpha': alpha, 'CondStab': CondStab,
			'tau': tau, 'zeta': z, 'psi': psi, 'beta': beta, 'loopP1': loopP1, 'loopP2': loopP2, 'discrP': discrP, 'ReLambSynctools': ReLambda, 'ImLambSynctools': ImLambda}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot Omega as parameter plot in the K - wc plot

#		 makePlotsFromSynctoolsResults(figID, x, y,  z, rescale_x, rescale_y, rescale_z, x_label, y_label, z_label, x_identifier, y_identifier, z_identifier)
paraPlot.makePlotsFromSynctoolsResults(100, paramsDict['tau'], paramsDict['fric'], paramsDict['Omeg'], w/(2.0*np.pi), 1, 1.0,
				r'$\frac{\omega\tau}{2\pi}$', r'$\gamma$', r'$\Omega$', 'tau', 'fric', 'Omeg', None, cm.coolwarm)
if dictNet['calcSynctoolsStab']:
	paraPlot.makePlotsFromSynctoolsResults(101, paramsDict['tau'], paramsDict['fric'], paramsDict['ReLambSynctools'], w/(2.0*np.pi), 1, w/(2.0*np.pi),
					r'$\frac{\omega\tau}{2\pi}$', r'$\gamma$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'tau', 'fric', 'ReLambda', None, cm.PuOr)
	paraPlot.makePlotsFromSynctoolsResults(102, paramsDict['tau'], paramsDict['fric'], paramsDict['ImLambSynctools'], w/(2.0*np.pi), 1, 1.0/w,
					r'$\frac{\omega\tau}{2\pi}$', r'$\gamma$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'tau', 'fric', 'ImLambda', None, cm.PuOr)
plt.draw(); #plt.show();

paraPlot.plotParametric(paramsDict)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
