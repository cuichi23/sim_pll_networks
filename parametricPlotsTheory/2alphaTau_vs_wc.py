#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import datetime
import os, gc, sys
now = datetime.datetime.now()

# plot parameter
axisLabel = 50;
titleLabel= 10;
dpi_val   = 150;
figwidth  = 10;
figheight = 5;

import parametricPlots as paraPlot
import synctools_interface_lib as synctools
import coupling_fct_lib as coupfct

dictNet={
	'Nx': 2,																	# oscillators in x-direction
	'Ny': 1,																	# oscillators in y-direction
	'mx': 1	,																	# twist/chequerboard in x-direction (depends on closed or open boundary conditions)
	'my': -999,																	# twist/chequerboard in y-direction
	'Tsim': 100,
	'topology': 'ring',															# 1d) ring, chain, 2d) square-open, square-periodic, hexagonal...
																				# 3) global, entrainOne, entrainAll, entrainPLLsHierarch, compareEntrVsMutual
	'zeta': -1, 																# real part of eigenvalue of slowest decaying perturbation mode for the set of parameters, also a fct. of tau!
	'psi': np.pi,																# real part of eigenvalue of slowest decaying perturbation
	'computeFreqAndStab': True													# compute linear stability and global frequency if possible: True or False
}

dictPLL={
	'intrF': 1.0,																# intrinsic frequency in Hz
	'syncF': 1.0,																# frequency of synchronized state in Hz
	'coupK': 0.65,																# [random.uniform(0.3, 0.4) for i in range(dictNet['Nx']*dictNet['Ny'])],# coupling strength in Hz float or [random.uniform(minK, maxK) for i in range(dictNet['Nx']*dictNet['Ny'])]
	'cutFc': 0.20,																# LF cut-off frequency in Hz, None for no LF, or e.g., N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05]
	'div': 1,																	# divisor of divider (int)
	'friction_coefficient': 1,													# friction coefficient of 2nd order Kuramoto models
	'fric_coeff_PRE_vs_PRR': 'PRE',												# 'PRR': friction coefficient multiplied to instant. AND intrin. freq, 'PRE': friction coefficient multiplied only to instant. freq
	'feedback_delay': 0,														# value of feedback delay in seconds
	'feedback_delay_var': None, 												# variance of feedback delay
	'transmission_delay': 0.65, 												# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]), OR [np.random.uniform(min,max) for i in range(dictNet['Nx']*dictNet['Ny'])]
	# choose from coupfct.<ID>: sine, cosine, neg_sine, neg_cosine, triangular, deriv_triangular, square_wave, pfd
	'coup_fct_sig': coupfct.neg_cosine,											# coupling function h(x) for PLLs with ideally filtered PD signals:
	'derivative_coup_fct': coupfct.sine											# derivative h'(x) of coupling function h(x)
}

print('Before use: debug plotting! Somehow the plots DO NOT represent the results computed, there must be an ERROR.'); sys.exit()

w 		= 2.0*np.pi*dictPLL['intrF']
K 		= dictPLL['coupK']
z 		= dictNet['zeta']														# eigenvalue of the perturbation mode
psi		= dictNet['psi']														# imaginary part of complex representation of zeta in polar coordinates

h  		= dictPLL['coup_fct_sig']
hp 		= dictPLL['derivative_coup_fct']

beta 	= np.pi																		# choose according to choice of mx, my and the topology!

tau 	= np.arange(0, 5, 0.1)
wc  	= 2.0*np.pi*np.arange( 0.001, 0.8, 0.6285/(2.0*np.pi) )

allSol = []
OmegIn2AlphaTauVsFc = np.zeros([len(tau), len(wc)]); alpha = np.zeros([len(tau), len(wc)]); ReLambda = np.zeros([len(tau), len(wc)]); ImLambda = np.zeros([len(tau), len(wc)]);
for i in range(len(tau)):
	dictPLL.update({'transmission_delay': tau[i]})								# set this temporarly to one value -- in Hz
	for j in range(len(wc)):
		isRadian 	= False														# set this False to get values returned in [Hz] instead of [rad * Hz]
		dictPLL.update({'cutFc': wc[j]/(2*np.pi)})								# set this temporarly to one value -- in Hz
		sf 			= synctools.SweepFactory(dictPLL, dictNet, isRadians=isRadian)
		fsl 		= sf.sweep()
		para_mat 	= fsl.get_parameter_matrix(isRadians=isRadian)
		if len(para_mat[:,4]) > 1:
			#print('Found multistability of synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dictPLL['coupK'], dictPLL['transmission_delay'], beta,')\nPick state with largest frequency!')
			index = np.argmax(para_mat[:,4], axis=0)
			OmegIn2AlphaTauVsFc[i,j] = 2.0*np.pi*para_mat[index,4];
			alpha[i,j] = ((2.0*np.pi*para_mat[index,1]/para_mat[index,12])*dictPLL['derivative_coup_fct']( (-2.0*np.pi*para_mat[index,4]*para_mat[index,3]+beta)/para_mat[index,12] ))
			ReLambda[i,j] = para_mat[index,5]
			ImLambda[i,j] = para_mat[index,6]
		else:
			#print('Found one synchronized state, Omega:', para_mat[:,4], '\tfor (alpha, tau, beta)=(', dictPLL['coupK'], dictPLL['transmission_delay'], beta,').')
			OmegIn2AlphaTauVsFc[i,j] = 2.0*np.pi*para_mat[:,4][0];
			alpha[i,j] = ((2.0*np.pi*para_mat[:,1]/para_mat[:,12])*dictPLL['derivative_coup_fct']( (-2.0*np.pi*para_mat[:,4]*para_mat[:,3]+beta)/para_mat[:,12] ))[0]
			ReLambda[i,j] = para_mat[:,5][0]
			ImLambda[i,j] = para_mat[:,6][0]
		allSol.append(para_mat)

dictPLL.update({'transmission_delay': tau})										# set coupling strength key in dictPLL back to the array
dictPLL.update({'cutFc': wc})													# set coupling strength key in dictPLL back to the array

loopP1	= 'alpha'																# x-axis -- NOTE: this needs to have the same order as the loops above!
loopP2 	= 'wc'																	# y-axis	otherwise, the Omega sorting will be INCORRECT!
discrP	= None																	# does not apply to parametric plots
rescale = '2alpha'																# set this in case you want to plot against a rescaled loopP variable

paramsDict = {'h': h, 'hp': hp, 'w': w, 'K': K, 'wc': wc, 'Omeg': OmegIn2AlphaTauVsFc, 'alpha': alpha,
			'tau': tau, 'zeta': z, 'psi': psi, 'beta': beta, 'loopP1': loopP1, 'loopP2': loopP2, 'discrP': discrP, 'rescale': rescale}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot Omega as parameter plot in the alpha - wc plot

# synctools.generate_delay_plot(dictPLL, dictNet, isRadians=False)
# #sys.exit()
#
# plt.plot(tau, alpha[:,0], 'b*');
# plt.plot(OmegIn2AlphaTauVsFc[:,0]*tau, alpha[:,0], 'r+');
# plt.xlabel(r'$\Omega\tau$ or $\tau$'); plt.ylabel(r'$\alpha$');
# plt.draw(); plt.show()
# sys.exit()

#		 makePlotsFromSynctoolsResults(figID, x, y,  z, rescale_x, rescale_y, rescale_z, x_label, y_label, z_label, x_identifier, y_identifier, z_identifier)
paraPlot.makePlotsFromSynctoolsResults(100, alpha, wc, OmegIn2AlphaTauVsFc, 2, 1.0/w, 1.0,
				r'$2\alpha(\tau)$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\Omega$', 'alphaTau', 'wc', 'Omeg', None, cm.coolwarm)
paraPlot.makePlotsFromSynctoolsResults(101, alpha, wc, ReLambda, 2, 1.0/w, w/(2.0*np.pi),
				r'$2\alpha(\tau)$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'alphaTau', 'wc', 'ReLambda', None, cm.PuOr)
paraPlot.makePlotsFromSynctoolsResults(102, alpha, wc, ImLambda, 2, 1.0/w, 1.0/w,
				r'$2\alpha(\tau)$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'alphaTau', 'wc', 'ImLambda', None, cm.PuOr)
plt.draw(); #plt.show();

paraPlot.plotParametric(paramsDict)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
