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

dictNet={
	'Nx': 2,																	# oscillators in x-direction
	'Ny': 1,																	# oscillators in y-direction
	'mx': 1	,																	# twist/chequerboard in x-direction (depends on closed or open boundary conditions)
	'my': -999,																	# twist/chequerboard in y-direction
	'Tsim': 100,
	'topology': 'ring',															# 1d) ring, chain, 2d) square-open, square-periodic, hexagonal...
																				# 3) global, entrainOne, entrainAll, entrainPLLsHierarch, compareEntrVsMutual
	'computeFreqAndStab': True													# compute linear stability and global frequency if possible: True or False
}

dictPLL={
	'intrF': 1.0,																# intrinsic frequency in Hz
	'syncF': 1.0,																# frequency of synchronized state in Hz
	'coupK': 0.65,																# [random.uniform(0.3, 0.4) for i in range(dictNet['Nx']*dictNet['Ny'])],# coupling strength in Hz float or [random.uniform(minK, maxK) for i in range(dictNet['Nx']*dictNet['Ny'])]
	'cutFc': 0.20,																# LF cut-off frequency in Hz, None for no LF, or e.g., N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05]
	'div': 1,																	# divisor of divider (int)
	'feedback_delay': 0,														# value of feedback delay in seconds
	'feedback_delay_var': None, 												# variance of feedback delay
	'transmission_delay': 0.65, 												# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]), OR [np.random.uniform(min,max) for i in range(dictNet['Nx']*dictNet['Ny'])]
	'coup_fct_sig': lambda x: -np.cos(x),										# coupling function for PLLs with ideally filtered PD signals:
	# mixer+1sig shift: np.sin(x), mixer: np.cos(x), XOR: sawtooth(x,width=0.5), PFD: 0.5*(np.sign(x)*(1+sawtooth(1*x*np.sign(x), width=1)))
	'derivative_coup_fct': lambda x: np.sin(x)									# derivative of coupling function h
}

w 		= 2.0*np.pi*dictPLL['intrF']
tau 	= dictPLL['transmission_delay']
z 		= -1																	# eigenvalue of the perturbation mode
psi		= -np.pi																# imaginary part of complex representation of zeta in polar coordinates

h  		= dictPLL['coup_fct_sig']
hp 		= dictPLL['derivative_coup_fct']

beta 	= 0																		# choose according to choice of mx, my and the topology!

K		= 2.0*np.pi*np.arange( 0.001, 0.8, 0.6285/(2.0*np.pi) )
wc  	= 2.0*np.pi*np.arange( 0.001, 0.8, 0.6285/(2.0*np.pi) )

#OmegIn2AlphaVsFc = []; alpha = []; ReLambda = []; ImLambda = [];
OmegIn2AlphaVsFc = np.zeros([len(K), len(wc)]); alpha = np.zeros([len(K), len(wc)]); ReLambda = np.zeros([len(K), len(wc)]); ImLambda = np.zeros([len(K), len(wc)]);
for i in range(len(K)):
	dictPLL.update({'coupK': K[i]/(2*np.pi)})									# set this temporarly to one value -- in Hz
	for j in range(len(wc)):
		isRadian 	= False														# set this False to get values returned in [Hz] instead of [rad * Hz]
		dictPLL.update({'cutFc': wc[j]/(2*np.pi)})								# set this temporarly to one value -- in Hz
		sf 			= synctools.SweepFactory(dictPLL, dictNet, isRadians=isRadian)
		fsl 		= sf.sweep()
		para_mat 	= fsl.get_parameter_matrix(isRadians=isRadian)
		if len(para_mat[:,4]) > 1:
			print('Found multistability of synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dictPLL['coupK'], dictPLL['transmission_delay'], beta,')\nPick state with largest frequency!')
			index = np.argmax(para_mat[:,4], axis=0)
			OmegIn2AlphaVsFc[i,j] = 2.0*np.pi*para_mat[index,4];
			#OmegIn2AlphaVsFc.append(para_mat[index,4].tolist());
			alpha[i,j] = ((2.0*np.pi*para_mat[index,1]/para_mat[index,12])*dictPLL['derivative_coup_fct']( (-2.0*np.pi*para_mat[index,4]*para_mat[index,3]+beta)/para_mat[index,12] ))
			#alpha.append( ((para_mat[index,1]/para_mat[index,12])*dictPLL['derivative_coup_fct']( -para_mat[index,4]*para_mat[index,3]+beta)).tolist() );
			ReLambda[i,j] = para_mat[index,5]
			#ReLambda.append(para_mat[index,5].tolist());
			ImLambda[i,j] = para_mat[index,6]
			#ImLambda.append(para_mat[index,6].tolist());
		else:
			print('Found one synchronized state, Omega:', para_mat[:,4], '\tfor (alpha, tau, beta)=(', dictPLL['coupK'], dictPLL['transmission_delay'], beta,').')
			OmegIn2AlphaVsFc[i,j] = 2.0*np.pi*para_mat[:,4][0];
			#OmegIn2AlphaVsFc.append(para_mat[:,4].tolist()[0]);
			alpha[i,j] = ((2.0*np.pi*para_mat[:,1]/para_mat[:,12])*dictPLL['derivative_coup_fct']( (-2.0*np.pi*para_mat[:,4]*para_mat[:,3]+beta)/para_mat[:,12] ))[0]
			#alpha.append( ((para_mat[:,1]/para_mat[:,12])*dictPLL['derivative_coup_fct']( -para_mat[:,4]*para_mat[:,3]+beta)).tolist()[0] );
			ReLambda[i,j] = para_mat[:,5][0]
			#ReLambda.append(para_mat[:,5].tolist()[0]);
			ImLambda[i,j] = para_mat[:,6][0]
			#ImLambda.append(para_mat[:,6].tolist()[0]);

dictPLL.update({'coupK': K})													# set coupling strength key in dictPLL back to the array
dictPLL.update({'cutFc': wc})													# set coupling strength key in dictPLL back to the array

loopP1	= 'alpha'																# x-axis -- NOTE: this needs to have the same order as the loops above!
loopP2 	= 'wc'																	# y-axis	otherwise, the Omega sorting will be INCORRECT!
discrP	= None																	# does not apply to parametric plots
rescale = '2alpha'																# set this in case you want to plot against a rescaled loopP variable

paramsDict = {'h': h, 'hp': hp, 'w': w, 'K': K, 'wc': wc, 'Omeg': OmegIn2AlphaVsFc, 'alpha': alpha,
			'tau': tau, 'zeta': z, 'psi': psi, 'beta': beta, 'loopP1': loopP1, 'loopP2': loopP2, 'discrP': discrP, 'rescale': rescale}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot Omega as parameter plot in the alpha - wc plot

#		 makePlotsFromSynctoolsResults(figID, x, y,  z, rescale_x, rescale_y, rescale_z, x_label, y_label, z_label, x_identifier, y_identifier, z_identifier)
paraPlot.makePlotsFromSynctoolsResults(100, alpha, wc, OmegIn2AlphaVsFc, 2, 1.0/w, 1.0,
				r'$2\alpha$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\Omega$', 'alpha', 'wc', 'Omeg', None, cm.coolwarm)
paraPlot.makePlotsFromSynctoolsResults(101, alpha, wc, ReLambda, 2, 1.0/w, w/(2.0*np.pi),
				r'$2\alpha$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'alpha', 'wc', 'ReLambda', None, cm.PuOr)
paraPlot.makePlotsFromSynctoolsResults(102, alpha, wc, ImLambda, 2, 1.0/w, 1.0/w,
				r'$2\alpha$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'alpha', 'wc', 'ImLambda', None, cm.PuOr)
plt.draw(); #plt.show();

paraPlot.plotParametric(paramsDict)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
