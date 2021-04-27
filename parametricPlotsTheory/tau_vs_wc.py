#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
import numpy as np
import datetime
import os, gc
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
	'cutFc': 0.4,																# LF cut-off frequency in Hz, None for no LF, or e.g., N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05]
	'div': 1,																	# divisor of divider (int)
	'feedback_delay': 0,														# value of feedback delay in seconds
	'feedback_delay_var': None, 												# variance of feedback delay
	'transmission_delay': 0.65, 												# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]), OR [np.random.uniform(min,max) for i in range(dictNet['Nx']*dictNet['Ny'])]
	'coup_fct_sig': lambda x: -np.cos(x),										# coupling function for PLLs with ideally filtered PD signals:
	# mixer+1sig shift: np.sin(x), mixer: np.cos(x), XOR: sawtooth(x,width=0.5), PFD: 0.5*(np.sign(x)*(1+sawtooth(1*x*np.sign(x), width=1)))
	'derivative_coup_fct': lambda x: np.sin(x)									# derivative of coupling function h
}

w 		= 2.0*np.pi*dictPLL['intrF']
K		= 2.0*np.pi*dictPLL['coupK']
z 		= -1																	# eigenvalue of the perturbation mode
psi		= -np.pi																# imaginary part of complex representation of zeta in polar coordinates

h  		= dictPLL['coup_fct_sig']
hp 		= dictPLL['derivative_coup_fct']

beta 	= 0#np.pi																	# choose according to choice of mx, my and the topology!

tau 	= np.arange(0, 5, 0.01)
wc  	= 2.0*np.pi*np.arange(0.0001, 0.75, 0.006285/(2.0*np.pi))

#OmegInTauVsFc = []; alpha = []; ReLambda = []; ImLambda = [];
OmegInTauVsFc = np.zeros([len(tau), len(wc)]); alpha = np.zeros([len(tau), len(wc)]); ReLambda = np.zeros([len(tau), len(wc)]); ImLambda = np.zeros([len(tau), len(wc)]);
for i in range(len(tau)):
	dictPLL.update({'transmission_delay': tau[i]})								# set this temporarly to one value -- in seconds
	for j in range(len(wc)):
		dictPLL.update({'cutFc': wc[j]/(2*np.pi)})								# set this temporarly to one value -- in Hz
		isRadian 	= False														# set this False to get values returned in [Hz] instead of [rad * Hz]
		sf 			= synctools.SweepFactory(dictPLL, dictNet, isRadians=isRadian)
		fsl 		= sf.sweep()
		para_mat 	= fsl.get_parameter_matrix(isRadians=isRadian)
		if len(para_mat[:,4]) > 1:
			print('Found multistability of synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dictPLL['coupK'], dictPLL['transmission_delay'], beta,')\nPick state with largest frequency!')
			index = np.argmax(para_mat[:,4], axis=0)
			OmegInTauVsFc[i,j] = 2.0*np.pi*para_mat[index,4];
			#OmegInTauVsFc.append(para_mat[index,4].tolist());
			alpha[i,j] = ((2.0*np.pi*para_mat[index,1]/para_mat[index,12])*dictPLL['derivative_coup_fct']( (-2.0*np.pi*para_mat[index,4]*para_mat[index,3]+beta)/para_mat[index,12] ))
			#alpha.append( ((para_mat[index,1]/para_mat[index,12])*dictPLL['derivative_coup_fct']( -para_mat[index,4]*para_mat[index,3]+beta)).tolist() );
			ReLambda[i,j] = para_mat[index,5]
			#ReLambda.append(para_mat[index,5].tolist());
			ImLambda[i,j] = para_mat[index,6]
			#ImLambda.append(para_mat[index,6].tolist());
		else:
			print('Found one synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dictPLL['coupK'], dictPLL['transmission_delay'], beta,').')
			OmegInTauVsFc[i,j] = 2.0*np.pi*para_mat[:,4][0];
			#OmegInTauVsFc.append(para_mat[:,4].tolist()[0]);
			alpha[i,j] = ((2.0*np.pi*para_mat[:,1]/para_mat[:,12])*dictPLL['derivative_coup_fct']( (-2.0*np.pi*para_mat[:,4]*para_mat[:,3]+beta)/para_mat[:,12] ))[0]
			#alpha.append( ((para_mat[:,1]/para_mat[:,12])*dictPLL['derivative_coup_fct']( -para_mat[:,4]*para_mat[:,3]+beta)).tolist()[0] );
			ReLambda[i,j] = para_mat[:,5][0]
			#ReLambda.append(para_mat[:,5].tolist()[0]);
			ImLambda[i,j] = para_mat[:,6][0]
			#ImLambda.append(para_mat[:,6].tolist()[0]);

dictPLL.update({'transmission_delay': tau})										# set coupling strength key in dictPLL back to the array
dictPLL.update({'cutFc': wc})													# set coupling strength key in dictPLL back to the array

loopP1	= 'tau'																	# x-axis
loopP2 	= 'wc'																	# y-axis
discrP	= None																	# does not apply to parametric plots
rescale = None																	# set this in case you want to plot against a rescaled loopP variable

paramsDict = {'h': h, 'hp': hp, 'w': w, 'K': K, 'wc': wc, 'Omeg': OmegInTauVsFc, 'alpha': alpha,
			'tau': tau, 'zeta': z, 'psi': psi, 'beta': beta, 'loopP1': loopP1, 'loopP2': loopP2, 'discrP': discrP}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot Omega as parameter plot in the tau - wc plot

#		 makePlotsFromSynctoolsResults(figID, x, y,  z, rescale_x, rescale_y, rescale_z, x_label, y_label, z_label, x_identifier, y_identifier, z_identifier)
paraPlot.makePlotsFromSynctoolsResults(100, tau, wc, OmegInTauVsFc, w/(2.0*np.pi), 1.0/w, 1.0,
				r'$\frac{\omega\tau}{2\pi}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\Omega$', 'tau', 'K', 'Omeg', None, cm.coolwarm)
paraPlot.makePlotsFromSynctoolsResults(101, tau, wc, ReLambda, w/(2.0*np.pi), 1.0/w, w/(2.0*np.pi),
				r'$\frac{\omega\tau}{2\pi}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'tau', 'wc', 'ReLambda', None, cm.PuOr)
paraPlot.makePlotsFromSynctoolsResults(102, tau, wc, ImLambda, w/(2.0*np.pi), 1.0/w, 1.0/w,
				r'$\frac{\omega\tau}{2\pi}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'tau', 'wc', 'ImLambda', None, cm.PuOr)
plt.draw(); #plt.show();

paraPlot.plotParametric(paramsDict)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
