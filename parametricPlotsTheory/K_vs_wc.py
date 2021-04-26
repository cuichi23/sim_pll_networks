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
	'my': 0,																	# twist/chequerboard in y-direction
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

beta 	= np.pi																	# choose according to choice of mx, my and the topology!

K		= 2.0*np.pi*np.arange( 0.001, 0.8, 0.006285/(2.0*np.pi) )
wc  	= 2.0*np.pi*np.arange( 0.001, 0.8, 0.006285/(2.0*np.pi) )

OmegInKvsFc = []; alpha = []; ReLambda = []; ImLambda = [];
for i in range(len(K)):
	dictPLL.update({'coupK': K[i]/(2*np.pi)})									# set this temporarly to one value -- in Hz
	for j in range(len(wc)):
		isRadian 	= False														# set this False to get values returned in [Hz] instead of [rad * Hz]
		dictPLL.update({'cutFc': wc[j]/(2*np.pi)})								# set this temporarly to one value -- in Hz
		sf 			= synctools.SweepFactory(dictPLL, dictNet, isRadians=isRadian)
		fsl 		= sf.sweep()
		para_mat 	= fsl.get_parameter_matrix(isRadians=isRadian)
		OmegInKvsFc.append(para_mat[:,4]);
		if len(para_mat[:,4]) > 1:
			print('Found multistability of synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dictPLL['coupK'], tau, beta,')')
		alpha.append( (para_mat[:,1]/para_mat[:,12])*dictPLL['derivative_coup_fct']( -para_mat[:,4]*para_mat[:,3]+beta ) );
		ReLambda.append(para_mat[:,5]);
		ImLambda.append(para_mat[:,6]);

dictPLL.update({'coupK': K/(2*np.pi)})											# set coupling strength key in dictPLL back to the array
dictPLL.update({'cutFc': wc/(2*np.pi)})											# set coupling strength key in dictPLL back to the array

loopP1	= 'K'																	# x-axis
loopP2 	= 'wc'																	# y-axis
discrP	= None																	# does not apply to parametric plots
rescale = 'K_to_2alpha'															# set this in case you want to plot against a rescaled loopP variable

paramsDict = {'h': h, 'hp': hp, 'w': w, 'K': K, 'wc': wc, 'Omeg': OmegInKvsFc, 'alpha': alpha,
			'tau': tau, 'zeta': z, 'psi': psi, 'beta': beta, 'loopP1': loopP1, 'loopP2': loopP2, 'discrP': discrP}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot Omega as parameter plot in the K tau plot

#		 makePlotsFromSynctoolsResults(figID, x, y,  z, rescale_x, rescale_y, rescale_z, x_label, y_label, z_label, x_identifier, y_identifier, z_identifier)
paraPlot.makePlotsFromSynctoolsResults(100, K, wc, OmegInKvsFc, 1.0/w, 1.0/w, 1.0,
				r'$\frac{K}{\omega}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\Omega$', 'K', 'wc', 'Omeg', None, cm.coolwarm)
paraPlot.makePlotsFromSynctoolsResults(101, K, wc, ReLambda, 1.0/w, 1.0/w, w/(2.0*np.pi),
				r'$\frac{K}{\omega}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'K', 'wc', 'ReLambda', None, cm.PuOr)
paraPlot.makePlotsFromSynctoolsResults(102, K, wc, ImLambda, 1.0/w, 1.0/w, 1.0/w,
				r'$\frac{K}{\omega}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'K', 'wc', 'ImLambda', None, cm.PuOr)
plt.draw(); plt.show();

paraPlot.plotParametric(paramsDict, dictPLL, dictNet)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
