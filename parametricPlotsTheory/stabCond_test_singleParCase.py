#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
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
import function_lib as fct_lib

dictNet={
	'Nx': 3,																	# oscillators in x-direction
	'Ny': 3,																	# oscillators in y-direction
	'mx': 0	,																	# twist/chequerboard in x-direction (depends on closed or open boundary conditions)
	'my': 0,																	# twist/chequerboard in y-direction
	'Tsim': 100,
	'topology': 'square-periodic',												# 1d) ring, chain, 2d) square-open, square-periodic, hexagonal...
																				# 3) global, entrainOne, entrainAll, entrainPLLsHierarch, compareEntrVsMutual
	'zeta': [-0.5, 0.25],														# real part of eigenvalue of slowest decaying perturbation mode for the set of parameters, also a fct. of tau!
	'psi': [np.pi, 0],															# real part of eigenvalue of slowest decaying perturbation
	'computeFreqAndStab': True													# compute linear stability and global frequency if possible: True or False
}

dictPLL={
	'analyzeFreq': 'max',														# choose from 'max', 'min', 'middle' --> which of up to three multistable Omega to analyze
	'intrF': 1.0,																# intrinsic frequency in Hz
	'syncF': 1.0,																# frequency of synchronized state in Hz
	'coupK': 0.4,																# [random.uniform(0.3, 0.4) for i in range(dictNet['Nx']*dictNet['Ny'])],# coupling strength in Hz float or [random.uniform(minK, maxK) for i in range(dictNet['Nx']*dictNet['Ny'])]
	'cutFc': 0.014,																# LF cut-off frequency in Hz, None for no LF, or e.g., N=9 with mean 0.015: [0.05,0.015,0.00145,0.001,0.0001,0.001,0.00145,0.015,0.05]
	'div': 1,																	# divisor of divider (int)
	'friction_coefficient': 2,													# friction coefficient of 2nd order Kuramoto models
	'feedback_delay': 0,														# value of feedback delay in seconds
	'feedback_delay_var': None, 												# variance of feedback delay
	'transmission_delay': 0.65, 												# value of transmission delay in seconds, float (single), list (tau_k) or list of lists (tau_kl): np.random.uniform(min,max,size=[dictNet['Nx']*dictNet['Ny'],dictNet['Nx']*dictNet['Ny']]), OR [np.random.uniform(min,max) for i in range(dictNet['Nx']*dictNet['Ny'])]
	# choose from coupfct.<ID>: sine, cosine, neg_sine, neg_cosine, triangular, deriv_triangular, square_wave, pfd, inverse_cosine, inverse_sine
	'coup_fct_sig': coupfct.sine,												# coupling function h(x) for PLLs with ideally filtered PD signals:
	'derivative_coup_fct': coupfct.cosine,										# derivative h'(x) of coupling function h(x)
	'inve_deriv_coup_fct': coupfct.inverse_cosine								# inverse of derivative of coupling function
}

#synctools.generate_delay_plot(dictPLL, dictNet, isRadians=False)
#sys.exit()

w 		= 2.0*np.pi*dictPLL['intrF']
wc		= 2.0*np.pi*dictPLL['cutFc']
z 		= dictNet['zeta']														# eigenvalue of the perturbation mode
psi		= dictNet['psi']														# imaginary part of complex representation of zeta in polar coordinates
fric 	= dictPLL['friction_coefficient']
tau 	= dictPLL['transmission_delay']
K		= 2.0*np.pi*dictPLL['coupK']

h  		= dictPLL['coup_fct_sig']
hp 		= dictPLL['derivative_coup_fct']

beta 	= 0#np.pi																# choose according to choice of mx, my and the topology!

fzeta = 1+np.sqrt(1-np.abs(z[0])**2)

isRadian 	= False																# set this False to get values returned in [Hz] instead of [rad * Hz]
sf 			= synctools.SweepFactory(dictPLL, dictNet, isRadians=isRadian)
fsl 		= sf.sweep()
para_mat 	= fsl.get_parameter_matrix(isRadians=isRadian)
if len(para_mat[:,4]) > 1:
	#print('Found multistability of synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dictPLL['coupK'], dictPLL['transmission_delay'], beta,')\nPick state with largest frequency!')
	if dictPLL['analyzeFreq'] == 'max':
		index = np.argmax(para_mat[:,4], axis=0)
	elif dictPLL['analyzeFreq'] == 'min':
		index = np.argmin(para_mat[:,4], axis=0)
	elif dictPLL['analyzeFreq'] == 'middle':
		index = np.where(para_mat[:,4]==sorted(para_mat[:,4])[1])[0][0]
	Omega = 2.0*np.pi*para_mat[index,4];
	#print('Picked frequency [Hz]: ', Omega[i,j]/(2.0*np.pi), '\tdivision: ', para_mat[index,12], '\tK [Hz]: ', para_mat[index,1])
	alpha    = ((2.0*np.pi*para_mat[index,1]/para_mat[index,12])*dictPLL['derivative_coup_fct']( (-2.0*np.pi*para_mat[index,4]*para_mat[index,3]+beta)/para_mat[index,12] ))
	ReLambda = para_mat[index,5]
	ImLambda = para_mat[index,6]
else:
	#print('Found one synchronized state, Omega:', para_mat[:,4], '\tfor (K, tau, beta)=(', dictPLL['coupK'], dictPLL['transmission_delay'], beta,').')
	Omega = 2.0*np.pi*para_mat[:,4][0];
	#print('Picked frequency [Hz]: ', Omega[i,j]/(2.0*np.pi), '\tdivision: ', para_mat[:,12], '\tK [Hz]: ', para_mat[:,1])
	alpha    = ((2.0*np.pi*para_mat[:,1]/para_mat[:,12])*dictPLL['derivative_coup_fct']( (-2.0*np.pi*para_mat[:,4]*para_mat[:,3]+beta)/para_mat[:,12] ))[0]
	ReLambda = para_mat[:,5][0]
	ImLambda = para_mat[:,6][0]
CondStab = []
for i in range(len(z)):
	if wc*fric**2/(2*alpha) > fzeta or wc*fric**2/(2*alpha) > 1:
		CondStab.append(1)
	else:
		CondStab.append(None)

#print('Omega', Omega, '\ttype(Omega)', type(Omega))
print('CondStab', CondStab)

solvStabCotan = fct_lib.stabFunctions('cotanFct')								# create object to solve for the mu -- use: 'realPart', 'imagPart' or 'cotanFct'
solvStabRealC = fct_lib.stabFunctions('realPart')								# create object to solve for the mu -- use: 'realPart', 'imagPart' or 'cotanFct'
solvStabImagC = fct_lib.stabFunctions('imagPart')								# create object to solve for the mu -- use: 'realPart', 'imagPart' or 'cotanFct'

muQuad = [];																	# create containers that hold the different results
muBFcota_initQuad = []; muBFreal_initQuad = []; muBFimag_initQuad = [];
muBFcota_initIndi = []; muBFreal_initIndi = []; muBFimag_initIndi = [];
testresMuQuadBFcota = []; testresMuQuadBFreal = []; testresMuQuadBFimag = [];
testresiniIndBFcota = []; testresiniIndBFreal = [];	testresiniIndBFimag = [];
testres_muQuadplot = [];

muBFcota_initQuad  = np.zeros([len(z), 4])
muBFreal_initQuad  = np.zeros([len(z), 4])
muBFimag_initQuad  = np.zeros([len(z), 4])
for i in range(len(z)):															# calculate numerically the mu using the solutions to the quadratic equation as initial guesses
	tempQuad = paraPlot.equationMu(tau, alpha, wc, z[i], fric)					# obtain Mu from quadratic equation
	print(tempQuad)
	for j in range(len(tempQuad)):
		if not np.isnan(tempQuad[j]):
			#muBFcota_initQuad[i,j] = solvStabCotan.solveRootsFct(tau, z[i], psi[i], alpha, wc, fric, tempQuad[j])
			muBFreal_initQuad[i,j] = solvStabRealC.solveRootsFct(tau, z[i], psi[i], alpha, wc, fric, tempQuad[j])
			muBFimag_initQuad[i,j] = solvStabImagC.solveRootsFct(tau, z[i], psi[i], alpha, wc, fric, tempQuad[j])
		else:
			#muBFcota_initQuad[i,j] = np.nan
			muBFreal_initQuad[i,j] = np.nan
			muBFimag_initQuad[i,j] = np.nan
	muQuad.append(tempQuad)

muQuadplot = np.array(muQuad)
muQuad 	   = np.concatenate(muQuad).tolist()

intervalNumbPiHalf = 10
scanRegimeReal = np.zeros(len(z)); scanRegimeImag = np.zeros(len(z));
muBFcota_initIndi  = np.zeros([len(z), intervalNumbPiHalf])
muBFreal_initIndi  = np.zeros([len(z), intervalNumbPiHalf])
muBFimag_initIndi  = np.zeros([len(z), intervalNumbPiHalf])
for i in range(len(z)):
	scanRegimeReal[i] = np.sqrt(alpha*wc)*(1.0+np.abs(z[i])/2.0)
	scanRegimeImag[i] = np.abs(z[i])*alpha / fric
	for j in range(intervalNumbPiHalf):											# calculate numerically the mu using multiples of pi/2 as initial guesses
		#muBFcota_initIndi[i,j] = solvStabCotan.solveRootsFct(tau, z[i], psi[i], alpha, wc, fric, j/2*np.pi)
		muBFreal_initIndi[i,j] = solvStabRealC.solveRootsFct(tau, z[i], psi[i], alpha, wc, fric, j*scanRegimeReal[i]/intervalNumbPiHalf)
		muBFimag_initIndi[i,j] = solvStabImagC.solveRootsFct(tau, z[i], psi[i], alpha, wc, fric, j*scanRegimeImag[i]/intervalNumbPiHalf)
print('Rethink! At the moment only one solution is returned!')

dmu = 0.01; min_mu = -10; max_mu = 10;
fct_lib.plotRealCharSigZero(dmu, min_mu, max_mu, muBFreal_initIndi, muBFimag_initIndi, muQuadplot, tau, z, psi, alpha, wc, fric, intervalNumbPiHalf, scanRegimeReal)
fct_lib.plotImagCharSigZero(dmu, min_mu, max_mu, muBFreal_initIndi, muBFimag_initIndi, muQuadplot, tau, z, psi, alpha, wc, fric, intervalNumbPiHalf, scanRegimeImag)

for i in range(len(z)):
	for j in range(len(muBFreal_initQuad[i,:])):								# test with the conditions all solutions mu obtained from real and imaginary part with initial conditions from quad. fct.
		#testresMuQuadBFcota.append( paraPlot.stabilityTest(tau, z[i], psi[i], alpha, wc, fric, muBFcota_initQuad[i,j]) )
		testresMuQuadBFreal.append( paraPlot.stabilityTest(tau, z[i], psi[i], alpha, wc, fric, muBFreal_initQuad[i,j]) )
		testresMuQuadBFimag.append( paraPlot.stabilityTest(tau, z[i], psi[i], alpha, wc, fric, muBFimag_initQuad[i,j]) )

	for j in range(len(muBFreal_initIndi[i,:])):								# test with the conditions all solutions mu obtained from real and imaginary part with initial conditions in the regime where intersection can exist
		#testresiniIndBFcota.append( paraPlot.stabilityTest(tau, z[i], psi[i], alpha, wc, fric, muBFcota_initIndi[i,j]) )
		testresiniIndBFreal.append( paraPlot.stabilityTest(tau, z[i], psi[i], alpha, wc, fric, muBFreal_initIndi[i,j]) )
	for j in range(len(muBFimag_initIndi[i,:])):
		testresiniIndBFimag.append( paraPlot.stabilityTest(tau, z[i], psi[i], alpha, wc, fric, muBFimag_initIndi[i,j]) )

	for j in range(len(muQuadplot[i,:])):
		testres_muQuadplot.append( paraPlot.stabilityTest(tau, z[i], psi[i], alpha, wc, fric, muQuadplot[i,j]) )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ print results from both analysis, brute force numerical and using the conditions

print('\n\nResults from synctools numerical stability analysis:', para_mat)
print('The case that was tested: {Omega, tau, K, gamma, wc, v, alpha, zeta, psi} resulted in {ReLambda, ImLambda}: \n{', Omega, ', ', tau , ', ', K, ', ', fric, ', ', wc, ', ',
dictPLL['div'], ', ', alpha, ', ', z, ', ', psi, '} \n--> {', ReLambda, ', ', ImLambda, '}')
print('muQuadplot', muQuadplot)
print('Case real part, initial quad. fct.! Results from checking conditions:', testresMuQuadBFreal, '\tfor mu-values: ', muBFreal_initQuad)
print('Case imag part, initial quad. fct.! Results from checking conditions:', testresMuQuadBFimag, '\tfor mu-values: ', muBFimag_initQuad)
print('Case real part, initial individual! Results from checking conditions:', testresiniIndBFreal, '\tfor mu-values: ', muBFreal_initIndi)
print('Case imag part, initial individual! Results from checking conditions:', testresiniIndBFimag, '\tfor mu-values: ', muBFreal_initIndi)
print('Case mu solutions from quad. fct.!  Results from checking conditions:', testres_muQuadplot,  '\tfor mu-values: ', muQuadplot)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.draw();
plt.show();
