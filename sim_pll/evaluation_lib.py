#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import sys, gc
import inspect

import networkx as nx
import numpy as np
import scipy
from scipy import signal
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
import itertools
from itertools import permutations as permu
from itertools import combinations as combi
import matplotlib
import os, pickle
if not os.environ.get('SGE_ROOT') is None:										# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.ndimage.filters import uniform_filter1d
import time
import datetime
import pandas as pd

from sim_pll import integer_mult_period_signal_lib as findIntTinSig
from sim_pll import plot_lib

now = datetime.datetime.now()

''' Enable automatic carbage collector '''
gc.enable();

''' All plots in latex mode '''
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.rcParams['agg.path.chunksize'] = 10000

''' STYLEPACKS '''
titlefont = {
		'family' : 'serif',
		'color'  : 'black',
		'weight' : 'normal',
		'size'   : 9,
		}

labelfont = {
		'family' : 'sans-serif',
		'color'  : 'black',
		'weight' : 'normal',
		'size'   : 16,
		}

annotationfont = {
		'family' : 'monospace',
		'color'  : (0, 0.27, 0.08),
		'weight' : 'normal',
		'size'   : 14,
		}

def plotTest(params):

	# plot parameter
	axisLabel = 50;
	titleLabel= 10;
	dpi_val   = 150;
	figwidth  = 6;
	figheight = 3;

	labeldict 	= {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 	= {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color		= ['blue', 'red', 'purple', 'cyan', 'green', 'yellow'] #'magenta'
	linet		= ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig1.canvas.manager.set_window_title('test plot phase')  							 # plot the phase
	plt.clf()

	plt.plot(params['x'], params['y']%(2*np.pi), color=color[0], linewidth=1, linestyle=linet[0], label=labeldict1[params['label']])
	plt.plot(params['x'][params['delay_steps']-1], params['y'][int(params['delay_steps'])-1,0]+0.05,'go')

	plt.xlabel(labeldict1[params['xlabel']], fontdict = labelfont)
	plt.ylabel(labeldict1[params['ylabel']], fontdict = labelfont)
	plt.legend()

	fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig2.canvas.manager.set_window_title('test plot frequency')							# plot the instantaneous frequency
	plt.clf()

	plt.plot(params['x'][0:-1], np.diff(params['y'], axis=0)/(2*np.pi*params['dt']), color=color[0], linewidth=1, linestyle=linet[0], label=r'f(t)')

	plt.xlabel(labeldict1[params['xlabel']], fontdict = labelfont)
	plt.ylabel(r'$f(t)$ Hz', fontdict = labelfont)
	plt.legend()

	plt.draw()
	plt.show()

################################################################################


def prepareDictsForPlotting(dictPLL, dictNet):

	if dictPLL['cutFc'] is None:
		dictPLL.update({'cutFc': np.inf})

	if not np.abs(np.min(dictPLL['intrF'])) > 1E-17:									# for f=0, there would otherwies be a float division by zero
		dictPLL.update({'intrF': 1})
		print('Since intrinsic frequency was zero: for plotting set to one to generate boundaries!')


	return dictPLL, dictNet


def saveDictionaries(dictToSave, name, K, tau, Fc, Nx, Ny, mx, my, topology):

	if Fc is None:
		Fc = np.inf

	N = int(Nx*Ny)
	filename = 'results/%s_K%.3f_tau%.3f_Fc%.3f_mx%i_my%i_N%i_topo%s_%d:%d_%d_%d_%d'%(name, np.mean(K), np.mean(tau), np.mean(Fc), mx, my, N, topology, now.hour, now.minute, now.year, now.month, now.day)
	f 		 = open(filename,'wb')
	pickle.dump(dictToSave,f, protocol=4)
	f.close()

	return None

################################################################################

def calculateEigenvalues(dictNet, dictPLL):
	''' Calculate eigenvalues zeta for networks of homogeneous PLL '''

	if dictNet['topology'] == 'global':											# test whether global coupling topology
		print('All to all coupling topology identified!')
		zeta = 1/(dictNet['Nx']*dictNet['Ny']-1)
		dictNet.update({'zeta': zeta})

	# if dictNet['Ny'] == 1:														# check whether 2D or 1D topology
	# 	print('1d network topology identified!')
	# 	if dictNet['topology'] == 'ring':
	# 		zeta = 1
	# 		dictNet.update({'zeta': zeta})
	# 	elif dictNet['topology'] == 'chain':
	# 		if dictNet['mx'] == 0:
	# 			zeta = 1
	# 			dictNet.update({'zeta': zeta})
	# 		elif dictNet['mx'] > 0:
	# 			zeta = np.cos(np.arange(0,dictNet['Nx'])*np.pi/(dictNet['Nx']-1))
	# 			dictNet.update({'zeta': zeta})
	# 	else:
	# 		print('Coupling topology not yet implemented, add expression for eigenvalues or brute-force solve!')
	#
	# elif dictNet['Ny'] > 1:
	# 	print('2d network topology identified!')
	# 	if dictNet['topology'] == 'square-open':
	# 		zeta = 1
	# 		dictNet.update({'zeta': zeta})
	# 	elif dictNet['topology'] == 'square-periodic':
	# 		zeta = 1
	# 		dictNet.update({'zeta': zeta})
	# 	else:
	# 		print('Coupling topology not yet implemented, add expression for eigenvalues or brute-force solve!')
	#
	return dictNet, dictPLL

################################################################################

''' CALCULATE SPECTRUM '''
def calcSpectrum( phi, dictPLL, dictNet, psd_id=0, percentOfTsim=0.5 ): #phi,Fsample,couplingfct,waveform=None,expectedFreq=-999,evalAllRealizations=False,decayTimeSlowestMode=None

	Pxx_dBm=[]; Pxx_dBV=[]; f=[];
	try:
		windowset='boxcar' # here we choose boxcar since a modification of the ends of the time-series is not necessary for an integer number of periods
		print('Trying to cut integer number of periods! Inside calcSpectrum.')
		if dictPLL['extra_coup_sig'] is None:
			analyzeL = findIntTinSig.cutTimeSeriesOfIntegerPeriod(dictPLL['sampleF'], dictNet['Tsim'], dictPLL['transmission_delay'], dictPLL['syncF'],
																np.max(dictPLL['coupK']), phi, psd_id, percentOfTsim)
		else:
			analyzeL = findIntTinSig.cutTimeSeriesOfIntegerPeriod(dictPLL['sampleF'], dictNet['Tsim'], dictPLL['transmission_delay'], dictPLL['syncF'],
																np.max([np.max(dictPLL['coupK']), np.max(dictPLL['coupStr_2ndHarm'])]), phi, psd_id, percentOfTsim)
	except:
		windowset='hamming' 													#'hamming' #'hamming', 'boxcar'
		print('\n\nError in cutTimeSeriesOfIntegerPeriod-function! Not picking integer number of periods for PSD! Using window %s!\n\n'%windowset)
		analyzeL= [ int( dictNet['Tsim']*dictPLL['sampleF']*(1-percentOfTsim) ), int( dictNet['Tsim']*dictPLL['sampleF'] )-1 ]

	window = scipy.signal.get_window(windowset, analyzeL[1]-analyzeL[0], fftbins=True);
	#print('Length window:', len(window), '\tshape window:', np.shape(window))
	print('\nCurrent window option is', windowset, 'for waveform', inspect.getsourcelines(dictPLL['PSD_from_signal'])[0][0],
			'NOTE: in principle can always choose to be sin() for cleaner PSD in first harmonic approximation of the signal.')
	print('Calculate spectrum for',percentOfTsim,'percent of the time-series. Implement better solution using decay times.')

	tsdata		= dictPLL['PSD_from_signal'](phi[analyzeL[0]:analyzeL[1]])
	#print('Length tsdata:', len(tsdata), '\tshape tsdata:', np.shape(tsdata))

	ftemp, Vxx 	= scipy.signal.periodogram(tsdata, dictPLL['sampleF'], return_onesided=True, window=window, scaling='density', axis=0) #  returns Pxx with dimensions [V^2] if scaling='spectrum' and [V^2/Hz] if if scaling='density'
	P0 = 1E-3; R=50; 															# for P0 in [mW/Hz] and R [ohm]

	Pxx_dBm.append( 10*np.log10((Vxx/R)/P0) )
	f.append( ftemp )

	return f, Pxx_dBm

################################################################################

def rotate_phases(phi0, isInverse=False):
	''' Rotates the phases such that the phase space direction phi_0 is rotated onto the main diagonal of the n dimensional phase space

		Author: Daniel Platz

	Parameters
	----------
	phi  :  np.array
			array of phases
	isInverse  :  bool
				  if True: returns coordinates of physical phase space in terms of the rotated coordinate system
				  (implies that isInverse=True gives you the coordinates in the rotated system)

	Returns
	-------
	phi_0_rotated  :  np.array
					  phases in rotated or physical phase space '''

	# Determine rotation angle
	n = len(phi0)
	if n <= 1:
		print('ERROR, 1d value cannot be rotated!')

	alpha = -np.arccos(1.0 / np.sqrt(n))

	# Construct rotation matrix
	v = np.zeros((n, n))
	v[0, 0] = 1.0
	v[1:, 1:] = 1.0 / float(n - 1)
	w = np.zeros((n, n))
	w[1:, 0] = -1.0 / np.sqrt(n - 1)
	w[0, 1:] = 1.0 / np.sqrt(n - 1)
	r = np.identity(n) + (np.cos(alpha) - 1) * v + np.sin(alpha) * w			# for N=3 -->
	# print('---------------------------------------')
	# print('---------------------------------------')
	# print(v)
	# print('---------------------------------------')
	# print(w)
	# print('---------------------------------------')
	# print(r)

	# Apply rotation matrix
	if not isInverse:															# if isInverse==False, this condition is True
		return np.dot(r, phi0)													# transform input into rotated phase space
	else:
		return np.dot(np.transpose(r), phi0)									# transform back into physical phase space

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_dphi_matrix(n):
	m = np.zeros((n * (n - 1), n))
	x0 = 0
	x1 = 0
	for i in range(n * (n - 1)):
		x0 = int(np.floor(i / float(n - 1)))
		m[i, x0] = 1
		if x1 == x0:
			x1 += 1
			x1 = np.mod(x1, n)

		m[i, x1] = -1
		x1 += 1
		x1 = np.mod(x1, n)
	return m

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_d_matrix(n):
	'''Constructs a matrix to compute the phase differences from a
	   vector of non-rotated phases'''
	d = np.zeros((n, n))
	for i in range(n):
		d[i, i] = -1
		d[i, np.mod(i + 1, n)] = 1
	return d

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PhaseDifferenceCell(object):
	def __init__(self, n):
		self.n = n
		self.dphi_matrix = get_dphi_matrix(n)
		self.d_min = -np.pi
		self.d_max = np.pi
		self.d = get_d_matrix(n)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	def is_inside(self, x, isRotated=False):
		# Check if vector has the correct length
		if len(x) != self.n:
			raise Exception('Vector has not the required length n.')

		# Rotate back to initial coordinate system if required
		if isRotated:
			x_tmp = rotate_phases(x, isInverse=False)
		else:
			x_tmp = x

		# Map to phase difference space
		dphi = np.dot(self.d, x_tmp)

		is_inside = True
		for i in range(len(dphi) - 1):
			if np.abs(dphi[i]) > self.d_max:
				is_inside = False
				break

		return is_inside

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	def is_inside_old(self, x, isRotated=False):
		'''Checks if a vector is inside the phase difference unit cell.

		Parameters
		----------
		x  :  np.array
				coordinate vector of length n which is the number of non-reduced dimensions
		isRotated  :  boolean
						True if x is given in rotated coordinates

		Returns
		-------
		is_inside  :  boolean
						True if point is inside unit cell
		'''
		# Check if vector has the correct length
		if len(x) != self.n:
			raise Exception('Vector has not the required length n.')

		# Rotate back to initial coordinate system if required
		if isRotated:
			x_tmp = rotate_phases(x, isInverse=False)
		else:
			x_tmp = x

		# Map to phase difference space
		d = np.dot(self.dphi_matrix, x_tmp)

		is_inside = True
		for i in range(len(d)):
			if d[i] < self.d_min:
				is_inside = False
				break
			if d[i] > self.d_max:
				is_inside = False
				break

		return is_inside

################################################################################

''' GET FILTER STATUS IN SYNCHRONISED STATE '''
def getFilterStatus(F,K,Fc,delay,Fsim,Tsim):
	dt = 1.0/Fsim
	Nsteps = int(Tsim*Fsim)
	delay_steps = int(delay/dt)
	pll_list = [ PhaseLockedLoop(
					Delayer(delay,dt),
					PhaseDetectorCombiner(idx_pll,[(idx_pll+1)%2]),
					LowPass(Fc,dt,y=0),
					VoltageControlledOscillator(F,K,dt,c=0,phi=0)
					)  for idx_pll in range(2) ]
	_  = simulatePhaseModel(Nsteps,2,pll_list)
	return pll_list[0].low_pass_filter.control_signal

################################################################################

''' MODEL FITTING: DEMIR MODEL '''
def fitModelDemir(f_model,d_model,fitrange=0):

	f_peak = f_model[np.argmax(d_model)]										# find main peak

	if fitrange != 0:															# mask data
		ma = np.ma.masked_inside(f_model, f_peak-fitrange, f_peak+fitrange)
		f_model_ma = f_model[ma.mask]
		d_model_ma = d_model[ma.mask]
	else:
		f_model_ma = f_model
		d_model_ma = d_model

	A = np.sqrt(2)																# calculate power of main peak for sine wave
	P_offset = 10*np.log10(A**2/2)

	optimize_func = lambda p: P_offset + 10*np.log10( (p[0]**2 * p[1])/(np.pi * p[0]**4 * p[1]**2 + (f_model_ma-p[0])**2 )) # model fit
	error_func = lambda p: optimize_func(p) - d_model_ma
	p_init = (f_peak, 1e-8)
	p_final,success = leastsq(error_func, p_init[:])

	f_model_ma = f_model														# restore data
	d_model_ma = d_model

	return f_model, optimize_func(p_final), p_final

################################################################################
################################################################################
################################################################################

def obtainOrderParam(dictPLL, dictNet, dictData):
	''' MODIFIED KURAMOTO ORDER PARAMETERS '''
	numb_av_T = 3																			   	# number of periods of free-running frequencies to average over
	if np.min(dictPLL['intrF']) > 0:														 	# for f=0, there would otherwise be a float division by zero
		F1=np.min(dictPLL['intrF'])
	else:
		F1=np.min(dictPLL['intrF'])+1E-3

	if dictNet['topology'] == "square-periodic" or dictNet['topology'] == "hexagon-periodic" or dictNet['topology'] == "octagon-periodic":
		r = oracle_mTwistOrderParameter2d(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):, :], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'])
		orderparam = oracle_mTwistOrderParameter2d(dictData['phi'][:, :], dictNet['Nx'], dictNet['Ny'], dictNet['mx'], dictNet['my'])
	elif dictNet['topology'] == "square-open" or dictNet['topology'] == "hexagon" or dictNet['topology'] == "octagon":
		if dictNet['mx']==1 and dictNet['my']==1:
			ktemp=2
		elif dictNet['mx']==1 and dictNet['my']==0:
			ktemp=0
		elif dictNet['mx']==0 and dictNet['my']==1:
			ktemp=1
		elif dictNet['mx']==0 and dictNet['my']==0:
			ktemp=3
		else:
			ktemp=4
		"""
				ktemp == 0 : x  checkerboard state
				ktemp == 1 : y  checkerboard state
				ktemp == 2 : xy checkerboard state
				ktemp == 3 : in-phase synchronized
			"""
		r = oracle_CheckerboardOrderParameter2d(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):, :], dictNet['Nx'], dictNet['Ny'], ktemp)
		# ry = np.nonzero(rmat > 0.995)[0]
		# rx = np.nonzero(rmat > 0.995)[1]
		orderparam = oracle_CheckerboardOrderParameter2d(dictData['phi'][:, :], dictNet['Nx'], dictNet['Ny'], ktemp)
	elif dictNet['topology'] == "compareEntrVsMutual":
		rMut 	 = oracle_mTwistOrderParameter(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):, 0:2], dictNet['mx']);
		orderMut = oracle_mTwistOrderParameter(dictData['phi'][:, 0:2], dictNet['mx']);
		rEnt 	 = oracle_mTwistOrderParameter(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):, 2:4], dictNet['mx']);
		orderEnt = oracle_mTwistOrderParameter(dictData['phi'][:, 2:4], dictNet['mx']);
		if isPlottingTimeSeries:
			figwidth  = 6; figheight = 5; t = np.arange(dictData['phi'].shape[0]); now = datetime.datetime.now();
			fig0 = plt.figure(num=0, figsize=(figwidth, figheight), dpi=150, facecolor='w', edgecolor='k')
			fig0.canvas.manager.set_window_title('order parameters mutual and entrained')			   # plot orderparameter
			plt.clf()
			plt.plot((dictData['t']*dictPLL['dt']), orderMut,'b-',  label='2 mutual coupled PLLs' )
			plt.plot((dictData['t']*dictPLL['dt']), orderEnt,'r--', label='one entrained PLL')
			plt.plot(dictPLL['transmission_delay'], orderMut[int(round(dictPLL['transmission_delay']/dictPLL['dt']))], 'yo', ms=5)						   # mark where the simulation starts
			plt.axvspan(dictData['t'][-int(2*1.0/(F1*dictPLL['dt']))]*dictPLL['dt'], dictData['t'][-1]*dictPLL['dt'], color='b', alpha=0.3)
			plt.xlabel(r'$t$ $[s]$'); plt.legend();
			plt.ylabel(r'$R( t,m = %d )$' % dictNet['mx'])
			plt.savefig('results/orderparam_mutual_entrained_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
			plt.savefig('results/orderparam_mutual_entrained_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
			r = np.zeros(len(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):,0]))
			orderparam = np.zeros(len(dictData['phi'][:, 0]))
	elif dictNet['topology'] == "chain":
		"""
				dictNet['mx']  > 0 : x  checkerboard state
				dictNet['mx'] == 0 : in-phase synchronized
			"""
		r = oracle_CheckerboardOrderParameter1d(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):, :], dictNet['mx'])
		orderparam = oracle_CheckerboardOrderParameter1d(dictData['phi'][:, :])							# calculate the order parameter for all times
	elif ( dictNet['topology'] == "ring" or dictNet['topology'] == 'global'):
		# print('Calculate order parameter for ring or global topology. For phases: ', dictData['phi'])
		time.sleep(5)
		r = oracle_mTwistOrderParameter(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):, :], dictNet['mx'])# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
		orderparam = oracle_mTwistOrderParameter(dictData['phi'][:, :], dictNet['mx'])					# calculate the m-twist order parameter for all times
	elif "entrain" in dictNet['topology']:
		# ( dictNet['topology'] == "entrainOne" or dictNet['topology'] == "entrainAll" or dictNet['topology'] == "entrainPLLsHierarch"):
		phi_constant_expected = dictNet['phiInitConfig']
		r = calcKuramotoOrderParEntrainSelfOrgState(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):, :], phi_constant_expected);
		orderparam = calcKuramotoOrderParEntrainSelfOrgState(dictData['phi'][:, :], phi_constant_expected);
	# r = oracle_mTwistOrderParameter(dictData['phi'][-int(2*1.0/(F1*dictPLL['dt'])):, :], dictNet['mx'])			# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
	# orderparam = oracle_mTwistOrderParameter(dictData['phi'][:, :], dictNet['mx'])					# calculate the m-twist order parameter for all times
	# print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])
	print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])

	return r, orderparam, F1

################################################################################

def evaluateSimulationsChrisHoyer(poolData):

	#print('poolData', poolData[0][0])
	# poolData = load(...)														# in principle saved pool data can be loaded and plotted

	# plot parameter
	axisLabel  = 60;
	tickSize   = 35;
	titleLabel = 10;
	dpi_val	   = 150;
	figwidth   = 6;
	figheight  = 5;
	alpha 	   = 0.5;
	linewidth  = 0.5;

	unit_cell = PhaseDifferenceCell(poolData[0][0]['dictNet']['Nx']*poolData[0][0]['dictNet']['Ny'])
	threshold_statState = np.pi/15

	fig16 = plt.figure(num=16, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig16.canvas.manager.set_window_title('HF (VCO output) basin attraction plot - 2pi periodic')			# basin attraction plot
	ax16 = fig16.add_subplot(111)

	fig161 = plt.figure(num=161, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig161.canvas.manager.set_window_title('HF (VCO output) basin attraction (diamond W>w, circle W<w)')	# basin attraction plot
	ax161 = fig161.add_subplot(111)

	fig17 = plt.figure(num=17, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig17.canvas.manager.set_window_title('HF (VCO output) output basin attraction plot')			 		# basin attraction plot
	ax17 = fig17.add_subplot(111)

	fig18 = plt.figure(num=18, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig18.canvas.manager.set_window_title('LF (cross-coupling) basin attraction plot - 2pi periodic')		# basin attraction plot
	ax18 = fig18.add_subplot(111)

	fig181 = plt.figure(num=181, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig181.canvas.manager.set_window_title('LF (cross-coupling) basin attraction (diamond W>w, circle W<w)')# basin attraction plot
	ax181 = fig181.add_subplot(111)

	fig19 = plt.figure(num=19, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig19.canvas.manager.set_window_title('LF (cross-coupling) basin attraction plot')			 			# basin attraction plot
	ax19 = fig19.add_subplot(111)

	delay_steps	= int( np.floor( poolData[0][0]['dictPLL']['transmission_delay'] / poolData[0][0]['dictPLL']['dt'] ) )
	stats_init_phase_conf_final_state = np.empty([len(poolData[0][:]), 5])
	deltaThetaDivSave		= np.empty( [len(poolData[0][:]), len(poolData[0][0]['dictData']['phi'][delay_steps+1::poolData[0][0]['dictPLL']['sampleFplot'],0])-0] )
	deltaThetaDivDotSave	= np.empty( [len(poolData[0][:]), len(poolData[0][0]['dictData']['phi'][delay_steps+1::poolData[0][0]['dictPLL']['sampleFplot'],0])-0] )

	for i in range(len(poolData[0][:])):
		#print('working on realization %i results from sim:'%i, poolData[0][i]['dictNet'], '\n', poolData[0][i]['dictPLL'], '\n', poolData[0][i]['dictData'],'\n\n')

		#print('Check whether perturbation is inside unit-cell (evaluation.py)! phiS:', poolData[0][i]['dictNet']['phiPerturb'], '\tInside? True/False:', unit_cell.is_inside((poolData[0][i]['dictNet']['phiPerturb']), isRotated=False)); time.sleep(2)
		#print('How about phi, is it a key to dictData?', 'phi' in poolData[0][i]['dictData'])
		if unit_cell.is_inside((poolData[0][i]['dictNet']['phiPerturb']), isRotated=False):	# NOTE this case is for scanValues set only in -pi to pi, we so not plot outside the unit cell

			# test whether frequency is larger or smaller than mean intrinsic frequency as a first distinction between multistable synced states with the same phase relations but
			# different frequency -- for more than 3 multistable in- or anti-phase synched states that needs to be reworked
			if ( poolData[0][i]['dictData']['phi'][-1,0] - poolData[0][i]['dictData']['phi'][-2,0] ) / poolData[0][i]['dictPLL']['dt'] > np.mean( poolData[0][i]['dictPLL']['intrF'] ):
				initmarker = 'd'
			else:
				initmarker = 'o'

			deltaTheta 			= poolData[0][i]['dictData']['phi'][:,0] - poolData[0][i]['dictData']['phi'][:,1]
			deltaThetaDot		= np.diff( deltaTheta, axis=0 ) / poolData[0][i]['dictPLL']['dt']
			deltaThetaDiv 		= poolData[0][i]['dictData']['phi'][:,0]/poolData[0][i]['dictPLL']['div'] - poolData[0][i]['dictData']['phi'][:,1]/poolData[0][i]['dictPLL']['div']
			deltaThetaDivDot	= np.diff( deltaThetaDiv, axis=0 ) / poolData[0][i]['dictPLL']['dt']

			if np.abs( np.abs( (deltaTheta[-1]+np.pi)%(2.*np.pi)-np.pi ) - np.pi ) < threshold_statState:
				color = 'r'														# anti-phase
			elif np.abs( (deltaTheta[-1]+np.pi)%(2.*np.pi)-np.pi ) - 0.0 < threshold_statState:
				color = 'b'														# in-phase
			else:
				color = 'k'														# neither in- nor anti-phase


			# plot for HF output
			ax16.plot((deltaTheta[delay_steps+1::poolData[0][0]['dictPLL']['sampleFplot']]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot[delay_steps::poolData[0][0]['dictPLL']['sampleFplot']], '-', color=color, alpha=alpha, linewidth=linewidth)	 	# plot trajectory
			ax16.plot((deltaTheta[delay_steps]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot[delay_steps], 'o', color=color, alpha=alpha)	 		# plot initial dot
			ax16.plot((deltaTheta[-1]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot[-1], 'x', color=color, alpha=alpha)						 	# plot final state cross
			#plot_lib.deltaThetaDot_vs_deltaTheta(poolData[0][i]['dictPLL'], poolData[0][i]['dictNet'], (deltaTheta[1:]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot, color, alpha)
			ax17.plot(deltaTheta[delay_steps+1::poolData[0][0]['dictPLL']['sampleFplot']], deltaThetaDot[delay_steps::poolData[0][0]['dictPLL']['sampleFplot']], '-', color=color, alpha=alpha, linewidth=linewidth)		# plot trajectory
			ax17.plot(deltaTheta[delay_steps], deltaThetaDot[delay_steps], 'o', color=color, alpha=alpha)			# plot initial dot
			ax17.plot(deltaTheta[-1], deltaThetaDot[-1], 'x', color=color, alpha=alpha)							# plot final state cross

			ax161.plot((deltaTheta[0]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot[0], initmarker, color=color, alpha=alpha) # plot initial dot

			if np.abs( np.abs( (deltaThetaDiv[-1]+np.pi)%(2.*np.pi)-np.pi ) - np.pi ) < threshold_statState:
				color = 'r'
				stats_init_phase_conf_final_state[i,4] = np.pi
			elif np.abs( (deltaThetaDiv[-1]+np.pi)%(2.*np.pi)-np.pi ) - 0.0 < threshold_statState:
				color = 'b'
				stats_init_phase_conf_final_state[i,4] = 0.0
			else:
				color = 'k'
				stats_init_phase_conf_final_state[i,4] = -999

			stats_init_phase_conf_final_state[i,0] = deltaThetaDiv[delay_steps] 							# save initial phase difference
			stats_init_phase_conf_final_state[i,1] = deltaThetaDivDot[delay_steps]							# save initial freq. difference
			stats_init_phase_conf_final_state[i,2] = deltaThetaDiv[-1] 										# save final phase difference
			stats_init_phase_conf_final_state[i,3] = deltaThetaDivDot[-1]									# save final freq. difference

			# plot for LF output
			ax18.plot((deltaThetaDiv[delay_steps+1::poolData[0][0]['dictPLL']['sampleFplot']]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[delay_steps::poolData[0][0]['dictPLL']['sampleFplot']], '-', color=color, alpha=alpha, linewidth=linewidth)	# plot trajectory
			ax18.plot((deltaThetaDiv[delay_steps]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[delay_steps], 'o', color=color, alpha=alpha)		# plot initial dot
			ax18.plot((deltaThetaDiv[-1]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[-1], 'x', color=color, alpha=alpha)							# plot final state cross
			#plot_lib.deltaThetaDivDot_vs_deltaThetaDiv(poolData[0][i]['dictPLL'], poolData[0][i]['dictNet'], (deltaThetaDiv[1:]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot, color, alpha)
			ax19.plot(deltaThetaDiv[delay_steps+1::poolData[0][0]['dictPLL']['sampleFplot']], deltaThetaDivDot[delay_steps::poolData[0][0]['dictPLL']['sampleFplot']], '-', color=color, alpha=alpha, linewidth=linewidth)		# plot trajectory
			ax19.plot(deltaThetaDiv[delay_steps], deltaThetaDivDot[delay_steps], 'o', color=color, alpha=alpha)			# plot initial dot
			ax19.plot(deltaThetaDiv[-1], deltaThetaDivDot[-1], 'x', color=color, alpha=alpha)								# plot final state cross

			ax181.plot((deltaThetaDiv[0]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[0], initmarker, color=color, alpha=alpha) # plot initial dot

			deltaThetaDivSave[i] 	= deltaThetaDiv[delay_steps+1::poolData[0][0]['dictPLL']['sampleFplot']]
			deltaThetaDivDotSave[i] = deltaThetaDivDot[delay_steps::poolData[0][0]['dictPLL']['sampleFplot']]

	#plt.xlabel(r'$\Delta\theta(t)$')
	#plt.ylabel(r'$\Delta\dot{\theta}(t)$')
	ax16.set_xlabel(r'$\Delta\theta(t)$ mod $2\pi$', fontsize=axisLabel)
	ax16.set_ylabel(r'$\Delta\dot{\theta}(t)$', fontsize=axisLabel)
	ax17.set_xlabel(r'$\Delta\theta(t)$', fontsize=axisLabel)
	ax17.set_ylabel(r'$\Delta\dot{\theta}(t)$', fontsize=axisLabel)
	ax18.set_xlabel(r'$\Delta\theta(t)/v$ mod $2\pi$', fontsize=axisLabel)
	ax18.set_ylabel(r'$\Delta\dot{\theta}(t)/v$', fontsize=axisLabel)
	ax19.set_xlabel(r'$\Delta\theta(t)/v$', fontsize=axisLabel)
	ax19.set_ylabel(r'$\Delta\dot{\theta}(t)/v$', fontsize=axisLabel)
	ax161.set_xlabel(r'$\Delta\theta(t)$ mod $2\pi$', fontsize=axisLabel)
	ax161.set_ylabel(r'$\Delta\dot{\theta}(t)$', fontsize=axisLabel)
	ax181.set_xlabel(r'$\Delta\theta(t)/v$ mod $2\pi$', fontsize=axisLabel)
	ax181.set_ylabel(r'$\Delta\dot{\theta}(t)/v$', fontsize=axisLabel)

	fig16.savefig('results/HF-2pi_periodic_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig16.savefig('results/HF-2pi_periodic_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig17.savefig('results/HF_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig17.savefig('results/HF_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig18.savefig('results/LF-2pi_periodic_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig18.savefig('results/LF-2pi_periodic_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig19.savefig('results/LF_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig19.savefig('results/LF_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig161.savefig('results/HF-multStabInfo_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig161.savefig('results/HF-multStabInfo_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig181.savefig('results/LF-multStabInfo_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig181.savefig('results/LF-multStabInfo_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	np.save('results/deltaThetaDivSave_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.npy' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), deltaThetaDivSave)
	np.save('results/deltaThetaDivDotSave_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.npy' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), deltaThetaDivDotSave)
	np.save('results/LF-stats_init_phase_conf_final_state_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.npy' %(np.mean(poolData[0][0]['dictPLL']['coupK']), np.mean(poolData[0][0]['dictPLL']['cutFc']), np.mean(poolData[0][0]['dictPLL']['syncF']), np.mean(poolData[0][0]['dictPLL']['transmission_delay']), np.mean(poolData[0][0]['dictPLL']['noiseVarVCO']), now.year, now.month, now.day), stats_init_phase_conf_final_state)
	plt.draw(); plt.show()

	return None

################################################################################

''' CALCULATE KURAMOTO ORDER PARAMETER '''
def calcPairwiseKurmOrder(phi):
	'''Computes the Kuramoto order parameter r for in-phase synchronized states

	   Parameters
	   ----------
	   phi:  np.array
			real-valued 2d matrix or 1d vector of phases
			in the 2d case the columns of the matrix represent the individual oscillators

	   Returns
	   -------
	   r  :  np.array
			real value or real-valued 1d vetor of the Kuramotot order parameter

	   Authors
	   -------
	   Lucas Wetzel'''
	# Complex phasor representation
	z = np.exp(1j * phi)

	# Kuramoto order parameter
	if len(phi.shape) == 1:
		r 	= np.abs(np.mean(z))
		psi	= np.arctan(np.imag(z)/np.real(z))
	elif len(phi.shape) == 2:
		r = np.abs(np.mean(z, axis=1))
		psi	= np.arctan(np.imag(z, axis=1)/np.real(z, axis=1))
	else:
		print( 'Error: phi with wrong dimensions' )
		r = None

	return r, psi

################################################################################

def calcKuramotoOrderParameter(phi):
	'''Computes the Kuramoto order parameter r for in-phase synchronized states

	   Parameters
	   ----------
	   phi:  np.array
			real-valued 2d matrix or 1d vector of phases
			in the 2d case the columns of the matrix represent the individual oscillators

	   Returns
	   -------
	   r  :  np.array
			real value or real-valued 1d vetor of the Kuramotot order parameter

	   Authors
	   -------
	   Lucas Wetzel, Daniel Platz'''
	# Complex phasor representation
	z = np.exp(1j * phi)

	# Kuramoto order parameter
	if len(phi.shape) == 1:
		r = np.abs(np.mean(z))
	elif len(phi.shape) == 2:
		r = np.abs(np.mean(z, axis=1))
	else:
		print( 'Error: phi with wrong dimensions' )
		r = None

	return r

################################################################################

''' CALCULATE KURAMOTO ORDER PARAMETER FOR ENTRAINMENT OF SYNCED STATES'''
def calcKuramotoOrderParEntrainSelfOrgState(phi, phi_constant_expected):
	'''Computes the Kuramoto order parameter r for in-phase synchronized states

	   Parameters
	   ----------
	   phi:  np.array
			real-valued 2d matrix or 1d vector of phases
			in the 2d case the columns of the matrix represent the individual oscillators

	   Returns
	   -------
	   r  :  np.array
			real value or real-valued 1d vetor of the Kuramotot order parameter

	   Authors
	   -------
	   Lucas Wetzel, Daniel Platz'''
	# Complex phasor representation
	z = np.exp(1j * (phi - phi_constant_expected))

	# Kuramoto order parameter
	if len(phi.shape) == 1:
		r = np.abs(np.mean(z))
	elif len(phi.shape) == 2:
		r = np.abs(np.mean(z, axis=1))
	else:
		print( 'Error: phi with wrong dimensions' )
		r = None

	return r

################################################################################

def mTwistOrderParameter(phi):
	'''Computes the Fourier order parameter 'rm' for all m-twist synchronized states

	   Parameters
	   ----------
	   phi  :  np.array
			   real-valued 2d matrix or 1d vector of phases
			   in the 2d case the columns of the matrix represent the individual oscillators

	   Returns
	   -------
	   rm  :  np.array
			  complex-valued 1d vector or 2d matrix of the order parameter

	   Authors
	   -------
	   Lucas Wetzel, Daniel Platz'''

	# Complex phasor representation
	zm = np.exp(1j * phi)

	# Fourier transform along the oscillator index axis
	if len(phi.shape) == 1:
		rm = np.fft.fft(zm) / len(phi)
	elif len(phi.shape) == 2:
		rm = np.fft.fft(zm, axis=1) / phi.shape[1]
	else:
		print('Error: phi with wrong dimensions')
		rm = None
	return rm

################################################################################

def _CheckerboardOrderParameter(phi):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi is supposed
	   to be 1d vector of phases without time evolution.
	'''
	r = 0.0
	for ix in range(len(phi)):
		r += np.exp(1j * phi[ix]) * np.exp(-1j * np.pi * ix)
	r = np.abs(r) / float(len(phi))
	return r

################################################################################

def CheckerboardOrderParameter(phi):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi can be a 1d or 2d vector whose first index
	   corresponds to different times.
	   :rtype: np.ndarray
	'''
	if len(phi.shape) == 1:
		return _CheckerboardOrderParameter(phi)
	else:
		r = np.zeros(phi.shape[0])
		for it in range(phi.shape[0]):
			r[it] = _CheckerboardOrderParameter(phi[it, :])
		return r

################################################################################

def _mTwistOrderParameter2d(phi, nx, ny):
	'''Computes the 2d twist order parameters for 2d states. Phi is supposed
	   to be 1d vector of phases. The result is returned as an array of shape (ny, nx)
	'''
	phi_2d = np.reshape(phi, (ny, nx))
	r = np.fft.fft2(np.exp(1j * phi_2d))
	return np.abs(r) / float(len(phi))


def mTwistOrderParameter2d(phi, nx, ny):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi can be a 1d or 2d vector whose first index
	   corresponds to different times.
	'''
	if len(phi.shape) == 1:
		return _mTwistOrderParameter2d(phi, nx, ny)
	else:
		r = []
		for it in range(phi.shape[0]):
			r.append(_mTwistOrderParameter2d(phi[it, :], nx ,ny))
		return np.array(r)

################################################################################

def _CheckerboardOrderParameter2d(phi, nx, ny):
	'''Computes the 2d checkerboard order parameters for 2d states. Phi is supposed
	   to be 1d vector of phases. Please note that there are three different checkerboard states in 2d.
	'''
	k = np.array([[0, np.pi], [np.pi, 0], [np.pi, np.pi]])
	r = np.zeros(3, dtype=np.complex)
	phi_2d = np.reshape(phi, (ny, nx))
	for ik in range(3):
		for iy in range(ny):
			for ix in range(nx):
				r[ik] += np.exp(1j * phi_2d[iy, ix]) * np.exp(-1j * (k[ik, 0] * iy + k[ik, 1] * ix))
	r = np.abs(r) / float(len(phi))
	return r

################################################################################

def CheckerboardOrderParameter2d(phi, nx, ny):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi can be a 1d or 2d vector whose first index
	   corresponds to different times.
	'''
	if len(phi.shape) == 1:
		return _CheckerboardOrderParameter2d(phi, nx, ny)
	else:
		r = []
		for it in range(phi.shape[0]):
			r.append(CheckerboardOrderParameter2d(phi[it, :], nx, ny))
		return np.array(r)

################################################################################

def oracle_mTwistOrderParameter(phi, k):  # , kx, ky
	'''Computes the absolute value of k-th Fourier order parameter 'rm' for all m-twist synchronized states

	   Parameters
	   ----------
	   phi: np.array
			 real-valued 2d matrix or 1d vector of phases
			 in the 2d case the columns of the matrix represent the individual oscillators
	   k  : integer
			 the index of the requested Fourier order parameter

	   Returns
	   -------
	   rm  : np.complex/np.array
			   real value/real-valued 1d vector of the k-th order parameter

	   Authors
	   -------
	   Lucas Wetzel, Daniel Platz'''

	r = mTwistOrderParameter(phi)
	if len(phi.shape) == 1:
		rk = np.abs(r[k])
	elif len(phi.shape) == 2:
		rk = np.abs(r[:, k])
	else:
		print('Error: phi with wrong dimensions')
		rk = None
	return rk

################################################################################

def oracle_CheckerboardOrderParameter1d(phi, k=1):
	"""
		k == 0 : global sync state
		k == 1 : checkerboard state
	"""
	if k == 0:
		return calcKuramotoOrderParameter(phi)
	elif k == 1:
		return CheckerboardOrderParameter(phi)
	else:
		raise Exception('Non-valid value for k')

################################################################################

def oracle_mTwistOrderParameter2d(phi, nx, ny, kx, ky):
	return mTwistOrderParameter2d(phi, nx, ny)[:, ky, kx]


def oracle_CheckerboardOrderParameter2d(phi, nx, ny, k):
	"""
			k == 0 : x checkerboard state
			k == 1 : y checkerboard state
			k == 2 : xy checkerboard state
			k == 3 : global sync state
		"""
	if k == 0 or k == 1 or k == 2:
		return CheckerboardOrderParameter2d(phi, nx, ny)[:, k]
	elif k == 3:
		return calcKuramotoOrderParameter(phi)
	else:
		raise Exception('Non-valid value for k')

################################################################################

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
	'''
	Function to offset the "center" of a colormap. Useful for
	data with a negative min and positive max and you want the
	middle of the colormap's dynamic range to be at zero.

	Input
	-----
	  cmap : The matplotlib colormap to be altered
	  start : Offset from lowest point in the colormap's range.
		  Defaults to 0.0 (no lower offset). Should be between
		  0.0 and `midpoint`.
	  midpoint : The new center of the colormap. Defaults to
		  0.5 (no shift). Should be between 0.0 and 1.0. In
		  general, this should be  1 - vmax / (vmax + abs(vmin))
		  For example if your data range from -15.0 to +5.0 and
		  you want the center of the colormap at 0.0, `midpoint`
		  should be set to  1 - 5/(5 + 15)) or 0.75
	  stop : Offset from highest point in the colormap's range.
		  Defaults to 1.0 (no upper offset). Should be between
		  `midpoint` and 1.0.
	'''
	cdict = {
		'red': [],
		'green': [],
		'blue': [],
		'alpha': []
	}

	# regular index to compute the colors
	reg_index = np.linspace(start, stop, 257)

	# shifted index to match the data
	shift_index = np.hstack([
		np.linspace(0.0, midpoint, 128, endpoint=False),
		np.linspace(midpoint, 1.0, 129, endpoint=True)
	])

	for ri, si in zip(reg_index, shift_index):
		r, g, b, a = cmap(ri)

		cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))

	newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	plt.register_cmap(cmap=newcmap)

	return newcmap

################################################################################

# def evaluateSimulationIsing(poolData, phase_wrap=0, number_of_bins=25, prob_density=False, order_param_solution=0.0, number_of_expected_oscis_in_one_group=10):
#
# 	# plot parameter
# 	axisLabel = 9
# 	legendLab = 6
# 	tickSize = 5
# 	titleLabel = 9
# 	dpi_val = 150
# 	figwidth = 6
# 	figheight = 5
# 	linewidth = 0.8
# 	plot_size_inches_x = 10
# 	plot_size_inches_y = 5
# 	labelpadxaxis = 10
# 	labelpadyaxis = 20
# 	alpha = 0.5
#
# 	threshold_realizations_plot = 8
#
# 	if phase_wrap == 1:				# plot phase-differences in [-pi, pi] interval
# 		shift2piWin = np.pi
# 	elif phase_wrap == 2:			# plot phase-differences in [-pi/2, 3*pi/2] interval
# 		shift2piWin = 0.5*np.pi
# 	elif phase_wrap == 3:			# plot phase-differences in [0, 2*pi] interval
# 		shift2piWin = 0
#
# 	#unit_cell = PhaseDifferenceCell(poolData[0][0]['dictNet']['Nx']*poolData[0][0]['dictNet']['Ny'])
# 	threshold_statState = np.pi/15
# 	plotEveryDt = 1
# 	numberColsPlt = 3
# 	numberColsPlt_widePlt = 1
# 	number_of_intrinsic_periods_smoothing = 1.5
# 	print('For smoothing of phase-differences and order parameters we average over %0.2f periods of the ensemble mean intrinsic frequency.' % number_of_intrinsic_periods_smoothing)
#
# 	fig16, ax16 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig16.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, phase relations')					# phase relations
# 	if isinstance( poolData[0][0]['dictPLL']['cutFc'], np.float):
# 		fig16.suptitle(r'parameters: $f=$%0.2f Hz, $K=$%0.2f Hz/V, $K^\textrm{I}=$%0.2f Hz/V, $f_c=$%0.2f Hz, $\tau=$%0.2f s'%(poolData[0][0]['dictPLL']['intrF'], poolData[0][0]['dictPLL']['coupK'], poolData[0][0]['dictPLL']['coupStr_2ndHarm'], poolData[0][0]['dictPLL']['cutFc'], poolData[0][0]['dictPLL']['transmission_delay']))
# 	fig16.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax16 = ax16.ravel()
#
# 	fig161, ax161 = plt.subplots(int(np.ceil(len(poolData[0][:]) / numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig161.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, smoothed out phase relations')  # phase relations
# 	if isinstance(poolData[0][0]['dictPLL']['cutFc'], np.float):
# 		fig161.suptitle(r'parameters: $f=$%0.2f Hz, $K=$%0.2f Hz/V, $K^\textrm{I}=$%0.2f Hz/V, $f_c=$%0.2f Hz, $\tau=$%0.2f s' % (
# 		poolData[0][0]['dictPLL']['intrF'], poolData[0][0]['dictPLL']['coupK'], poolData[0][0]['dictPLL']['coupStr_2ndHarm'], poolData[0][0]['dictPLL']['cutFc'],
# 		poolData[0][0]['dictPLL']['transmission_delay']))
# 	fig161.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax161 = ax161.ravel()
#
# 	fig17, ax17 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt_widePlt)), numberColsPlt_widePlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig17.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, inst. frequencies')					# inst. frequencies
# 	fig17.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax17 = ax17.ravel()
#
# 	fig18, ax18 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig18.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, order parameter')					# order parameter
# 	fig18.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax18 = ax18.ravel()
#
# 	fig19, ax19 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt_widePlt)), numberColsPlt_widePlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig19.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, signals')							# signals
# 	fig19.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax19 = ax19.ravel()
#
# 	fig20, ax20 = plt.subplots(int(np.ceil(len(poolData[0][:]) / numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig20.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, histograms')  # signals
# 	fig20.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax20 = ax20.ravel()
#
# 	fig99, ax99 = plt.subplots(1, 1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig99.canvas.manager.set_window_title('Network view of result.')  # network
#
# 	if len(poolData[0][:]) > threshold_realizations_plot: # only plot when many realizations are computed for overview
# 		fig21, ax21 = plt.subplots(1, 1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 		fig21.canvas.manager.set_window_title('all order parameters (solution correct: solid, incorrect: dashed line)')  # all order parameters
# 		fig21.set_size_inches(plot_size_inches_x, plot_size_inches_y)
#
# 		fig211, ax211 = plt.subplots(1, 1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 		fig211.canvas.manager.set_window_title('all order parameters smoothed (solution correct: solid, incorrect: dashed line)')  # all order parameters
# 		fig211.set_size_inches(plot_size_inches_x, plot_size_inches_y)
#
# 		fig22, ax22 = plt.subplots(1, 1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 		fig22.canvas.manager.set_window_title('distribution of times to solution')  # all order parameters
# 		fig22.set_size_inches(plot_size_inches_x, plot_size_inches_y)
#
# 	# print('poolData in eva.evaluateSimulationIsing(poolData):', poolData)
#
# 	print('For evaluation of asymptotic order parameter we average over %0.2f periods of the ensemble mean intrinsic frequency.' % number_of_intrinsic_periods_smoothing)
#
# 	sol_time = []
# 	success_count = 0
# 	success_count_test1 = 0
# 	group_oscillators_maxcut = np.zeros([len(poolData[0][:]), len(poolData[0][0]['dictData']['phi'][0, :])])
#
# 	# loop over the realizations
# 	for i in range(len(poolData[0][:])):
# 		deltaTheta = np.zeros([len(poolData[0][i]['dictData']['phi'][0, :]), len(poolData[0][i]['dictData']['phi'][:, 0])])
# 		signalOut  = np.zeros([len(poolData[0][i]['dictData']['phi'][0, :]), len(poolData[0][i]['dictData']['phi'][:, 0])])
#
# 		thetaDot = np.diff( poolData[0][i]['dictData']['phi'][:, :], axis=0 ) / poolData[0][i]['dictPLL']['dt']				# compute frequencies and order parameter
# 		r, orderparam, F1 = obtainOrderParam(poolData[0][i]['dictPLL'], poolData[0][i]['dictNet'], poolData[0][i]['dictData'])
#
# 		ax18[i].plot( poolData[0][i]['dictData']['t'][::plotEveryDt], orderparam[::plotEveryDt], label=r'$R_\textrm{final}=%0.2f$'%(orderparam[-1]), linewidth=linewidth )
#
# 		# HOWTO 1) to determine whether the correct solution has be found, we test for the asymptotic value of the order parameter
# 		order_param_diff_expected_value_threshold = 0.01
# 		correct_solution_test0 = False
# 		if np.abs(np.mean(orderparam[-int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']):]) - order_param_solution) < order_param_diff_expected_value_threshold:
# 			print('Order parameter predicted for solution=%0.2f has been reached. Averaged over last %i periods of the intrinsic frequency for realization %i.'%(order_param_solution, number_of_intrinsic_periods_smoothing, i))
# 			success_count += 1					# to calculate the probability of finding the correct solutions
# 			correct_solution_test0 = True		# this is needed to decide for which realizations we need to measure the time to solution
#
# 		# HOWTO 2) to determine whether the correct solution has be found, we also test for mutual phase-differences between the oscillators
# 		group1 = 0
# 		group2 = 0
# 		correct_solution_test1 = False
# 		for j in range(len(poolData[0][i]['dictData']['phi'][0, :])):
# 			# calculate mean phase difference over an interval of 'number_of_intrinsic_periods_smoothing' periods at the end of all oscillators with respect to oscillator zero
# 			# interval_index = -int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt'])
# 			temp_phase_diff = np.mean(poolData[0][i]['dictData']['phi'][-int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']):, 0] - poolData[0][i]['dictData']['phi'][-int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']):, j])
# 			print('Realization %i, mean phase difference {mod 2pi into [-pi, pi)} between k=0 and k=%i is deltaPhi=%0.2f'%(i, j, ((temp_phase_diff+np.pi) % (2*np.pi))-np.pi))
# 			if np.abs(((temp_phase_diff+np.pi) % (2*np.pi))-np.pi) < np.pi/2:
# 				group1 += 1
# 			else:
# 				group2 += 1
# 		if not group1+group2 == len(poolData[0][i]['dictData']['phi'][0, :]):
# 			print('ERROR: check!')
# 			sys.exit()
# 		if group1 == number_of_expected_oscis_in_one_group or group2 == number_of_expected_oscis_in_one_group:
# 			success_count_test1 += 1
# 			correct_solution_test1 = True
#
# 		# HOWTO 3) to determine the time to solutions; find the time at the which the asymptotic value of the order parameter has been reached, we count from the start time increasing one of the coupling strengths
# 		order_param_std_threshold = 0.005
# 		smoothing = True
# 		if smoothing:
# 			if correct_solution_test1:
# 				# when the derivative of the order parameter is close to zero, we expect that the asymptotic state has been reached
# 				# here we look for the cases where this is NOT the case yet, then the last entry of the resulting vector will be the transition time from transient to asymptotic dynamics
# 				derivative_order_param_smoothed = (np.diff( uniform_filter1d( orderparam[poolData[0][i]['dictNet']['max_delay_steps']:],
# 					size=int(15 * number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect') ) / poolData[0][i]['dictPLL']['dt'])
#
# 				rolling_std_derivative_order_param_smoothed = pd.Series(derivative_order_param_smoothed).rolling(int(15 * number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt'])).std()
#
# 				#temp = np.where( (np.diff( uniform_filter1d( orderparam[(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps']):],
# 				#	size=int(15 * number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect') ) / poolData[0][i]['dictPLL']['dt']) > order_param_change_threshold )
# 				min_std = np.min(rolling_std_derivative_order_param_smoothed)
# 				print('min_std:', min_std)
# 				max_std = np.max(rolling_std_derivative_order_param_smoothed)
# 				order_param_std_threshold = 0.1 * (max_std - min_std) + min_std
# 				print('Realization %i, order_param_std_threshold to %0.02f, for {min_std, max_std} = {%0.2f,%0.2f} '%(i, order_param_std_threshold, min_std, max_std))
# 				temp = np.where(rolling_std_derivative_order_param_smoothed[poolData[0][i]['dictData']['tstep_annealing_start']:] > order_param_std_threshold)
#
# 				plt.plot(derivative_order_param_smoothed, 'b')
# 				plt.plot(rolling_std_derivative_order_param_smoothed, 'r--')
# 				plt.plot(temp[0][-1]-poolData[0][i]['dictData']['tstep_annealing_start'], 0, 'cd')
#
# 				# print('temp=', temp[0])
# 				if not len(temp[0]) == 0:
# 					# subtract from the last time when the transient dynamics caused order parameter fluctuations above the threshold the time when the annealing process started
# 					# the substraction of the initial delay history is already done since we only search from tau onwards for the time at which the fluctuations fulfill the conditions
# 					sol_time.append((temp[0][-1]) * poolData[0][i]['dictPLL']['dt'])
# 				else:
# 					sol_time.append(np.inf)
# 				# print('sol_time=', sol_time)
# 				plt.draw()
# 				plt.show()
# 			else:
# 				sol_time.append(np.inf)
#
# 			ax18[i].plot( poolData[0][i]['dictData']['t'][(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps']):-1:plotEveryDt], uniform_filter1d((np.diff(
# 										orderparam[(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps']):]) / poolData[0][i]['dictPLL']['dt']),
# 										size=int(0.5 * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect')[::plotEveryDt], 'r', linewidth=0.5, alpha=0.35 )
# 		else:
# 			if correct_solution_test1:
# 				temp = np.where( np.diff(orderparam[(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps']):]) / poolData[0][i]['dictPLL']['dt'] > order_param_change_threshold )
# 				sol_time.append(temp[0][-1] * poolData[0][i]['dictPLL']['dt'])
# 			else:
# 				sol_time.append(np.inf)
#
# 			ax18[i].plot( poolData[0][i]['dictData']['t'][(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps']):-1:plotEveryDt], (np.diff(
# 								orderparam[(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps']):]) / poolData[0][i]['dictPLL']['dt'])[::plotEveryDt], 'r', linewidth=0.5, alpha=0.35 )
#
# 		ax18[i].plot(poolData[0][i]['dictData']['t'][(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps'])], 0, 'cd', markersize=1)
# 		if correct_solution_test0 and sol_time[i] != np.inf:
# 			ax18[i].plot(sol_time[i] + poolData[0][i]['dictData']['t'][(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps'])], 0, 'c*', markersize=1)
#
# 		if len(poolData[0][:]) > threshold_realizations_plot:
# 			ax21.plot(poolData[0][i]['dictData']['t'][(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps'])], -0.05, 'cd', markersize=1)
# 			if correct_solution_test0:
# 				ax21.plot(poolData[0][i]['dictData']['t'][::plotEveryDt], orderparam[::plotEveryDt], '-', label=r'$R_\textrm{final}=%0.2f$' % (orderparam[-1]), linewidth=linewidth)
# 				ax211.plot(poolData[0][i]['dictData']['t'][::plotEveryDt], uniform_filter1d(orderparam[::plotEveryDt],
# 							size=int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect'), '-', label=r'$R_\textrm{final}=%0.2f$' % (orderparam[-1]), linewidth=linewidth)
# 				if sol_time[i] != np.inf:
# 					ax21.plot(sol_time[i] + poolData[0][i]['dictData']['t'][(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps'])], -0.05, 'c*', markersize=1)
# 			else:
# 				ax21.plot(poolData[0][i]['dictData']['t'][::plotEveryDt], orderparam[::plotEveryDt], '--', label=r'$R_\textrm{final}=%0.2f$' % (orderparam[-1]), linewidth=linewidth)
# 				ax211.plot(poolData[0][i]['dictData']['t'][::plotEveryDt], uniform_filter1d(orderparam[::plotEveryDt],
# 							size=int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect'), '--', label=r'$R_\textrm{final}=%0.2f$' % (orderparam[-1]), linewidth=linewidth)
#
#
# 		if phase_wrap == 0:  # plot phase differences in [-inf, inf), i.e., we use the unwrapped phases that have counted the cycles/periods
# 			ax20[i].hist(poolData[0][i]['dictData']['phi'][-3, :] - poolData[0][i]['dictData']['phi'][-2, 0], bins=number_of_bins, rwidth=0.9, density=prob_density)
# 		elif phase_wrap != 0:
# 			# print('histogram_data (wrapping if phase):', ((dictData['phi'][at_index, plotlist] + shift2piWin) % (2 * np.pi)) - shift2piWin)
# 			ax20[i].hist((((poolData[0][i]['dictData']['phi'][-3, :] - poolData[0][i]['dictData']['phi'][-2, 0] + shift2piWin) % (2.0 * np.pi)) - shift2piWin), bins=number_of_bins, rwidth=0.9, density=prob_density)
#
# 		final_phase_oscillator = []
# 		for j in range(len(poolData[0][i]['dictData']['phi'][0, :])):
# 			if shift2piWin != 0:
# 				deltaTheta[j] = (((poolData[0][i]['dictData']['phi'][:, 0] - poolData[0][i]['dictData']['phi'][:, j]) + shift2piWin) % (2.0 * np.pi)) - shift2piWin 		# calculate phase-differnce w.r.t. osci k=0
# 			else:
# 				deltaTheta[j] = poolData[0][i]['dictData']['phi'][:, 0] - poolData[0][i]['dictData']['phi'][:, j]
# 			signalOut[j] = poolData[0][i]['dictPLL']['vco_out_sig'](poolData[0][i]['dictData']['phi'][:, j])				# generate signals for all phase histories
#
# 			# save in which binarized state the oscillator was at the end of the realization, averaged over
# 			if np.mean(deltaTheta[j][-int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']):-1]) - 0 < 0.2:
# 				group_oscillators_maxcut[i, j] = -1
# 				final_phase_oscillator.append('zero')
# 			elif np.mean(deltaTheta[j][-int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']):-1]) - np.pi < 0.2:
# 				group_oscillators_maxcut[i, j] = 1
# 				final_phase_oscillator.append('pi')
# 			else:
# 				group_oscillators_maxcut[i, j] = 0
# 				final_phase_oscillator.append('diff')
#
# 			if j == 0:
# 				linestyle = '--'
# 			else:
# 				linestyle = '-'
#
# 			ax16[i].plot( poolData[0][i]['dictData']['t'][::plotEveryDt], deltaTheta[j, ::plotEveryDt], linestyle, linewidth=linewidth, label='sig PLL%i' %(j))
# 			ax161[i].plot(poolData[0][i]['dictData']['t'], uniform_filter1d(deltaTheta[j, :], size=int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect'), linestyle, linewidth=linewidth, label='sig PLL%i' % (j))
# 			ax19[i].plot( poolData[0][i]['dictData']['t'][::plotEveryDt], poolData[0][i]['dictPLL']['vco_out_sig'](poolData[0][i]['dictData']['phi'][::plotEveryDt, j]), linewidth=linewidth, label='sig PLL%i' %(j))
# 			ax17[i].plot( poolData[0][i]['dictData']['t'][1::plotEveryDt], thetaDot[::plotEveryDt, j], linewidth=linewidth, label='sig PLL%i' %(j))
#
# 		print('working on realization %i results from sim:'%i, poolData[0][i]['dictNet'], '\n', poolData[0][i]['dictPLL'], '\n', poolData[0][i]['dictData'], '\n\n')
#
# 		if i == int( len(poolData[0][:]) / 2 ):
# 			ax16[i].set_ylabel(r'$\Delta\theta(t)$', fontsize=axisLabel)
# 			ax161[i].set_ylabel(r'$\langle\Delta\theta(t)\rangle_{%0.1f T}$'%(number_of_intrinsic_periods_smoothing), fontsize=axisLabel)
# 			ax17[i].set_ylabel(r'$\dot{\theta}(t)$ in radHz', fontsize=axisLabel)
# 			ax18[i].set_ylabel(r'$R(t)$', fontsize=axisLabel)
# 			ax19[i].set_ylabel(r'$s(t)$', fontsize=axisLabel)
# 			ax20[i].set_ylabel(r'$H\left(\Delta\theta(t)\right)$', fontsize=axisLabel)
# 		if i == len(poolData[0][:])-2:
# 			ax16[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax161[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax18[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax20[i].set_xlabel(r'$\Delta\theta(t)$ in $[rad]$', fontsize=axisLabel)
# 		if i == len(poolData[0][:])-1:
# 			ax17[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax19[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 		if len(poolData[0][:]) > threshold_realizations_plot and i == len(poolData[0][:])-1:
# 			ax21.set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax21.set_ylabel(r'$R(t)$', fontsize=axisLabel)
# 			ax21.tick_params(labelsize=tickSize)
# 			ax211.set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax211.set_ylabel(r'$\langle R(t) \rangle_{%0.1f T}$'%(number_of_intrinsic_periods_smoothing), fontsize=axisLabel)
# 			ax211.tick_params(labelsize=tickSize)
# 			ax22.set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax22.set_ylabel(r'$p(t)$', fontsize=axisLabel)
# 			ax22.tick_params(labelsize=tickSize)
#
# 		ax16[i].tick_params(labelsize=tickSize)
# 		ax161[i].tick_params(labelsize=tickSize)
# 		ax17[i].tick_params(labelsize=tickSize)
# 		ax18[i].tick_params(labelsize=tickSize)
# 		ax18[i].legend(loc='lower right', fontsize=legendLab)
# 		ax19[i].tick_params(labelsize=tickSize)
# 		ax20[i].tick_params(labelsize=tickSize)
#
# 		#if i == 0:
# 		print('Plotting the network and its binarized asymptotic state.')
# 		color_map = []
# 		if isinstance(poolData[0][i]['dictPLL']['gPDin'], list):
# 			network_graph = nx.from_numpy_array(np.array(poolData[0][i]['dictPLL']['gPDin']))
# 		else:
# 			network_graph = nx.from_numpy_array(poolData[0][i]['dictPLL']['gPDin'])
# 		print('len(final_phase_oscillator)=', len(final_phase_oscillator))
# 		for node in network_graph:
# 			print('Working on node %i'%(node))
# 			if final_phase_oscillator[node] == 'zero':
# 				color_map.append('green')
# 			elif final_phase_oscillator[node] == 'pi':
# 				color_map.append('blue')
# 			else:
# 				color_map.append('red')
# 		plt.figure(9999-i)
# 		nx.draw(network_graph, node_color=color_map, with_labels=True, pos=nx.spring_layout(network_graph))
# 		plt.savefig('results/network_asymptotic_state_r%i_%d_%d_%d.svg' % (i, now.year, now.month, now.day), dpi=dpi_val)
# 		plt.savefig('results/network_asymptotic_state_r%i_%d_%d_%d.png' % (i, now.year, now.month, now.day), dpi=dpi_val)
#
# 	sol_time = np.array(sol_time)
# 	sol_time_without_inf_entries = sol_time[sol_time != np.inf]
# 	if len(sol_time_without_inf_entries) == 0:
# 		print('All times to solution were evaluated as np.inf and hence the mean time so solution is np.inf!')
# 		sol_time_without_inf_entries = np.array([np.inf])
# 	print('success_count: ', success_count, 'len(poolData[0][:]: ', len(poolData[0][:]), 'sol_time:', sol_time)
# 	results_string = 'Final evaluation:\n1) for a total of %i realizations, success probability (evaluation R) = %0.4f\n2) and success probability evaluating groups separated by pi = %0.4f\n3) average time to solution = %0.4f seconds, i.e., %0.2f mean intrinsic periods.\n4) average time to solution without infinity entries= %0.4f seconds, i.e., %0.2f mean intrinsic periods.\n5) fastest and slowest time to solution in multiples of periods: {%0.2f, %0.2f}'%(
# 		len(poolData[0][:]), success_count / len(poolData[0][:]), success_count_test1 / len(poolData[0][:]),
# 		np.mean(sol_time),
# 		np.mean(sol_time)/np.mean(poolData[0][i]['dictPLL']['intrF']),
# 		np.mean(sol_time_without_inf_entries),
# 		np.mean(sol_time_without_inf_entries)/np.mean(poolData[0][i]['dictPLL']['intrF']),
# 		np.min(sol_time)/np.mean(poolData[0][i]['dictPLL']['intrF']),
# 		np.max(sol_time)/np.mean(poolData[0][i]['dictPLL']['intrF']))
# 	if len(poolData[0][:]) > threshold_realizations_plot:
# 		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# 		#ax21.text(0.25*poolData[0][0]['dictData']['t'][-1], 0.2, results_string, horizontalalignment='left', verticalalignment='bottom', bbox=props, fontsize=9)
# 		ax211.text(0.25 * poolData[0][0]['dictData']['t'][-1], 0.2, results_string, horizontalalignment='left', verticalalignment='bottom', bbox=props, fontsize=9)
#
# 	print(results_string)
#
# 	if np.any(sol_time[:]) == np.inf:
# 		ax22.hist(sol_time, bins=15, rwidth=0.9, density=prob_density)
#
# 	ax16[0].legend(loc='upper right', fontsize=legendLab)
# 	ax17[0].legend(loc='upper right', fontsize=legendLab)
# 	ax19[0].legend(loc='upper right', fontsize=legendLab)
#
# 	fig16.savefig('results/phase_relations_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig161.savefig('results/phase_relations_smoothed_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig17.savefig('results/frequencies_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig18.savefig('results/order_parameter_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig19.savefig('results/signals_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig20.savefig('results/histograms_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
# 	#fig99.savefig('results/network_asymptotic_state_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig16.savefig('results/phase_relations_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig161.savefig('results/phase_relations_smoothed_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig17.savefig('results/frequencies_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig18.savefig('results/order_parameter_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig19.savefig('results/signals_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
# 	fig20.savefig('results/histograms_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
# 	# fig99.savefig('results/network_asymptotic_state_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
# 	if len(poolData[0][:]) > threshold_realizations_plot:
# 		fig21.savefig('results/all_order_parameters_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
# 		fig21.savefig('results/all_order_parameters_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
# 		fig211.savefig('results/all_order_parameters_smoothed_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
# 		fig211.savefig('results/all_order_parameters_smoothed_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
# 		fig22.savefig('results/hist_soltime_density_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
# 		fig22.savefig('results/hist_soltime_density_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
#
# 	plt.draw()
# 	plt.show()
#
# 	return None