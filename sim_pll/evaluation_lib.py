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
gc.enable()

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
	axisLabel = 50
	titleLabel= 10
	dpi_val   = 150
	figwidth  = 6
	figheight = 3

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


def prepareDictsForPlotting(dict_pll, dict_net):

	if dict_pll['cutFc'] is None:
		dict_pll.update({'cutFc': np.inf})

	if not np.abs(np.min(dict_pll['intrF'])) > 1E-17:									# for f=0, there would otherwies be a float division by zero
		dict_pll.update({'intrF': 1})
		print('Since intrinsic frequency was zero: for plotting set to one to generate boundaries!')


	return dict_pll, dict_net


def saveDictionaries(dictToSave, name, K, tau, Fc, Nx, Ny, mx, my, topology):

	if Fc is None:
		Fc = np.inf

	N = int(Nx*Ny)
	filename = 'results/%s_K%.3f_tau%.3f_Fc%.3f_mx%i_my%i_N%i_topo%s_%d:%d_%d_%d_%d' % (name, np.mean(K), np.mean(tau), np.mean(Fc), mx, my, N, topology, now.hour, now.minute, now.year, now.month, now.day)
	f = open(filename,'wb')
	pickle.dump(dictToSave, f, protocol=4)
	f.close()

	return None

################################################################################

def calculateEigenvalues(dict_net, dict_pll):
	''' Calculate eigenvalues zeta for networks of homogeneous PLL '''

	if dict_net['topology'] == 'global':											# test whether global coupling topology
		print('All to all coupling topology identified!')
		zeta = 1/(dict_net['Nx']*dict_net['Ny']-1)
		dict_net.update({'zeta': zeta})

	# if dict_net['Ny'] == 1:														# check whether 2D or 1D topology
	# 	print('1d network topology identified!')
	# 	if dict_net['topology'] == 'ring':
	# 		zeta = 1
	# 		dict_net.update({'zeta': zeta})
	# 	elif dict_net['topology'] == 'chain':
	# 		if dict_net['mx'] == 0:
	# 			zeta = 1
	# 			dict_net.update({'zeta': zeta})
	# 		elif dict_net['mx'] > 0:
	# 			zeta = np.cos(np.arange(0,dict_net['Nx'])*np.pi/(dict_net['Nx']-1))
	# 			dict_net.update({'zeta': zeta})
	# 	else:
	# 		print('Coupling topology not yet implemented, add expression for eigenvalues or brute-force solve!')
	#
	# elif dict_net['Ny'] > 1:
	# 	print('2d network topology identified!')
	# 	if dict_net['topology'] == 'square-open':
	# 		zeta = 1
	# 		dict_net.update({'zeta': zeta})
	# 	elif dict_net['topology'] == 'square-periodic':
	# 		zeta = 1
	# 		dict_net.update({'zeta': zeta})
	# 	else:
	# 		print('Coupling topology not yet implemented, add expression for eigenvalues or brute-force solve!')
	#
	return dict_net, dict_pll

################################################################################

''' CALCULATE SPECTRUM '''
def calcSpectrum(phase_or_signal: np.ndarray, dict_pll: dict, dict_net: dict, dict_algo: dict, psd_id: np.int=0, percentOfTsim: np.float=0.5, signal_given: bool=False): #phi,Fsample,couplingfct,waveform=None,expectedFreq=-999,evalAllRealizations=False,decayTimeSlowestMode=None

	Pxx_dBm = []
	Pxx_dBV = []
	f = []
	try:
		windowset = 'boxcar' 		# here we choose boxcar since a modification of the ends of the time-series is not necessary for an integer number of periods
		print('Trying to cut integer number of periods! Inside calcSpectrum.')
		if dict_pll['extra_coup_sig'] is None:
			analyzeL = findIntTinSig.cutTimeSeriesOfIntegerPeriod(dict_pll['sampleF'], dict_net['Tsim'], dict_pll['transmission_delay'], dict_pll['syncF'],
																np.max(dict_pll['coupK']), phase_or_signal, psd_id, percentOfTsim, signal_given)
		else:
			analyzeL = findIntTinSig.cutTimeSeriesOfIntegerPeriod(dict_pll['sampleF'], dict_net['Tsim'], dict_pll['transmission_delay'], dict_pll['syncF'],
																np.max([np.max(dict_pll['coupK']), np.max(dict_pll['coupStr_2ndHarm'])]), phase_or_signal, psd_id, percentOfTsim, signal_given)
	except:
		windowset = 'hamming' 													#'hamming' #'hamming', 'boxcar'
		print('\n\nError in cutTimeSeriesOfIntegerPeriod-function! Not picking integer number of periods for PSD! Using window %s!\n\n'%windowset)
		analyzeL = [int(dict_net['Tsim'] * dict_pll['sampleF'] * (1-percentOfTsim)), int(dict_net['Tsim'] * dict_pll['sampleF'])-1]

	window = scipy.signal.get_window(windowset, analyzeL[1]-analyzeL[0], fftbins=True)

	if signal_given:
		print('phase_or_signal', phase_or_signal)
		if 'entrain' in dict_net['topology'] and psd_id == 0 and (isinstance(dict_pll['intrF'], list) or
						isinstance(dict_pll['intrF'], np.ndarray)) and (dict_algo['param_id_0'] == 'intrF' or dict_algo['param_id_1'] == 'intrF'):
			print('Caution - may lead to: FloatingPointError: divide by zero encountered in log10 since the control signal of the reference is usually constant!')
			# tsdata = phase_or_signal[analyzeL[0]:analyzeL[1], 1:]
		else:
			tsdata = phase_or_signal[analyzeL[0]:analyzeL[1]]

		print('\nCurrent window option is', windowset,
			'for the waveform of the signal provided.\nNOTE: in principle can always choose to be sin() for cleaner PSD in first harmonic approximation of the signal.')
	else:
		tsdata = dict_pll['PSD_from_signal'](phase_or_signal[analyzeL[0]:analyzeL[1]])
		print('\nCurrent window option is', windowset, 'for waveform', inspect.getsourcelines(dict_pll['PSD_from_signal'])[0][0],
			  'NOTE: in principle can always choose to be sin() for cleaner PSD in first harmonic approximation of the signal.')
	#print('Length tsdata:', len(tsdata), '\tshape tsdata:', np.shape(tsdata))

	#print('Length window:', len(window), '\tshape window:', np.shape(window))
	print('Calculate spectrum for', percentOfTsim, 'percent of the time-series. Implement better solution using decay times.')

	ftemp, Vxx = scipy.signal.periodogram(tsdata, dict_pll['sampleF'], return_onesided=True, window=window, scaling='density', axis=0) #  returns Pxx with dimensions [V^2] if scaling='spectrum' and [V^2/Hz] if if scaling='density'
	P0 = 1E-3														# 1000 mW
	R = 50 															# 50 Ohms --> for P0 in [mW/Hz] and R [ohm]

	Pxx_dBm.append(10*np.log10((Vxx/R)/P0))
	f.append(ftemp)

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
		isRotated  :  bool
						True if x is given in rotated coordinates

		Returns
		-------
		is_inside  :  bool
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

def compute_order_parameter(dict_pll: dict, dict_net: dict, dict_data: dict):
	''' MODIFIED KURAMOTO ORDER PARAMETERS '''
	numb_av_T = 2.5																			   	# number of periods of free-running frequencies to average over
	if np.min(dict_pll['intrF']) > 0:														 	# for f=0, there would otherwise be a float division by zero
		F1 = np.min(dict_pll['intrF'])
	else:
		F1 = np.min(dict_pll['intrF'])+1E-3

	division = dict_pll['div']

	if dict_net['topology'] == "square-periodic" or dict_net['topology'] == "hexagon-periodic" or dict_net['topology'] == "octagon-periodic":
		r = oracle_mTwistOrderParameter2d(dict_data['phi'][-int(numb_av_T*1.0/(F1*dict_pll['dt'])):, :], dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'])
		order_parameter = oracle_mTwistOrderParameter2d(dict_data['phi'][:, :], dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'])
		order_parameter_divided_phases = oracle_mTwistOrderParameter2d(dict_data['phi'][:, :]/division, dict_net['Nx'], dict_net['Ny'], dict_net['mx'], dict_net['my'])
	elif dict_net['topology'] == "square-open" or dict_net['topology'] == "hexagon" or dict_net['topology'] == "octagon":
		if dict_net['mx'] == 1 and dict_net['my'] == 1:
			ktemp=2
		elif dict_net['mx'] == 1 and dict_net['my'] == 0:
			ktemp=0
		elif dict_net['mx'] == 0 and dict_net['my'] == 1:
			ktemp = 1
		elif dict_net['mx'] == 0 and dict_net['my'] == 0:
			ktemp = 3
		else:
			ktemp = 4
		"""
				ktemp == 0 : x  checkerboard state
				ktemp == 1 : y  checkerboard state
				ktemp == 2 : xy checkerboard state
				ktemp == 3 : in-phase synchronized
			"""
		r = oracle_CheckerboardOrderParameter2d(dict_data['phi'][-int(numb_av_T*1.0/(F1*dict_pll['dt'])):, :], dict_net['Nx'], dict_net['Ny'], ktemp)
		# ry = np.nonzero(rmat > 0.995)[0]
		# rx = np.nonzero(rmat > 0.995)[1]
		order_parameter = oracle_CheckerboardOrderParameter2d(dict_data['phi'][:, :], dict_net['Nx'], dict_net['Ny'], ktemp)
		order_parameter_divided_phases = oracle_CheckerboardOrderParameter2d(dict_data['phi'][:, :]/division, dict_net['Nx'], dict_net['Ny'], ktemp)
	elif dict_net['topology'] == "compareEntrVsMutual":
		rMut 	 = oracle_mTwistOrderParameter(dict_data['phi'][-int(numb_av_T*1.0/(F1*dict_pll['dt'])):, 0:2], dict_net['mx'])
		orderMut = oracle_mTwistOrderParameter(dict_data['phi'][:, 0:2], dict_net['mx'])
		rEnt 	 = oracle_mTwistOrderParameter(dict_data['phi'][-int(numb_av_T*1.0/(F1*dict_pll['dt'])):, 2:4], dict_net['mx'])
		orderEnt = oracle_mTwistOrderParameter(dict_data['phi'][:, 2:4], dict_net['mx'])
		if isPlottingTimeSeries:
			figwidth = 6; figheight = 5; t = np.arange(dict_data['phi'].shape[0]); now = datetime.datetime.now();
			fig0 = plt.figure(num=0, figsize=(figwidth, figheight), dpi=150, facecolor='w', edgecolor='k')
			fig0.canvas.manager.set_window_title('order parameters mutual and entrained')			   # plot orderparameter
			plt.clf()
			plt.plot((dict_data['t']*dict_pll['dt']), orderMut, 'b-',  label='2 mutual coupled PLLs')
			plt.plot((dict_data['t']*dict_pll['dt']), orderEnt, 'r--', label='one entrained PLL')
			plt.plot(dict_pll['transmission_delay'], orderMut[int(round(dict_pll['transmission_delay']/dict_pll['dt']))], 'yo', ms=5)						   # mark where the simulation starts
			plt.axvspan(dict_data['t'][-int(2*1.0/(F1*dict_pll['dt']))]*dict_pll['dt'], dict_data['t'][-1]*dict_pll['dt'], color='b', alpha=0.3)
			plt.xlabel(r'$t$ $[s]$')
			plt.ylabel(r'$R( t,m = %d )$' % dict_net['mx'])
			plt.legend()
			plt.savefig('results/orderparam_mutual_entrained_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day))
			plt.savefig('results/orderparam_mutual_entrained_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
			r = np.zeros(len(dict_data['phi'][-int(numb_av_T*1.0/(F1*dict_pll['dt'])):,0]))
			order_parameter = np.zeros(len(dict_data['phi'][:, 0]))
	elif dict_net['topology'] == "chain":
		"""
				dict_net['mx']  > 0 : x  checkerboard state
				dict_net['mx'] == 0 : in-phase synchronized
			"""
		print('Computing order parameter for a 1d chain of coupled oscillators with open boundary conditions.')
		r = oracle_chequerboard_order_parameter_one_dimension(dict_data['phi'][-int(numb_av_T*1.0/(F1*dict_pll['dt'])):, :], dict_net['mx'])
		order_parameter = oracle_chequerboard_order_parameter_one_dimension(dict_data['phi'][:, :], dict_net['mx'])									# calculate the order parameter for all times
		order_parameter_divided_phases = oracle_chequerboard_order_parameter_one_dimension(dict_data['phi'][:, :]/division, dict_net['mx'])			# calculate the order parameter for all times of divided phases
	elif dict_net['topology'] == "ring" or dict_net['topology'] == 'global':
		# print('Calculate order parameter for ring or global topology. For phases: ', dict_data['phi'])
		time.sleep(5)
		r = oracle_mTwistOrderParameter(dict_data['phi'][-int(numb_av_T*1.0/(F1*dict_pll['dt'])):, :], dict_net['mx'])		# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
		order_parameter = oracle_mTwistOrderParameter(dict_data['phi'][:, :], dict_net['mx'])								# calculate the m-twist order parameter for all times
		order_parameter_divided_phases = oracle_mTwistOrderParameter(dict_data['phi'][:, :]/division, dict_net['mx'])		# calculate the m-twist order parameter for all times of divided phases
	elif "entrain" in dict_net['topology']:
		# ( dict_net['topology'] == "entrainOne" or dict_net['topology'] == "entrainAll" or dict_net['topology'] == "entrainPLLsHierarch"):
		phi_constant_expected = dict_net['phiInitConfig']
		r = calcKuramotoOrderParEntrainSelfOrgState(dict_data['phi'][-int(numb_av_T*1.0/(F1*dict_pll['dt'])):, :], phi_constant_expected)
		order_parameter = calcKuramotoOrderParEntrainSelfOrgState(dict_data['phi'][:, :], phi_constant_expected)
		order_parameter_divided_phases = calcKuramotoOrderParEntrainSelfOrgState(dict_data['phi'][:, :]/division, phi_constant_expected)
	# r = oracle_mTwistOrderParameter(dict_data['phi'][-int(2*1.0/(F1*dict_pll['dt'])):, :], dict_net['mx'])							# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
	# order_parameter = oracle_mTwistOrderParameter(dict_data['phi'][:, :], dict_net['mx'])												# calculate the m-twist order parameter for all times
	# print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])
	print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])

	return order_parameter, order_parameter_divided_phases, F1

################################################################################

def evaluateSimulationsChrisHoyer(pool_data: dict) -> None:

	#print('pool_data', pool_data[0][0])
	# pool_data = load(...)														# in principle saved pool data can be loaded and plotted

	# plot parameter
	axisLabel  = 60
	tickSize   = 35
	titleLabel = 10
	dpi_val	   = 150
	figwidth   = 6
	figheight  = 5
	alpha 	   = 0.5
	linewidth  = 0.5

	unit_cell = PhaseDifferenceCell(pool_data[0][0]['dict_net']['Nx']*pool_data[0][0]['dict_net']['Ny'])
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

	delay_steps	= int( np.floor( pool_data[0][0]['dict_pll']['transmission_delay'] / pool_data[0][0]['dict_pll']['dt'] ) )
	stats_init_phase_conf_final_state = np.empty([len(pool_data[0][:]), 5])
	deltaThetaDivSave		= np.empty( [len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][delay_steps+1::pool_data[0][0]['dict_pll']['sampleFplot'],0])-0] )
	deltaThetaDivDotSave	= np.empty( [len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][delay_steps+1::pool_data[0][0]['dict_pll']['sampleFplot'],0])-0] )

	for i in range(len(pool_data[0][:])):
		#print('working on realization %i results from sim:'%i, pool_data[0][i]['dict_net'], '\n', pool_data[0][i]['dict_pll'], '\n', pool_data[0][i]['dict_data'],'\n\n')

		#print('Check whether perturbation is inside unit-cell (evaluation.py)! phiS:', pool_data[0][i]['dict_net']['phiPerturb'], '\tInside? True/False:', unit_cell.is_inside((pool_data[0][i]['dict_net']['phiPerturb']), isRotated=False)); time.sleep(2)
		#print('How about phi, is it a key to dict_data?', 'phi' in pool_data[0][i]['dict_data'])
		if unit_cell.is_inside((pool_data[0][i]['dict_net']['phiPerturb']), isRotated=False):	# NOTE this case is for scanValues set only in -pi to pi, we so not plot outside the unit cell

			# test whether frequency is larger or smaller than mean intrinsic frequency as a first distinction between multistable synced states with the same phase relations but
			# different frequency -- for more than 3 multistable in- or anti-phase synched states that needs to be reworked
			if ( pool_data[0][i]['dict_data']['phi'][-1, 0] - pool_data[0][i]['dict_data']['phi'][-2, 0] ) / pool_data[0][i]['dict_pll']['dt'] > np.mean( pool_data[0][i]['dict_pll']['intrF'] ):
				initmarker = 'd'
			else:
				initmarker = 'o'

			deltaTheta 			= pool_data[0][i]['dict_data']['phi'][:,0] - pool_data[0][i]['dict_data']['phi'][:,1]
			deltaThetaDot		= np.diff( deltaTheta, axis=0 ) / pool_data[0][i]['dict_pll']['dt']
			deltaThetaDiv 		= pool_data[0][i]['dict_data']['phi'][:, 0]/pool_data[0][i]['dict_pll']['div'] - pool_data[0][i]['dict_data']['phi'][:, 1]/pool_data[0][i]['dict_pll']['div']
			deltaThetaDivDot	= np.diff( deltaThetaDiv, axis=0 ) / pool_data[0][i]['dict_pll']['dt']

			if np.abs( np.abs( (deltaTheta[-1]+np.pi)%(2.*np.pi)-np.pi ) - np.pi ) < threshold_statState:
				color = 'r'														# anti-phase
			elif np.abs( (deltaTheta[-1]+np.pi)%(2.*np.pi)-np.pi ) - 0.0 < threshold_statState:
				color = 'b'														# in-phase
			else:
				color = 'k'														# neither in- nor anti-phase


			# plot for HF output
			ax16.plot((deltaTheta[delay_steps+1::pool_data[0][0]['dict_pll']['sampleFplot']]+np.pi) % (2.*np.pi)-np.pi, deltaThetaDot[delay_steps::pool_data[0][0]['dict_pll']['sampleFplot']], '-', color=color, alpha=alpha, linewidth=linewidth)	 	# plot trajectory
			ax16.plot((deltaTheta[delay_steps]+np.pi) % (2.*np.pi)-np.pi, deltaThetaDot[delay_steps], 'o', color=color, alpha=alpha)	 		# plot initial dot
			ax16.plot((deltaTheta[-1]+np.pi) % (2.*np.pi)-np.pi, deltaThetaDot[-1], 'x', color=color, alpha=alpha)						 	# plot final state cross
			#plot_lib.deltaThetaDot_vs_deltaTheta(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], (deltaTheta[1:]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot, color, alpha)
			ax17.plot(deltaTheta[delay_steps+1::pool_data[0][0]['dict_pll']['sampleFplot']], deltaThetaDot[delay_steps::pool_data[0][0]['dict_pll']['sampleFplot']], '-', color=color, alpha=alpha, linewidth=linewidth)		# plot trajectory
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
			ax18.plot((deltaThetaDiv[delay_steps+1::pool_data[0][0]['dict_pll']['sampleFplot']]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[delay_steps::pool_data[0][0]['dict_pll']['sampleFplot']], '-', color=color, alpha=alpha, linewidth=linewidth)	# plot trajectory
			ax18.plot((deltaThetaDiv[delay_steps]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[delay_steps], 'o', color=color, alpha=alpha)		# plot initial dot
			ax18.plot((deltaThetaDiv[-1]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[-1], 'x', color=color, alpha=alpha)							# plot final state cross
			#plot_lib.deltaThetaDivDot_vs_deltaThetaDiv(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], (deltaThetaDiv[1:]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot, color, alpha)
			ax19.plot(deltaThetaDiv[delay_steps+1::pool_data[0][0]['dict_pll']['sampleFplot']], deltaThetaDivDot[delay_steps::pool_data[0][0]['dict_pll']['sampleFplot']], '-', color=color, alpha=alpha, linewidth=linewidth)		# plot trajectory
			ax19.plot(deltaThetaDiv[delay_steps], deltaThetaDivDot[delay_steps], 'o', color=color, alpha=alpha)			# plot initial dot
			ax19.plot(deltaThetaDiv[-1], deltaThetaDivDot[-1], 'x', color=color, alpha=alpha)								# plot final state cross

			ax181.plot((deltaThetaDiv[0]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[0], initmarker, color=color, alpha=alpha) # plot initial dot

			deltaThetaDivSave[i] 	= deltaThetaDiv[delay_steps+1::pool_data[0][0]['dict_pll']['sampleFplot']]
			deltaThetaDivDotSave[i] = deltaThetaDivDot[delay_steps::pool_data[0][0]['dict_pll']['sampleFplot']]

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

	fig16.savefig('results/HF-2pi_periodic_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig16.savefig('results/HF-2pi_periodic_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig17.savefig('results/HF_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig17.savefig('results/HF_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig18.savefig('results/LF-2pi_periodic_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig18.savefig('results/LF-2pi_periodic_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig19.savefig('results/LF_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig19.savefig('results/LF_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig161.savefig('results/HF-multStabInfo_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig161.savefig('results/HF-multStabInfo_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig181.savefig('results/LF-multStabInfo_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig181.savefig('results/LF-multStabInfo_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	np.save('results/deltaThetaDivSave_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.npy' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), deltaThetaDivSave)
	np.save('results/deltaThetaDivDotSave_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.npy' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), deltaThetaDivDotSave)
	np.save('results/LF-stats_init_phase_conf_final_state_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.npy' %(np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']), np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), stats_init_phase_conf_final_state)
	plt.draw()
	plt.show()

	return None

################################################################################

def evaluate_entrainment_of_mutual_sync(pool_data: dict, phase_diff_wrap_to_interval: int=3, average_time_for_time_series_in_periods: np.float=3.5) -> None:
	# plot parameter
	axisLabel = 60
	tickSize = 35
	titleLabel = 10
	dpi_val = 150
	figwidth = 10
	figheight = 5
	plot_size_inches_x = 10
	plot_size_inches_y = 5
	alpha = 0.5
	linewidth = 0.5
	labelpadxaxis = 10
	labelpadyaxis = 20

	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']
	marker_list_blue = ['b^', 'bd', 'bo']
	marker_list_red = ['r^', 'rd', 'ro']
	marker_list = ['kd', 'bo', 'r.']

	if phase_diff_wrap_to_interval == 1:  # plot phase-differences in [-pi, pi] interval
		shift2piWin = np.pi
	elif phase_diff_wrap_to_interval == 2:  # plot phase-differences in [-pi/2, 3*pi/2] interval
		shift2piWin = 0.5 * np.pi
	elif phase_diff_wrap_to_interval == 3:  # plot phase-differences in [0, 2*pi] interval
		shift2piWin = 0.0

	plot_time_series = False

	# set a threshold below which the peaks are considered to be originating from the noise and not an actual stable oscillation
	peak_power_noise_threshold = -60

	# setup container array with dimensions: number of realizations, oscillators in each realization WITHOUT the reference
	freq_of_ctrl_signals = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][0, 1:])])
	peak_power_of_ctrl_signal_freq = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][0, 1:])])
	vp2p_of_ctrl_signals = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][0, 1:])])
	mean_phase_differences = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][0, 1:])])
	std_phase_differences = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][0, 1:])])

	mean_ensemble_frequency_mutual_coup_oscis = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][:, 0])-1])
	std_ensemble_frequency_mutual_coup_oscis = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][:, 0])-1])
	mean_frequency_mutual_coup_oscis = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][0, 1:])])
	std_frequency_mutual_coup_oscis = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][0, 1:])])
	mean_ensemble_controlsig_mutual_coup_oscis = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['ctrl'][:, 0])])
	std_ensemble_controlsig_mutual_coup_oscis = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['ctrl'][:, 0])])
	mean_controlsig_mutual_coup_oscis = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][0, 1:])])
	std_controlsig_mutual_coup_oscis = 999 + np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][0, 1:])])
	# loop over all realizations
	for i in range(len(pool_data[0][:])):
		plt.close('all')
		# calculate the range of indexes to be averaged corresponding to a given number of periods of the mean intrinsic frequency (excluding the reference)
		average_time_for_time_series_as_indexes = np.int(average_time_for_time_series_in_periods *
														np.mean(pool_data[0][i]['dict_pll']['intrF'][1:]) / pool_data[0][i]['dict_pll']['dt'])

		# compute instantaneous frequencies for all oscillators, including the reference from the phase-time series
		instantaneous_frequencies = np.diff(pool_data[0][i]['dict_data']['phi'], axis=0) / (2 * np.pi * pool_data[0][i]['dict_pll']['dt'])
		# compute time-series of ensemble averages and standard deviations of instantaneous frequency
		if plot_time_series:
			mean_ensemble_frequency_mutual_coup_oscis[i, :] = np.mean(instantaneous_frequencies[:, 1:], axis=1)
			std_ensemble_frequency_mutual_coup_oscis[i, :] = np.std(instantaneous_frequencies[:, 1:], axis=1)
		# compute the mean and std of instantaneous frequency over time for all oscillators
		mean_frequency_mutual_coup_oscis[i, :] = np.mean(instantaneous_frequencies[-average_time_for_time_series_as_indexes:, 1:], axis=0)
		std_frequency_mutual_coup_oscis[i, :] = np.std(instantaneous_frequencies[-average_time_for_time_series_as_indexes:, 1:], axis=0)
		# compute time-series of ensemble averages and standard deviations of the ctrl signal
		if plot_time_series:
			mean_ensemble_controlsig_mutual_coup_oscis[i, :] = np.mean(pool_data[0][i]['dict_data']['ctrl'][:, 1:], axis=1)
			std_ensemble_controlsig_mutual_coup_oscis[i, :] = np.std(pool_data[0][i]['dict_data']['ctrl'][:, 1:], axis=1)
		# compute the mean and std of the ctrl signal over time for all oscillators
		mean_controlsig_mutual_coup_oscis[i, :] = np.mean(pool_data[0][i]['dict_data']['ctrl'][-average_time_for_time_series_as_indexes:, 1:], axis=0)
		std_controlsig_mutual_coup_oscis[i, :] = np.std(pool_data[0][i]['dict_data']['ctrl'][-average_time_for_time_series_as_indexes:, 1:], axis=0)
		# compute phase mean differences and stds
		mean_phase_differences[i, 0] = np.mean(((pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 1] / pool_data[0][i]['dict_pll']['div'] -
												pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 2] / pool_data[0][i]['dict_pll']['div'] + shift2piWin) % (
															2 * np.pi)) - shift2piWin)
		mean_phase_differences[i, 1] = np.mean(((pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 1] / pool_data[0][i]['dict_pll']['div'] -
												pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 3] / pool_data[0][i]['dict_pll']['div'] + shift2piWin) % (
															2 * np.pi)) - shift2piWin)
		mean_phase_differences[i, 2] = np.mean(((pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 2] / pool_data[0][i]['dict_pll']['div'] -
												pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 3] / pool_data[0][i]['dict_pll']['div'] + shift2piWin) % (
															2 * np.pi)) - shift2piWin)

		std_phase_differences[i, 0] = np.std(((pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 1] / pool_data[0][i]['dict_pll']['div'] -
											pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 2] / pool_data[0][i]['dict_pll']['div'] + shift2piWin) % (
														  2 * np.pi)) - shift2piWin)
		std_phase_differences[i, 1] = np.std(((pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 1] / pool_data[0][i]['dict_pll']['div'] -
											pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 3] / pool_data[0][i]['dict_pll']['div'] + shift2piWin) % (
														  2 * np.pi)) - shift2piWin)
		std_phase_differences[i, 2] = np.std(((pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 2] / pool_data[0][i]['dict_pll']['div'] -
											pool_data[0][i]['dict_data']['phi'][-average_time_for_time_series_as_indexes:, 3] / pool_data[0][i]['dict_pll']['div'] + shift2piWin) % (
														  2 * np.pi)) - shift2piWin)

		# calculate power spectra of the control signal
		f = []
		Pxx_db = []
		# print('pool_data[0][i][*dict_data*][*ctrl*]', pool_data[0][i]['dict_data']['ctrl'])
		# plt.plot(pool_data[0][i]['dict_data']['t'] , pool_data[0][i]['dict_data']['ctrl'][:, 0])
		# plt.draw()
		# plt.show()
		for j in range(1, len(pool_data[0][i]['dict_data']['phi'][0, :])):  # calculate spectrum of signals for all oscillators
			ftemp, Pxx_temp = calcSpectrum(pool_data[0][i]['dict_data']['ctrl'][:, j], pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'],
										pool_data[0][i]['dict_algo'], psd_id=j, percentOfTsim=0.5, signal_given=True)
			# print('Test ftemp[0]:', ftemp[0])
			f.append(ftemp[0])
			# print('Test Pxx_temp[0]:', Pxx_temp[0])
			# time.sleep(2)
			Pxx_db.append(Pxx_temp[0])

		fig0 = plt.figure(num=0, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig0.canvas.manager.set_window_title('PSD plots of the realizations')  # plot spectrum
		fig0.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		plt.xlabel('frequencies [Hz]', fontdict=labelfont, labelpad=labelpadxaxis)
		plt.ylabel('P [dBm]', fontdict=labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)

		peak_power_val = []
		index_of_highest_peak = []
		frequency_of_max_peak = []
		for j in range(len(f)):		# do not evaluate for the reference
			# print('Test:', Pxx_db[i])
			index_of_highest_peak.append(np.argmax(Pxx_db[j]))  			# find the principle peak
			frequency_of_max_peak.append(f[j][index_of_highest_peak[j]])  	# save the frequency where the maximum peak is found
			peak_power_val.append(Pxx_db[j][index_of_highest_peak[j]])  	# save the peak power value
			peak_power_of_ctrl_signal_freq[i, j] = peak_power_val[-1]		# save the peak power value
			if peak_power_val[-1] >= peak_power_noise_threshold:			# only save the frequency if the peak power is above a threshold
				freq_of_ctrl_signals[i, j] = frequency_of_max_peak[-1]		# save the frequency of the first harmonic
			else:
				freq_of_ctrl_signals[i, j] = 0
			time_window_to_detect_min_max_of_signal_if_periodic = int((2.3 / frequency_of_max_peak[-1]) / pool_data[0][i]['dict_pll']['dt'])
			vp2p_of_ctrl_signals[i, j] = (np.max(pool_data[0][i]['dict_data']['ctrl'][-time_window_to_detect_min_max_of_signal_if_periodic:, j+1], axis=0) -
											np.min(pool_data[0][i]['dict_data']['ctrl'][-time_window_to_detect_min_max_of_signal_if_periodic:, j+1], axis=0)) 	# peak to peak amplitude of the control signal at within the averaging range

			plt.plot(f[j], Pxx_db[j], '-', label='PLL%i' % (j+1))

		#print('f[0][2] - f[0][1]), peak_power_val[0])', f[0][2] - f[0][1], peak_power_val[0])
		plt.title(r'power spectrum $\Delta f=$%0.5E' % (f[0][2] - f[0][1]), fontdict=labelfont)
		plt.legend(loc='upper right')
		plt.grid()

		try:
			plt.ylim([np.min(Pxx_db[0][index_of_highest_peak[0]:]), np.max(peak_power_val) + 5])
		except:
			print('Could not set ylim accordingly!')
		plt.xlim(0, 2.5 * np.min(pool_data[0][i]['dict_pll']['intrF']))

		plt.savefig('results/PSD_realization%i_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (i, np.mean(pool_data[0][i]['dict_pll']['coupK']),
			np.mean(pool_data[0][i]['dict_pll']['cutFc']), np.mean(pool_data[0][i]['dict_pll']['syncF']), np.mean(pool_data[0][i]['dict_pll']['transmission_delay']),
			np.mean(pool_data[0][i]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/PSD_realization%i_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (i, np.mean(pool_data[0][i]['dict_pll']['coupK']),
			np.mean(pool_data[0][i]['dict_pll']['cutFc']), np.mean(pool_data[0][i]['dict_pll']['syncF']), np.mean(pool_data[0][i]['dict_pll']['transmission_delay']),
			np.mean(pool_data[0][i]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
		plt.close('all')

		# for a few cases plot the time-series
		data_points = len(pool_data[0][0]['dict_algo']['allPoints'])
		if i == int(0.2*data_points) or i == int(0.5*data_points) or i == int(0.8*data_points):
			plot_lib.plot_inst_frequency_and_phase_difference(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_algo'],
				pool_data[0][i]['dict_data'], False, [], 1, 0.995, 1.005, plot_id=i)
			plt.close('all')
			if 'ctrl' in pool_data[0][0]['dict_data']:
				plot_lib.plot_control_signal_dynamics(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'], plot_id=i)
			plot_lib.plot_inst_frequency_and_order_parameter(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'], [], True, plot_id=i)
			#plot_lib.plot_inst_frequency_and_phase_difference(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_algo'], pool_data[0][i]['dict_data'], False, [], 1, 0.995, 1.005, plot_id=i)
			plt.close('all')

	x_values = pool_data[0][0]['dict_algo']['allPoints']

	print('x_values:', x_values, 'freq_of_ctrl_signals:', freq_of_ctrl_signals)
	print('np.mean(freq_of_ctrl_signals, axis=1):', np.mean(freq_of_ctrl_signals, axis=1), 'np.std(freq_of_ctrl_signals, axis=1):', np.std(freq_of_ctrl_signals, axis=1))
	print('np.mean(vp2p_of_ctrl_signals, axis=1):', np.mean(vp2p_of_ctrl_signals, axis=1), 'np.std(vp2p_of_ctrl_signals, axis=1):', np.std(vp2p_of_ctrl_signals, axis=1))

	# plot the results for each realization, corresponding to the difference x_values: frequencies and frequency of tuning voltages
	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	if isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], list) or isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.ndarray):
		fig1.canvas.manager.set_window_title('results reference frequency scan for {<tau>=%0.02f, topology=%s}' % (np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), pool_data[0][0]['dict_net']['topology']))
	elif isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.float) or isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.int):
		fig1.canvas.manager.set_window_title('results reference frequency scan for {tau=%0.02f, topology=%s}' % (pool_data[0][0]['dict_pll']['transmission_delay'], pool_data[0][0]['dict_net']['topology']))
	ax1 = fig1.add_subplot(111)
	ax2 = ax1.twinx()

	# plot ref frequencies
	ax1.plot(x_values, x_values, 'kx', label=r'$\frac{f_\textrm{\tiny ref}}{\langle\omega_k\rangle_k}$')
	for i in range(len(f)):
		#ax1.plot(x_values, np.mean(freq_of_ctrl_signals, axis=1), 'kd', label=r'$\frac{\langle f_k^\textrm{ctrl} \rangle_k}{\langle\omega_k\rangle_k}$')
		ax2.plot(x_values, freq_of_ctrl_signals[:, i], marker_list[i], label=r'$\frac{f_%i^\textrm{\tiny ctrl}}{\omega_k}$' % i)
		# plot amplitudes
		#ax2.plot(x_values, np.mean(vp2p_of_ctrl_signals, axis=1), 'b^', label=r'$\langle V_\textrm{pp} \rangle_k$')
		#ax2.errorbar(x_values, np.mean(vp2p_of_ctrl_signals, axis=1), np.std(vp2p_of_ctrl_signals, axis=1), None, 'b^', capsize=3, label=r'$\langle V_\textrm{pp} \rangle_k$')

	ax1.set_xlabel(r'$\frac{\omega_\textrm{\small ref}}{\bar{\omega_k}}$', fontsize=axisLabel, color='k')
	ax1.set_ylabel(r'$\frac{\dot{\theta}(t)}{\bar{\omega_k}}$', fontsize=axisLabel, color='k')
	#ax2.set_ylabel(r'$V_\textrm{\tiny pp}$', fontsize=axisLabel, color='b')
	ax2.set_ylabel(r'$\frac{\langle f_k^\textrm{\small ctrl} \rangle_k}{\langle\omega_k\rangle_k}$', fontsize=axisLabel, color='b')

	ax1.legend(loc='center left')
	ax2.legend(loc='center right')
	fig1.savefig('results/evaluation_ctrl_signals_and_frequencies_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']),
	np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig1.savefig('results/evaluation_ctrl_signals_and_frequencies_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']),
	np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	# plot the results for each realization, corresponding to the difference x_values: frequencies and peak-2-peak amplitudes of tuning voltages
	fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	if isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], list) or isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.ndarray):
		fig2.canvas.manager.set_window_title('results ctrl sig scan for {<tau>=%0.02f, topology=%s}' % (np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), pool_data[0][0]['dict_net']['topology']))
	elif isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.float) or isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.int):
		fig2.canvas.manager.set_window_title('results ctrl sig scan for {tau=%0.02f, topology=%s}' % (pool_data[0][0]['dict_pll']['transmission_delay'], pool_data[0][0]['dict_net']['topology']))
	ax1 = fig2.add_subplot(111)
	ax2 = ax1.twinx()

	# plot ref frequencies
	ax1.plot(x_values, x_values, 'kx', label=r'$\frac{\omega_\textrm{\tiny ref}}{\bar{\omega_k}}$')
	for i in range(len(f)):
		# plot the ensemble mean and std of the frequencies of the mutually coupled Plls
		#ax1.plot(x_values[:, 0], mean_frequency_mutual_coup_oscis[:, i], marker_list_red[i], markersize=2.5, label=r'$\frac{\dot{\theta}_%i(t_e)}{\omega_k}$' % i)
		ax1.errorbar(x_values[:, 0], mean_frequency_mutual_coup_oscis[:, i], std_frequency_mutual_coup_oscis[:, i], None, marker_list_red[i], capsize=3, markersize=2.5, label=r'$\frac{\dot{\theta}_%i(t_e)}{\omega_k}$' % i)
		# plot the ensemble mean and std of the control signal peak to peak voltages
		ax2.plot(x_values[:, 0], vp2p_of_ctrl_signals[:, i], marker_list_blue[i], label=r'$V_\textrm{pp}$ PLL%i' % i)
		# ax2.plot(x_values, np.mean(vp2p_of_ctrl_signals, axis=1), 'b^', label=r'$V_\textrm{pp}$')


	ax1.set_xlabel(r'$\frac{\omega_\textrm{\small ref}}{\bar{\omega}_k}$', fontsize=axisLabel, color='k')
	ax1.set_ylabel(r'$\frac{\dot{\theta}(t)}{\bar{\omega_k}}$', fontsize=axisLabel, color='k')
	ax2.set_ylabel(r'$V_\textrm{\small pp}$', fontsize=axisLabel, color='b')

	ax1.legend(loc='center left')
	ax2.legend(loc='center right')
	fig2.savefig('results/evaluation_ctrl_signals_and_amplitudes_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']),
		np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig2.savefig('results/evaluation_ctrl_signals_and_amplitudes_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']),
		np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	# plot the results for each realization, corresponding to the difference x_values: asymptotic phase differences
	fig3 = plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	if isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], list) or isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.ndarray):
		fig3.canvas.manager.set_window_title('phase-differences reference frequency scan for {<tau>=%0.02f, topology=%s}' % (np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), pool_data[0][0]['dict_net']['topology']))
	elif isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.float) or isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.int):
		fig3.canvas.manager.set_window_title('phase-differences reference frequency scan for {tau=%0.02f, topology=%s}' % (pool_data[0][0]['dict_pll']['transmission_delay'], pool_data[0][0]['dict_net']['topology']))
	ax1 = fig3.add_subplot(111)
	# ax2 = ax1.twinx()

	# plot ref frequencies
	#ax2.plot(x_values, x_values, 'kx', label=r'$\frac{\omega_\textrm{\tiny ref}}{\bar{\omega_k}}$')
	for i in range(len(mean_phase_differences[0, :])):
		ax1.errorbar(x_values[:, 0], mean_phase_differences[:, i], std_phase_differences[:, i], None, marker_list[i], capsize=3, markersize=2.5) #, label=r'$\Delta\phi_{%i3}$' % (i+1))
	#ax1.plot(x_values[:, 0], mean_phase_differences[:, 0], 'b^', markersize=3.5, label=r'$\Delta\phi_{AC}$')
	#ax1.plot(x_values[:, 0], mean_phase_differences[:, 1], 'ro', markersize=3.5, label=r'$\Delta\phi_{BC}$')


	# ax2.set_xlabel(r'$\frac{\omega_\textrm{\small ref}}{\bar{\omega}_k}$', fontsize=axisLabel, color='k')
	# ax2.set_ylabel(r'$\frac{\dot{\theta}(t)}{\bar{\omega_k}}$', fontsize=axisLabel, color='k')
	ax1.set_ylabel(r'$\Delta\phi_{kl}$', fontsize=axisLabel, color='k')

	ax1.legend([r'$\Delta\phi_{AB}$', r'$\Delta\phi_{AC}$', r'$\Delta\phi_{BC}$'], loc='center left')
	# ax2.legend(loc='center right')
	fig3.savefig('results/evaluation_ctrl_phase_differences_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']),
		np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	fig3.savefig('results/evaluation_ctrl_phase_differences_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']),
		np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	# plot the results for each realization, corresponding to the difference x_values: time-series of frequencies and control signals
	if plot_time_series:
		for i in range(len(pool_data[0][:])):
			fig4 = plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
			if isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], list) or isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.ndarray):
				fig4.canvas.manager.set_window_title('time series for {<tau>=%0.02f, topology=%s}' % (
				np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), pool_data[0][0]['dict_net']['topology']))
			elif isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.float) or isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.int):
				fig4.canvas.manager.set_window_title('time series for {tau=%0.02f, topology=%s}' % (pool_data[0][0]['dict_pll']['transmission_delay'], pool_data[0][0]['dict_net']['topology']))
			ax1 = fig4.add_subplot(111)
			ax2 = ax1.twinx()

			ax1.errorbar(pool_data[0][0]['dict_data']['t'], mean_ensemble_frequency_mutual_coup_oscis[i, :], std_ensemble_frequency_mutual_coup_oscis[i, :], None, 'kx', capsize=3, label=r'$$')

			ax2.plot(pool_data[0][0]['dict_data']['t'], mean_ensemble_controlsig_mutual_coup_oscis[i, :], std_ensemble_controlsig_mutual_coup_oscis[i, :], 'b^', capsize=3, label=r'$V_\textrm{pp}$')

			ax1.set_xlabel(r'$t$', fontsize=axisLabel, color='k')
			ax1.set_ylabel(r'$\frac{\dot{theta}(t)}{\bar{\omega_k}}$', fontsize=axisLabel, color='k')
			ax2.set_ylabel(r'$V_\textrm{pp}$', fontsize=axisLabel, color='b')

			plt.legend()
			fig4.savefig('results/timeseries_ctrl_signals_and_amplitudes_fref%0.2f_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (x_values[i],
				np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']),
				np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
			fig4.savefig('results/timeseries_ctrl_signals_and_amplitudes_fref%0.2f_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (x_values[i],
				np.mean(pool_data[0][0]['dict_pll']['coupK']), np.mean(pool_data[0][0]['dict_pll']['cutFc']), np.mean(pool_data[0][0]['dict_pll']['syncF']),
				np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), np.mean(pool_data[0][0]['dict_pll']['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	# @Chris: I can now run several individual realizations for different omega_R in parallel - then I use the data and extract:
	# 	the mean and std of the frequencies of the PLLs in the ring/chain, compute the power spectrum of the control signal
	# 	and measure the peak to peak amplitude/std of the control signal?!
	#
	# analyze peak to peak amplitude of ctrl signal
	#
	# extract frequency of ctrl signal
	#
	# check whether mean frequencies of
	#
	# Dict with cross coupling frequency of each node + ref, vtune vpp and frequency

	dict_for_chris = {'fref': x_values, 'instantaneous_frequencies_time_series': instantaneous_frequencies, 'freq_ctrl_sig': freq_of_ctrl_signals,
						'peak_power_frist_harm_ctrl_signal': peak_power_of_ctrl_signal_freq, 'p2p_amplitude_ctrl_sig': vp2p_of_ctrl_signals,
					  	'time_averaged_inst_osci_frequency_values': mean_frequency_mutual_coup_oscis, 'std_of_time_averaged_inst_osci_frequency_values': std_frequency_mutual_coup_oscis,
					  	'mean_phase_differences_wrt_osci_3': mean_phase_differences, 'std_phase_differences_wrt_osci_3': std_phase_differences}
	if isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], list) or isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.ndarray):
		np.save('results/dict_for_chris_tau-%0.3f_topology-%s.npy' % (np.mean(pool_data[0][0]['dict_pll']['transmission_delay']), pool_data[0][0]['dict_net']['topology']), dict_for_chris)
	elif isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.float) or isinstance(pool_data[0][0]['dict_pll']['transmission_delay'], np.int):
		np.save('results/dict_for_chris_tau-%0.3f_topology-%s.npy' % (pool_data[0][0]['dict_pll']['transmission_delay'], pool_data[0][0]['dict_net']['topology']), dict_for_chris)

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
		print('Error: phi with wrong dimensions')
		r = None

	return r, psi

################################################################################

def real_part_kuramoto_order_parameter(phi):
	'''Computes the Kuramoto order parameter r for in-phase synchronized states

	   Parameters
	   ----------
	   phi:  np.array
			real-valued 2d matrix or 1d vector of phases
			in the 2d case the columns of the matrix represent the individual oscillators

	   Returns
	   -------
	   r  :  np.array
			real value or real-valued 1d vetor of the Kuramoto order parameter

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
		print('Error: phi with wrong dimensions')
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

def _calculate_kuramoto_order_parameter_for_one_dim_chequerboard_patterns(phi):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi is supposed
	   to be 1d vector of phases without time evolution.
	'''
	r = 0.0
	for ix in range(len(phi)):
		r += np.exp(1j * phi[ix]) * np.exp(-1j * np.pi * ix)
	r = np.abs(r) / float(len(phi))
	return r

################################################################################

def calculate_kuramoto_order_parameter_for_one_dim_chequerboard_patterns(phi):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi can be a 1d or 2d vector whose first index
	   corresponds to different times.
	   r: type: np.ndarray
	'''
	if len(phi.shape) == 1:
		return _calculate_kuramoto_order_parameter_for_one_dim_chequerboard_patterns(phi)
	else:
		r = np.zeros(phi.shape[0])
		for it in range(phi.shape[0]):
			r[it] = _calculate_kuramoto_order_parameter_for_one_dim_chequerboard_patterns(phi[it, :])
		return r

################################################################################

def _mTwistOrderParameter2d(phi, kx, ky):
	'''Computes the 2d twist order parameters for 2d states. Phi is supposed
	   to be 1d vector of phases. The result is returned as an array of shape (ny, nx)
	'''
	phi_2d = np.reshape(phi, (ky, kx))
	r = np.fft.fft2(np.exp(1j * phi_2d))

	return np.abs(r) / float(len(phi))


def mTwistOrderParameter2d(phi, kx, ky):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi can be a 1d or 2d vector whose first index
	   corresponds to different times.
	'''
	if len(phi.shape) == 1:

		return _mTwistOrderParameter2d(phi, kx, ky)
	else:
		r = []
		for it in range(phi.shape[0]):
			r.append(_mTwistOrderParameter2d(phi[it, :], kx, ky))

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

def oracle_chequerboard_order_parameter_one_dimension(phi, k=0):
	"""
		k == 0 : global sync state
		k == 1 : checkerboard state
	"""
	if k == 0:
		return real_part_kuramoto_order_parameter(phi)
	elif k == 1:
		return calculate_kuramoto_order_parameter_for_one_dim_chequerboard_patterns(phi)
	else:
		raise Exception('Non-valid value for k.')

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
		return real_part_kuramoto_order_parameter(phi)
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

# def evaluateSimulationIsing(pool_data, phase_wrap=0, number_of_bins=25, prob_density=False, order_param_solution=0.0, number_of_expected_oscis_in_one_group=10):
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
# 	#unit_cell = PhaseDifferenceCell(pool_data[0][0]['dict_net']['Nx']*pool_data[0][0]['dict_net']['Ny'])
# 	threshold_statState = np.pi/15
# 	plotEveryDt = 1
# 	numberColsPlt = 3
# 	numberColsPlt_widePlt = 1
# 	number_of_intrinsic_periods_smoothing = 1.5
# 	print('For smoothing of phase-differences and order parameters we average over %0.2f periods of the ensemble mean intrinsic frequency.' % number_of_intrinsic_periods_smoothing)
#
# 	fig16, ax16 = plt.subplots(int(np.ceil(len(pool_data[0][:])/numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig16.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, phase relations')					# phase relations
# 	if isinstance( pool_data[0][0]['dict_pll']['cutFc'], np.float):
# 		fig16.suptitle(r'parameters: $f=$%0.2f Hz, $K=$%0.2f Hz/V, $K^\textrm{I}=$%0.2f Hz/V, $f_c=$%0.2f Hz, $\tau=$%0.2f s'%(pool_data[0][0]['dict_pll']['intrF'], pool_data[0][0]['dict_pll']['coupK'], pool_data[0][0]['dict_pll']['coupStr_2ndHarm'], pool_data[0][0]['dict_pll']['cutFc'], pool_data[0][0]['dict_pll']['transmission_delay']))
# 	fig16.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax16 = ax16.ravel()
#
# 	fig161, ax161 = plt.subplots(int(np.ceil(len(pool_data[0][:]) / numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig161.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, smoothed out phase relations')  # phase relations
# 	if isinstance(pool_data[0][0]['dict_pll']['cutFc'], np.float):
# 		fig161.suptitle(r'parameters: $f=$%0.2f Hz, $K=$%0.2f Hz/V, $K^\textrm{I}=$%0.2f Hz/V, $f_c=$%0.2f Hz, $\tau=$%0.2f s' % (
# 		pool_data[0][0]['dict_pll']['intrF'], pool_data[0][0]['dict_pll']['coupK'], pool_data[0][0]['dict_pll']['coupStr_2ndHarm'], pool_data[0][0]['dict_pll']['cutFc'],
# 		pool_data[0][0]['dict_pll']['transmission_delay']))
# 	fig161.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax161 = ax161.ravel()
#
# 	fig17, ax17 = plt.subplots(int(np.ceil(len(pool_data[0][:])/numberColsPlt_widePlt)), numberColsPlt_widePlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig17.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, inst. frequencies')					# inst. frequencies
# 	fig17.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax17 = ax17.ravel()
#
# 	fig18, ax18 = plt.subplots(int(np.ceil(len(pool_data[0][:])/numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig18.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, order parameter')					# order parameter
# 	fig18.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax18 = ax18.ravel()
#
# 	fig19, ax19 = plt.subplots(int(np.ceil(len(pool_data[0][:])/numberColsPlt_widePlt)), numberColsPlt_widePlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig19.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, signals')							# signals
# 	fig19.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax19 = ax19.ravel()
#
# 	fig20, ax20 = plt.subplots(int(np.ceil(len(pool_data[0][:]) / numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig20.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, histograms')  # signals
# 	fig20.subplots_adjust(hspace=0.4, wspace=0.4)
# 	ax20 = ax20.ravel()
#
# 	fig99, ax99 = plt.subplots(1, 1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# 	fig99.canvas.manager.set_window_title('Network view of result.')  # network
#
# 	if len(pool_data[0][:]) > threshold_realizations_plot: # only plot when many realizations are computed for overview
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
# 	# print('pool_data in eva.evaluateSimulationIsing(pool_data):', pool_data)
#
# 	print('For evaluation of asymptotic order parameter we average over %0.2f periods of the ensemble mean intrinsic frequency.' % number_of_intrinsic_periods_smoothing)
#
# 	sol_time = []
# 	success_count = 0
# 	success_count_test1 = 0
# 	group_oscillators_maxcut = np.zeros([len(pool_data[0][:]), len(pool_data[0][0]['dict_data']['phi'][0, :])])
#
# 	# loop over the realizations
# 	for i in range(len(pool_data[0][:])):
# 		deltaTheta = np.zeros([len(pool_data[0][i]['dict_data']['phi'][0, :]), len(pool_data[0][i]['dict_data']['phi'][:, 0])])
# 		signalOut  = np.zeros([len(pool_data[0][i]['dict_data']['phi'][0, :]), len(pool_data[0][i]['dict_data']['phi'][:, 0])])
#
# 		thetaDot = np.diff( pool_data[0][i]['dict_data']['phi'][:, :], axis=0 ) / pool_data[0][i]['dict_pll']['dt']				# compute frequencies and order parameter
# 		order_parameter, order_parameter_divided_phases, F1 = compute_order_parameter(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_data'])
#
# 		ax18[i].plot( pool_data[0][i]['dict_data']['t'][::plotEveryDt], order_parameter[::plotEveryDt], label=r'$R_\textrm{final}=%0.2f$'%(order_parameter[-1]), linewidth=linewidth )
#
# 		# HOWTO 1) to determine whether the correct solution has be found, we test for the asymptotic value of the order parameter
# 		order_param_diff_expected_value_threshold = 0.01
# 		correct_solution_test0 = False
# 		if np.abs(np.mean(order_parameter[-int(number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']):]) - order_param_solution) < order_param_diff_expected_value_threshold:
# 			print('Order parameter predicted for solution=%0.2f has been reached. Averaged over last %i periods of the intrinsic frequency for realization %i.'%(order_param_solution, number_of_intrinsic_periods_smoothing, i))
# 			success_count += 1					# to calculate the probability of finding the correct solutions
# 			correct_solution_test0 = True		# this is needed to decide for which realizations we need to measure the time to solution
#
# 		# HOWTO 2) to determine whether the correct solution has be found, we also test for mutual phase-differences between the oscillators
# 		group1 = 0
# 		group2 = 0
# 		correct_solution_test1 = False
# 		for j in range(len(pool_data[0][i]['dict_data']['phi'][0, :])):
# 			# calculate mean phase difference over an interval of 'number_of_intrinsic_periods_smoothing' periods at the end of all oscillators with respect to oscillator zero
# 			# interval_index = -int(number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt'])
# 			temp_phase_diff = np.mean(pool_data[0][i]['dict_data']['phi'][-int(number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']):, 0] - pool_data[0][i]['dict_data']['phi'][-int(number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']):, j])
# 			print('Realization %i, mean phase difference {mod 2pi into [-pi, pi)} between k=0 and k=%i is deltaPhi=%0.2f'%(i, j, ((temp_phase_diff+np.pi) % (2*np.pi))-np.pi))
# 			if np.abs(((temp_phase_diff+np.pi) % (2*np.pi))-np.pi) < np.pi/2:
# 				group1 += 1
# 			else:
# 				group2 += 1
# 		if not group1+group2 == len(pool_data[0][i]['dict_data']['phi'][0, :]):
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
# 				derivative_order_param_smoothed = (np.diff( uniform_filter1d( order_parameter[pool_data[0][i]['dict_net']['max_delay_steps']:],
# 					size=int(15 * number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']), mode='reflect') ) / pool_data[0][i]['dict_pll']['dt'])
#
# 				rolling_std_derivative_order_param_smoothed = pd.Series(derivative_order_param_smoothed).rolling(int(15 * number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt'])).std()
#
# 				#temp = np.where( (np.diff( uniform_filter1d( order_parameter[(pool_data[0][i]['dict_data']['tstep_annealing_start'] + pool_data[0][i]['dict_net']['max_delay_steps']):],
# 				#	size=int(15 * number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']), mode='reflect') ) / pool_data[0][i]['dict_pll']['dt']) > order_param_change_threshold )
# 				min_std = np.min(rolling_std_derivative_order_param_smoothed)
# 				print('min_std:', min_std)
# 				max_std = np.max(rolling_std_derivative_order_param_smoothed)
# 				order_param_std_threshold = 0.1 * (max_std - min_std) + min_std
# 				print('Realization %i, order_param_std_threshold to %0.02f, for {min_std, max_std} = {%0.2f,%0.2f} '%(i, order_param_std_threshold, min_std, max_std))
# 				temp = np.where(rolling_std_derivative_order_param_smoothed[pool_data[0][i]['dict_data']['tstep_annealing_start']:] > order_param_std_threshold)
#
# 				plt.plot(derivative_order_param_smoothed, 'b')
# 				plt.plot(rolling_std_derivative_order_param_smoothed, 'r--')
# 				plt.plot(temp[0][-1]-pool_data[0][i]['dict_data']['tstep_annealing_start'], 0, 'cd')
#
# 				# print('temp=', temp[0])
# 				if not len(temp[0]) == 0:
# 					# subtract from the last time when the transient dynamics caused order parameter fluctuations above the threshold the time when the annealing process started
# 					# the substraction of the initial delay history is already done since we only search from tau onwards for the time at which the fluctuations fulfill the conditions
# 					sol_time.append((temp[0][-1]) * pool_data[0][i]['dict_pll']['dt'])
# 				else:
# 					sol_time.append(np.inf)
# 				# print('sol_time=', sol_time)
# 				plt.draw()
# 				plt.show()
# 			else:
# 				sol_time.append(np.inf)
#
# 			ax18[i].plot( pool_data[0][i]['dict_data']['t'][(pool_data[0][i]['dict_data']['tstep_annealing_start'] + pool_data[0][i]['dict_net']['max_delay_steps']):-1:plotEveryDt], uniform_filter1d((np.diff(
# 										order_parameter[(pool_data[0][i]['dict_data']['tstep_annealing_start'] + pool_data[0][i]['dict_net']['max_delay_steps']):]) / pool_data[0][i]['dict_pll']['dt']),
# 										size=int(0.5 * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']), mode='reflect')[::plotEveryDt], 'r', linewidth=0.5, alpha=0.35 )
# 		else:
# 			if correct_solution_test1:
# 				temp = np.where( np.diff(order_parameter[(pool_data[0][i]['dict_data']['tstep_annealing_start'] + pool_data[0][i]['dict_net']['max_delay_steps']):]) / pool_data[0][i]['dict_pll']['dt'] > order_param_change_threshold )
# 				sol_time.append(temp[0][-1] * pool_data[0][i]['dict_pll']['dt'])
# 			else:
# 				sol_time.append(np.inf)
#
# 			ax18[i].plot( pool_data[0][i]['dict_data']['t'][(pool_data[0][i]['dict_data']['tstep_annealing_start'] + pool_data[0][i]['dict_net']['max_delay_steps']):-1:plotEveryDt], (np.diff(
# 								order_parameter[(pool_data[0][i]['dict_data']['tstep_annealing_start'] + pool_data[0][i]['dict_net']['max_delay_steps']):]) / pool_data[0][i]['dict_pll']['dt'])[::plotEveryDt], 'r', linewidth=0.5, alpha=0.35 )
#
# 		ax18[i].plot(pool_data[0][i]['dict_data']['t'][(pool_data[0][i]['dict_data']['tstep_annealing_start'] + pool_data[0][i]['dict_net']['max_delay_steps'])], 0, 'cd', markersize=1)
# 		if correct_solution_test0 and sol_time[i] != np.inf:
# 			ax18[i].plot(sol_time[i] + pool_data[0][i]['dict_data']['t'][(pool_data[0][i]['dict_data']['tstep_annealing_start'] + pool_data[0][i]['dict_net']['max_delay_steps'])], 0, 'c*', markersize=1)
#
# 		if len(pool_data[0][:]) > threshold_realizations_plot:
# 			ax21.plot(pool_data[0][i]['dict_data']['t'][(pool_data[0][i]['dict_data']['tstep_annealing_start'] + pool_data[0][i]['dict_net']['max_delay_steps'])], -0.05, 'cd', markersize=1)
# 			if correct_solution_test0:
# 				ax21.plot(pool_data[0][i]['dict_data']['t'][::plotEveryDt], order_parameter[::plotEveryDt], '-', label=r'$R_\textrm{final}=%0.2f$' % (order_parameter[-1]), linewidth=linewidth)
# 				ax211.plot(pool_data[0][i]['dict_data']['t'][::plotEveryDt], uniform_filter1d(order_parameter[::plotEveryDt],
# 							size=int(number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']), mode='reflect'), '-', label=r'$R_\textrm{final}=%0.2f$' % (order_parameter[-1]), linewidth=linewidth)
# 				if sol_time[i] != np.inf:
# 					ax21.plot(sol_time[i] + pool_data[0][i]['dict_data']['t'][(pool_data[0][i]['dict_data']['tstep_annealing_start'] + pool_data[0][i]['dict_net']['max_delay_steps'])], -0.05, 'c*', markersize=1)
# 			else:
# 				ax21.plot(pool_data[0][i]['dict_data']['t'][::plotEveryDt], order_parameter[::plotEveryDt], '--', label=r'$R_\textrm{final}=%0.2f$' % (order_parameter[-1]), linewidth=linewidth)
# 				ax211.plot(pool_data[0][i]['dict_data']['t'][::plotEveryDt], uniform_filter1d(order_parameter[::plotEveryDt],
# 							size=int(number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']), mode='reflect'), '--', label=r'$R_\textrm{final}=%0.2f$' % (order_parameter[-1]), linewidth=linewidth)
#
#
# 		if phase_wrap == 0:  # plot phase differences in [-inf, inf), i.e., we use the unwrapped phases that have counted the cycles/periods
# 			ax20[i].hist(pool_data[0][i]['dict_data']['phi'][-3, :] - pool_data[0][i]['dict_data']['phi'][-2, 0], bins=number_of_bins, rwidth=0.9, density=prob_density)
# 		elif phase_wrap != 0:
# 			# print('histogram_data (wrapping if phase):', ((dict_data['phi'][at_index, plotlist] + shift2piWin) % (2 * np.pi)) - shift2piWin)
# 			ax20[i].hist((((pool_data[0][i]['dict_data']['phi'][-3, :] - pool_data[0][i]['dict_data']['phi'][-2, 0] + shift2piWin) % (2.0 * np.pi)) - shift2piWin), bins=number_of_bins, rwidth=0.9, density=prob_density)
#
# 		final_phase_oscillator = []
# 		for j in range(len(pool_data[0][i]['dict_data']['phi'][0, :])):
# 			if shift2piWin != 0:
# 				deltaTheta[j] = (((pool_data[0][i]['dict_data']['phi'][:, 0] - pool_data[0][i]['dict_data']['phi'][:, j]) + shift2piWin) % (2.0 * np.pi)) - shift2piWin 		# calculate phase-differnce w.r.t. osci k=0
# 			else:
# 				deltaTheta[j] = pool_data[0][i]['dict_data']['phi'][:, 0] - pool_data[0][i]['dict_data']['phi'][:, j]
# 			signalOut[j] = pool_data[0][i]['dict_pll']['vco_out_sig'](pool_data[0][i]['dict_data']['phi'][:, j])				# generate signals for all phase histories
#
# 			# save in which binarized state the oscillator was at the end of the realization, averaged over
# 			if np.mean(deltaTheta[j][-int(number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']):-1]) - 0 < 0.2:
# 				group_oscillators_maxcut[i, j] = -1
# 				final_phase_oscillator.append('zero')
# 			elif np.mean(deltaTheta[j][-int(number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']):-1]) - np.pi < 0.2:
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
# 			ax16[i].plot( pool_data[0][i]['dict_data']['t'][::plotEveryDt], deltaTheta[j, ::plotEveryDt], linestyle, linewidth=linewidth, label='sig PLL%i' %(j))
# 			ax161[i].plot(pool_data[0][i]['dict_data']['t'], uniform_filter1d(deltaTheta[j, :], size=int(number_of_intrinsic_periods_smoothing * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt']), mode='reflect'), linestyle, linewidth=linewidth, label='sig PLL%i' % (j))
# 			ax19[i].plot( pool_data[0][i]['dict_data']['t'][::plotEveryDt], pool_data[0][i]['dict_pll']['vco_out_sig'](pool_data[0][i]['dict_data']['phi'][::plotEveryDt, j]), linewidth=linewidth, label='sig PLL%i' %(j))
# 			ax17[i].plot( pool_data[0][i]['dict_data']['t'][1::plotEveryDt], thetaDot[::plotEveryDt, j], linewidth=linewidth, label='sig PLL%i' %(j))
#
# 		print('working on realization %i results from sim:'%i, pool_data[0][i]['dict_net'], '\n', pool_data[0][i]['dict_pll'], '\n', pool_data[0][i]['dict_data'], '\n\n')
#
# 		if i == int( len(pool_data[0][:]) / 2 ):
# 			ax16[i].set_ylabel(r'$\Delta\theta(t)$', fontsize=axisLabel)
# 			ax161[i].set_ylabel(r'$\langle\Delta\theta(t)\rangle_{%0.1f T}$'%(number_of_intrinsic_periods_smoothing), fontsize=axisLabel)
# 			ax17[i].set_ylabel(r'$\dot{\theta}(t)$ in radHz', fontsize=axisLabel)
# 			ax18[i].set_ylabel(r'$R(t)$', fontsize=axisLabel)
# 			ax19[i].set_ylabel(r'$s(t)$', fontsize=axisLabel)
# 			ax20[i].set_ylabel(r'$H\left(\Delta\theta(t)\right)$', fontsize=axisLabel)
# 		if i == len(pool_data[0][:])-2:
# 			ax16[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax161[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax18[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax20[i].set_xlabel(r'$\Delta\theta(t)$ in $[rad]$', fontsize=axisLabel)
# 		if i == len(pool_data[0][:])-1:
# 			ax17[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 			ax19[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
# 		if len(pool_data[0][:]) > threshold_realizations_plot and i == len(pool_data[0][:])-1:
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
# 		if isinstance(pool_data[0][i]['dict_pll']['gPDin'], list):
# 			network_graph = nx.from_numpy_array(np.array(pool_data[0][i]['dict_pll']['gPDin']))
# 		else:
# 			network_graph = nx.from_numpy_array(pool_data[0][i]['dict_pll']['gPDin'])
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
# 	print('success_count: ', success_count, 'len(pool_data[0][:]: ', len(pool_data[0][:]), 'sol_time:', sol_time)
# 	results_string = 'Final evaluation:\n1) for a total of %i realizations, success probability (evaluation R) = %0.4f\n2) and success probability evaluating groups separated by pi = %0.4f\n3) average time to solution = %0.4f seconds, i.e., %0.2f mean intrinsic periods.\n4) average time to solution without infinity entries= %0.4f seconds, i.e., %0.2f mean intrinsic periods.\n5) fastest and slowest time to solution in multiples of periods: {%0.2f, %0.2f}'%(
# 		len(pool_data[0][:]), success_count / len(pool_data[0][:]), success_count_test1 / len(pool_data[0][:]),
# 		np.mean(sol_time),
# 		np.mean(sol_time)/np.mean(pool_data[0][i]['dict_pll']['intrF']),
# 		np.mean(sol_time_without_inf_entries),
# 		np.mean(sol_time_without_inf_entries)/np.mean(pool_data[0][i]['dict_pll']['intrF']),
# 		np.min(sol_time)/np.mean(pool_data[0][i]['dict_pll']['intrF']),
# 		np.max(sol_time)/np.mean(pool_data[0][i]['dict_pll']['intrF']))
# 	if len(pool_data[0][:]) > threshold_realizations_plot:
# 		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# 		#ax21.text(0.25*pool_data[0][0]['dict_data']['t'][-1], 0.2, results_string, horizontalalignment='left', verticalalignment='bottom', bbox=props, fontsize=9)
# 		ax211.text(0.25 * pool_data[0][0]['dict_data']['t'][-1], 0.2, results_string, horizontalalignment='left', verticalalignment='bottom', bbox=props, fontsize=9)
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
# 	if len(pool_data[0][:]) > threshold_realizations_plot:
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