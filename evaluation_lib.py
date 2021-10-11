#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import sys, gc
import inspect
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
import itertools
from itertools import permutations as permu
from itertools import combinations as combi
import matplotlib
import os, pickle
if not os.environ.get('SGE_ROOT') == None:										# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import time
import datetime
now = datetime.datetime.now()
import scipy
from scipy import signal
import integer_mult_period_signal_lib as findIntTinSig
import plot_lib

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
	fig1.canvas.set_window_title('test plot phase')  							 # plot the phase
	plt.clf()

	plt.plot(params['x'], params['y']%(2*np.pi), color=color[0], linewidth=1, linestyle=linet[0], label=labeldict1[params['label']])
	plt.plot(params['x'][params['delay_steps']-1], params['y'][int(params['delay_steps'])-1,0]+0.05,'go')

	plt.xlabel(labeldict1[params['xlabel']], fontdict = labelfont)
	plt.ylabel(labeldict1[params['ylabel']], fontdict = labelfont)
	plt.legend()

	fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig2.canvas.set_window_title('test plot frequency')							# plot the instantaneous frequency
	plt.clf()

	plt.plot(params['x'][0:-1], np.diff(params['y'], axis=0)/(2*np.pi*params['dt']), color=color[0], linewidth=1, linestyle=linet[0], label=r'f(t)')

	plt.xlabel(labeldict1[params['xlabel']], fontdict = labelfont)
	plt.ylabel(r'$f(t)$ Hz', fontdict = labelfont)
	plt.legend()

	plt.draw()
	plt.show()

################################################################################


def prepareDictsForPlotting(dictPLL, dictNet):

	if dictPLL['cutFc'] == None:
		dictPLL.update({'cutFc': np.inf})

	if not np.abs(np.min(dictPLL['intrF'])) > 1E-17:									# for f=0, there would otherwies be a float division by zero
		dictPLL.update({'intrF': 1})
		print('Since intrinsic frequency was zero: for plotting set to one to generate boundaries!')


	return dictPLL, dictNet


def saveDictionaries(dictToSave, name, K, tau, Fc, Nx, Ny, mx, my, topology):

	if Fc == None:
		Fc = np.inf

	N = int(Nx*Ny)
	filename = 'results/%s_K%.3f_tau%.3f_Fc%.3f_mx%i_my%i_N%i_topo%s_%d:%d_%d_%d_%d'%(name, np.mean(K), np.mean(tau), np.mean(Fc), mx, my, N, topology, now.hour, now.minute, now.year, now.month, now.day)
	f 		 = open(filename,'wb')
	pickle.dump(dictToSave,f)
	f.close()

	return None

################################################################################

def calculateEigenvalues(dictNet, dictPLL):
	''' Calculate eigenvalues zeta for networks of homogeneous PLL '''

	if dictNet['topology'] == 'global':											# test wheter global coupling topology
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
def calcSpectrum( phi, dictPLL, dictNet, percentOfTsim=0.5 ): #phi,Fsample,couplingfct,waveform=None,expectedFreq=-999,evalAllRealizations=False,decayTimeSlowestMode=None

	Pxx_dBm=[]; Pxx_dBV=[]; f=[];
	windowset='hamming' #'hamming' #'hamming', 'boxcar'
	print('\nCurrent window option is', windowset, 'for waveform', inspect.getsourcelines(dictPLL['vco_out_sig'])[0][0],
			'NOTE: in principle can always choose to be sin() for cleaner PSD in first harmonic approximation of the signal.')
	print('Calculate spectrum for',percentOfTsim,'percent of the time-series. Implement better solution using decay times.')
	try:
		analyzeL = findIntTinSig.cutTimeSeriesOfIntegerPeriod(dictPLL['sampleF'], dictNet['Tsim'], dictPLL['syncF'],
																np.max([dictPLL['coupK'], dictPLL['coupStr_2ndHarm']]), phi, percentOfTsim);
		window	 = scipy.signal.get_window('boxcar', int(dictPLL['sampleF']), fftbins=True); # here we choose boxcar since a modification of the ends of the time-series is not necessary for an integer number of periods
	except:
		print('\n\nError in cutTimeSeriesOfIntegerPeriod-function! Not picking integer number of periods for PSD!\n\n')
		analyzeL= [ int( dictNet['Tsim']*dictPLL['sampleF']*(1-percentOfTsim) ), int( dictNet['Tsim']*dictPLL['sampleF'] )-1 ]
		window	= scipy.signal.get_window(windowset, int(dictPLL['sampleF']), fftbins=True);

	tsdata		= dictPLL['vco_out_sig'](phi[analyzeL[0]:analyzeL[1]])

	ftemp, Vxx 	= scipy.signal.periodogram(tsdata, dictPLL['sampleF'], return_onesided=True, window=windowset, scaling='density', axis=0) #  returns Pxx with dimensions [V^2] if scaling='spectrum' and [V^2/Hz] if if scaling='density'
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
	return pll_list[0].low_pass_filter.y

################################################################################

''' MODEL FITTING: DEMIR MODEL '''
def fitModelDemir(f_model,d_model,fitrange=0):

	f_peak = f_model[np.argmax(d_model)]										# find main peak

	if fitrange != 0:															# mask data
		ma = np.ma.masked_inside(f_model,f_peak-fitrange,f_peak+fitrange)
		f_model_ma = f_model[ma.mask]
		d_model_ma = d_model[ma.mask]
	else:
		f_model_ma = f_model
		d_model_ma = d_model

	A = np.sqrt(2)																# calculate power of main peak for sine wave
	P_offset = 10*np.log10(A**2/2)

	optimize_func = lambda p: P_offset + 10*np.log10( (p[0]**2 * p[1])/(np.pi * p[0]**4 * p[1]**2 + (f_model_ma-p[0])**2 )) # model fit
	error_func = lambda p: optimize_func(p) - d_model_ma
	p_init = (f_peak,1e-8)
	p_final,success = leastsq(error_func,p_init[:])

	f_model_ma = f_model														# restore data
	d_model_ma = d_model

	return f_model, optimize_func(p_final), p_final

################################################################################
################################################################################
################################################################################

def obtainOrderParam(dictPLL, dictNet, dictData):
	''' MODIFIED KURAMOTO ORDER PARAMETERS '''
	numb_av_T = 3;																			   # number of periods of free-running frequencies to average over
	if np.min(dictPLL['intrF']) > 0:																				   # for f=0, there would otherwies be a float division by zero
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
			ktemp=4;
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
			fig0.canvas.set_window_title('order parameters mutual and entrained')			   # plot orderparameter
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
			orderparam = np.zeros(len(dictData['phi'][:,0]))
	elif dictNet['topology'] == "chain":
		"""
				dictNet['mx']  > 0 : x  checkerboard state
				dictNet['mx'] == 0 : in-phase synchronized
			"""
		r = oracle_CheckerboardOrderParameter1d(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):, :], dictNet['mx'])
		orderparam = oracle_CheckerboardOrderParameter1d(dictData['phi'][:, :])							# calculate the order parameter for all times
	elif ( dictNet['topology'] == "ring" or dictNet['topology'] == 'global'):
		r = oracle_mTwistOrderParameter(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):, :], dictNet['mx'])# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
		orderparam = oracle_mTwistOrderParameter(dictData['phi'][:, :], dictNet['mx'])					# calculate the m-twist order parameter for all times
	elif ( dictNet['topology'] == "entrainOne" or dictNet['topology'] == "entrainAll" or dictNet['topology'] == "entrainPLLsHierarch"):
		phi_constant_expected = dictNet['phiInitConfig'];
		r = calcKuramotoOrderParEntrainSelfOrgState(dictData['phi'][-int(numb_av_T*1.0/(F1*dictPLL['dt'])):, :], phi_constant_expected);
		orderparam = calcKuramotoOrderParEntrainSelfOrgState(dictData['phi'][:, :], phi_constant_expected);
	# r = oracle_mTwistOrderParameter(dictData['phi'][-int(2*1.0/(F1*dictPLL['dt'])):, :], dictNet['mx'])			# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
	# orderparam = oracle_mTwistOrderParameter(dictData['phi'][:, :], dictNet['mx'])					# calculate the m-twist order parameter for all times
	# print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])
	print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])

	return r, orderparam, F1

################################################################################

def evaluateSimulationIsing(poolData):

	# plot parameter
	axisLabel  = 9;
	legendLab  = 6
	tickSize   = 5;
	titleLabel = 9;
	dpi_val	   = 150;
	figwidth   = 6;
	figheight  = 5;
	alpha 	   = 0.5;

	#unit_cell = PhaseDifferenceCell(poolData[0][0]['dictNet']['Nx']*poolData[0][0]['dictNet']['Ny'])
	treshold_statState = np.pi/15
	plotEveryDt		   = 1;
	numberColsPlt	   = 3;

	fig16, ax16 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig16.canvas.set_window_title('Ising different initial conditions, fixed topology, phase relations')	# phase relations
	if isinstance( poolData[0][0]['dictPLL']['cutFc'], np.float):
		fig16.suptitle(r'parameters: $f=$%0.2f Hz, $K=$%0.2f Hz/V, $K^\textrm{I}=$%0.2f Hz/V, $f_c=$%0.2f Hz, $\tau=$%0.2f s'%(poolData[0][0]['dictPLL']['intrF'], poolData[0][0]['dictPLL']['coupK'], poolData[0][0]['dictPLL']['coupStr_2ndHarm'], poolData[0][0]['dictPLL']['cutFc'], poolData[0][0]['dictPLL']['transmission_delay']))
	fig16.subplots_adjust(hspace=0.4, wspace=0.4)
	ax16 = ax16.ravel()

	fig17, ax17 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig17.canvas.set_window_title('Ising different initial conditions, fixed topology, inst. frequencies')	# inst. frequencies
	fig17.subplots_adjust(hspace=0.4, wspace=0.4)
	ax17 = ax17.ravel()

	fig18, ax18 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig18.canvas.set_window_title('Ising different initial conditions, fixed topology, order parameter')	# order parameter
	fig18.subplots_adjust(hspace=0.4, wspace=0.4)
	ax18 = ax18.ravel()

	fig19, ax19 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig19.canvas.set_window_title('Ising different initial conditions, fixed topology, signals')			# signals
	fig19.subplots_adjust(hspace=0.4, wspace=0.4)
	ax19 = ax19.ravel()

	print('poolData in eva.evaluateSimulationIsing(poolData):', poolData)

	for i in range(len(poolData[0][:])):
		deltaTheta = np.zeros([len(poolData[0][i]['dictData']['phi'][0,:]), len(poolData[0][i]['dictData']['phi'][:,0])]);
		signalOut  = np.zeros([len(poolData[0][i]['dictData']['phi'][0,:]), len(poolData[0][i]['dictData']['phi'][:,0])]);

		thetaDot 			= np.diff( poolData[0][i]['dictData']['phi'][:,:], axis=0 ) / poolData[0][i]['dictPLL']['dt']	# compute frequencies and order parameter
		r, orderparam, F1 	= obtainOrderParam(poolData[0][i]['dictPLL'], poolData[0][i]['dictNet'], poolData[0][i]['dictData'])

		ax18[i].plot( poolData[0][i]['dictData']['t'][::plotEveryDt], orderparam[::plotEveryDt] )

		for j in range(len(poolData[0][i]['dictData']['phi'][0,:])):
			deltaTheta[j] 	= poolData[0][i]['dictData']['phi'][:,0] - poolData[0][i]['dictData']['phi'][:,j] 				# calculate phase-differnce w.r.t. osci k=0
			signalOut[j]	= poolData[0][i]['dictPLL']['vco_out_sig'](poolData[0][i]['dictData']['phi'][:,j])				# generate signals for all phase histories

			ax16[i].plot( poolData[0][i]['dictData']['t'][::plotEveryDt], deltaTheta[j,::plotEveryDt], label='sig PLL%i' %(j))
			ax19[i].plot( poolData[0][i]['dictData']['t'][::plotEveryDt], poolData[0][i]['dictPLL']['vco_out_sig'](poolData[0][i]['dictData']['phi'][::plotEveryDt,j]), label='sig PLL%i' %(j))
			ax17[i].plot( poolData[0][i]['dictData']['t'][1::plotEveryDt], thetaDot[::plotEveryDt,j], label='sig PLL%i' %(j))

		print('working on realization %i results from sim:'%i, poolData[0][i]['dictNet'], '\n', poolData[0][i]['dictPLL'], '\n', poolData[0][i]['dictData'],'\n\n')

		ax16[i].set_xlabel(r'$t$ in $[s]$', 				fontsize=axisLabel)
		ax16[i].set_ylabel(r'$\Delta\theta(t)$', 			fontsize=axisLabel)
		ax17[i].set_xlabel(r'$t$ in $[s]$', 				fontsize=axisLabel)
		ax17[i].set_ylabel(r'$\dot{\theta}(t)$ in radHz', 	fontsize=axisLabel)
		ax18[i].set_xlabel(r'$t$ in $[s]$', 				fontsize=axisLabel)
		ax18[i].set_ylabel(r'$R(t)$', 						fontsize=axisLabel)
		ax19[i].set_xlabel(r'$t$ in $[s]$', 				fontsize=axisLabel)
		ax19[i].set_ylabel(r'$s(t)$', 						fontsize=axisLabel)

		ax16[i].tick_params(labelsize=tickSize)
		ax17[i].tick_params(labelsize=tickSize)
		ax18[i].tick_params(labelsize=tickSize)
		ax19[i].tick_params(labelsize=tickSize)

	ax16[0].legend(loc='upper right', fontsize=legendLab)
	ax17[0].legend(loc='upper right', fontsize=legendLab)
	ax18[0].legend(loc='upper right', fontsize=legendLab)
	ax19[0].legend(loc='upper right', fontsize=legendLab)

	plt.draw(); plt.show()

	return None

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

	unit_cell = PhaseDifferenceCell(poolData[0][0]['dictNet']['Nx']*poolData[0][0]['dictNet']['Ny'])
	treshold_statState = np.pi/15

	fig16 = plt.figure(num=16, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig16.canvas.set_window_title('HF (VCO output) basin attraction plot - 2pi periodic')			# basin attraction plot
	ax16 = fig16.add_subplot(111)

	fig161 = plt.figure(num=161, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig161.canvas.set_window_title('HF (VCO output) basin attraction (diamond W>w, circle W<w)')	# basin attraction plot
	ax161 = fig161.add_subplot(111)

	fig17 = plt.figure(num=17, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig17.canvas.set_window_title('HF (VCO output) output basin attraction plot')			 		# basin attraction plot
	ax17 = fig17.add_subplot(111)

	fig18 = plt.figure(num=18, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig18.canvas.set_window_title('LF (cross-coupling) basin attraction plot - 2pi periodic')		# basin attraction plot
	ax18 = fig18.add_subplot(111)

	fig181 = plt.figure(num=181, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig181.canvas.set_window_title('LF (cross-coupling) basin attraction (diamond W>w, circle W<w)')# basin attraction plot
	ax181 = fig181.add_subplot(111)

	fig19 = plt.figure(num=19, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig19.canvas.set_window_title('LF (cross-coupling) basin attraction plot')			 			# basin attraction plot
	ax19 = fig19.add_subplot(111)

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

			delay_steps			= int( np.floor( poolData[0][0]['dictPLL']['transmission_delay'] / poolData[0][0]['dictPLL']['dt'] ) )
			deltaTheta 			= poolData[0][i]['dictData']['phi'][:,0] - poolData[0][i]['dictData']['phi'][:,1]
			deltaThetaDot		= np.diff( deltaTheta, axis=0 ) / poolData[0][i]['dictPLL']['dt']
			deltaThetaDiv 		= poolData[0][i]['dictData']['phi'][:,0]/poolData[0][i]['dictPLL']['div'] - poolData[0][i]['dictData']['phi'][:,1]/poolData[0][i]['dictPLL']['div']
			deltaThetaDivDot	= np.diff( deltaThetaDiv, axis=0 ) / poolData[0][i]['dictPLL']['dt']

			if np.abs( np.abs( (deltaTheta[-1]+np.pi)%(2.*np.pi)-np.pi ) - np.pi ) < treshold_statState:
				color = 'r'														# anti-phase
			elif np.abs( (deltaTheta[-1]+np.pi)%(2.*np.pi)-np.pi ) - 0.0 < treshold_statState:
				color = 'b'														# in-phase
			else:
				color = 'k'														# neither in- nor anti-phase

			# plot for HF output
			ax16.plot((deltaTheta[delay_steps+1:]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot[delay_steps:], '-', color=color, alpha=alpha, linewidth='1.2')	 	# plot trajectory
			ax16.plot((deltaTheta[delay_steps]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot[delay_steps], 'o', color=color, alpha=alpha, linewidth='1.2')	 		# plot initial dot
			ax16.plot((deltaTheta[-1]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot[-1], 'x', color=color, alpha=alpha, linewidth='1.2')						 	# plot final state cross
			#plot_lib.deltaThetaDot_vs_deltaTheta(poolData[0][i]['dictPLL'], poolData[0][i]['dictNet'], (deltaTheta[1:]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot, color, alpha)
			ax17.plot(deltaTheta[delay_steps+1:], deltaThetaDot[delay_steps:], '-', color=color, alpha=alpha, linewidth='1.2')		# plot trajectory
			ax17.plot(deltaTheta[delay_steps], deltaThetaDot[delay_steps], 'o', color=color, alpha=alpha, linewidth='1.2')			# plot initial dot
			ax17.plot(deltaTheta[-1], deltaThetaDot[-1], 'x', color=color, alpha=alpha, linewidth='1.2')							# plot final state cross

			ax161.plot((deltaTheta[0]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDot[0], initmarker, color=color, alpha=alpha, linewidth='1.2') # plot initial dot

			if np.abs( np.abs( (deltaThetaDiv[-1]+np.pi)%(2.*np.pi)-np.pi ) - np.pi ) < treshold_statState:
				color = 'r'
			elif np.abs( (deltaThetaDiv[-1]+np.pi)%(2.*np.pi)-np.pi ) - 0.0 < treshold_statState:
				color = 'b'
			else:
				color = 'k'

			# plot for LF output
			ax18.plot((deltaThetaDiv[delay_steps+1:]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[delay_steps:], '-', color=color, alpha=alpha, linewidth='1.2')	# plot trajectory
			ax18.plot((deltaThetaDiv[delay_steps]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[delay_steps], 'o', color=color, alpha=alpha, linewidth='1.2')		# plot initial dot
			ax18.plot((deltaThetaDiv[-1]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[-1], 'x', color=color, alpha=alpha, linewidth='1.2')							# plot final state cross
			#plot_lib.deltaThetaDivDot_vs_deltaThetaDiv(poolData[0][i]['dictPLL'], poolData[0][i]['dictNet'], (deltaThetaDiv[1:]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot, color, alpha)
			ax19.plot(deltaThetaDiv[delay_steps+1:], deltaThetaDivDot[delay_steps:], '-', color=color, alpha=alpha, linewidth='1.2')		# plot trajectory
			ax19.plot(deltaThetaDiv[delay_steps], deltaThetaDivDot[delay_steps], 'o', color=color, alpha=alpha, linewidth='1.2')			# plot initial dot
			ax19.plot(deltaThetaDiv[-1], deltaThetaDivDot[-1], 'x', color=color, alpha=alpha, linewidth='1.2')								# plot final state cross

			ax181.plot((deltaThetaDiv[0]+np.pi)%(2.*np.pi)-np.pi, deltaThetaDivDot[0], initmarker, color=color, alpha=alpha, linewidth='1.2') # plot initial dot

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
