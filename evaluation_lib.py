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
import os
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
import scipy
from scipy import signal
import integer_mult_period_signal_lib as findIntTinSig

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


def obtainOrderParam(dictPLL, dictNet, dictData):
	''' MODIFIED KURAMOTO ORDER PARAMETERS '''
	numb_av_T = 3;																			   # number of periods of free-running frequencies to average over
	if dictPLL['intrF'] > 0:																				   # for f=0, there would otherwies be a float division by zero
		F1=dictPLL['intrF']
	else:
		F1=dictPLL['intrF']+1E-3

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


def get_d_matrix(n):
	'''Constructs a matrix to compute the phase differences from a
	   vector of non-rotated phases'''
	d = np.zeros((n, n))
	for i in range(n):
		d[i, i] = -1
		d[i, np.mod(i + 1, n)] = 1
	return d

class PhaseDifferenceCell(object):
	def __init__(self, n):
		self.n = n
		self.dphi_matrix = get_dphi_matrix(n)
		self.d_min = -np.pi
		self.d_max = np.pi
		self.d = get_d_matrix(n)

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


def _CheckerboardOrderParameter(phi):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi is supposed
	   to be 1d vector of phases without time evolution.
	'''
	r = 0.0
	for ix in range(len(phi)):
		r += np.exp(1j * phi[ix]) * np.exp(-1j * np.pi * ix)
	r = np.abs(r) / float(len(phi))
	return r


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


''' CALCULATE SPECTRUM '''
def calcSpectrum( phi, dictPLL, dictNet, percentOfTsim=0.75 ): #phi,Fsample,couplingfct,waveform=None,expectedFreq=-999,evalAllRealizations=False,decayTimeSlowestMode=None

	Pxx_dBm=[]; Pxx_dBV=[]; f=[];
	windowset='hamming' #'hamming' #'hamming', 'boxcar'
	print('\nCurrent window option is', windowset, 'for waveform', inspect.getsourcelines(dictPLL['vco_out_sig'])[0][0],
			'NOTE: in principle can always choose to be sin() for cleaner PSD in first harmonic approximation of the signal.')
	print('Calculate spectrum for',1-percentOfTsim,'percent of the time-series. Implement better solution using decay times.')
	try:
		analyzeL= findIntTinSig.cutTimeSeriesOfIntegerPeriod(dictPLL['sampleF'], dictNet['Tsim'], dictPLL['syncF'],
																np.max([dictPLL['coupK'], dictPLL['coupStr_2ndHarm']]), phi, percentOfTsim);
		window	 	= scipy.signal.get_window('boxcar', int(dictPLL['sampleF']), fftbins=True);
	except:
		print('\n\nError in cutTimeSeriesOfIntegerPeriod-function! Not picking integer number of periods for PSD!\n\n')
		analyzeL= [ int( dictNet['Tsim']*(1-percentOfTsim)*dictPLL['sampleF'] ), int( dictNet['Tsim']*dictPLL['sampleF'] )-1 ]
		window	= scipy.signal.get_window(windowset, int(dictPLL['sampleF']), fftbins=True);

	tsdata		= dictPLL['vco_out_sig'](phi[analyzeL[0]:analyzeL[1]])

	ftemp, Vxx 	= scipy.signal.periodogram(tsdata, dictPLL['sampleF'], return_onesided=True, window=windowset, scaling='density', axis=0) #  returns Pxx with dimensions [V^2] if scaling='spectrum' and [V^2/Hz] if if scaling='density'
	P0 = 1E-3; R=50; 															# for P0 in [mW/Hz] and R [ohm]

	Pxx_dBm.append( 10*np.log10((Vxx/R)/P0) )
	f.append( ftemp )

	return f, Pxx_dBm

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
	return pll_list[0].lf.y

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
