#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)

#!/usr/bin/python
from numpy import pi, sin
import numpy as np
from sympy import *
# from sympy import solve, nroots, I
# from sympy import simplify, Symbol, pprint, collect_const
from sympy.abc import q
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.signal import sawtooth
from scipy.signal import square
import scipy.optimize as optimize
from scipy.optimize import fsolve
from scipy.optimize import root
import cmath
from scipy import signal

import  gc, os, time

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

# plot parameter
axisLabel = 60;
tickSize  = 35;
titleLabel= 10;
dpi_val	  = 150;
figwidth  = 6;
figheight = 5;

################################################################################

def initial_guess(discr=75):
	return [np.linspace( -1e-3, 1e-3, discr), np.linspace(-0.28*2.0*np.pi, 0.20*2.0*np.pi, discr)];

################################################################################

def linStabEq(l_vec, params):

	x = np.zeros(2)

	l = l_vec[0] + 1J * l_vec[1]
	f = l*(1.0+l*(1.0/params['wc'])) + params['a']*(1.0-params['zeta']*np.exp(-l*params['tau']))

	x[0] = np.real(f)
	x[1] = np.imag(f)

	return x

################################################################################

def solveLinStab(params):

	init = initial_guess();

	tempRe = []; tempIm = [];

	for i in range(len(init[0])):

		temp =  optimize.root(linStabEq, (init[0][i],init[1][i]), args=(params), tol=1e-14, method='hybr')

		if ( temp.success == True and np.round(temp.x[0], 16) != 0.0 and np.round(temp.x[1], 16) != 0.0 ):
			tempRe.append(temp.x[0])
			tempIm.append(temp.x[1])

	if len(tempRe) != 0:
		lambsolReMax = np.real(np.max(tempRe));
		if np.array(tempRe).argmax() < len(tempIm):
			lambsolImMax = tempIm[np.array(tempRe).argmax()]
	else:
		lambsolReMax = 0.0;
		lambsolImMax = 0.0;

	solArray = np.array([lambsolReMax, np.abs(lambsolImMax)])

	return solArray

################################################################################

class stabFunctions:
	def __init__(self, functionId):

		if   functionId == 'realPart':
			self.f = lambda mu, tau, z, psi, a, wc, fric: -mu**2.0 + a*wc*(1.0-np.abs(z)*np.cos(mu*tau-psi))

		elif functionId == 'imagPart':
			self.f = lambda mu, tau, z, psi, a, wc, fric: +mu*wc*fric + a*wc*np.abs(z)*np.sin(mu*tau-psi)

		elif functionId == 'cotanFct':
			self.f = lambda mu, tau, z, psi, a, wc, fric: np.cos(mu*tau-psi)/np.sin(mu*tau-psi) - a/(mu*fric) + mu/(wc*fric)

		else:
			raise Exception('Function to solved not defined!')

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	def solveFct(self, mu, tau, z, psi, a, wc, fric):

		x = np.zeros(1)
		f = self.f(mu, tau, z, psi, a, wc, fric)

		x = np.real(f)

		return x
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	def solveRootsFct(self, tau, z, psi, a, wc, fric, initguess, method='hybr'):

		temp = optimize.root(self.solveFct, initguess, args=(tau, z, psi, a, wc, fric), tol=1e-14, method=method)
		if temp.success:
			print('Found solutions from root finding:', temp.x)
			result = temp.x
			if len(temp.x) > 1:
				print('FOUND MORE THAN ONE SOLUTION WITH THE SOLVER!'); time.sleep(25)
		else:
			print('Found no solution for %0.2f, result stated: '%(initguess), temp)
			result = np.nan

		return result

###############################################################################

def plotRealCharSigZero(dmu, min_mu, max_mu, mu_sols1, mu_sols2, mu_sols3, tau, z, psi, a, wc, fric, intervalNumbPiHalf, scanRegimeReal):

	linestyles = ['-', '--', '-.', ':']
	if isinstance(z, np.int) or  isinstance(z, np.float):
		z = [z]; psi = [psi];

	mu 		= np.arange(min_mu, max_mu, dmu)
	f_lhs 	= lambda mu, z, psi: (mu**2.0 - a*wc)/(-a*wc*np.abs(z))
	f_rhs 	= lambda mu, z, psi: np.cos(mu*tau-psi)

	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig1.canvas.set_window_title('real part char. eq., sigma=0')

	for i in range(len(z)):
		plt.vlines(scanRegimeReal[i], -1, 1, linewidth=1, linestyle=linestyles[i], color='y')
		plt.plot(mu, f_lhs(mu, z[i], psi[i]), linewidth=1, linestyle=linestyles[i], color='b')
		plt.plot(mu, f_rhs(mu, z[i], psi[i]), linewidth=1, linestyle=linestyles[i], color='r')
		plt.plot(mu_sols3[i], f_lhs(mu_sols3[i], z[i], psi[i]), 'k+', markersize=15)#, label='quad')
		for j in range(intervalNumbPiHalf):
			plt.plot(mu_sols1[i,j], f_lhs(mu_sols1[i,j], z[i], psi[i]), 'c*', markersize=8)#, label='real')
			#plt.plot(mu_sols1, f_rhs(mu_sols1), )
			plt.plot(mu_sols2[i,j], f_lhs(mu_sols2[i,j], z[i], psi[i]), 'gd', markersize=2)#, label='imag')
			#plt.plot(mu_sols2, f_rhs(mu_sols2), )

	plt.xlabel(r'$\mu$', fontdict = labelfont)
	plt.ylabel(r'$f(\mu)$', fontdict = labelfont)
	plt.ylim([-5, 5])
	#plt.legend()

	#plt.savefig('results/name_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	#plt.savefig('results/name_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=300)

	return None

################################################################################

def plotImagCharSigZero(dmu, min_mu, max_mu, mu_sols1, mu_sols2, mu_sols3, tau, z, psi, a, wc, fric, intervalNumbPiHalf, scanRegimeImag):

	linestyles = ['-', '--', '-.', ':']
	if isinstance(z, np.int) or  isinstance(z, np.float):
		z = [z]; psi = [psi];

	mu 		= np.arange(min_mu, max_mu, dmu)
	f_lhs 	= lambda mu, z, psi: -mu*wc*fric / (a*wc*np.abs(z))
	f_rhs 	= lambda mu, z, psi: np.sin(mu*tau-psi)

	fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig2.canvas.set_window_title('imag part char. eq., sigma=0')

	for i in range(len(z)):
		plt.vlines(scanRegimeImag[i], -1, 1, linewidth=1, linestyle=linestyles[i], color='y')
		plt.plot(mu, f_lhs(mu, z[i], psi[i]), linewidth=1, linestyle=linestyles[i], color='b')
		plt.plot(mu, f_rhs(mu, z[i], psi[i]), linewidth=1, linestyle=linestyles[i], color='r')
		plt.plot(mu_sols3[i], f_lhs(mu_sols3[i], z[i], psi[i]), 'k+', markersize=15)#, label='quad')
		for j in range(intervalNumbPiHalf):
			plt.plot(mu_sols1[i,j], f_lhs(mu_sols1[i,j], z[i], psi[i]), 'c*', markersize=2)#, label='real')
			#plt.plot(mu_sols1, f_rhs(mu_sols1), )
			plt.plot(mu_sols2[i,j], f_lhs(mu_sols2[i,j], z[i], psi[i]), 'gd', markersize=8)#, label='imag')
			#plt.plot(mu_sols2, f_rhs(mu_sols2), )

	plt.xlabel(r'$\mu$', fontdict = labelfont)
	plt.ylabel(r'$f(\mu)$', fontdict = labelfont)
	plt.ylim([-5, 5])
	#plt.legend('lhs', 'rhs', 'quad', 'real', 'imag')

	#plt.savefig('results/name_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	#plt.savefig('results/name_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=300)

	return None

################################################################################

def plotCotanSigZero(dmu, min_mu, max_mu, mu_sols1, mu_sols2, mu_sols3, tau, z, psi, a, wc, fric):

	if isinstance(z, np.int) or  isinstance(z, np.float):
		z = [z]; psi = [psi];

	mu 		= np.arange(min_mu, max_mu, dmu)
	f_lhs 	= lambda mu, psi: a/(mu*fric) - mu/(wc*fric)
	f_rhs 	= lambda mu, psi: np.cos(mu*tau-psi)/np.sin(mu*tau-psi)

	fig3 = plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig3.canvas.set_window_title('real part char. eq., sigma=0')

	for i in range(len(z)):
		plt.plot(mu, f_lhs(mu, z, psi[i]), linewidth=1, linestyle='-', color='b')
		plt.plot(mu, f_rhs(mu, z, psi[i]), linewidth=1, linestyle='-', color='r')
	plt.plot(mu_sols1, f_lhs(mu_sols1), 'c*', markersize=4)
	#plt.plot(mu_sols1, f_rhs(mu_sols1), )
	plt.plot(mu_sols2, f_lhs(mu_sols2), 'gd', markersize=4)
	#plt.plot(mu_sols2, f_rhs(mu_sols2), )

	plt.xlabel(r'$\mu$', fontdict = labelfont)
	plt.ylabel(r'$f(\mu)$', fontdict = labelfont)
	# plt.legend()

	#plt.savefig('results/name_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	#plt.savefig('results/name_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=300)

	return None

################################################################################

# def conditionBoundary(paramVec, mu, tau, zeta, psi, a, fric):
#
# 	K  = paramVec[0]
# 	wc = paramVec[1]
#
# 	f = ( mu )**2 - K * np.abs(a/K) * wc * (1.0 - np.abs(zeta)*np.cos( mu*tau-psi) )
#
# 	return f
#
# ################################################################################
#
# def solveRootsFct(tau, psi, a, wc, fric, initguess):
#
# 	temp = optimize.root(conditionBoundary, (K, wc), args=(mu, tau, zeta, psi, a, fric), tol=1e-14, method='hybr')
# 	print('Found solutions from root finding:', temp.x)
#
# 	return temp.x[0]
