#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq, linStabEq, initial_guess, xrange, solveLinStabSinglePLL
from LiStaGloF_lib2 import LoopGain, LoopBDWidth, solveLoopBandwidth, solveLinStabBandw
from numpy import pi, sin
import numpy as np
import sympy
from sympy import solve, nroots, I
from sympy.abc import q
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.signal import sawtooth
from scipy.signal import square
import scipy.optimize as optimize
from scipy.optimize import fsolve
from scipy.optimize import root
# from mpmath import findroot
# from sympy import I, limit
# from sympy.solvers import solveset
# from sympy.solvers import solve
# from sympy import Symbol, Function, nsolve
# from sympy import sin, cos, exp
# import mpmath
# mpmath.mp.dps = 25
# choose digital vs analog
# choose digital vs analog
digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;

inphase1= True;
inphase2= False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
w    	  	= 2.0*np.pi*62E9; 			# intrinsic	frequency
Kvco      	= 2.0*np.pi*(0.8181E9); 	# Sensitivity of VCO
AkPD	  	= 0.162*1.0					#  Amplitude of the output of the PD --
GkLF		= 1.0
Gvga	  	= 0.1							# Gain of the first adder
tauf 	  	= 0.0						# tauf = sum of all processing delays in the feedback
order 	  	= 1.0						# the order of the Loop Filter
tauc	  	= 1.0/(2.0*np.pi*1E6); 		# the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
v	 	  	= 1024;					# the division
c		  	= 3E8						# speed of light
min_delay 	= 0.1E-9
INV		  	= 0.0*np.pi					# Inverter
#			# Inverter
#
maxp 	  	= 4.0E3;
wnormy 		= 2.0*np.pi;				# this is the frequency with which we rescale the y-axis, choose either 1, 2pi or w (y-axis: radHz, Hz, dimless)
wnormx 		= 2.0*np.pi;				# this is the frequency with which we rescale the x-axis, choose either w/2pi-rescaling (2.0*np.pi) or no rescaling of tau (w)

filter		= 1
figwidth  	= 6;
figheight 	= 6;
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'

# print(globalFreqINV(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'])
fig0    = plt.figure(num=0, figsize=(figwidth, figheight), facecolor='w', edgecolor='k')
# print('coupling strength K=', K(Kvco, AkPD,  GkLF, Gvga),' radHz and K_Hz=', K(Kvco, AkPD, GkLF, Gvga)/(2.0*np.pi),' Hz')
# print(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'][np.where(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 			globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter)['Re'][0]>0.0)][0])

#*******************************************************************************
fig         = plt.figure(figsize=(figwidth,figheight))
ax          = fig.add_subplot(111)

if digital == True:
	plt.title(r'digital case for $\omega=2\pi\cdot$%2.4E Hz and min_delay= %0.2f' %(w/(2.0*np.pi),min_delay*(w)/(2.0*np.pi)/v));
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'digital case for $\omega=2\pi\cdot$%2.4E Hz and min_delay= %0.2f' %(w/(2.0*np.pi),min_delay*(w)/(2.0*np.pi)/v));
# adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
if wnormy == 1:
	plt.ylabel(r'$\Omega$', fontsize=18)
elif wnormy == 2.0*np.pi:
	plt.ylabel(r'$f_{\Omega}$', fontsize=18)
elif wnormy == w:
	plt.ylabel(r'$\Omega/\omega$', fontsize=18)
# 	plt.axhline(y=w, color='green', linestyle='-',linewidth=3)
x = min_delay*(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
				 tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegUnst']/(np.asarray(2.0*np.pi*v)))

# print(solveLinStabBandw(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD,  GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter))
# print(solveLoopBandwidth(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 				 tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['Re'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				  globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				  globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		  		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'],
# 				 Kvco, AkPD,  GkLF, Gvga, tauc, v, maxp, digital, inphase1, INV)['LoopBand'])


# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later
# #*************************************************************************************************************************************************************************


[lineOmegStabIn] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegStab'],
				solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegStab']/np.asarray(wnormy),
				 'o',ms=7, color='blue',  label=r'Inphase Stable')

[lineOmegUnstIn] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegUnst'],
				solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegUnst']/np.asarray(wnormy),
				'o',ms=1, color='blue', label=r'Inphase Unstable')



[lineOmegStabAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['OmegStab'],
				solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['OmegStab']/np.asarray(wnormy),
				 'o',ms=7, color='red',  label=r'Antiphase Stable')

[lineOmegUnstAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['OmegUnst'],
				solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['OmegUnst']/np.asarray(wnormy),
				'o',ms=1, color='red', label=r'Antiphase Unstable')

#
# [line]=ax.plot((x,x),(0.0001+max(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 							globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 							 tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegStab']/np.asarray(wnormy)),
# 							 -0.0001+min(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 							  globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 							  tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegStab']/np.asarray(wnormy))),'k.-',linestyle='--', linewidth=3)


# #*************************************************************************************************************************************************************************
fig1         = plt.figure(figsize=(figwidth,figheight))
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.suptitle(r'Analysis of full expression of characteristic equation, min_delay= %0.2f' %(min_delay*(w)/(2.0*np.pi)/v));
plt.title(r'v= %0.2f, fc=%3.2E Hz, Kvco=%3.2E Hz/V, Gvga=%0.1f' %(v,1.0/(2.0*np.pi*tauc),Kvco/(2.0*np.pi),Gvga));
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)

plt.axhline(y=0, color='green', linestyle='-',linewidth=3)



[lineSigma1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase1, expansion, INV, filter)['Re'],
	linewidth=4, color='red', label=r'$\sigma$=Re$(\lambda)$ (in-phase)')
[lineGamma1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase1, expansion, INV, filter)['Im'],
	'+', ms=4, color='blue', label=r'$\gamma$=Im$(\lambda)$ (in-phase)')


[lineSigma2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter)['Re'],
	'--', linewidth=1, color='red', label=r'$\sigma$=Re$(\lambda)$ (antiphase)')
[lineGamma2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter)['Im'],
	 '.',ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$ (antiphase)')
#
# [LoopBandWdthIn1Norm] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['BDWidth1']/w, '--',ms=1, color='purple', label=r'Loop Bandwidth1 (inphase)')
# [LoopBandWdthIn2Norm] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['BDWidth2']/w, '-.',ms=1, color='purple', label=r'Loop Bandwidth2 (inphase)')
# #
#
# [LoopBandWdthIn3Norm] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['BDWidth3']/w, 'd',ms=1, color='purple', label=r'Loop Bandwidth3 (inphase)')
# [LoopBandWdthIn4Norm] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['BDWidth4']/w, '.',ms=1, color='purple', label=r'Loop Bandwidth4 (inphase)')



# [LoopBandWdthIn4NormFull]=ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],solveLinStabBandw(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD,  GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['Im']/w, '-.',ms=3, color='purple', label=r'Loop Bandwidth Network')
# #
# [LoopBandWdthGenearalNorm]= ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],solveLoopBandwidth(solveLinStabSinglePLL(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		 tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['Re'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		  globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		  globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'],
# 		 Kvco, AkPD,  GkLF, Gvga, tauc, v, maxp, digital, inphase1, INV)['LoopBand']/w,'d', ms=4, color='magenta', label=r'LoopBand (in-phase) One PLL')
# #
# #
# # #
# #
# [lineSigma1zeta0] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStabSinglePLL(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase1, expansion, INV, filter)['Re'],
# 	linewidth=4, color='black', label=r'$\sigma$=Re$(\lambda)$ (in-phase)One PLL')
# [lineGamma1zeta0] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStabSinglePLL(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase1, expansion, INV, filter)['Im'],
# 	'+', ms=4, color='green', label=r'$\gamma$=Im$(\lambda)$ (in-phase) One PLL')
# #

# [lineSigma2zeta0] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStabSinglePLL(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter)['Re'],
# 	'--', linewidth=1, color='black', label=r'$\sigma$=Re$(\lambda)$ (antiphase) One PLL')
# #
# [lineGamma2zeta0] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStabSinglePLL(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter)['Im'],
# 	 '.',ms=1, color='green', label=r'$\gamma$=Im$(\lambda)$ (antiphase) One PLL')

# [LoopBandWdthGenearalNorm] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], solveLoopBandwidth(np.zeros(len(xrange(maxp, tauf)),dtype=np.float64), globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				  globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				  globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		  		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'],
# 				 Kvco, AkPD,  GkLF, Gvga, tauc, v, maxp, digital, inphase1, INV)['LoopBand']/w, '-',ms=1, color='black', label=r'Loop BandwidthGeneral (in-phase)')


# fig2         = plt.figure(figsize=(figwidth,figheight))
# ax2          = fig2.add_subplot(111)
# # plot grid, labels, define intial values
# plt.suptitle(r'Steady state Loop Gain');
# plt.title(r'v= %0.2f, fc=%3.2E Hz, Kvco=%3.2E Hz/V, Gvga=%0.1f' %(v,1.0/(2.0*np.pi*tauc),Kvco/(2.0*np.pi),Gvga));
# plt.grid()
# plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
# plt.ylabel(r'Loop Gain', fontsize=18)
#
# plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
#
# [LoopGainIn] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'], '.',ms=1, color='red', label=r'Loop Gain (inphase)')
#
# [LoopGainAnti] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['LPGain'], '.',ms=1, color='blue', label=r'Loop Gain (Antiphase)')
#
# fig3         = plt.figure(figsize=(figwidth,figheight))
# ax3          = fig3.add_subplot(111)
# # plot grid, labels, define intial values
# plt.suptitle(r'Loop Bandwidth (Inphase)');
# plt.title(r'v= %0.2f, fc=%3.2E Hz, Kvco=%3.2E Hz/V, Gvga=%0.1f' %(v,1.0/(2.0*np.pi*tauc),Kvco/(2.0*np.pi),Gvga));
# plt.grid()
# plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
# plt.ylabel(r'Loop Bandwidth', fontsize=18)
#
# plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
# #
# [lineGamma20] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter)['Im'],
# 	 '.',ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$ (antiphase)')
# [LoopBandWdthIn1] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['BDWidth1'], '--',ms=1, color='red', label=r'Loop Bandwidth1 (inphase)')
# [LoopBandWdthIn2] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['BDWidth2'], '-',ms=1, color='red', label=r'Loop Bandwidth2 (inphase)')

#
# [LoopBandWdthIn3] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['BDWidth3'], '-',ms=1, color='black', label=r'Loop Bandwidth3 (inphase)')
# [LoopBandWdthIn4] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['BDWidth4'], '-',ms=1, color='purple', label=r'Loop Bandwidth4 (inphase)')

#
#
# #
#
# [LoopBandWdthGenearal] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], solveLoopBandwidth(solveLinStabSinglePLL(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 				 tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['Re'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				  globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				  globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		  		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'],
# 				 Kvco, AkPD,  GkLF, Gvga, tauc, v, maxp, digital, inphase1, INV)['LoopBand'], '-',ms=1, color='black', label=r'Loop BandwidthGeneral (in-phase)')

# [LoopBandWdthGenearal] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], solveLoopBandwidth(np.zeros(len(xrange(maxp, tauf)),dtype=np.float64), globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				  globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				  globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 		  		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase1, INV)['LPGain'],
# 				 Kvco, AkPD,  GkLF, Gvga, tauc, v, maxp, digital, inphase1, INV)['LoopBand'], '-',ms=1, color='black', label=r'Loop BandwidthGeneral (in-phase)')

# fig4         = plt.figure(figsize=(figwidth,figheight))
# ax4          = fig4.add_subplot(111)
# # plot grid, labels, define intial values
# plt.suptitle(r'Loop Bandwidth (Anti-phase)');
# plt.title(r'v= %0.2f, fc=%3.2E Hz, Kvco=%3.2E Hz/V, Gvga=%0.1f' %(v,1.0/(2.0*np.pi*tauc),Kvco/(2.0*np.pi),Gvga));
# plt.grid()
# plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
# plt.ylabel(r'Loop Bandwidth', fontsize=18)
#
# plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
# [LoopBandWdthAnti1] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['BDWidth1'], '-',ms=1, color='blue', label=r'Loop Bandwidth1 (Anti-phase)')
# [LoopBandWdthAnti2] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# 		Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'],
# 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['BDWidth2'], '--',ms=1, color='blue', label=r'Loop Bandwidth2 (Anti-phase)')
# # [LoopBandWdthAnti3] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# # 		Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# # 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'],
# # 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['BDWidth3'], '-',ms=1, color='black', label=r'Loop Bandwidth3 (Anti-phase)')
# # [LoopBandWdthAnti4] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,
# # 		Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# # 		globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'],
# # 		tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['BDWidth4'], '-',ms=1, color='purple', label=r'Loop Bandwidth4 (Anti-phase)')
#



#
#
#
# print(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'],
# tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['LPGain'], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase2, INV)['BDWidth1'])



# add two sliders for tweaking the parameters
# define an axes area and draw a slider in it
v_slider_ax   = fig0.add_axes([0.15, 0.80, 0.65, 0.15], facecolor=axis_color)
v_slider      = Slider(v_slider_ax, r'$v$', 4, 1024, valinit=v, valstep=4)
# Draw another slider
tauf_slider_ax  = fig0.add_axes([0.15, 0.60, 0.65, 0.15], facecolor=axis_color)
tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf, valfmt='%1.2E')
# Draw another slider
Gvga_slider_ax = fig0.add_axes([0.15, 0.40, 0.65, 0.15], facecolor=axis_color)
Gvga_slider    = Slider(Gvga_slider_ax, r'$G^{vga}$', 0.5, 2.0, valinit=Gvga)
# Draw another slider
wc_slider_ax  = fig0.add_axes([0.15, 0.20, 0.65, 0.15], facecolor=axis_color)
wc_slider     = Slider(wc_slider_ax, r'$\omega_c$ in MHz', 0.00100, 800, valinit=(1/(2.0E6*np.pi*tauc)))

INV_slider_ax  = fig0.add_axes([0.15, 0.02, 0.65, 0.15], facecolor=axis_color)
INV_slider     = Slider(INV_slider_ax, r'$INV$', 0.0, np.pi, valinit=INV, valstep=np.pi)


# Draw another slider


# define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
	global digital
	initial_guess()
	lineOmegStabIn.set_ydata(np.asarray(1.0/wnormy)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter)['OmegStab']);
	lineOmegStabIn.set_xdata(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter)['OmegStab']);
	lineOmegUnstIn.set_ydata(np.asarray(1.0/wnormy)*solveLinStab(globalFreq(w,Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter)['OmegUnst']);
	lineOmegUnstIn.set_xdata(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val, filter)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val, filter)['OmegUnst']);


	lineOmegStabAnti.set_ydata(np.asarray(1.0/wnormy)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion,INV_slider.val, filter)['OmegStab']);
	lineOmegStabAnti.set_xdata(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion,INV_slider.val, filter)['OmegStab']);
	lineOmegUnstAnti.set_ydata(np.asarray(1.0/wnormy)*solveLinStab(globalFreq(w,Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter)['OmegUnst']);
	lineOmegUnstAnti.set_xdata(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val, filter)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val, filter)['OmegUnst']);







	lineSigma1.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter)['Re']);
	lineSigma1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);

	lineGamma1.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter)['Im']);
	lineGamma1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
	#




	lineSigma2.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter)['Re']);
	lineSigma2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);

	lineGamma2.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter)['Im']);
	lineGamma2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);


	#
	#
	# lineSigma1zeta0.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStabSinglePLL(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter)['Re']);
	# lineSigma1zeta0.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
	#
	# lineGamma1zeta0.set_ydata(np.asarray(1/w)*solveLinStabSinglePLL(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter)['Im']);
	# lineGamma1zeta0.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
	# #
	#
	#
	#
	#
	# lineSigma2zeta0.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStabSinglePLL(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter)['Re']);
	# lineSigma2zeta0.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
	#
	# lineGamma2zeta0.set_ydata(np.asarray(1/w)*solveLinStabSinglePLL(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter)['Im']);
	# lineGamma2zeta0.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
	#
	#
	#
	#





#**********************************************************************************************************************************************************************************
	#
	# LoopGainIn.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'])
	#
	# LoopGainIn.set_ydata(LoopGain(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['LPGain'])
	#
	# LoopGainAnti.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'])
	#
	# LoopGainAnti.set_ydata(LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase2, INV_slider.val)['LPGain'])
	#
	#
	#
	#
	# LoopBandWdthIn1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'])
	#
	# LoopBandWdthIn1.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['BDWidth1'])
	#
	#
	#
	# LoopBandWdthIn2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'])
	# #
	# LoopBandWdthIn2.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['BDWidth2'])
	# #
	# #
	# LoopBandWdthIn3.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'])
	# #
	# LoopBandWdthIn3.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['BDWidth3'])
	# #
	# #
	# LoopBandWdthIn4.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'])
	# #
	# LoopBandWdthIn4.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['BDWidth4'])
	#
	#
	#
	#
	#
	#
	#
	#
	# LoopBandWdthIn1Norm.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'])
	#
	# LoopBandWdthIn1Norm.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['BDWidth1']/w)
	#
	#
	#
	# LoopBandWdthIn2Norm.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'])
	# #
	# LoopBandWdthIn2Norm.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['BDWidth2']/w)
	# #
	# #
	# LoopBandWdthIn3Norm.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'])
	# #
	# LoopBandWdthIn3Norm.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['BDWidth3']/w)
	# #
	# #
	# LoopBandWdthIn4Norm.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'])
	# #
	# LoopBandWdthIn4Norm.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase1, INV_slider.val)['BDWidth4']/w)
	#
	#
	#
	#
	#
	#
	#
	#
	#
	#
	# LoopBandWdthAnti1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'])
	#
	# LoopBandWdthAnti1.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase2, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase2, INV_slider.val)['BDWidth1'])
	#
	#
	#
	# LoopBandWdthAnti2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'])
	# #
	# LoopBandWdthAnti2.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase2, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase2, INV_slider.val)['BDWidth2'])
	# #
	# #
	# LoopBandWdthAnti3.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'])
	# #
	# LoopBandWdthAnti3.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase2, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase2, INV_slider.val)['BDWidth3'])
	# #
	# #
	# LoopBandWdthAnti4.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'])
	# #
	# LoopBandWdthAnti4.set_ydata(LoopBDWidth(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val,
	# LoopGain(globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# globalFreq(w, Kvco, AkPD, GkLF, Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'],
	# tauf_slider.val, Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase2, INV_slider.val)['LPGain'],
	# Kvco, AkPD,  GkLF, Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital, inphase2, INV_slider.val)['BDWidth4'])
	#
	#
	#
	#
	#
	#
	#
	#
	#
	#
	#
	#
	#
	#







	# line.set_xdata(min_delay*(w/(2.0*np.pi)/v_slider.val))
	ax.set_title(r'Analysis of full expression of characteristic equation, min_delay= %0.2f' %(min_delay*(w)/(2.0*np.pi)/v_slider.val));
	ax1.set_title(r'analog case for $\omega=2\pi\cdot$%2.4E Hz and min_delay= %0.2f' %(w/(2.0*np.pi),min_delay*(w)/(2.0*np.pi)/v_slider.val));
	# ax2.set_title(r'v= %0.2f, fc=%3.2E Hz, Kvco=%3.2E Hz/V, Gvga=%0.1f' %(v_slider.val, wc_slider.val/(2.0*np.pi), Kvco/(2.0*np.pi), Gvga_slider.val));
	# ax3.set_title(r'v= %0.2f, fc=%3.2E Hz, Kvco=%3.2E Hz/V, Gvga=%0.1f' %(v_slider.val, wc_slider.val/(2.0*np.pi), Kvco/(2.0*np.pi), Gvga_slider.val));
	# ax4.set_title(r'v= %0.2f, fc=%3.2E Hz, Kvco=%3.2E Hz/V, Gvga=%0.1f' %(v_slider.val, wc_slider.val/(2.0*np.pi), Kvco/(2.0*np.pi), Gvga_slider.val));

	# 	# recompute the ax.dataLim
	ax.relim();
	ax1.relim();
	# ax2.relim();
	# ax3.relim();
	# ax4.relim();

# 	# update ax.viewLim using the new dataLim
	ax.autoscale_view();
	ax1.autoscale_view();
	# ax2.autoscale_view();
	# ax3.autoscale_view();
	# ax4.autoscale_view();

	plt.draw()
	fig.canvas.draw_idle();
	fig1.canvas.draw_idle();
	# fig2.canvas.draw_idle();
	# fig3.canvas.draw_idle();
	# fig4.canvas.draw_idle();

v_slider.on_changed(sliders_on_changed)
tauf_slider.on_changed(sliders_on_changed)
Gvga_slider.on_changed(sliders_on_changed)
wc_slider.on_changed(sliders_on_changed)
INV_slider.on_changed(sliders_on_changed)

# add a button for resetting the parameters
# PLLtype_button_ax = fig.add_axes([0.8, 0.9, 0.1, 0.04])
# PLLtype_button = Button(PLLtype_button_ax, 'dig/ana', color=axis_color, hovercolor='0.975')
# def PLLtype_button_on_clicked(mouse_event):
# 	global digital
# 	digital = not digital;
# 	print('state digital:', digital)
# 	if digital == True:
# 		ax.set_title(r'digital case for $\omega$=%.3f' %w);
# 		ax1.set_title(r'digital case for $\omega$=%.3f, linear stability' %w);
# 		#ax2.set_title(r'digital case for $\omega$=%.3f, linear stability' %w);
# 		#ax5.set_title(r'digital case for $\omega$=%.3f, Nyquist' %w);
# 		#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# 	else:
# 		ax.set_title(r'analog case for $\omega$=%.3f' %w);
# 		ax1.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
# 		#ax2.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
# 		#ax5.set_title(r'analog case for $\omega$=%.3f, Nyquist' %w);
# 	fig.canvas.draw_idle()
# PLLtype_button.on_clicked(PLLtype_button_on_clicked)

# # add a set of radio buttons for changing color
# color_radios_ax = fig.add_axes([0.025, 0.75, 0.15, 0.15], facecolor=axis_color)
# color_radios = RadioButtons(color_radios_ax, ('red', 'green'), active=0)
# def color_radios_on_clicked(label):
#     lineBeta12.set_color(label)
#     ax.legend()
#     fig.canvas.draw_idle()
# color_radios.on_clicked(color_radios_on_clicked)

ax.legend()
ax1.legend()
# ax2.legend()
# ax3.legend()
# ax4.legend()
#ax5.legend()
plt.show()
