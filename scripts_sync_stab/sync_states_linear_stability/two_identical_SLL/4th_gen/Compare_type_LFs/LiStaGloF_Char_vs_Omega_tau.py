#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq,linStabEq,initial_guess
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
digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;

# choose phase or anti-phase synchroniazed states,
digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;

inphase1= True;
inphase2= False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
w    	  	= 2.0*np.pi*60E9; 			# intrinsic	frequency
Kvco      	= 2.0*np.pi*(1.0E9); 		# Sensitivity of VCO
GkPD	  	= 3.2						# Gain of the PD -- the factor of 2 here is related to the IMS 2020 paper, AkPD as defined in that paper already includes the 0.5 that we later account for when calculating K
Gk		  	= 5.8
Gl		  	= 5.8
Ak		 	= 0.113/2.0
Al			= 0.113/2.0
GkLF		= 1.0
Gvga	  	= 2.0						# Gain of the first adder
tauf 	  	= 0.0						# tauf = sum of all processing delays in the feedback
order 	  	= 1.0						# the order of the Loop Filter
tauc	  	= 1.0/(2.0*np.pi*100E6); 	# the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
v	 	  	= 4.0;						# the division
c		  	= 3E8						# speed of light
min_delay 	= 0.1E-9
INV		  	= 0.0*np.pi					# Inverter
#			# Inverter
#
maxp 	  	= 2900;
wnormy 		= 2.0*np.pi;				# this is the frequency with which we rescale the y-axis, choose either 1, 2pi or w (y-axis: radHz, Hz, dimless)
			# this is the frequency with which we rescale the y-axis, choose either 1, 2pi or w (y-axis: radHz, Hz, dimless)
filter1		=1
filter2		=2
filter3		=3
figwidth  	= 6;
figheight 	= 6;
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'

# print(globalFreqINV(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'])
# fig0    = plt.figure(num=0, figsize=(figwidth, figheight), facecolor='w', edgecolor='k')


#*******************************************************************************
# fig         = plt.figure(figsize=(figwidth,figheight))
# ax          = fig.add_subplot(111)
# #*******************************************************************************
# if digital == True:
# 	plt.title(r'digital case for $\omega$=%.3f' %w);
# 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# else:
# 	plt.title(r'analog case for $\omega$=%.3f' %w);
# # adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.12, bottom=0.25)
# # plot grid, labels, define intial values
# plt.grid()
# plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$\Omega/\omega$', fontsize=18)
# # 	plt.axhline(y=w, color='green', linestyle='-',linewidth=3)
#
# # # draw the initial plot
# # # the 'lineXXX' variables are used for modifying the lines later
#
# # #*************************************************************************************************************************************************************************
#
#
# [lineOmegStabIn] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegStab'],
# 				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegStab']/np.asarray(w),
# 				 'o',ms=3, color='blue',  label=r'Inphase Stable')
#
# [lineOmegUnstIn] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegUnst'],
# 				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegUnst']/np.asarray(w),
# 				'o',ms=3, color='red', label=r'Inphase Unstable')
#
#
#
# [lineOmegStabAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab'],
# 				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab']/np.asarray(w),
# 				 'o',ms=1, color='blue',  label=r'Antiphase Stable')
#
# [lineOmegUnstAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst'],
# 				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst']/np.asarray(w),
# 				'o',ms=1, color='red', label=r'Antiphase Unstable')
#
#
#


# #*************************************************************************************************************************************************************************


fig1         = plt.figure(figsize=(figwidth,figheight))
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'The analytic expression characteristic equation');
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)

plt.axhline(y=0, color='green', linestyle='-',linewidth=3)

# [lineSigmaIn1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase1, expansion, INV, filter1 )['Re'],
# 	linewidth=4, color='red', label=r'$\sigma$=Re$(\lambda)$ (in-phase, Chebychev Filter)')
# [lineGammaIn1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase1, expansion, INV, filter1 )['Im'],
# 	'+', ms=4, color='blue', label=r'$\gamma$=Im$(\lambda)$ (in-phase, Chebychev Filter)')

#
# [lineSigmaAnti1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter1 )['Re'],
# 	'--', linewidth=1, color='green', label=r'$\sigma$=Re$(\lambda)$ (antiphase, Chebychev Filter )')
# [lineGammaAnti1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter1 )['Im'],
# 	 '.',ms=1, color='yellow', label=r'$\gamma$=Im$(\lambda)$ (antiphase, Chebychev Filter)')

[lineSigmaIn2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase1, expansion, INV, filter2 )['Re'],
	linewidth=4, color='red', label=r'$\sigma$=Re$(\lambda)$ (in-phase, RC Filter)')
# [lineGammaIn2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase1, expansion, INV, filter2 )['Im'],
# 	'+', ms=4, color='blue', label=r'$\gamma$=Im$(\lambda)$ (in-phase, RC Filter)')
#
#
# [lineSigmaAnti2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter2 )['Re'],
# 	'--', linewidth=1, color='red', label=r'$\sigma$=Re$(\lambda)$ (antiphase, RC Filter)')
# [lineGammaAnti2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter2 )['Im'],
# 	 '.',ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$ (antiphase, RC Filter)')



[lineSigmaIn3] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase1, expansion, INV, filter3 )['Re'],
	linewidth=4, color='blue', label=r'$\sigma$=Re$(\lambda)$ (in-phase, Lead-Lag Filter)')
# [lineGammaIn3] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase1, expansion, INV, filter3 )['Im'],
# 	'+', ms=4, color='blue', label=r'$\gamma$=Im$(\lambda)$ (in-phase, Lead-Lag Filter)')
#
#
# [lineSigmaAnti3] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter3 )['Re'],
# 	'--', linewidth=1, color='red', label=r'$\sigma$=Re$(\lambda)$ (antiphase, Lead-Lag Filter)')
# [lineGammaAnti3] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV, filter3 )['Im'],
# 	 '.',ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$ (antiphase, Lead-Lag Filter)')

# # add two sliders for tweaking the parameters
# # define an axes area and draw a slider in it
# v_slider_ax   = fig0.add_axes([0.25, 0.67, 0.65, 0.1], facecolor=axis_color)
# v_slider      = Slider(v_slider_ax, r'$v$', 1, 16, valinit=v, valstep=1)
# # Draw another slider
# tauf_slider_ax  = fig0.add_axes([0.25, 0.45, 0.65, 0.1], facecolor=axis_color)
# tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf, valfmt='%1.2E')
# # Draw another slider
# Ga1_slider_ax = fig0.add_axes([0.25, 0.23, 0.65, 0.1], facecolor=axis_color)
# Ga1_slider    = Slider(Ga1_slider_ax, r'$G^{a,1}$', 0.01*Ga1, 2.0*Ga1, valinit=Ga1)
# # Draw another slider
# tauc_slider_ax  = fig0.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
# tauc_slider     = Slider(tauc_slider_ax, r'$\tau^c$', 0.005*tauc, 3.0*tauc, valinit=tauc, valfmt='%1.2E')
#
# INV_slider_ax  = fig0.add_axes([0.25, 0.01, 0.65, 0.02], facecolor=axis_color)
# INV_slider     = Slider(INV_slider_ax, r'$INV$', 0.0, np.pi, valinit=INV, valstep=np.pi)
#
# # define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 	global digital
# 	initial_guess()
# 	# lineOmegStabIn.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 	# 					tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['OmegStab']);
# 	# lineOmegStabIn.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 	# 					tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['OmegStab']);
# 	# lineOmegUnstIn.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w,Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['OmegUnst']);
# 	# lineOmegUnstIn.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['OmegUnst']);
# 	#
# 	#
# 	# lineOmegStabAnti.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 	# 					tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion,INV_slider.val, zeta)['OmegStab']);
# 	# lineOmegStabAnti.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 	# 					tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion,INV_slider.val, zeta)['OmegStab']);
# 	# lineOmegUnstAnti.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w,Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['OmegUnst']);
# 	# lineOmegUnstAnti.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	# 					globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['OmegUnst']);
# 	#
#
#
#
# 	lineSigmaIn1.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter1)['ReMax']);
# 	lineSigmaIn1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
#
# 	lineGammaIn1.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter1)['ImMax']);
# 	lineGammaIn1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
#
#
# 	lineSigmaAnti1.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter1)['ReMax']);
# 	lineSigmaAnti1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
# 	lineGammaAnti1.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter1)['ImMax']);
# 	lineGammaAnti1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
#
#
#
#
# 	lineSigmaIn2.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter2)['ReMax']);
# 	lineSigmaIn2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
#
# 	lineGammaIn2.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter2)['ImMax']);
# 	lineGammaIn2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
#
#
# 	lineSigmaAnti2.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter2)['ReMax']);
# 	lineSigmaAnti2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
# 	lineGammaAnti2.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter2)['ImMax']);
# 	lineGammaAnti2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
#
#
#
# 	lineSigmaIn3.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter3)['ReMax']);
# 	lineSigmaIn3.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
#
# 	lineGammaIn3.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, filter3)['ImMax']);
# 	lineGammaIn3.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
# 	#
#
#
#
#
# 	lineSigmaAnti3.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter3)['ReMax']);
# 	lineSigmaAnti3.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
# 	lineGammaAnti3.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, filter3)['ImMax']);
# 	lineGammaAnti3.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
# 	# 	# recompute the ax.dataLim
# 	ax.relim();
# 	ax1.relim();
#
# # 	# update ax.viewLim using the new dataLim
# 	ax.autoscale_view();
# 	ax1.autoscale_view();
#
# 	plt.draw()
# 	fig.canvas.draw_idle();
# 	fig1.canvas.draw_idle();
#
# v_slider.on_changed(sliders_on_changed)
# tauf_slider.on_changed(sliders_on_changed)
# Ga1_slider.on_changed(sliders_on_changed)
# tauc_slider.on_changed(sliders_on_changed)
# INV_slider.on_changed(sliders_on_changed)
#


# # add a button for resetting the parameters
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

# ax.legend()
ax1.legend()
#ax2.legend()
#ax5.legend()
plt.show()
