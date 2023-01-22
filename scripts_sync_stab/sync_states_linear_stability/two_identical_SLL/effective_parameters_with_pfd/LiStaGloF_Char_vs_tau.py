#Linear Stability of Global Frequency (LiStaGloF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq, linStabEq
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
import mpmath
import time
import csv
mpmath.mp.dps = 25
import datetime
now = datetime.datetime.now()
# from mpmath import findroot
# from sympy import I, limit
# from sympy.solvers import solveset
# from sympy.solvers import solve
# from sympy import Symbol, Function, nsolve
# from sympy import sin, cos, exp
# import mpmath
# mpmath.mp.dps = 25

# choose digital vs analog
# digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# choose phase or anti-phase synchroniazed states,
PD 		 = 'pfd';
inphase	 = True;
inphase2 = False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider

w    = 1.0*2.0*np.pi				#2.0*np.pi*24E9;
K    = 1.0*2.0*np.pi*1.24972E-2		#2.0*np.pi*250E6;;	# Sensitivity of VCO 3rd Gen 1.26273E-2
tauf = 0.0
tauc = 1.0/(2.0*np.pi*3.97938E-5);	#3,97938×10⁻⁵
v	 = 512.0;
c	 = 3E8;
maxp = 80;



####################################################################################################################################################################################

print('System of two mutually delay-coupled oscilltors')
if PD=='digital':
	print('Each PLL has an XOR as a Phase Detector (PD)')
elif PD=='analog':
	print('Each PLL has a multiplier as Phase Detector (PD)')
elif PD=='pfd':
	print('Each PLL has a Phase Frequency Detector (PFD)')
dpi_value = 300;
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
#*******************************************************************************
fig         = plt.figure()
ax          = fig.add_subplot(111)

#
# if digital == True:
# 	plt.title(r'digital case for $\omega$=%.3f' %w);
# 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# else:
# 	plt.title(r'analog case for $\omega$=%.3f' %w);
# adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\omega\tau/2\pi$', fontsize=26)
plt.ylabel(r'$\Omega/\omega$', fontsize=26)
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
fig.set_size_inches(22,8.5)
# tauf  = tauf;#tauf*w/(2.0*np.pi);
# tauc_0  = tauc;
# K_0     = K;
# v_0		= v;
# c_0		= c;

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*******************************************************************************

[lineOmegStabIn] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'], globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['tauStab'],
						 solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'], globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['OmegStab']/np.asarray(w),
						'o',ms=2, color='blue',  label=r'In-phase Stable')

[lineOmegUnstIn] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['tauUnst'],
						 solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['OmegUnst']/np.asarray(w),
						 'o',ms=2, color='red', label=r'In-phase Unstable')
#
# #
#
# [lineOmegStabAnti] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase2)['Omeg'], globalFreq(w, K, tauf, v, PD, maxp, inphase2)['tau'], tauf, K, tauc, v, PD, maxp, inphase2, expansion)['tauStab'],
# 						 solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase2)['Omeg'], globalFreq(w, K, tauf, v, PD, maxp, inphase2)['tau'], tauf, K, tauc, v, PD, maxp, inphase2, expansion)['OmegStab']/np.asarray(w),
# 						'.',ms=1, color='blue',  label=r'Antiphase Stable')
#
# [lineOmegUnstAnti] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase2)['Omeg'],globalFreq(w, K, tauf, v, PD, maxp, inphase2)['tau'], tauf, K, tauc, v, PD, maxp, inphase2, expansion)['tauUnst'],
# 						 solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase2)['Omeg'],globalFreq(w, K, tauf, v, PD, maxp, inphase2)['tau'], tauf, K, tauc, v, PD, maxp, inphase2, expansion)['OmegUnst']/np.asarray(w),
# 						 '.',ms=1, color='red', label=r'Antiphase Unstable')


plt.savefig('plots/GlobalFreq_noINV_K%.2f_fc%.4f_v_%d_%d_%d_%d.pdf' %(K/(2.0*np.pi), (1.0)/(tauc*2.0*np.pi), v, now.year, now.month, now.day), dpi=dpi_value)
plt.savefig('plots/GlobalFreq_noINV_K%.2f_fc%.4f_v_%d_%d_%d_%d.png' %(K/(2.0*np.pi), (1.0)/(tauc*2.0*np.pi), v, now.year, now.month, now.day), dpi=dpi_value)



fig1         = plt.figure()
ax1          = fig1.add_subplot(111)

# plot grid, labels, define intial values
# plt.title(r' Characteristic equation inphase');
plt.grid()
plt.xlabel(r'$\omega\tau/2\pi$', fontsize=26)
plt.ylabel(r'$\sigma/\omega$, $\gamma/\omega$', fontsize=26)
ax1.xaxis.set_tick_params(labelsize=24)
ax1.yaxis.set_tick_params(labelsize=24)
fig1.set_size_inches(22,8.5)
# draw the initial plot
# [lineSigmaIn] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['Re'][0], '.', color='red', label=r'$\sigma$=Re$(\lambda)$')
#
# [lineSigmaIn2] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['Re'][10],'.', color='black', label=r'$\sigma$=Re$(\lambda)$')
[lineSigmaIn1] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['ReMax'], 'o', color='green', label=r'$\sigma$=Re$(\lambda_max)$')
[lineGammaIn] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['ImMax'], 'o', color='blue', label=r'$\gamma$=Im$(\lambda)$')


plt.savefig('plots/Stability_noINV_K%.2f_fc%.4f_v_%d_%d_%d_%d.pdf' %(K/(2.0*np.pi), (1.0)/(tauc*2.0*np.pi), v, now.year, now.month, now.day), dpi=dpi_value)
plt.savefig('plots/Stability_noINV_K%.2f_fc%.4f_v_%d_%d_%d_%d.png' %(K/(2.0*np.pi), (1.0)/(tauc*2.0*np.pi), v, now.year, now.month, now.day), dpi=dpi_value)





ax.legend()
ax1.legend()
# ax2.legend()
#ax5.legend()
plt.show()
