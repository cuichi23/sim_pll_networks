#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq, linStabEq, initial_guess, linStabEq,globalFreqNonlinear, solveLinStabNonliner,NonlinearFreqResponse
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
digital = True;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;

inphase1= True;
inphase2= False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider

w    	= 2.0*np.pi*24.25E9;	# intrinsic	frequency
Kvco    = 2.0*np.pi*(757.64E6);	# Sensitivity of VCO
AkPD	= 0.8					# amplitude of the PD -- the factor of 2 here is related to the IMS 2020 paper, AkPD as defined in that paper already includes the 0.5 that we later account for when calculating K
Ga1     = 1.0					 	# Gain of the first adder
order	= 2.0					# the order of the Loop Filter
tauf    = 0.0					# tauf = sum of all processing delays in the feedback
tauc	= 1.0/(2.0*np.pi*0.965E6);  # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
v		= 512;					# the division
c		= 0.63*3E8				# speed of light
maxp 	= 250;
INV		= 1.0*np.pi				# Inverter

wnormy = 2.0*np.pi;				# this is the frequency with which we rescale the y-axis, choose either 1, 2pi or w (y-axis: radHz, Hz, dimless)
wnormx = 2.0*np.pi;				# this is the frequency with which we rescale the x-axis, choose either 2pi or w
Vbias  = 2.34
zeta	= -1
figwidth  =	6;
figheight = 6;
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
syncstateInphase = globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)
stabIn           = solveLinStab(syncstateInphase['Omeg'], syncstateInphase['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)
syncstateInphaseNonlin= globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)

stabInNonlin = solveLinStabNonliner(syncstateInphaseNonlin['Omeg'], syncstateInphaseNonlin['tau'], Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)
# print(globalFreqINV(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'])
# fig0    = plt.figure(num=0, figsize=(figwidth, figheight), facecolor='w', edgecolor='k')


#*******************************************************************************
fig         = plt.figure(figsize=(figwidth,figheight))
ax          = fig.add_subplot(111)

if digital == True:
	plt.title(r'digital case for $\omega=2\pi\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'analog case for $\omega=2\pi\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
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



# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later
# #*************************************************************************************************************************************************************************


[lineOmegStabIn] = ax.plot(np.asarray(1.0/(2.0*np.pi))*stabIn['tauStab']*stabIn['OmegStab'], stabIn['OmegStab']/np.asarray(wnormy),
						'o',ms=7, color='blue',  label=r'linear Response Stable')


[lineOmegUnstIn] = ax.plot(np.asarray(1.0/(2.0*np.pi))*stabIn['tauUnst']*stabIn['OmegUnst'],
						 stabIn['OmegUnst']/np.asarray(wnormy),
						  'o',ms=1, color='blue', label=r'linear Response Unstable')


[lineOmegStabInNon] = ax.plot(np.asarray(1.0/(2.0*np.pi))*stabInNonlin['tauStab']*stabInNonlin['OmegStab'],
						 np.asarray(1/(2.0*np.pi))*stabInNonlin['OmegStab'],
						 'o',ms=7, color='red',  label=r'Inphase Stable Nonlinear Response')

[lineOmegUnstInNon] = ax.plot(np.asarray(1.0/(2.0*np.pi))*stabInNonlin['tauUnst']*stabInNonlin['OmegUnst'],
						 np.asarray(1/(2.0*np.pi))*stabInNonlin['OmegUnst'],
						 'o',ms=1, color='red', label=r'Inphase Unstable Nonlinear Response')




#*************************************************************************************************************************************************************************
fig1         = plt.figure(figsize=(figwidth,figheight))
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'Analysis of full expression of characteristic equation');
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)

plt.axhline(y=0, color='green', linestyle='-',linewidth=3)



[lineSigma1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*syncstateInphase['Omeg']*syncstateInphase['tau'], np.asarray(2.0*np.pi/w)*stabIn['Re'],
	linewidth=4, color='red', label=r'$\sigma$, linear Response')
[lineGamma1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*syncstateInphase['Omeg']*syncstateInphase['tau'], np.asarray(1/w)*stabIn['Im'],
	'+', ms=4, color='blue', label=r'$\gamma$, linear Response')

#
[lineSigmaNon] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*syncstateInphase['Omeg']*syncstateInphase['tau'], np.asarray(2.0*np.pi/w)*stabInNonlin['Re'],
	linewidth=4, color='red', label=r'$\sigma$, Nonlinear Response')
[lineGammaNon] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*syncstateInphase['Omeg']*syncstateInphase['tau'], np.asarray(1/w)*stabInNonlin['Im'],
	'+', ms=4, color='blue', label=r'$\gamma$, Nonlinear Response')


# [lineSigma2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV )['Re'],
# 	'--', linewidth=1, color='red', label=r'$\sigma$=Re$(\lambda)$ (antiphase)')
# [lineGamma2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital,maxp, inphase2, expansion, INV )['Im'],
# 	 '.',ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$ (antiphase)')
#



# add two sliders for tweaking the parameters
# define an axes area and draw a slider in it
# v_slider_ax   = fig0.add_axes([0.15, 0.80, 0.65, 0.15], facecolor=axis_color)
# v_slider      = Slider(v_slider_ax, r'$v$', 4, 8, valinit=v, valstep=4)
# # Draw another slider
# tauf_slider_ax  = fig0.add_axes([0.15, 0.60, 0.65, 0.15], facecolor=axis_color)
# tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf, valfmt='%1.2E')
# # Draw another slider
# Gvga_slider_ax = fig0.add_axes([0.15, 0.40, 0.65, 0.15], facecolor=axis_color)
# Gvga_slider    = Slider(Gvga_slider_ax, r'$G^{vga}$', 0.5, 2.0, valinit=Gvga)
# # Draw another slider
# wc_slider_ax  = fig0.add_axes([0.15, 0.20, 0.65, 0.15], facecolor=axis_color)
# wc_slider     = Slider(wc_slider_ax, r'$\omega_c$ in MHz', 100, 800, valinit=(1/(2.0E6*np.pi*tauc)))
#
# INV_slider_ax  = fig0.add_axes([0.15, 0.02, 0.65, 0.15], facecolor=axis_color)
# INV_slider     = Slider(INV_slider_ax, r'$INV$', 0.0, np.pi, valinit=INV, valstep=np.pi)
#

# Draw another slider


# define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 	global digital
# 	initial_guess()
# 	lineOmegStabIn.set_ydata(np.asarray(1.0/wnormy)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['OmegStab']);
# 	lineOmegStabIn.set_xdata(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['OmegStab']);
# 	lineOmegUnstIn.set_ydata(np.asarray(1.0/wnormy)*solveLinStab(globalFreq(w,Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['OmegUnst']);
# 	lineOmegUnstIn.set_xdata(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val)['OmegUnst']);
#
#
# 	lineOmegStabAnti.set_ydata(np.asarray(1.0/wnormy)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion,INV_slider.val)['OmegStab']);
# 	lineOmegStabAnti.set_xdata(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion,INV_slider.val)['OmegStab']);
# 	lineOmegUnstAnti.set_ydata(np.asarray(1.0/wnormy)*solveLinStab(globalFreq(w,Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['OmegUnst']);
# 	lineOmegUnstAnti.set_xdata(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val)['OmegUnst']);
#
#
#
#
#
#
#
# 	lineSigma1.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['Re']);
# 	lineSigma1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
#
# 	lineGamma1.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['Im']);
# 	lineGamma1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
# 	#
#
#
#
#
# 	lineSigma2.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['Re']);
# 	lineSigma2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
# 	lineGamma2.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['Im']);
# 	lineGamma2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
# 	# line.set_xdata(min_delay*(w/(2.0*np.pi)/v_slider.val))
# 	ax.set_title(r'Analysis of full expression of characteristic equation, min_delay= %0.2f' %(min_delay*(w)/(2.0*np.pi)/v_slider.val));
# 	ax1.set_title(r'analog case for $\omega=2\pi\cdot$%2.4E Hz and min_delay= %0.2f' %(w/(2.0*np.pi),min_delay*(w)/(2.0*np.pi)/v_slider.val));
#
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
# Gvga_slider.on_changed(sliders_on_changed)
# wc_slider.on_changed(sliders_on_changed)
# INV_slider.on_changed(sliders_on_changed)

# add a button for resetting the parameters


# # add a set of radio buttons for changing color
# color_radios_ax = fig.add_axes([0.025, 0.75, 0.15, 0.15], facecolor=axis_color)
# color_radios = RadioButtons(color_radios_ax, ('red', 'green'), active=0)
# def color_radios_on_clicked(label):
#     lineBeta12.set_color(label)
#     ax.legend()
#     fig.canvas.draw_idle()
# color_radios.on_clicked(color_radios_on_clicked)

ax.legend()
# ax1.legend()
#ax2.legend()
#ax5.legend()
plt.show()
