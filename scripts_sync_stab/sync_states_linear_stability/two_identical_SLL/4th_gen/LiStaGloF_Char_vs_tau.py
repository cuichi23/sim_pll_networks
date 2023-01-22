#Linear Stability of Global Frequency (LiStaGloF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq, linStabEq #,globalFreqKrange
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
digital = 'False';

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;

inphase1= True;
inphase2= False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
w    	  	= 2.0*np.pi*62E9; 			# intrinsic	frequency
Kvco      	= 2.0*np.pi*(0.70E9); 		# Sensitivity of VCO
AkPD	  	= 0.162*1.0					# Amplitude of the output of the PD --
GkLF		= 1.0
Gvga	  	= 2						# Gain of the first adder
tauf 	  	= 0.0						# tauf = sum of all processing delays in the feedback
order 	  	= 1.0						# the order of the Loop Filter
tauc	  	= 1.0/(2.0*np.pi*10E6); 	# the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
v	 	  	= 512.0;						# the division
c		  	= 3E8						# speed of light
min_delay 	= 0.1E-9
INV		  	= 0.0*np.pi					# Inverter
#			# Inverter
#
maxp 	  	=950;
wnormy 		= 2.0*np.pi;				# this is the frequency with which we rescale the y-axis, choose either 1, 2pi or w (y-axis: radHz, Hz, dimless)
wnormx 		= 2.0*np.pi;				# this is the frequency with which we rescale the x-axis, choose either w/2pi-rescaling (2.0*np.pi) or no rescaling of tau (w)

filter		= 1
figwidth  	= 6;
figheight 	= 6;
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'

# print(globalFreqINV(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'])
# fig0    = plt.figure(num=0, figsize=(figwidth, figheight), facecolor='w', edgecolor='k')


#*******************************************************************************
fig         = plt.figure(figsize=(figwidth,figheight))
ax          = fig.add_subplot(111)
if digital == True:
	plt.title(r'digital case for $\omega=2\pi\cdot$%2.4E Hz and min_delay= %0.2f' %(w/(2.0*np.pi),min_delay*(w)/(2.0*np.pi)/v));
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'analog case for $\omega=2\pi\cdot$%2.4E Hz and min_delay= %0.2f' %(w/(2.0*np.pi),min_delay*(w)/(2.0*np.pi)/v));
# adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
if wnormx == 1:
	plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
	x = min_delay*(w/(2.0*np.pi)/v)
else:
	plt.xlabel(r'$\tau$', fontsize=18)
	x = min_delay*(1.0/v)
if wnormy == 1:
	plt.ylabel(r'$\Omega$', fontsize=18)
elif wnormy == 2.0*np.pi:
	plt.ylabel(r'$f_{\Omega}$', fontsize=18)
elif wnormy == w:
	plt.ylabel(r'$\Omega/\omega$', fontsize=18)

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later
# #*******************************************************************************

[lineOmegStabIn] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['tauStab'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, GkLF,Gvga,tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegStab']/np.asarray(w),
						'o',ms=7, color='blue',  label=r'Inphase Stable')

[lineOmegUnstIn] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['tauUnst'],
						 solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['OmegUnst']/np.asarray(w),
						  'o',ms=1, color='blue', label=r'Inphase Unstable')
#
# [lineOmegStabAnti] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['tauStab'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, GkLF,Gvga,tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['OmegStab']/np.asarray(w),
# 						  'o',ms=7, color='red',  label=r'Antiphase Stable')
#
# [lineOmegUnstAnti] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'],
#  							tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['tauUnst'],
# 						 solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2,INV)['tau'],
# 						 	tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['OmegUnst']/np.asarray(w),
# 						 'o',ms=1, color='red', label=r'Antiphase Unstable')
#
# [line]=ax.plot((x,x),(0.0001+max(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2,INV)['tau'],
# 						tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['OmegUnst']/np.asarray(w)),-0.0001+min(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2,INV)['tau'],
# 						tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['OmegUnst']/np.asarray(w))),'k.-',linestyle='--', linewidth=3)#label=r'min_delay= %0.2f' %(min_delay*(w)/(2.0*np.pi)/v))
# # [line]=ax.plot((x,x),(0.998,1.0021),'k--',linestyle='--')#, label=r'min_delay= %0.2f' %(min_delay*(w)/(2.0*np.pi)/v))

fig1         = plt.figure(figsize=(figwidth,figheight))
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.suptitle(r'Analysis of full expression of characteristic equation, min_delay= %0.2f' %(min_delay*(w)/(2.0*np.pi)/v));
plt.title(r'v= %0.2f, fc=%0.4E Hz, Kvco=%0.2f Hz/V, Gvga=%0.1f' %(v,1.0/(2.0*np.pi*tauc),Kvco,Gvga));

plt.grid()
if wnormx == w:
	plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
else:
	plt.xlabel(r'$\tau$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)

# draw the initial plot
[lineSigmaIn] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1,INV)['Omeg'],
	globalFreq(w,Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['Re'],
	 linewidth=4, color='red', label=r'$\sigma$=Re$(\lambda)$ (in-phase)')

[lineGammaIn] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1,INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w,Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase1, expansion, INV, filter)['Im'],
	'.', ms=4, color='blue', label=r'$\gamma$=Im$(\lambda)$ (in-phase)')


# [lineSigmaAnti] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['Re'],
# 	 linewidth=1, color='red', label=r'$\sigma$=Re$(\lambda)$ (anti-phase)')
#
# [lineGammaAnti] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase1,INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, order, digital, maxp, inphase2, expansion, INV, filter)['Im'],
# 	 '.',ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$ (anti-phase)')
#
#
#
# # add two sliders for tweaking the parameters
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
# wc_slider     = Slider(wc_slider_ax, r'$\omega_c$ in MHz', 0.1*1/(2.0E6*np.pi*tauc), 800, valinit=(1/(2.0E6*np.pi*tauc)))
#
# INV_slider_ax  = fig0.add_axes([0.15, 0.02, 0.65, 0.15], facecolor=axis_color)
# INV_slider     = Slider(INV_slider_ax, r'$INV$', 0.0, np.pi, valinit=INV, valstep=np.pi)
#
#
# # define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 	global digital
#
# 	lineSigmaIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'],
# 		tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val)['Re']*np.asarray(2.0*np.pi/w));
#
# 	lineSigmaIn.set_xdata(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau']);
#
# 	lineGammaIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val)['Im']/np.asarray(w));
# 	lineGammaIn.set_xdata(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau']);
#
# 	lineSigmaAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val)['Re']*np.asarray(2.0*np.pi/w));
# 	lineSigmaAnti.set_xdata(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau']);
#
# 	lineGammaAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val)['Im']/np.asarray(w));
# 	lineGammaAnti.set_xdata(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
#
#
#
# 	lineOmegStabIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val,Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val)['OmegStab']/np.asarray(w));
# 	lineOmegStabIn.set_xdata(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['tauStab']);
#
# 	lineOmegUnstIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['OmegUnst']/np.asarray(w));
# 	lineOmegUnstIn.set_xdata(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['tauUnst']);
#
#
#
#
#
# 	lineOmegStabAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val,Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val)['OmegStab']/np.asarray(w));
# 	lineOmegStabAnti.set_xdata(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['tauStab']);
#
# 	lineOmegUnstAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['OmegUnst']/np.asarray(w));
# 	lineOmegUnstAnti.set_xdata(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['tauUnst']);
#
# 	if wnormx == 1:
# 		line.set_xdata(min_delay*(w/(2.0*np.pi)/v_slider.val))
# 	else:
# 		line.set_xdata(min_delay*(1.0/v_slider.val))
#
# 	line.set_ydata((0.0001+max(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['OmegUnst']/np.asarray(w)),-0.0001+min(solveLinStab(globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 											globalFreq(w, Kvco, AkPD, GkLF,Gvga_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, GkLF,Gvga_slider.val,
# 											1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['OmegUnst']/np.asarray(w))))
# 	ax.set_title(r'Analysis of full expression of characteristic equation, min_delay= %0.2f' %(min_delay*(w)/(2.0*np.pi)/v_slider.val));
# 	ax1.set_title(r'analog case for $\omega=2\pi\cdot$%2.4E Hz and min_delay= %0.2f' %(w/(2.0*np.pi),min_delay*(w)/(2.0*np.pi)/v_slider.val));
#
# 	ax.relim();
# 	ax1.relim();
# 	# ax2.relim()
# 	# ax3.relim();
# 	# ax5.relim()
# # 	# update ax.viewLim using the new dataLim
# 	ax.autoscale_view();
# 	ax1.autoscale_view();
# 	# ax2.autoscale_view();
# 	# ax3.autoscale_view();
# 	# ax5.autoscale_view()
# 	plt.draw()
# 	fig.canvas.draw_idle();
# 	fig1.canvas.draw_idle();
# 	# fig2.canvas.draw_idle();
# 	# fig3.canvas.draw_idle();
# 	# fig5.canvas.draw_idle();
#
# v_slider.on_changed(sliders_on_changed)
# tauf_slider.on_changed(sliders_on_changed)
# Gvga_slider.on_changed(sliders_on_changed)
# wc_slider.on_changed(sliders_on_changed)
# INV_slider.on_changed(sliders_on_changed)
#
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
# 		# ax2.set_title(r'digital case for $\omega$=%.3f, linear stability' %w);
# 		#ax5.set_title(r'digital case for $\omega$=%.3f, Nyquist' %w);
# 		#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# 	else:
# 		ax.set_title(r'analog case for $\omega$=%.3f' %w);
# 		ax1.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
# 		# ax2.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
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
#ax5.legend()
plt.show()
