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
Ga1     = 0.01					# Gain of the first adder
order	= 2.0					# the order of the Loop Filter
tauf    = 0.0					# tauf = sum of all processing delays in the feedback
tauc	= 1.0/(2.0*np.pi*0.965E6);  # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
v		= 512;					# the division
c		= 0.63*3E8				# speed of light
maxp 	= 4E4;
INV		= 1.0*np.pi				# Inverter

wnormy = 2.0*np.pi;				# this is the frequency with which we rescale the y-axis, choose either 1, 2pi or w (y-axis: radHz, Hz, dimless)
wnormx = 2.0*np.pi;				# this is the frequency with which we rescale the x-axis, choose either 2pi or w


figwidth  =	6;
figheight = 6;
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'

# print(globalFreqINV(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'])
# fig0    = plt.figure(num=0, figsize=(figwidth, figheight), facecolor='w', edgecolor='k')


#*******************************************************************************
fig         = plt.figure(figsize=(figwidth,figheight))
ax          = fig.add_subplot(111)
if digital == True:
	plt.title(r'digital case for $\omega=2\pi\cdot$%2.4E Hz, v=%0.1E and $G^{a,1}=$%0.5f' %(w/(2.0*np.pi),v,Ga1));
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'analog case for $\omega=2\pi\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
if wnormx == w:
	plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
else:
	plt.xlabel(r'$\tau$', fontsize=18)
if wnormy == 1:
	plt.ylabel(r'$\Omega$', fontsize=18)
elif wnormy == 2.0*np.pi:
	plt.ylabel(r'$f_{\Omega}$', fontsize=18)
elif wnormy == w:
	plt.ylabel(r'$\Omega/\omega$', fontsize=18)

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*******************************************************************************

[lineOmegStabIn] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['tauStab'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegStab']/np.asarray(wnormy),
						'o',ms=7, color='blue',  label=r'Inphase Stable')

[lineOmegUnstIn] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['tauUnst'],
						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegUnst']/np.asarray(wnormy),
						  'o',ms=1, color='blue', label=r'Inphase Unstable')

# [lineOmegStabAnti] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['tauStab'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['OmegStab']/np.asarray(wnormy),
# 						  'o',ms=7, color='red',  label=r'Antiphase Stable')
#
# [lineOmegUnstAnti] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['tauUnst'],
# 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['OmegUnst']/np.asarray(wnormy),
# 						 'o',ms=1, color='red', label=r'Antiphase Unstable')


fig1         =plt.figure(figsize=(figwidth,figheight))
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'Analysis of full expression of characteristic equation for v=%0.1E and $G^{a,1}=$%0.5f' %(v,Ga1));
plt.grid()
if wnormx == w:
	plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
else:
	plt.xlabel(r'$\tau$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)

# draw the initial plot
[lineSigmaIn] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['Omeg'],
	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['ReMax'],
	 linewidth=4, color='red', label=r'$\sigma$=Re$(\lambda)$ (in-phase)')

[lineGammaIn] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['ImMax'],
	'.', ms=4, color='blue', label=r'$\gamma$=Im$(\lambda)$ (in-phase)')
#
#
# [lineSigmaAnti] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['Re'],
# 	 linewidth=1, color='red', label=r'$\sigma$=Re$(\lambda)$ (anti-phase)')
#
# [lineGammaAnti] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['Im'],
# 	 '.',ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$ (anti-phase)')



# # add two sliders for tweaking the parameters
# define an axes area and draw a slider in it
# v_slider_ax   = fig0.add_axes([0.25, 0.13, 0.65, 0.02], facecolor=axis_color)
# v_slider      = Slider(v_slider_ax, r'$v$', 1, 512, valinit=v)
# # Draw another slider
# tauf_slider_ax  = fig0.add_axes([0.25, 0.10, 0.65, 0.02], facecolor=axis_color)
# tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf, valfmt='%1.2E')
# # Draw another slider
# Ga1_slider_ax = fig0.add_axes([0.25, 0.07, 0.65, 0.02], facecolor=axis_color)
# Ga1_slider    = Slider(Ga1_slider_ax, r'$G^{a,1}$', 0.01*Ga1, 100*Ga1, valinit=Ga1)
# # Draw another slider
# tauc_slider_ax  = fig0.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
# tauc_slider     = Slider(tauc_slider_ax, r'$\tau^c$', 0.05*tauc, 3.00*tauc, valinit=tauc, valfmt='%1.2E')
#
# INV_slider_ax  = fig0.add_axes([0.25, 0.01, 0.65, 0.02], facecolor=axis_color)
# INV_slider     = Slider(INV_slider_ax, r'$INV$', 0.0, np.pi, valinit=INV, valstep=np.pi)
#
#
# # define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 	global digital
#
# 	lineSigmaIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val)['Re']*np.asarray(2.0*np.pi/w));
# 	lineSigmaIn.set_xdata(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau']);
#
# 	lineGammaIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val)['Im']/np.asarray(w));
# 	lineGammaIn.set_xdata(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau']);
#
# 	lineSigmaAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val)['Re']*np.asarray(2.0*np.pi/w));
# 	lineSigmaAnti.set_xdata(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau']);
#
# 	lineGammaAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val)['Im']/np.asarray(w));
# 	lineGammaAnti.set_xdata(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
#
#
#
# 	lineOmegStabIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val,Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val)['OmegStab']/np.asarray(wnormy));
# 	lineOmegStabIn.set_xdata(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['tauStab']);
#
# 	lineOmegUnstIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['OmegUnst']/np.asarray(wnormy));
# 	lineOmegUnstIn.set_xdata(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val)['tauUnst']);
#
#
#
#
#
# 	lineOmegStabAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val,Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val)['OmegStab']/np.asarray(wnormy));
# 	lineOmegStabAnti.set_xdata(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['tauStab']);
#
# 	lineOmegUnstAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['OmegUnst']/np.asarray(wnormy));
# 	lineOmegUnstAnti.set_xdata(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val)['tauUnst']);
#
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
# Ga1_slider.on_changed(sliders_on_changed)
# tauc_slider.on_changed(sliders_on_changed)
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
