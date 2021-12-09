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
PD = 'pfd';
inphase=True;
inphase2=False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider

w    = 1.0*2.0*np.pi				#2.0*np.pi*24E9;
K    = 0.4*2.0*np.pi			#2.0*np.pi*250E6;
tauf = 0.0
tauc = 1.0/(2.0*np.pi*0.014);
v	 = 1.0;
c	 = 3E8;
maxp = 47;

print('System of two mutually delay-coupled oscilltors')
print('Each PLL has a Phase Frequency Detector (PfD)')

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
plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)
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



fig1         = plt.figure()
ax1          = fig1.add_subplot(111)

# plot grid, labels, define intial values
plt.title(r'The analytic expression characteristic equation inphase');
plt.grid()
plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\sigma/\omega$, $\gamma/\omega$', fontsize=18)

# draw the initial plot
# [lineSigmaIn] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['Re'][0], '.', color='red', label=r'$\sigma$=Re$(\lambda)$')
#
# [lineSigmaIn2] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['Re'][10],'.', color='black', label=r'$\sigma$=Re$(\lambda)$')
[lineSigmaIn1] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['ReMax'], linewidth=2, color='green', label=r'$\sigma$=Re$(\lambda_max)$')
[lineGammaIn] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['ImMax'], '.', color='blue', label=r'$\gamma$=Im$(\lambda)$')


#
#
# fig2         = plt.figure()
# ax2          = fig2.add_subplot(111)
#
# # plot grid, labels, define intial values
# plt.title(r'Im($\lambda$) vs Re($\lambda$) inphase');
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$\sigma/\omega$, $\gamma/\omega$', fontsize=18)
#
# # draw the initial plot
# [lineNyq] = ax2.plot(solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'], globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['ImMax'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['ReMax'], '.', color='red', label=r'$\sigma$=Re$(\lambda)$')
#
# [lineNyq2] = ax2.plot(solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'], globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['Im'][0], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['Re'][0], '.', color='blue', label=r'$\sigma$=Re$(\lambda)$')
# #
# # fig2         = plt.figure()
# [lineNyq3] = ax2.plot(solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'], globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['Im'][10], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp, inphase)['tau'], tauf, K, tauc, v, PD, maxp, inphase, expansion)['Re'][11], '.', color='green', label=r'$\sigma$=Re$(\lambda)$')
#
# ax2          = fig2.add_subplot(111)
# # plot grid, labels, define intial values
# plt.title(r'The analytic expression characteristic equation Antiphase');
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$\sigma/\omega$, $\gamma/\omega$', fontsize=18)
#
#
# [lineSigmaAnti] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, PD, maxp, inphase2)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase2)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp, inphase2)['tau'], tauf, K, tauc, v, PD, maxp, inphase2, expansion)['ReMax'], linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda)$')
# [lineGammaAnti] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, PD, maxp, inphase2)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp, inphase2)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp, inphase2)['tau'], tauf, K, tauc, v, PD, maxp, inphase2, expansion)['ImMax'], '.', color='blue', label=r'$\gamma$=Im$(\lambda)$')


# [lineSigma1] = ax1.plot(globalFreq(w, K, tauf, v, PD, maxp)['Omeg']*globalFreq(w, K, tauf, v, PD, maxp)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp)['tau'], tauf, K, tauc, v, PD,al)['Re'],	linewidth=2, color='red', label=r'Re$(\lambda)$')
# [lineGamma1] = ax1.plot(globalFreq(w, K, tauf, v, PD, maxp)['Omeg']*globalFreq(w, K, tauf, v, PD, maxp)['tau'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp)['tau'], tauf, K, tauc, v, PD,al)['Im'],	linewidth=2, color='blue', label=r'Im$(\lambda)$')
#
#
# fig2         = plt.figure()
# ax2          = fig2.add_subplot(111)
# # plot grid, labels, define intial values
# plt.grid()
# plt.xlabel(r'Re$(\lambda)$', fontsize=18)
# plt.ylabel(r'Im$(\lambda)$', fontsize=18)
#
#
#
# [lineNyq] = ax2.plot(solveLinStab(globalFreq(w, K, tauf, v, PD, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp)['tau'], tauf, K, tauc, v, PD,maxp,al)['Re'], solveLinStab(globalFreq(w, K, tauf, v, PD, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, PD, maxp)['tau'], tauf, K, tauc, v, PD,maxp,al)['Im'],	linewidth=2, color='red', label=r'Im$(\lambda)$')
#
#

# # add two sliders for tweaking the parameters
# define an axes area and draw a slider in it
# v_slider_ax   = fig.add_axes([0.25, 0.10, 0.65, 0.02], facecolor=axis_color)
# v_slider      = Slider(v_slider_ax, r'$v$', 1, 512, valinit=v)
# # Draw another slider
# tauf_slider_ax  = fig.add_axes([0.25, 0.07, 0.65, 0.02], facecolor=axis_color)
# tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf)
# # Draw another slider
# K_slider_ax = fig.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
# K_slider    = Slider(K_slider_ax, r'$K$', 0.001*w, 1.5*w, valinit=K)
# # Draw another slider
# tauc_slider_ax  = fig.add_axes([0.25, 0.01, 0.65, 0.02], facecolor=axis_color)
# tauc_slider     = Slider(tauc_slider_ax, r'$\tau^c$', 0.05*(1.0/w), 10.0*(1.0/w), valinit=tauc)
#
# # define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 	global PD
#
# 	lineSigmaIn.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['Omeg'],
# 		globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, PD, maxp, inphase, expansion)['Re']/np.asarray(w));
# 	lineSigmaIn.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau']);
#
# 	lineGammaIn.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['Omeg'],
# 		globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, PD, maxp, inphase, expansion)['Im']/np.asarray(w));
# 	lineGammaIn.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau']);
#
# 	lineOmegStabIn.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, PD, maxp, inphase, expansion)['OmegStab']/np.asarray(w));
# 	lineOmegStabIn.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, PD, maxp, inphase, expansion)['tauStab']);
#
# 	lineOmegUnstIn.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, PD, maxp, inphase, expansion)['OmegUnst']/np.asarray(w));
# 	lineOmegUnstIn.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, PD, maxp, inphase, expansion)['tauUnst']);
#
# 	lineOmegStabAnti.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, PD, maxp, inphase, expansion)['OmegStab']/np.asarray(w));
# 	lineOmegStabAnti.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, PD, maxp, inphase, expansion)['tauStab']);
#
# 	lineOmegUnstAnti.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, PD, maxp, inphase, expansion)['OmegUnst']/np.asarray(w));
# 	lineOmegUnstAnti.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp, inphase)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, PD, maxp, inphase, expansion)['tauUnst']);


	# lineSigma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['Omeg'],
	# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, PD,al)['Re']);
	# lineSigma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['tau']);
	#
	# lineGamma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['Omeg'],
	# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, PD,al)['Im']);
	# lineGamma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['tau']);
	#
	# lineNyq.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, PD,al)['Im']);
	# lineNyq.set_xdata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, PD, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, PD,al)['Re']);

# 	# recompute the ax.dataLim
# 	ax.relim();
# 	ax1.relim();
# 	ax2.relim()
# 	# ax3.relim();
# 	# ax5.relim()
# # 	# update ax.viewLim using the new dataLim
# 	ax.autoscale_view();
# 	ax1.autoscale_view();
# 	ax2.autoscale_view();
# 	# ax3.autoscale_view();
# 	# ax5.autoscale_view()
# 	plt.draw()
# 	fig.canvas.draw_idle();
# 	fig1.canvas.draw_idle();
# 	fig2.canvas.draw_idle();
# 	# fig3.canvas.draw_idle();
# 	# fig5.canvas.draw_idle();
#
# v_slider.on_changed(sliders_on_changed)
# tauf_slider.on_changed(sliders_on_changed)
# K_slider.on_changed(sliders_on_changed)
# tauc_slider.on_changed(sliders_on_changed)

# add a button for resetting the parameters
# PLLtype_button_ax = fig.add_axes([0.8, 0.9, 0.1, 0.04])
# PLLtype_button = Button(PLLtype_button_ax, 'dig/ana', color=axis_color, hovercolor='0.975')
# def PLLtype_button_on_clicked(mouse_event):
# 	global PD
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
