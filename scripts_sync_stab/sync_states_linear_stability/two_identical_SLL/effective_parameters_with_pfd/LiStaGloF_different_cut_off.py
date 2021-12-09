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
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
w    = 1.0*2.0*np.pi				#2.0*np.pi*24E9;
K    = 0.25*2.0*np.pi			#2.0*np.pi*250E6;
tauf = 0.0
tauc_0 = 1.0/(2.0*np.pi*0.02);
tauc_1 = 1.0/(2.0*np.pi*0.7)
tauc_2 = 1.0/(2.0*np.pi*1.1)
v	 = 1.0;
c	 = 3E8;
maxp = 14.3;


# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
#*******************************************************************************
fig         = plt.figure()
ax          = fig.add_subplot(111)
if digital == True:
	plt.title(r'digital case for $\omega$=%.3f' %w);
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'analog case for $\omega$=%.3f' %w);
# adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)
# 	plt.axhline(y=w, color='green', linestyle='-',linewidth=3)
tauf_0  = tauf;#tauf*w/(2.0*np.pi);
tauc_0  = tauc_0;
K_0     = K;
v_0		= v;
c_0		= c;
# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*******************************************************************************

[lineOmegStab] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['tauStab']*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'],tauf, K, tauc_0, v, digital, maxp, expansion)['OmegStab'],
				solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'], globalFreq(w, K, tauf, v, digital, maxp)['tau'],tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['OmegStab']/np.asarray(w),
				'.',ms=1, color='blue',  label=r'Stable')

[lineOmegUnst] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf, K, tauc_0, v, digital, maxp, expansion)['tauUnst']*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'],tauf_0, K_0, tauc_0, v_0,digital, maxp, expansion)['OmegUnst'],
				solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf, K, tauc_0, v, digital, maxp, expansion)['OmegUnst']/np.asarray(w),
				'o',ms=1, color='red', label=r'Unstable')



[lineOmegStab1] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_1, v_0, digital, maxp, expansion)['tauStab']*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'],tauf, K, tauc_1, v, digital, maxp, expansion)['OmegStab'],
				solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'], globalFreq(w, K, tauf, v, digital, maxp)['tau'],tauf_0, K_0, tauc_1, v_0, digital, maxp, expansion)['OmegStab']/np.asarray(w),
				'.',ms=1, color='yellow',  label=r'Stable,$\tau^c=1/0.7')

[lineOmegUnst1] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf, K, tauc_1, v, digital, maxp, expansion)['tauUnst']*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'],tauf_0, K_0, tauc_1, v_0,digital, maxp, expansion)['OmegUnst'],
				solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf, K, tauc_1, v, digital, maxp, expansion)['OmegUnst']/np.asarray(w),
				'o',ms=1, color='black', label=r'Unstable')


[lineOmegStab2] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_2, v_0, digital, maxp, expansion)['tauStab']*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'],tauf, K, tauc_2, v, digital, maxp, expansion)['OmegStab'],
				solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'], globalFreq(w, K, tauf, v, digital, maxp)['tau'],tauf_0, K_0,tauc_2, v_0, digital, maxp, expansion)['OmegStab']/np.asarray(w),
				'.',ms=1, color='blue',  label=r'Stable')

[lineOmegUnst2] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf, K, tauc_2, v, digital, maxp, expansion)['tauUnst']*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, K, tauf, v, digital, maxp)['tau'],tauf_0, K_0, tauc_2, v_0,digital, maxp, expansion)['OmegUnst'],
				solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf, K, tauc_2, v, digital, maxp, expansion)['OmegUnst']/np.asarray(w),
				'o',ms=1, color='red', label=r'Unstable')



fig1         = plt.figure()
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'The analytic expression characteristic equation');
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)

plt.axhline(y=0, color='green', linestyle='-',linewidth=3)


[lineSigma1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp, expansion)['Re'],	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda) \tau^c=1/(2\pi\;0.02)$')
[lineGamma1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp, expansion)['Im'],	linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda) \tau^c=1/(2\pi\;0.02)$')

[lineSigma2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_1, v_0, digital,maxp, expansion)['Re'],'-.',	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;0.7)$')
[lineGamma2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_1, v_0, digital,maxp, expansion)['Im'],'-.',	linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda)\tau^c=1/(2\pi\;0.7)$')

[lineSigma3] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_2, v_0, digital,maxp, expansion)['Re'],'--',	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;1.1)$')
[lineGamma3] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_2, v_0, digital,maxp, expansion)['Im'],'--',	linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda)\tau^c=1/(2\pi\;1.1)$')






fig2         = plt.figure()
ax2          = fig2.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'The analytic expression characteristic equation');
plt.grid()
plt.xlabel(r'$\tau$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)

plt.axhline(y=0, color='green', linestyle='-',linewidth=3)


[lineSigma_1] = ax2.plot(globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp, expansion)['Re'],	linewidth=3, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;0.02)$')
[lineGamma_1] = ax2.plot(globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp, expansion)['Im'],	linewidth=3, color='blue', label=r'$\gamma$=Im$(\lambda),\tau^c=1/(2\pi\;0.02)$')

[lineSigma_2] = ax2.plot(globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_1, v_0, digital,maxp, expansion)['Re'],'-.',	markersize=1, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;0.7)$')
[lineGamma_2] = ax2.plot(globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_1, v_0, digital,maxp, expansion)['Im'],'-.',	markersize=1, color='blue', label=r'$\gamma$=Im$(\lambda),\tau^c=1/(2\pi\;0.7)$')

[lineSigma_3] = ax2.plot(globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_2, v_0, digital,maxp, expansion)['Re'],'--',	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;1.1)$')
[lineGamma_3] = ax2.plot(globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_2, v_0, digital,maxp, expansion)['Im'],'--',	linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda),\tau^c=1/(2\pi\;1.1)$')





# add two sliders for tweaking the parameters
# define an axes area and draw a slider in it
# v_slider_ax   = fig.add_axes([0.25, 0.10, 0.65, 0.02], facecolor=axis_color)
# v_slider      = Slider(v_slider_ax, r'$v$', 1, 512, valinit=v_0)
# # Draw another slider
# tauf_slider_ax  = fig.add_axes([0.25, 0.07, 0.65, 0.02], facecolor=axis_color)
# tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf_0)#25*(2.0*np.pi/w)
# # Draw another slider
# K_slider_ax = fig.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
# K_slider    = Slider(K_slider_ax, r'$K$', 0.001*w, 1.5*w, valinit=K_0)
# # Draw another slider
# tauc_slider_ax  = fig.add_axes([0.25, 0.01, 0.65, 0.02], facecolor=axis_color)
# tauc_slider     = Slider(tauc_slider_ax, r'$\tau^c$', 0.05*(1.0/w), 10.0*(1.0/w), valinit=tauc_0)
#
# define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 	global digital
# 	initial_guess()
# 	lineOmegStab.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, digital,maxp, expansion)['OmegStab']);
# 	lineOmegStab.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion)['tauStab']*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, digital,maxp, expansion)['OmegStab']);
# 	lineOmegUnst.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion)['OmegUnst']);
# 	lineOmegUnst.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion)['tauUnst']*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion)['OmegUnst']);
#
#
# 	lineSigma1.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion)['Re']);
# 	lineSigma1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
#
# 	lineGamma1.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion)['Im']);
# 	lineGamma1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
# 	#
	# lineNyq.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Im']);
	# lineNyq.set_xdata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Re']);

# 	# recompute the ax.dataLim
# 	ax.relim();
# 	ax1.relim();
# 	#ax2.relim()
# # 	ax3.relim();
# #ax5.relim()
# # 	# update ax.viewLim using the new dataLim
# 	ax.autoscale_view();
# 	ax1.autoscale_view();
# 	#ax2.autoscale_view();
# # 	ax3.autoscale_view();
# #ax5.autoscale_view()
# 	plt.draw()
# 	fig.canvas.draw_idle();
# 	fig1.canvas.draw_idle();
# 	#fig2.canvas.draw_idle();
# # 	fig3.canvas.draw_idle();
# #fig5.canvas.draw_idle();
# v_slider.on_changed(sliders_on_changed)
# tauf_slider.on_changed(sliders_on_changed)
# K_slider.on_changed(sliders_on_changed)
# tauc_slider.on_changed(sliders_on_changed)

# add a button for resetting the parameters
PLLtype_button_ax = fig.add_axes([0.8, 0.9, 0.1, 0.04])
PLLtype_button = Button(PLLtype_button_ax, 'dig/ana', color=axis_color, hovercolor='0.975')
def PLLtype_button_on_clicked(mouse_event):
	global digital
	digital = not digital;
	print('state digital:', digital)
	if digital == True:
		ax.set_title(r'digital case for $\omega$=%.3f' %w);
		ax1.set_title(r'digital case for $\omega$=%.3f, linear stability' %w);
		ax2.set_title(r'digital case for $\omega$=%.3f, linear stability' %w);
		#ax5.set_title(r'digital case for $\omega$=%.3f, Nyquist' %w);
		#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
	else:
		ax.set_title(r'analog case for $\omega$=%.3f' %w);
		ax1.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
		ax2.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
		#ax5.set_title(r'analog case for $\omega$=%.3f, Nyquist' %w);
	fig.canvas.draw_idle()
PLLtype_button.on_clicked(PLLtype_button_on_clicked)

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
ax2.legend()
#ax5.legend()
plt.show()
