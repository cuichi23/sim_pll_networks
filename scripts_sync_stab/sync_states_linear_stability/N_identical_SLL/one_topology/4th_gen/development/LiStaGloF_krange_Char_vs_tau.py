#Linear Stability of Global Frequency (LiStaGloF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib2 import solveLinStab, globalFreq, linStabEq,Krange
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
w    = 2.0*np.pi*24E9+2*np.pi*(754.64E6)*2.12;
K    = 2*np.pi*(754.64E6)*0.4;
tauf = 0.1
tauc =0.1/w;
v	 = 32.0;
c=		3E8
maxp = 70;
print(solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf, K, tauc, v, digital, maxp, expansion)['ReKra'])
print('Omega=',globalFreq(w, K, tauf, v, digital, maxp)['Omeg'])
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
plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)
tauf_0  = tauf;#tauf*w/(2.0*np.pi);
tauc_0  = tauc;
K_0     = K;
v_0		= v;
c_0		= c;

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*******************************************************************************
# print(globalFreqKrange(w, tauf, v, digital, maxp))
# print(globalFreq(w,K, tauf, v, digital, maxp))
[lineOmegStab] = ax.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'],globalFreq(w, K, tauf, v, digital, maxp)['Omeg']/np.asarray(w),
						 '.',ms=1, color='blue',  label=r'Stable')
#
# [lineOmegStab] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreqKrange(w, tauf, v, digital, maxp)['OmegK'], globalFreqKrange(w, tauf, v, digital, maxp)['tauK'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['tauStab'],
# 						 solveLinStab(globalFreqKrange(w, tauf, v, digital, maxp)['OmegK'], globalFreqKrange(w, tauf, v, digital, maxp)['tauK'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['OmegStab']/np.asarray(w),
# 						 '.',ms=1, color='blue',  label=r'Stable')
#
# [lineOmegUnst] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['tauUnst'],
# 						 solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['OmegUnst']/np.asarray(w),
# 						 'o',ms=1, color='red', label=r'Unstable')

fig1         = plt.figure()
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'The analytic expression characteristic equation');
plt.grid()
plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\sigma/\omega$, $\gamma/\omega$', fontsize=18)

# draw the initial plot
# [lineSigma] = ax1.plot(Krange(), solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['Re'], linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda)$')
# [lineGamma] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	# globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['Im'], linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda)$')
# [lineSigma1] = ax1.plot(globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,al)['Re'],	linewidth=2, color='red', label=r'Re$(\lambda)$')
# [lineGamma1] = ax1.plot(globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,al)['Im'],	linewidth=2, color='blue', label=r'Im$(\lambda)$')
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
# [lineNyq] = ax2.plot(solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp,al)['Re'], solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp,al)['Im'],	linewidth=2, color='red', label=r'Im$(\lambda)$')
#
#

# # add two sliders for tweaking the parameters
# define an axes area and draw a slider in it


# define an action for modifying the line when any slider's value changes


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
		# ax2.set_title(r'digital case for $\omega$=%.3f, linear stability' %w);
		#ax5.set_title(r'digital case for $\omega$=%.3f, Nyquist' %w);
		#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
	else:
		ax.set_title(r'analog case for $\omega$=%.3f' %w);
		ax1.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
		# ax2.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
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
# ax2.legend()
#ax5.legend()
plt.show()
