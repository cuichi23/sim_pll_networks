#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq,linStabEq,linStabEq_expansion
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
from matplotlib.legend import Legend
from plotly import graph_objs as go
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
w    	= 2.0*np.pi*24E9;
Kvco    = 2*np.pi*(754.64E6);
AkPD	= 1.6
Ga1		= 0.85
tauf 	= 0.0
tauc	= 1.0/(2.0*np.pi*120E6);
v	 	= 32.0;
c		= 3E8
maxp 	= 70;


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
plt.xlabel(r'$d=c\tau$ in [mm]', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)
tauf_0  = tauf;
tauc_0  = tauc;
# K_0     = K;
v_0		= v;

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*********************************************************************************


[lineOmegStab] = ax.plot(np.asarray(c)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['tauStab'],
						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'],tauf, Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['OmegStab']/np.asarray(w),
						 '.',ms=1, color='blue', label=r'$\Omega_{st} v	 = 32.0, c=		3E8$')

[lineOmegUnst] = ax.plot(np.asarray(c)*solveLinStab(globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['tauUnst'],
						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['OmegUnst']/np.asarray(w),
						 'o',ms=1, color='red', label=r'$\Omega_{un},  w  = 2.0\pi*24 GHz$')

# draw the initial plot
# the 'lineXXX' variables are used for modifying the lines later
#*******************************************************************************


fig1         = plt.figure()
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$d=c\tau$ in [m]', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega,\gamma/\omega$', fontsize=18)

# draw the initial plot
# the 'lineXXX' variables are used for modifying the lines later
# [lineBeta12ax1] = ax1.plot(xrange(K), np.real(solveLinStab(xrange(K), tau_0, tauf_0, K_0, tauc, digital)),
# 	linewidth=2, color='red', label=r'Re$(\lambda)$')
# [lineBetaR1ax1] = ax1.plot(xrange(K), np.imag(solveLinStab(xrange(K), tau_0, tauf_0, K_0, tauc, digital)),
# 	linewidth=2, color='blue', label=r'Im$(\lambda)$')
[lineSigma] = ax1.plot(np.asarray(c)*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0,Kvco, AkPD, Ga1, tauc_0, v_0, digital,maxp, expansion)['Re'],	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda)$')
[lineGamma] = ax1.plot(np.asarray(c)*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'],np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0,Kvco, AkPD, Ga1, tauc_0, v_0, digital,maxp, expansion)['Im'],	'.', ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$')
#**********************************************************************************************************************************************************
#

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
		# ax5.set_title(r'digital case for $\omega$=%.3f, Nyquist' %w);
		#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
	else:
		ax.set_title(r'analog case for $\omega$=%.3f' %w);
		ax1.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
		# ax2.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
		# ax5.set_title(r'analog case for $\omega$=%.3f, Nyquist' %w);
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
# ax5.legend()
plt.show()
