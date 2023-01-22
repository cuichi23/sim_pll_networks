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
#False is the expansion
expansion1=False;
expansion2=True;
# choose phase or anti-phase synchroniazed states,
inphase1 = True;
inphase2 = False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider

w    = 1.0*2.0*np.pi				#2.0*np.pi*24E9;
Kvco = 0.055*2.0*np.pi			#2.0*np.pi*250E6;
AkPD = 1.0
Ga1  = 1.0;
tauf = 0.0
tauc = 1.0/(2.0*np.pi*0.40);
order= 1.0
v	 = 1.0;
c	 = 3E8;
maxp = 25;

INV  = 0.0*np.pi;
zeta = -1.0


# ADD PLOTS
####################################################################################################################################################################################


figwidth  =	6;
figheight = 6;
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'

# print(globalFreqINV(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'])
fig0    = plt.figure(num=0, figsize=(figwidth, figheight), facecolor='w', edgecolor='k')


#*******************************************************************************
fig         = plt.figure(figsize=(figwidth,figheight))
ax          = fig.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'Comparison of linear stability: full char. eq. vs 3rd order expansion, $\sigma=$Re$(\lambda)$ $\gamma$=Im$(\lambda)$');
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)

[lineSigmaExpan] = ax.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion2, INV, zeta)['ReMax'],
	'--',linewidth=4, color='red', label=r'$\sigma$=Re$(\lambda)$ (3rd Order Expansion) ')
[lineGammaExpan] = ax.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion2, INV, zeta)['ImMax'],
	'.', ms=4, color='blue', label=r'$\gamma$=Im$(\lambda)$(3rd Order Expansion)')


[lineSigmaFull] = ax.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion1, INV, zeta)['ReMax'],
	linewidth=4, color='red', label=r'$\sigma$=Re$(\lambda)$ (Full Expression)')
[lineGammaFull] = ax.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion1, INV, zeta)['ImMax'],
	'.', ms=4, color='blue', label=r'$\gamma$=Im$(\lambda)$Full Expression)')
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc, v_0, digital,al)['Im'],	linewidth=2, color='blue', label=r'Im$(\lambda)$')

# add two sliders for tweaking the parameters
# define an axes area and draw a slider in it
v_slider_ax   = fig0.add_axes([0.25, 0.67, 0.65, 0.1], facecolor=axis_color)
v_slider      = Slider(v_slider_ax, r'$v$', 1, 16, valinit=v, valstep=1)
# Draw another slider
tauf_slider_ax  = fig0.add_axes([0.25, 0.45, 0.65, 0.1], facecolor=axis_color)
tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf, valfmt='%1.2E')
# Draw another slider
Ga1_slider_ax = fig0.add_axes([0.25, 0.23, 0.65, 0.1], facecolor=axis_color)
Ga1_slider    = Slider(Ga1_slider_ax, r'$G^{a,1}$', 0.01*Ga1, 2.0*Ga1, valinit=Ga1)
# Draw another slider
tauc_slider_ax  = fig0.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
tauc_slider     = Slider(tauc_slider_ax, r'$\tau^c$', 0.005*tauc, 3.0*tauc, valinit=tauc, valfmt='%1.2E')

INV_slider_ax  = fig0.add_axes([0.25, 0.01, 0.65, 0.02], facecolor=axis_color)
INV_slider     = Slider(INV_slider_ax, r'$INV$', 0.0, np.pi, valinit=INV, valstep=np.pi)

# define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
	global digital


	lineSigmaExpan.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion2, INV_slider.val, zeta)['ReMax']);
	lineSigmaExpan.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);

	lineGammaExpan.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion2, INV_slider.val, zeta)['ImMax']);
	lineGammaExpan.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
	#




	lineSigmaFull.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion1, INV_slider.val, zeta)['ReMax']);
	lineSigmaFull.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);

	lineGammaFull.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion1, INV_slider.val, zeta)['ImMax']);
	lineGammaFull.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau']);
	# 	# recompute the ax.dataLim

# 	# recompute the ax.dataLim
	ax.relim();
	# ax1.relim();
	#ax2.relim()
# 	ax3.relim();
#ax5.relim()
# 	# update ax.viewLim using the new dataLim
	ax.autoscale_view();
	# ax1.autoscale_view();
	#ax2.autoscale_view();
# 	ax3.autoscale_view();
#ax5.autoscale_view()
	plt.draw()
	fig.canvas.draw_idle();
	# fig1.canvas.draw_idle();
	#fig2.canvas.draw_idle();
# 	fig3.canvas.draw_idle();
#fig5.canvas.draw_idle();

v_slider.on_changed(sliders_on_changed)
tauf_slider.on_changed(sliders_on_changed)
Ga1_slider.on_changed(sliders_on_changed)
tauc_slider.on_changed(sliders_on_changed)
INV_slider.on_changed(sliders_on_changed)


# add a button for resetting the parameters
PLLtype_button_ax = fig.add_axes([0.8, 0.9, 0.1, 0.04])
PLLtype_button = Button(PLLtype_button_ax, 'dig/ana', color=axis_color, hovercolor='0.975')
def PLLtype_button_on_clicked(mouse_event):
	global digital
	digital = not digital;
	print('state digital:', digital)
	if digital == True:
		ax.set_title(r'digital case for $\omega$=%.3f' %w);
		# ax1.set_title(r'digital case for $\omega$=%.3f, linear stability' %w);
		#ax2.set_title(r'digital case for $\omega$=%.3f, linear stability' %w);
		#ax5.set_title(r'digital case for $\omega$=%.3f, Nyquist' %w);
		#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
	else:
		ax.set_title(r'analog case for $\omega$=%.3f' %w);
		# ax1.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
		#ax2.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
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
# ax1.legend()
#ax2.legend()
#ax5.legend()
plt.show()
