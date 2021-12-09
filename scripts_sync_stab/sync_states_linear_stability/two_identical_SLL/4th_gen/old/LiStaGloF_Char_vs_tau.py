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
digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
w    	  = 2.0*np.pi*60E9;
Kvco      = 2*np.pi*(1E9);
AkPD	  = (350E-3)*(350E-3)*3.236
Ga1		  = 10.0
tauf 	  = 0.0
order 	  = 1.0
tauc	  = 1.0/(2.0*np.pi*800E6);
v	 	  = 8.0;
c		  = 3E8
maxp 	  = 150;
min_delay = 0.1E-9
# print(globalFreqKrange(w, K, tauf, v, digital, maxp))

# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
axcolor = 'lightgoldenrodyellow'
#*******************************************************************************
fig         = plt.figure()
ax          = fig.add_subplot(111)
if digital == True:
	plt.title(r'digital case for $\omega=$%0.2f GHz' %(w/(2.0E9*np.pi)) );
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'analog case for $\omega=$%0.2f GHz and Kvco=%0.2f GHz' %(w/(2.0E9*np.pi), Kvco/(2.0E9*np.pi)) );
# adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)

x = min_delay*(w/(2.0*np.pi)/v)  # line at this x position

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*******************************************************************************

[lineOmegStab] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, expansion)['tauStab'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, expansion)['OmegStab']/np.asarray(w),
						 '.',ms=1, color='blue',  label=r'Stable')

[lineOmegUnst] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, expansion)['tauUnst'],
						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, expansion)['OmegUnst']/np.asarray(w),
						 'o',ms=1, color='red', label=r'Unstable')
# line=plt.axvline(x=min_delay*(w/(2.0*np.pi)/v), label=r'$min(\tau)$ =%0.2f' %(min_delay*(w/(2.0*np.pi)/v)))

[line]=ax.plot((x,x),(0.998,1.0021),'k--',linestyle='--', label=r'min_delay= %0.2f' %(min_delay*(w)/(2.0*np.pi)/v))

# line=ax.axvline(min_delay*(w/(2.0*np.pi)/v), color='k', linestyle='--')
#, label=r'min_delay= %0.2f s' %(min_delay*(w)/(2.0*np.pi)/v))
# x1,y1=[np.asarray(min_delay)*np.asarray((w/(2.0*np.pi)/v)),np.asarray(min_delay)*np.asarray((w/(2.0*np.pi))/v)],[min(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc_0, v_0, digital, maxp, expansion)['OmegUnst']/np.asarray(w)),max(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc_0, v_0, digital, maxp, expansion)['OmegUnst']/np.asarray(w))]
#
# [line]=ax.plot(x1, y1,'--',ms=1, color='red', label=r'min_delay= %0.2f s' % (min_delay*(w)/(2.0*np.pi)/v))
# plt.annotate(df.iloc[pqr,0]+', (%.2f, %.2f)' % (x.ix[pqr], y.ix[pqr]), xy=(x.ix[pqr], y.ix[pqr]), xycoords='data', xytext=(x.ix[pqr], y.ix[pqr]+0.3), arrowprops=dict(arrowstyle='-|>'), horizontalalignment='center')


fig1         = plt.figure()
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'Analysis of full expression of characteristic equation');
plt.grid()
plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
# draw the initial plot
[lineSigma] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, expansion)['Re'], linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda)$')
[lineGamma] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, expansion)['Im'], '.', ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$')

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
v_slider_ax   = fig.add_axes([0.25, 0.10, 0.65, 0.02], facecolor=axis_color)
v_slider      = Slider(v_slider_ax, r'$v$', 4, 8, valinit=v, valfmt='%1.2f', valstep=4)

# Draw another slider
tauf_slider_ax  = fig.add_axes([0.25, 0.07, 0.65, 0.02], facecolor=axis_color)
tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf)
# Draw another slider
Ga1_slider_ax = fig.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
Ga1_slider    = Slider(Ga1_slider_ax, r'$G^{a,1}$', 0.01, 16.00, valinit=Ga1,valstep=0.0625)
# Draw another slider
wc_slider_ax  = fig.add_axes([0.25, 0.01, 0.65, 0.02], facecolor=axis_color)
wc_slider     = Slider(wc_slider_ax, r'$\omega_c$ in MHz', 100, 800, valinit=(1/(2.0E6*np.pi*tauc)))

# define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
	global digital
	# line.set_ydata(min(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	# 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, 1/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital,maxp, expansion)['OmegUnst']/np.asarray(w)))
	# line.set_xdata(np.asarray(min_delay)*np.asarray((w/(2.0*np.pi))/v_slider.val))
	# line.set_data(np.asarray(min_delay*(w/(2.0*np.pi)/v_slider.val)))
	lineSigma.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, expansion)['Re']/np.asarray(w));
	lineSigma.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);

	lineGamma.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, 1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, expansion)['Im']/np.asarray(w));
	lineGamma.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);

	lineOmegStab.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val,Kvco, AkPD, Ga1_slider.val,
						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, expansion)['OmegStab']/np.asarray(w));
	lineOmegStab.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, expansion)['tauStab']);

	lineOmegUnst.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, expansion)['OmegUnst']/np.asarray(w));
	lineOmegUnst.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
						1.0/(2.0E6*np.pi*wc_slider.val), v_slider.val, order, digital,maxp, expansion)['tauUnst']);
	line.set_xdata(min_delay*(w/(2.0*np.pi)/v_slider.val))
	# period.set_xdata(min_delay*(w/(2.0*np.pi)/v_slider.val), (w/(2.0*np.pi))*max(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']))
	# print(v_slider.val)

	# lineSigma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, 1/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital,al)['Re']);
	# lineSigma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	# lineGamma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, 1/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital,al)['Im']);
	# lineGamma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	# lineNyq.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, 1/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital,al)['Im']);
	# lineNyq.set_xdata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, 1/(2.0E6*np.pi*wc_slider.val), v_slider.val, digital,al)['Re']);

	# recompute the ax.dataLim
	ax.relim();
	ax1.relim();
	# ax2.relim()
	# ax3.relim();
	# ax5.relim()
# 	# update ax.viewLim using the new dataLim
	ax.autoscale_view();
	ax1.autoscale_view();
	# ax2.autoscale_view();
	# ax3.autoscale_view();
	# ax5.autoscale_view()
	plt.draw()
	fig.canvas.draw_idle();
	fig1.canvas.draw_idle();
	# fig2.canvas.draw_idle();
	# fig3.canvas.draw_idle();
	# fig5.canvas.draw_idle();

v_slider.on_changed(sliders_on_changed)
tauf_slider.on_changed(sliders_on_changed)
Ga1_slider.on_changed(sliders_on_changed)
wc_slider.on_changed(sliders_on_changed)

# add a button for resetting the parameters
PLLtype_button_ax = fig.add_axes([0.1, 0.9, 0.1, 0.04])
PLLtype_button = Button(PLLtype_button_ax, 'dig/ana', color=axis_color, hovercolor='0.975')
def PLLtype_button_on_clicked(mouse_event):
	global digital
	digital = not digital;
	print('state digital:', digital)
	if digital == True:
		ax.set_title(r'digital case for $\omega= 2.0 \pi 60GHz$');
		ax1.set_title(r'digital case for $\omega= 2.0 \pi 60GHz$, linear stability');
		# ax2.set_title(r'digital case for $\omega$=%.3f, linear stability' %w);
		#ax5.set_title(r'digital case for $\omega$=%.3f, Nyquist' %w);
		#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
	else:
		ax.set_title(r'analog case for $\omega= 2.0 \pi 60GHz$');
		ax1.set_title(r'analog case for $$\omega= 2.0 \pi 60GHz$, linear stability');
		# ax2.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
		#ax5.set_title(r'analog case for $\omega$=%.3f, Nyquist' %w);
	fig.canvas.draw_idle()
PLLtype_button.on_clicked(PLLtype_button_on_clicked)

def reset(event):
	Ga1_slider.reset()
	v_slider.reset();
	tauf_slider.reset();
	wc_slider.reset();

button.on_clicked(reset)

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
