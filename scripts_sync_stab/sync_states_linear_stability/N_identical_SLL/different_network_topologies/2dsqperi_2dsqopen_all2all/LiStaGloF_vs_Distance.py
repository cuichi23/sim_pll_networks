#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq,linStabEq,solveLinStab_topology_comparison
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
import topology
from topology import eigenvalzeta

# choose digital vs analog
digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#False is the expansion
expansion=False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
w    = 2.0*np.pi		#2.0*np.pi*24E9;
K    = 2.0*np.pi*0.04 	#2.0*np.pi*250E6;
tauf = 0.0
tauc = 0.014;
v	 = 1.0;
maxp = 65;

Nx=3;
Ny=1;


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
plt.xlabel(r'$d/\lambda=\tau \omega/2\pi$', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)
tauf_0  = tauf;
tauc_0  = tauc;
K_0     = K;
v_0		= v;

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*********************************************************************************


[lineOmeg] = ax.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'],
					globalFreq(w, K, tauf, v, digital, maxp)['Omeg']/np.asarray(w), '.', ms=1, color='blue', label=r'$\Omega/omega$')
# draw the initial plot
# the 'lineXXX' variables are used for modifying the lines later
#*******************************************************************************


fig1         = plt.figure()
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'Stability vs Distance for $N_x=$%d, $N_y=$%d, $N=N_x\,N_y=$%d' %(Nx, Ny, Nx*Ny));
plt.grid()
plt.xlabel(r'$d/\lambda=\tau \omega/2\pi$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)
lt.axhline(y=0, color='blue', linestyle='-',linewidth=1)
# draw the initial plot


[lineSigmaAll2All] = ax1.plot(np.asarray(1/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp, expansion, eigenvalzeta('global',Nx,Ny)['zeta'])['ReMax'],'-', linewidth=1.5, color='red', label=r'Re$(\lambda)$ all2all')
# #
[lineSigmaSquareOpen] = ax1.plot(np.asarray(1/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp, expansion, eigenvalzeta('square-open',Nx,Ny)['zeta'])['ReMax'], '--',linewidth=1.5, color='black', label=r'Re$(\lambda)$ square-open')


[lineSigmaSquarePeriodic] = ax1.plot(np.asarray(1/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp, expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ReMax'], '-',linewidth=1.5, color='green', label=r'Re$\lambda$ square-periodic')


# **********************************************************************************************************
fig2         = plt.figure()
ax2          = fig2.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'Imaginary parts vs Distance for $N_x=$%d, $N_y=$%d, $N=N_x\,N_y=$%d' %(Nx, Ny, Nx*Ny));
plt.grid()
plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\gamma/\omega$ ', fontsize=18)
lt.axhline(y=0, color='blue', linestyle='-',linewidth=3)


[lineGammaAll2All] = ax2.plot(np.asarray(1/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp, expansion, eigenvalzeta('global',Nx,Ny)['zeta'])['ImMax'], '-',linewidth=1.5, color='red', label=r'Im$(\lambda)$ all2all')
[lineGammaSquareOpen] = ax2.plot(np.asarray(1/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp, expansion, eigenvalzeta('square-open',Nx,Ny)['zeta'])['ImMax'], '--',linewidth=1.5, color='blue', label=r'Im$(\lambda)$ square-open')
[lineGammaSquarePeriodic] = ax2.plot(np.asarray(1/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp, expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ImMax'], '-',linewidth=1.5, color='green', label=r'Im$(\lambda)$ square-periodic')

#*******************************************************************************


# add two sliders for tweaking the parameters
# define an axes area and draw a slider in it
v_slider_ax1   = fig.add_axes([0.25, 0.13, 0.65, 0.02], facecolor=axis_color)
v_slider      = Slider(v_slider_ax1, 'v', 1, 512, valinit=v_0)
# Draw another slider
tauf_slider_ax1  = fig.add_axes([0.25, 0.10, 0.65, 0.02], facecolor=axis_color)
tauf_slider     = Slider(tauf_slider_ax1, 'tauf', 0.0, 25/w, valinit=tauf_0)
# Draw another slider
K_slider_ax1 = fig.add_axes([0.25, 0.07, 0.65, 0.02], facecolor=axis_color)
K_slider    = Slider(K_slider_ax1, 'K', 0.001*w, 1.5*w, valinit=K_0)
# Draw another slider
tauc_slider_ax1  = fig.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
tauc_slider     = Slider(tauc_slider_ax1, 'tauc', 0.05*(1.0/w), 10.0*(1.0/w), valinit=tauc_0)

# define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
	global digital
	lineOmeg.set_ydata(solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val,
						tauc_slider.val, v_slider.val, digital,maxp, expansion, eigenvalzeta('global',Nx,Ny)['zeta'])['Omeg']/np.asarray(w));
	lineOmeg.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val,
						tauf_slider.val, v_slider.val, digital,maxp, expansion, eigenvalzeta('global',Nx,Ny)['zeta'])['tau']);
	lineSigmaAll2All.set_ydata(np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion, eigenvalzeta('global',Nx,Ny)['zeta'])['ReMax']);
	lineSigmaAll2All.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	lineGammaAll2All.set_ydata(np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion, eigenvalzeta('global',Nx,Ny)['zeta'])['ImMax']);
	lineGammaAll2All.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);

	# lineSigmaSquare2d.set_ydata(np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion, eigenvalzeta('square-open',Nx,Ny)['zeta'])['ReMax']);
	# lineSigmaSquare2d.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);

	# lineGammaSquare2d.set_ydata(np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion, eigenvalzeta('square-open',Nx,Ny)['zeta'])['ImMax']);
	# lineGammaSquare2d.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);

	#
	lineSigmaSquareOpen.set_ydata(np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion, eigenvalzeta('square-open',Nx,Ny)['zeta'])['ReMax']);
	lineSigmaSquareOpen.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	# #
	lineGammaSquareOpen.set_ydata(np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion, eigenvalzeta('square-open',Nx,Ny)['zeta'])['ImMax']);
	lineGammaSquareOpen.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	# #

	lineSigmaSquarePeriodic.set_ydata(np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ReMax']);
	lineSigmaSquarePeriodic.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	lineGammaSquarePeriodic.set_ydata(np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ImMax']);
	lineGammaSquarePeriodic.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#

	# recompute the ax.dataLim
	ax.relim();
	ax1.relim();
	# ax2.relim();
	# ax3.relim();
	# ax5.relim()
	# update ax.viewLim using the new dataLim
	ax.autoscale_view()
	ax1.autoscale_view()
	# ax2.autoscale_view();
	# ax3.autoscale_view();
	# ax5.autoscale_view();
	plt.draw()
	fig.canvas.draw_idle();
	fig1.canvas.draw_idle();
	# fig2.canvas.draw_idle();
	# fig3.canvas.draw_idle();
	# fig5.canvas.draw_idle();
v_slider.on_changed(sliders_on_changed)
tauf_slider.on_changed(sliders_on_changed)
K_slider.on_changed(sliders_on_changed)
tauc_slider.on_changed(sliders_on_changed)

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
