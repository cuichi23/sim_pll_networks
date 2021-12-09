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
w    	= 2.0*np.pi*24E9;
Kvco    = 2*np.pi*(754.64E6);
AkPD	= 1.6
Ga1		= 0.2
tauf 	= 0.0
order1  = 1
order2  = 2
order3  = 5
order4  = 10
tauc1	= 1.0/(order1*2.0*np.pi*120E6);
tauc2	= 1.0/(order2*2.0*np.pi*120E6);
tauc3	= 1.0/(order3*2.0*np.pi*120E6);
tauc4	= 1.0/(order4*2.0*np.pi*120E6);
v	 	= 32.0;
c		= 3E8
maxp 	= 120;


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
tauc_0  = tauc1;
Ga1_0	= Ga1;
v_0		= v;
c_0		= c;
# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*******************************************************************************

[lineOmegStab] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc1, v_0,order1, digital, maxp, expansion)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'],tauf, Kvco, AkPD, Ga1, tauc1, v,order1, digital, maxp, expansion)['OmegStab'],
				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'], globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'],tauf_0, Kvco, AkPD, Ga1_0, tauc1, v_0,order1, digital, maxp, expansion)['OmegStab']/np.asarray(w),
				'.',ms=1, color='blue',  label=r'Stable')

[lineOmegUnst] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf, Kvco, AkPD, Ga1, tauc1, v,order1, digital, maxp, expansion)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'],tauf_0, Kvco, AkPD, Ga1_0, tauc1, v_0, order1, digital, maxp, expansion)['OmegUnst'],
				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,Kvco, AkPD, Ga1, tauc1, v,order1, digital, maxp, expansion)['OmegUnst']/np.asarray(w),
				'o',ms=1, color='red', label=r'Unstable')

fig1         = plt.figure()
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'solutions to char. eq. for different cases of the chain topology , where $\sigma=$Re$(\lambda)$ vs $\tau$ for $ Kvco=$%.3f, $w^c=120E6Hz$, $v=$%d and $\tau^f=$%.3f where a is the order of the LF' %( Kvco, v,tauf));
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$', fontsize=18)

plt.axhline(y=0, color='green', linestyle='-',linewidth=3)


[lineSigma1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc1, v_0, order1, digital,maxp, expansion)['Re'],	linewidth=2, color='blue', label=r'$\sigma$=Re$(\lambda)$, $a=$%d'%(order1))
[lineSigma2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc2, v_0, order2, digital,maxp, expansion)['Re'],	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda)$, $a=$%d'%(order2))

[lineSigma3] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc3, v_0, order3, digital,maxp, expansion)['Re'],	linewidth=2, color='green', label=r'$\sigma$=Re$(\lambda)$, $a=$%d'%(order3))


[lineSigma4] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc4, v_0,order4, digital,maxp, expansion)['Re'],'--',	linewidth=2, color='purple', label=r'$\sigma$=Re$(\lambda)$, $a=$%d'%(order4))


fig2         = plt.figure()
ax2          = fig2.add_subplot(111)
plt.title(r' solutions to char. eq. for different cases of the chain topology , where  $\gamma$=Im$(\lambda)$ vs $\tau$ for $Kvco=$%.3f,  $w^c=120E6Hz$, $v=$%d and $\tau^f=$%.3f where a is the order of the LF' %( Kvco, v,tauf));
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\gamma/\omega$', fontsize=18)

plt.axhline(y=0, color='green', linestyle='-',linewidth=3)


[lineGamma1] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc1, v_0, order1, digital,maxp, expansion)['Im'], '.',	ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$, $a=$%d'%(order1))

[lineGamma2] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc2, v_0, order2, digital,maxp, expansion)['Im'], '.',	ms=1, color='red', label=r'$\gamma$=Im$(\lambda)$, $a=$%d'%(order2))

# add two sliders for tweaking the parameters
# define an axes area and draw a slider in it
v_slider_ax   = fig.add_axes([0.25, 0.20, 0.65, 0.02], facecolor=axis_color)
v_slider      = Slider(v_slider_ax, r'$v$', 1, 512, valinit=v_0)
# Draw another slider
tauf_slider_ax  = fig.add_axes([0.25, 0.17, 0.65, 0.02], facecolor=axis_color)
tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf_0)#25*(2.0*np.pi/w)
# Draw another slider
Ga1_slider_ax = fig.add_axes([0.25, 0.14, 0.65, 0.02], facecolor=axis_color)
Ga1_slider    = Slider(Ga1_slider_ax, r'$G^{a,1}$', 0.01, 0.85, valinit=Ga1_0)
# Draw another slider
tauc1_slider_ax  = fig.add_axes([0.25, 0.10, 0.65, 0.02], facecolor=axis_color)
tauc1_slider     = Slider(tauc1_slider_ax, r'$b1=\tau^c/a1$', 0.05*1.0/(2.0*np.pi*120E6), 1.0/(2.0*np.pi*120E6), valinit=tauc1)
tauc2_slider_ax  = fig.add_axes([0.25, 0.07, 0.65, 0.02], facecolor=axis_color)
tauc2_slider     = Slider(tauc2_slider_ax, r'$b2=\tau^c/a2$', 0.05*1.0/(2.0*np.pi*120E6), 1.0/(2.0*np.pi*120E6), valinit=tauc2)
tauc3_slider_ax  = fig.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
tauc3_slider     = Slider(tauc3_slider_ax, r'$b1=\tau^c/a1$', 0.05*1.0/(2.0*np.pi*120E6), 1.0/(2.0*np.pi*120E6), valinit=tauc3)
tauc4_slider_ax  = fig.add_axes([0.25, 0.01, 0.65, 0.02], facecolor=axis_color)
tauc4_slider     = Slider(tauc4_slider_ax, r'$b2=\tau^c/a2$', 0.05*1.0/(2.0*np.pi*120E6), 1.0/(2.0*np.pi*120E6), valinit=tauc4)




# define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
	global digital
	initial_guess()
	lineOmegStab.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
						tauc1_slider.val, v_slider.val,order1, digital,maxp, expansion)['OmegStab']);
	lineOmegStab.set_xdata(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc1_slider.val, v_slider.val,order1, digital,maxp, expansion)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
						tauc1_slider.val, v_slider.val, order1, digital,maxp, expansion)['OmegStab']);


	lineOmegUnst.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w,Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc1_slider.val, v_slider.val,order1, digital,maxp, expansion)['OmegUnst']);
	lineOmegUnst.set_xdata(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc1_slider.val, v_slider.val,order1, digital,maxp, expansion)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc1_slider.val, v_slider.val,order1, digital,maxp, expansion)['OmegUnst']);



	lineSigma1.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc1_slider.val, v_slider.val, order1, digital,maxp, expansion)['Re']);
	lineSigma1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);

	lineSigma2.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	 	globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc2_slider.val, v_slider.val,order2, digital,maxp, expansion)['Re']);
	lineSigma2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	lineSigma3.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc3_slider.val, v_slider.val,order3, digital,maxp, expansion)['Re']);
	lineSigma3.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);

	lineSigma4.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc4_slider.val, v_slider.val,order4, digital,maxp, expansion)['Re']);
	lineSigma4.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);







	lineGamma1.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc1_slider.val, v_slider.val, order1, digital,maxp, expansion)['Im']);
	lineGamma1.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);

	lineGamma2.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc2_slider.val, v_slider.val, order2, digital,maxp, expansion)['Im']);
	lineGamma2.set_xdata(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	# lineNyq.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Im']);
	# lineNyq.set_xdata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Re']);

# 	# recompute the ax.dataLim
	ax.relim();
	ax1.relim();
	ax2.relim()
# 	ax3.relim();
#ax5.relim()
# 	# update ax.viewLim using the new dataLim
	ax.autoscale_view();
	ax1.autoscale_view();
	ax2.autoscale_view();
# 	ax3.autoscale_view();
#ax5.autoscale_view()
	plt.draw()
	fig.canvas.draw_idle();
	fig1.canvas.draw_idle();
	fig2.canvas.draw_idle();
# 	fig3.canvas.draw_idle();
#fig5.canvas.draw_idle();
v_slider.on_changed(sliders_on_changed)
tauf_slider.on_changed(sliders_on_changed)
Ga1_slider.on_changed(sliders_on_changed)
tauc1_slider.on_changed(sliders_on_changed)
tauc2_slider.on_changed(sliders_on_changed)
tauc3_slider.on_changed(sliders_on_changed)
tauc4_slider.on_changed(sliders_on_changed)

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
