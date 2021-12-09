#Linear Stability of Global Frequency (LiStaGloF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib2
from LiStaGloF_lib2 import solveLinStab, globalFreq, linStabEq,Krange #,globalFreqKrange
from numpy import pi, sin
import numpy as np
import sympy
from sympy import solve, nroots, I
from sympy.abc import q
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib as plt1
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
results     	 = [];
# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
w    	= 2.0*np.pi*24E9;
Kvco    = 2*np.pi*(754.64E6);
AkPD	= 1.6
Ga1		= 0.01
tauf 	= 0.0
tauc	= 1.0/(2.0*np.pi*120E6);
v	 	= 32.0;
c		= 3E8
maxp 	= 70;
# print(globalFreqKrange(w, K, tauf, v, digital, maxp))
results = solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,w, Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['results']
print('0=',results[0])
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
#*******************************************************************************

#
# X= Krange();
Y=np.asarray(w/2.0*np.pi)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,w,  Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['tauGa1'];
Z=2.0*np.pi*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,w,  Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['ReKra']/np.asarray(w);
print(len(results[:,3]),len(results[:,2]),len(results[:,0]))
# print('X=',X,'\nY=',Y,'\nZ=',Z)
dpi_value = 300
cm = plt.cm.get_cmap('RdYlBu')
cdict = {
   'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
   'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
   'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
 }

colormap = plt1.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

# ''' IMPORTANT: since we add the perturbations additively, here we need to shift allPoints around the initial phases of the respective m-twist state, using phiMr '''
plt.figure(1)                # plot the mean of the order parameter over a period 2T
plt.clf()
ax = plt.subplot(111)
# ax.set_aspect('equal')
# ad=plt.scatter(X,Y,c=Z)#, Z, alpha=0.5, edgecolor='', cmap=cm, vmin=0, vmax=1)
ad=plt.scatter(results[:,3],results[:,2], c=results[:,0])
# plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(int(k) ,initPhiPrime0) )
# if N==3:
# plt.xlabel(r'$\phi_1^{\prime}$')
# plt.ylabel(r'$\phi_2^{\prime}$')
# elif N==2:
# plt.xlabel(r'$\phi_0^{\prime}$')
# plt.ylabel(r'$\phi_1^{\prime}$')
# if N==3 and topology != "square-open" and topology != "chain":
# plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
# plt.xlim([1.05*allPoints[:,0].min()+phiMr[d1], 1.05*allPoints[:,0].max()+phiMr[d1]])
# plt.ylim([1.05*allPoints[:,1].min()+phiMr[d2], 1.05*allPoints[:,1].max()+phiMr[d2]])
plt.colorbar(ad)
# plt.savefig('results/rot_red_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
# plt.savefig('results/rot_red_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
#
# plt.figure(3)
# plt.clf()
# ax = plt.subplot(1, 1, 1)
# ax.set_aspect('equal')
# tempresults = results[:,0].reshape((paramDiscretization, paramDiscretization))   #np.flipud()
# tempresults = np.transpose(tempresults)
# tempresults_ma = ma.masked_where(tempresults < 0, tempresults)    # Create masked array
# plt.imshow(tempresults_ma, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower', extent=(allPoints[:,0].min()+phiMr[d1], allPoints[:,0].max()+phiMr[d1], allPoints[:,1].min()+phiMr[d2], allPoints[:,1].max()+phiMr[d2]), vmin=0, vmax=1)
# plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(int(k) ,initPhiPrime0) )
# if N==3:
# plt.xlabel(r'$\phi_1^{\prime}$')
# plt.ylabel(r'$\phi_2^{\prime}$')
# elif N==2:
# plt.xlabel(r'$\phi_0^{\prime}$')
# plt.ylabel(r'$\phi_1^{\prime}$')
# if N==3 and topology != "square-open" and topology != "chain":
# plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
# plt.xlim([1.05*allPoints[:,0].min()+phiMr[d1], 1.05*allPoints[:,0].max()+phiMr[d1]])
# plt.ylim([1.05*allPoints[:,1].min()+phiMr[d2], 1.05*allPoints[:,1].max()+phiMr[d2]])
# plt.colorbar()
# plt.savefig('results/imsh_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
# plt.savefig('results/imsh_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)














# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*******************************************************************************
#
# [lineOmegStab] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc_0, v_0, digital, maxp, expansion)['tauStab'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc_0, v_0, digital, maxp, expansion)['OmegStab']/np.asarray(w),
# 						 '.',ms=1, color='blue',  label=r'Stable')
#
# [lineOmegUnst] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc_0, v_0, digital, maxp, expansion)['tauUnst'],
# 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc_0, v_0, digital, maxp, expansion)['OmegUnst']/np.asarray(w),
# 						 'o',ms=1, color='red', label=r'Unstable')
#
# fig1         = plt.figure()
# ax1          = fig1.add_subplot(111)
# # plot grid, labels, define intial values
# plt.title(r'Analysis of full expression of characteristic equation');
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\omega$', fontsize=18)
#
# # draw the initial plot
# [lineSigma] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc_0, v_0, digital, maxp, expansion)['Re'], linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda)$')
# [lineGamma] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf_0, Kvco, AkPD, Ga1_0, tauc_0, v_0, digital, maxp, expansion)['Im'], '.', ms=1, color='blue', label=r'$\gamma$=Im$(\lambda)$')
#


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
# v_slider_ax   = fig.add_axes([0.25, 0.10, 0.65, 0.02], facecolor=axis_color)
# v_slider      = Slider(v_slider_ax, r'$v$', 1, 512, valinit=v_0)
# # Draw another slider
# tauf_slider_ax  = fig.add_axes([0.25, 0.07, 0.65, 0.02], facecolor=axis_color)
# tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf_0)
# # Draw another slider
# Ga1_slider_ax = fig.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
# Ga1_slider    = Slider(Ga1_slider_ax, r'$G^{a,1}$', 0.01, 0.85, valinit=Ga1_0)
# # Draw another slider
# tauc_slider_ax  = fig.add_axes([0.25, 0.01, 0.65, 0.02], facecolor=axis_color)
# tauc_slider     = Slider(tauc_slider_ax, r'$\tau^c$', 0.005*1.0/(2.0*np.pi*120E6), 1.0/(2.0*np.pi*120E6), valinit=tauc_0)
#
# # define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 	global digital
#
# 	lineSigma.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion)['Re']/np.asarray(w));
# 	lineSigma.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
#
# 	lineGamma.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, digital,maxp, expansion)['Im']/np.asarray(w));
# 	lineGamma.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
#
# 	lineOmegStab.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val,Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, digital,maxp, expansion)['OmegStab']/np.asarray(w));
# 	lineOmegStab.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, digital,maxp, expansion)['tauStab']);
#
# 	lineOmegUnst.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, digital,maxp, expansion)['OmegUnst']/np.asarray(w));
# 	lineOmegUnst.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, digital,maxp, expansion)['tauUnst']);
#
#
# 	# lineSigma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 	# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Re']);
# 	# lineSigma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
# 	#
# 	# lineGamma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 	# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Im']);
# 	# lineGamma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
# 	#
# 	# lineNyq.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Im']);
# 	# lineNyq.set_xdata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Re']);
#
# 	# recompute the ax.dataLim
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
# ax1.legend()
# ax2.legend()
#ax5.legend()
plt.show()
