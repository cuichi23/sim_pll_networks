# Linear Stability of Global Frequency (LiStaGoF.py)


# This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
# There is a slider for the parameters of the system in order to see how the the stability and the system changes



# !/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStabbetain, globalFreq,linStabEq,solveLinStabbetaanti, globalFreqdigital, xrange,solveglobalFreqdigital
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
import time
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
digital1 = True;
# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;


inphase1 = True;
inphase2 = False;


# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
# w1  	= 2.0*np.pi*24.48E9
# w2		= 2.0*np.pi*24.60E9
#
# wmean   = (w1+w2)/2.0
# Dw		= w2-w1
# Kvco    = 2.0*np.pi*(757.64E6);
# AkPD	= 1.6
# Ga1     = 0.6
# order	= 2.0
#
# tauf    = 0.0																	# tauf = sum of all processing delays in the feedback
# tauc	= 1.0/(2.0*np.pi*1E6);
# v		= 512.0;
# c		= 3E8
# maxp 	= 3;

w1  	= 2.0*np.pi*(1.0+0.01)#24.48E9##
w2		= 2.0*np.pi*(1.0-0.01)#24.60E9##

wmean   = (w1+w2)/2.0
Dw		= w2-w1
Kvco    = 2.0*np.pi*0.1#(757.64E6);
AkPD	= 1.6
Ga1     = 0.6
order	= 2.0

tauf    = 0.0																	# tauf = sum of all processing delays in the feedback
tauc	= 1.0/(2.0*np.pi*0.1)#*1E6);
v		= 1.0;
c		= 3E8
maxp 	= 2.0*np.pi;
cfDig        = lambda x: sawtooth(x,width=0.5)



taudig	= xrange(maxp, tauf)

start_time = time.time();
# # ADD PLOTS
# ####################################################################################################################################################################################
# axis_color  = 'lightgoldenrodyellow'
# #*******************************************************************************
# fig         = plt.figure()
# ax          = fig.add_subplot(111)
# if digital == True:
# 	plt.title(r'digital case for $\omega$=%.3f' %wmean);
# 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# else:
# 	plt.title(r'analog case for $\omega$=%.3f' %wmean);
# # adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.12, bottom=0.25)
# # plot grid, labels, define intial values
# plt.grid()
# plt.xlabel(r'$\bar{\omega}\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$\Omega/\bar{\omega}$', fontsize=18)
#
# # v_0		= v;
# # c_0		= c;
#
# # # draw the initial plot
# # # the 'lineXXX' variables are used for modifying the lines later
#
# # #*******************************************************************************
#
# # print(phasediff(wmean,K, tauf,  digital, maxp, Dw, inphase))
# # choose phase or anti-phase synchroniazed states,
#
# [lineOmeginphase] 	  = ax.plot( (wmean/(2.0*np.pi))*globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['taubetain'],
# 							globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['Omegabetain']/wmean, '.',
# 							linewidth=1, color='blue', label=r'Inphase' )
#
# [lineOmegantiinphase] = ax.plot( (wmean/(2.0*np.pi))*globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['taubetaanti'],
# 							globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['Omegabetaanti']/wmean,'.',
# 							linewidth=1, color='red', label=r'Antiphase' )
#
#
# fig1         = plt.figure()
# ax1          = fig1.add_subplot(111)
# # plot grid, labels, define intial values
# plt.title(r'The phase difference vs $\tau$');
# plt.grid()
# plt.xlabel(r'$\bar{\omega}\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$\beta/v$', fontsize=18)
#
# [lineBetainphase] 	  = ax1.plot((wmean/(2.0*np.pi))*globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['taubetain'],
# 									np.asarray(1.0/v)*(globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['betain']), '.',
# 									markersize=10, color='blue', label='Inphase')
#
# [lineBetaantiinphase] = ax1.plot((wmean/(2.0*np.pi))*globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['taubetaanti'],
# 									np.asarray(1.0/v)*(globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['betaanti']), '.',
# 									color='red', label='Antiphase')
# # # #**********************************************************************************************************************************************
#
#
#
#
#
# fig2         = plt.figure()
# ax2          = fig2.add_subplot(111)
# # plot grid, labels, define intial values
# plt.title(r'The phase difference vs $\tau$');
# plt.grid()
# plt.xlabel(r'$\bar{\omega}\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$\beta_{\rm HF}$ mod $2\pi$', fontsize=18)
#
# [lineBetainphase] 	  = ax2.plot((wmean/(2.0*np.pi))*globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['taubetain'],
# 									np.mod((globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['betain']), 2.0*np.pi), '.',
# 									markersize=10, color='blue', label='Inphase')
#
# [lineBetaantiinphase] = ax2.plot((wmean/(2.0*np.pi))*globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['taubetaanti'],
# 									np.mod((globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['betaanti']), 2.0*np.pi), '.',
# 									color='red', label='Antiphase')
# # # #



fig3         = plt.figure()
ax3          = fig3.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'digital case for $\omega$=%.3f' %wmean);
plt.grid()
plt.xlabel(r'$\tau$', fontsize=18)
plt.ylabel(r'$\Omega/\bar{\omega}$', fontsize=18)

[lineDigitalOmegainphase] 	  = ax3.plot(taudig, (solveglobalFreqdigital(wmean,  Kvco, AkPD, Ga1, tauf, v,  digital1, maxp, Dw, inphase1)['Omega'])/wmean, '.',
									markersize=10, color='blue', label='Inphase')


# [CouplingFunction]			=ax3.plot(taudig,cfDig(taudig),'.',	markersize=10, color='red', label='Coupling Function')
#
#
# [lineDigitalOmegaintiinphase] = ax3.plot(taudig, (solveglobalFreqdigital(wmean,  Kvco, AkPD, Ga1, tauf, v,  digital1, maxp, Dw, inphase2)['Omega'])/wmean, '.',
# 									color='red', label='Antiphase')


# fig4         = plt.figure()
# ax4          = fig4.add_subplot(111)
# # plot grid, labels, define intial values
# plt.title(r'digital case for $\omega$=%.3f' %wmean);
# plt.grid()
# plt.xlabel(r'$\tau$', fontsize=18)
# plt.ylabel(r'$\beta$', fontsize=18)
#
# [lineDigitalOmegainphase] 	  = ax4.plot(taudig, (solveglobalFreqdigital(wmean,  Kvco, AkPD, Ga1, tauf, v,  digital1, maxp, Dw, inphase1)['Beta']), '.',
# 									markersize=10, color='blue', label='Inphase')
#

# [lineDigitalOmegaintiinphase] = ax3.plot(taudig, (solveglobalFreqdigital(wmean,  Kvco, AkPD, Ga1, tauf, v,  digital1, maxp, Dw, inphase2)['Beta']), '.',
# 									color='red', label='Antiphase')




print('tau=',taudig)
print('Omega=',solveglobalFreqdigital(wmean,  Kvco, AkPD, Ga1, tauf, v,  digital1, maxp, Dw, inphase1)['Omega'])
print('Beta=',solveglobalFreqdigital(wmean,  Kvco, AkPD, Ga1, tauf, v,  digital1, maxp, Dw, inphase1)['Beta'])


# # #





# # [lineOmegStab] = ax.plot(np.asarray(wmean/(2.0*np.pi))*solveLinStab(globalFreq(wmean, K, tauf, v, digital, maxp, Dw)['Omeg'], globalFreq(wmean, K, tauf, v, digital, maxp, Dw)['tau'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['tauStab'],
# 						 solveLinStab(globalFreq(wmean, K, tauf, v, digital, maxp, Dw)['Omeg'], globalFreq(wmean, K, tauf, v, digital, maxp, Dw)['tau'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['OmegStab']/np.asarray(wmean),
# 						 '.',ms=1, color='blue',  label=r'Stable')
#
# [lineOmegUnst] = ax.plot(np.asarray(wmean/(2.0*np.pi))*solveLinStab(globalFreq(wmean, K, tauf, v, digital, maxp, Dw)['Omeg'],globalFreq(wmean, K, tauf, v, digital, maxp, Dw)['tau'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['tauUnst'],
# 						 solveLinStab(globalFreq(wmean, K, tauf, v, digital, maxp, Dw)['Omeg'],globalFreq(wmean, K, tauf, v, digital, maxp, Dw)['tau'], tauf_0, K_0, tauc_0, v_0, digital, maxp, expansion)['OmegUnst']/np.asarray(wmean),
# 						 'o',ms=1, color='red', label=r'Unstable')
#
# ************************************************************************************************************
# fig3         = plt.figure()
# ax3          = fig3.add_subplot(111)
# # plot grid, labels, define intial values
# plt.title(r'The analytic expression characteristic equation $\sigma$=Re$(\lambda)$');
# plt.grid()
# plt.xlabel(r'$\bar{\omega}\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$2\pi\sigma_{in}/\bar{\omega}$, $2\pi\sigma_{anti}/\bar{\omega}$,', fontsize=18)
#
# # draw the initial plot
#
# [lineSigmainphase] = ax3.plot((wmean/(2.0*np.pi))*globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['taubetain'],
# 							(2.0*np.pi/wmean)*solveLinStabbetain(globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['Omegabetain'],
# 							globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['taubetain'], tauf, Kvco, AkPD, Ga1, tauc, Dw,
# 							globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['betain'],v, order, digital, maxp, expansion)['RebetainMax'],
# 							linewidth=2, color='blue',label='Inphase')
#
#
# [lineSigmaantiphase] = ax3.plot((wmean/(2.0*np.pi))*globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['taubetaanti'],
# 							(2.0*np.pi/wmean)*solveLinStabbetaanti(globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['Omegabetaanti'],
# 							globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['taubetaanti'], tauf,  Kvco, AkPD, Ga1, tauc, Dw,
# 							globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['betaanti'],v, order, digital, maxp, expansion)['RebetaantiMax'],
# 							linewidth=2, color='red', label=r'Antiphase')
#
#
#
# 	# ************************************************************************************************************
# fig4         = plt.figure()
# ax4          = fig4.add_subplot(111)
# # plot grid, labels, define intial values
# plt.title(r'The analytic expression characteristic equation $\sigma$=Re$(\lambda)$');
# plt.grid()
# plt.xlabel(r'$\bar{\omega}\tau/2\pi$', fontsize=18)
# plt.ylabel(r' $\gamma_{in}/\bar{\omega}$, $\gamma_{anti}/\bar{\omega}$', fontsize=18)
#
# # draw the initial plot
#
#
# [lineGammainphase] = ax4.plot((wmean/(2.0*np.pi))*globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['taubetain'],
# 							solveLinStabbetain(globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['Omegabetain'],
# 							globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['taubetain'], tauf, Kvco, AkPD, Ga1, tauc, Dw,
# 							globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase1)['betain'],v, order, digital, maxp,
# 							expansion)['ImbetainMax']/wmean,'.', linewidth=2, color='blue', label='Inphase')
#
# #
#
# [lineGammaantiinphase] = ax4.plot((wmean/(2.0*np.pi))*globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['taubetaanti'],#
# 							solveLinStabbetaanti(globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['Omegabetaanti'],
# 							globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['taubetaanti'], tauf,  Kvco, AkPD, Ga1, tauc, Dw,
# 							globalFreq(wmean,  Kvco, AkPD, Ga1, tauf,v,  digital, maxp, Dw, inphase2)['betaanti'],v, order, digital, maxp,
# 							expansion)['ImbetaantiMax']/wmean,'.', linewidth=2, color='red', label=r'Antiphase')

print('time for execution:', time.time() - start_time, ' seconds');



# [lineGammainphase] = ax2.plot(globalFreq(wmean, K, tauf, digital, maxp, Dw, inphase)['taubetain'], solveLinStabbetain(globalFreq(wmean, K, tauf, digital, maxp, Dw, inphase)['Omegabetain'],
	# globalFreq(wmean, K, tauf, digital, maxp, Dw, inphase)['taubetain'], tauf_0, K_0, tauc_0, Dw, globalFreq(wmean, K, tauf, digital, maxp, Dw, inphase)['betain'], digital, maxp, expansion)['Imbetain'],'.', linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda)$')
# [lineSigma1] = ax1.plot(globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,al)['Re'],	linewidth=2, color='red', label=r'Re$(\lambda)$')
# [lineGamma1] = ax1.plot(globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	# globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,al)['Im'],	linewidth=2, color='blue', label=r'Im$(\lambda)$')

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
# tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/wmean, valinit=tauf_0)#25*(2.0*np.pi/w)
# # Draw another slider
# K_slider_ax = fig.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
# K_slider    = Slider(K_slider_ax, r'$K$', 0.001*wmean, 1.5*wmean, valinit=K_0)
# # Draw another slider
# tauc_slider_ax  = fig.add_axes([0.25, 0.01, 0.65, 0.02], facecolor=axis_color)
# tauc_slider     = Slider(tauc_slider_ax, r'$\tau^c$', 0.05*(1.0/wmean), 10.0*(1.0/wmean), valinit=tauc_0)

# define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 	global digital
# 	lineOmeg.set_ydata(globalFreq(wmean, K_slider.val, tauf_slider.val,  digital, maxp, Dw)['Omeg']);
#
# 	# lineOmeg.set_xdata(globalFreq(wmean, K_slider.val, tauf_slider.val, digital, maxp,Dw)['tau']);
#
# 	# lineOmegStab.set_ydata(solveLinStab(globalFreq(wmean, K_slider.val, tauf_slider.val,digital, maxp, Dw)['Omeg'],
# 	# 					globalFreq(wmean, K_slider.val, tauf_slider.val, digital, maxp, Dw)['tau'], tauf_slider.val, K_slider.val,
# 	# 					tauc_slider.val, v_0,digital, maxp, expansion)['OmegStab']/np.asarray(wmean));
# 	# lineOmegStab.set_xdata(np.asarray(wmean/(2.0*np.pi))*solveLinStab(globalFreq(wmean, K_slider.val, tauf_slider.val, digital, maxp, Dw)['Omeg'],
# 	# 					globalFreq(wmean, K_slider.val, tauf_slider.val,v, digital, maxp, Dw)['tau'], tauf_slider.val, K_slider.val,
# 	# 					tauc_slider.val, Dw, digital,maxp, expansion)['tauStab']);
# 	#
# 	# lineOmegUnst.set_ydata(solveLinStab(globalFreq(wmean, K_slider.val, tauf_slider.val,v, digital, maxp, Dw)['Omeg'],
# 	# 					globalFreq(wmean, K_slider.val, tauf_slider.val,v, digital, maxp, Dw)['tau'], tauf_slider.val, K_slider.val,
# 	# 					tauc_slider.val, Dw, digital,maxp, expansion)['OmegUnst']/np.asarray(wmean));
# 	# lineOmegUnst.set_xdata(np.asarray(wmean/(2.0*np.pi))*solveLinStab(globalFreq(wmean, K_slider.val, tauf_slider.val, digital, maxp, Dw)['Omeg'],
# 	# 					globalFreq(wmean, K_slider.val, tauf_slider.val,v, digital, maxp, Dw)['tau'], tauf_slider.val, K_slider.val,
# 	# 					tauc_slider.val, Dw, digital,maxp, expansion)['tauUnst']);
# 	# # lineOmegStab.set_ydata(solveLinStab(globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['Omeg'],
# 	# 					globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['tau'], tauf, K_slider.val,
# 	# 					tauf_slider.val, v, digital,maxp, expansion)['OmegStab']/np.asarray(wmean));
# 	# lineOmegStab.set_xdata(np.asarray(wmean/(2.0*np.pi))*solveLinStab(globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['Omeg'],
# 	# 					globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['tau'], tauf, K_slider.val,
# 	# 					tauf_slider.val, v, digital,maxp, expansion)['tauStab']);
# 	# lineOmegUnst.set_ydata(solveLinStab(globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['Omeg'],
# 	# 					globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['tau'], tauf, K_slider.val,
# 	# 					tauf_slider.val, v, digital,maxp, expansion)['OmegUnst']/np.asarray(wmean));
# 	# lineOmegUnst.set_xdata(np.asarray(wmean/(2.0*np.pi))*solveLinStab(globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['Omeg'],
# 	# 					globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['tau'], tauf, K_slider.val,
# 	# 					tauf_slider.val, v, digital,maxp, expansion)['tauUnst']);
# 	lineSigma.set_ydata(solveLinStab(globalFreq(wmean, K_slider.val, tauf_slider.val, digital, maxp, Dw)['Omeg'],
# 		globalFreq(wmean, K_slider.val, tauf_slider.val,  digital, maxp, Dw)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, Dw, digital,maxp, expansion)['Re']);
# 	lineSigma.set_xdata(globalFreq(wmean, K_slider.val, tauf_slider.val,  digital, maxp, Dw)['tau']);
#
# 	# lineGamma.set_ydata(solveLinStab(globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['Omeg'],
# 	# 	globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, Dw, digital,maxp, expansion)['Im']);
# 	# lineGamma.set_xdata(globalFreq(wmean, K_slider.val, tauf_slider.val, v, digital, maxp, Dw)['tau']);
#
# 	# lineSigma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['Omeg'],
# 	# globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v, digital,al)['Re']);
# 	# lineSigma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['tau']);
# 	#
# 	# lineGamma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['Omeg'],
# 	# globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v, digital,al)['Im']);
# 	# lineGamma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['tau']);
# 	#
# 	# lineNyq.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v, digital,al)['Im']);
# 	# lineNyq.set_xdata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v, digital,al)['Re']);
#
# # 	# recompute the ax.dataLim
# 	ax.relim();
# 	ax1.relim();
# 	# ax2.relim()
# # 	ax3.relim();
# #ax5.relim()
# # 	# update ax.viewLim using the new dataLim
# 	ax.autoscale_view();
# 	ax1.autoscale_view();
# 	# ax2.autoscale_view();
# # 	ax3.autoscale_view();
# #ax5.autoscale_view()
# 	plt.draw()
# 	fig.canvas.draw_idle();
# 	fig1.canvas.draw_idle();
# 	# fig2.canvas.draw_idle();
# # 	fig3.canvas.draw_idle();
# #fig5.canvas.draw_idle();
# # v_slider.on_changed(sliders_on_changed)
# tauf_slider.on_changed(sliders_on_changed)
# K_slider.on_changed(sliders_on_changed)
# tauc_slider.on_changed(sliders_on_changed)

# add a button for resetting the parameters
# PLLtype_button_ax = fig.add_axes([0.8, 0.9, 0.1, 0.04])
# PLLtype_button = Button(PLLtype_button_ax, 'dig/ana', color=axis_color, hovercolor='0.975')
# def PLLtype_button_on_clicked(mouse_event):
# 	global digital
# 	digital = not digital;
# 	print('state digital:', digital)
# 	if digital == True:
# 		ax.set_title(r'digital case for $\omega$=%.3f' %wmean);
# 		ax1.set_title(r'digital case for $\omega$=%.3f, linear stability' %wmean);
# 		ax2.set_title(r'digital case for $\omega$=%.3f, linear stability' %w);
# 		ax3.set_title(r'digital case for $\omega$=%.3f, Nyquist' %w);
# 		#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# 	else:
# 		ax.set_title(r'analog case for $\omega$=%.3f' %wmean);
# 		ax1.set_title(r'analog case for $\omega$=%.3f, linear stability' %wmean);
# 		ax2.set_title(r'analog case for $\omega$=%.3f, linear stability' %w);
# 		ax3.set_title(r'analog case for $\omega$=%.3f, Nyquist' %w);
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

# ax.legend()
# ax1.legend()
# ax2.legend()
ax3.legend()
plt.show()
