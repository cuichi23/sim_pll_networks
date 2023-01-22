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

# choose phase or anti-phase synchroniazed states,
inphase1 = True;
inphase2 = False;
w    = 1.0*2.0*np.pi				#2.0*np.pi*24E9;
Kvco = 0.055*2.0*np.pi			#2.0*np.pi*250E6;
AkPD = 1.0
Ga1  = 1.0;
tauf = 0.0
tauc_0 	= 1.0/(2.0*np.pi*0.02);
tauc_1 	= 1.0/(2.0*np.pi*0.7)
tauc_2 	= 1.0/(2.0*np.pi*1.1)
order= 1.0
v	 = 1.0;
c	 = 3E8;
maxp = 25;

INV  = 0.0*np.pi;
zeta = -1.0


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
if digital == True:
	plt.title(r'Digital case for $\omega$=%2.4E $\tau^c=$%.3f' %(w/(2.0*np.pi),tauc_0));
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'Analog case for $\omega=2\cdot\pi$%2.1E $\tau^c=$%.3f' %(w/(2.0*np.pi),tauc_0));
# adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)
# 	plt.axhline(y=w, color='green', linestyle='-',linewidth=3)

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #****************************************************************************************************************************************************************************


[lineOmegStabIn] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegStab'],
				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegStab']/np.asarray(w),
				 'o',ms=7, color='blue',  label=r'Inphase Stable')

[lineOmegUnstIn] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegUnst'],
				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegUnst']/np.asarray(w),
				'o',ms=1 ,color='blue',  label=r'Inphase Stable')



[lineOmegStabAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab'],
				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab']/np.asarray(w),
				 'o',ms=7, color='red',  label=r'Antiphase Stable')

[lineOmegUnstAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst'],
				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst']/np.asarray(w),
				'o',ms=1, color='red', label=r'Antiphase Unstable')





fig1         = plt.figure()
ax1          = fig1.add_subplot(111)
if digital == True:
	plt.title(r'The analytic expression characteristic equation of the Antiphase synchronised states for $\omega$=%2.1E' %(w/(2.0*np.pi)));
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'The analytic expression characteristic equation of the Antiphase synchronised states for $\omega$=%2.1E' %(w/(2.0*np.pi)));
# adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)
# 	plt.axhline(y=w, color='green', linestyle='-',linewidth=3)

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #****************************************************************************************************************************************************************************

plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
[lineSigmaAnti1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'],np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],'-.',	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda) \tau^c=1/(2\pi\;0.02)$')
[lineGammaAnti1] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ImMax'],'-.',	linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda) \tau^c=1/(2\pi\;0.02)$')

[lineSigmaAnti2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_1, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],'.',	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;0.7)$')
[lineGammaAnti2] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_1, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ImMax'],'.',	linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda)\tau^c=1/(2\pi\;0.7)$')

[lineSigmaAnti3] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_2, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],'--',	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;1.1)$')
[lineGammaAnti3] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_2, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ImMax'],'--',	linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda)\tau^c=1/(2\pi\;1.1)$')





#
fig2         = plt.figure()
ax2          = fig2.add_subplot(111)
if digital == True:
	plt.title(r'The analytic expression characteristic equation of the Inphase synchronised states for $\omega$=%2.1E' %(w/(2.0*np.pi)));
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'The analytic expression characteristic equation of the Inphase synchronised states for $\omega$=%2.1E' %(w/(2.0*np.pi)));
# adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)
# 	plt.axhline(y=w, color='green', linestyle='-',linewidth=3)

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #****************************************************************************************************************************************************************************

plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
[lineSigmain1] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'],'-.',	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda) \tau^c=1/(2\pi\;0.02)$')
[lineGammain1] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ImMax'],'-.',	linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda) \tau^c=1/(2\pi\;0.02)$')

[lineSigmain2] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_1, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'],'.',	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;0.7)$')
[lineGammain2] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_1, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ImMax'],'.',	linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda)\tau^c=1/(2\pi\;0.7)$')

[lineSigmain3] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_2, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'],'--',	linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;1.1)$')
[lineGammain3] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_2, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ImMax'],'--',	linewidth=2, color='blue', label=r'$\gamma$=Im$(\lambda)\tau^c=1/(2\pi\;1.1)$')





fig3         = plt.figure()
ax3          = fig3.add_subplot(111)
if digital == True:
	plt.title(r'The analytic expression characteristic equation of the in-phase synchronised states for $\omega$=%2.1E' %(w/(2.0*np.pi)));
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'The analytic expression characteristic equation of the in-phase synchronised states for $\omega$=%2.1E' %(w/(2.0*np.pi)));
# adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)
# 	plt.axhline(y=w, color='green', linestyle='-',linewidth=3)

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #****************************************************************************************************************************************************************************

plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
[SigmaAnti] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'],np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],'-.',	linewidth=4, color='red', label=r'$\sigma$=Re$(\lambda) \tau^c=1/(2\pi\;0.02)$ (Antiphase)')
# [lineSigmaAnti2] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_1, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],'.',	ms=4, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;0.7)$')
#
# [lineSigmaAnti3] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_2, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],'--',	linewidth=4, color='red', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;1.1)$')

[Sigmain] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'],'-.',	linewidth=1, color='blue', label=r'$\sigma$=Re$(\lambda) \tau^c=1/(2\pi\;0.02) (in-phase)$')
#
# [lineSigmain2] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_1, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'],'.',	ms=1, color='blue', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;0.7)$')
#
# [lineSigmain3] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc_2, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'],'--',	linewidth=1, color='blue', label=r'$\sigma$=Re$(\lambda), \tau^c=1/(2\pi\;1.1)$')


# add two sliders for tweaking the parameters
# define an axes area and draw a slider in it
INV_slider_ax   = fig0.add_axes([0.25, 0.10, 0.65, 0.02], facecolor=axis_color)
INV_slider      = Slider(INV_slider_ax, r'$INV$', 0.0, np.pi, valinit=INV, valstep=np.pi)
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
def sliders_on_changed(val):
	global digital
	initial_guess()

	lineOmegStabIn.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
								globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'],
								tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['OmegStab']);


	lineOmegStabIn.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
					globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'],
					tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
					globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'],
					tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['OmegStab']);


	lineOmegUnstIn.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
								globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'],
								tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['OmegUnst']);

	lineOmegUnstIn.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
					globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'],
					tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
									globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'],
									tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['OmegUnst']);


	lineOmegStabAnti.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
								globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'],
								tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['OmegStab']);


	lineOmegStabAnti.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
					globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'],
					tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
					globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'],
					tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['OmegStab']);


	lineOmegUnstAnti.set_ydata(np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
								globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'],
								tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['OmegUnst']);
	lineOmegUnstAnti.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
					globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'],
					tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
									globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'],
									tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['OmegUnst']);

	lineSigmain1.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['ReMax']);
	lineSigmain1.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV_slider.val)['tau']);
	#
	# lineSigmain11.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
	# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['ReMax']);
	# lineSigmain11.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV_slider.val)['tau']);
	#

	lineGammain1.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['ImMax']);
	lineGammain1.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV_slider.val)['tau']);





	lineSigmain2.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_1, v, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['ReMax']);
	lineSigmain2.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV_slider.val)['tau']);

	lineGammain2.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_1, v, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['ImMax']);
	lineGammain2.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV_slider.val)['tau']);
	#


	lineSigmain3.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_2, v, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['ReMax']);
	lineSigmain3.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV_slider.val)['tau']);

	lineGammain3.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_2, v, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['ImMax']);
	lineGammain3.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV_slider.val)['tau']);
	#

	lineSigmaAnti1.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['ReMax']);
	lineSigmaAnti1.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV_slider.val)['tau']);

	#
	# lineSigmaAnti11.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
	# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['ReMax']);
	# lineSigmaAnti11.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV_slider.val)['tau']);
	#

	lineGammaAnti1.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['ImMax']);
	lineGammaAnti1.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV_slider.val)['tau']);


	lineSigmaAnti2.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_1, v, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['ReMax']);
	lineSigmaAnti2.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV_slider.val)['tau']);

	lineGammaAnti2.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_1, v, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['ImMax']);
	lineGammaAnti2.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV_slider.val)['tau']);
	#


	lineSigmaAnti3.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_2, v, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['ReMax']);
	lineSigmaAnti3.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV_slider.val)['tau']);

	lineGammaAnti3.set_ydata(np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_2, v, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['ImMax']);
	lineGammaAnti3.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV_slider.val)['tau']);
	#








	SigmaAnti.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['ReMax']);
	SigmaAnti.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV_slider.val)['tau']);

	Sigmain.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg'],
		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['tau'], tauf, Kvco, AkPD, Ga1, tauc_0, v, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['ReMax']);
	Sigmain.set_xdata(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV_slider.val)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV_slider.val)['tau']);

	#





	# recompute the ax.dataLim
	ax.relim();
	ax1.relim();
	ax2.relim()
	ax3.relim();
#ax5.relim()
# 	# update ax.viewLim using the new dataLim
	ax.autoscale_view();
	ax1.autoscale_view();
	ax2.autoscale_view();
	ax3.autoscale_view();
#ax5.autoscale_view()
	plt.draw()
	fig.canvas.draw_idle();
	fig1.canvas.draw_idle();
	fig2.canvas.draw_idle();
	fig3.canvas.draw_idle();
#fig5.canvas.draw_idle();
INV_slider.on_changed(sliders_on_changed)
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
ax2.legend()
ax3.legend()
plt.show()
