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
# from mpmath import findroot
# from sympy import I, limit
# from sympy.solvers import solveset
# from sympy.solvers import solve
# from sympy import Symbol, Function, nsolve
# from sympy import sin, cos, exp
# import mpmath
# mpmath.mp.dps = 25
import topology
from topology import eigenvalzeta
# choose digital vs analog
digital = False;


# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
w1    = 1.0*2.0*np.pi				#2.0*np.pi*24E9;
w2    = 1*2.0*np.pi
w3    = 1.0*2.0*np.pi
K    = 0.02*2.0*np.pi			#2.0*np.pi*250E6;
tauf = 0.0
tauc = 1.0/(0.05*w1);
v1	 = 1.0;
v2	 = 2.0;
v3	 = 1.0;
c	 = 3E8;
maxp = 67;

Nx1=2;
Nx2=2;
Nx3=3;
Ny=1;

# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
axcolor = 'lightgoldenrodyellow'
#*******************************************************************************
fig         = plt.figure()
ax          = fig.add_subplot(111)
if digital == True:
	plt.title(r'digital case for $\omega$=%.3f' %w1);
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'analog case for $\omega$=%.3f' %w1);
# adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\omega\tau^{cc}/2\pi (Hz*sec)$', fontsize=18)
plt.ylabel(r'$\Omega/\omega$', fontsize=18)
tauf_0  = tauf;
tauc_0  = tauc;
K_0     = K;
v_0	= v1;

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*******************************************************************************
[lineOmegStab1] = ax.plot(np.asarray(w1/(2.0*np.pi))*solveLinStab(globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg'], globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v1, digital, maxp, expansion)['tauStab'],
						 solveLinStab(globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg'], globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v1, digital, maxp, expansion)['OmegStab']/np.asarray(w1),
						 '.',ms=1, color='blue',  label=r'Stable')

[lineOmegUnst1] = ax.plot(np.asarray(w1/(2.0*np.pi))*solveLinStab(globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg'],globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v1, digital, maxp, expansion)['tauUnst'],
						 solveLinStab(globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg'],globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v1, digital, maxp, expansion)['OmegUnst']/np.asarray(w1),
						 'o',ms=1, color='red', label=r'Unstable')
# [lineOmeg] = ax.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], globalFreq(w, K, tauf, v, digital, maxp)['Omeg']/np.asarray(w), '.', ms=1, color='blue', label=r'$\Omega$')

fig1         = plt.figure()
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values

plt.title(r'solutions to char. eq. for different cases of the chain topology , where $\sigma=$Re$(\lambda)$ and $\gamma$=Im$(\lambda)$ vs $\tau$ for $K=$%.3f, $\tau^c=1.0/(0.05*\omega_1)=$%.3f and $\tau^f=$%.3f' %(K ,tauc,tauf));
plt.grid()
plt.xlabel(r'$\omega\tau^{cc}/2\pi$ (Hz*sec)', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$ ', fontsize=18)
plt.axhline(y=0, color='black', linestyle='-',linewidth=3)
# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# draw the initial plot
[lineSigmachain1] = ax1.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w1)*solveLinStab_topology_comparison(globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg'],
	globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v1, digital,maxp,  expansion, eigenvalzeta('chain',Nx1,Ny)['zeta'])['ReMax'],'-', linewidth=1.5, color='red', label=r'Re$(\lambda)$ chain $N_x=$%d, $N_y=$%d and $v=$%d'%(Nx1, Ny,v1))

[lineSigmachain2] = ax1.plot(np.asarray(w2/(2.0*np.pi))*globalFreq(w2, K, tauf, v2, digital, maxp/v2)['tau'], np.asarray((2.0*np.pi)/(w2))*solveLinStab_topology_comparison(globalFreq(w2, K, tauf, v2, digital, maxp/v2)['Omeg'],
	globalFreq(w2, K, tauf, v2, digital, maxp/v2)['tau'], tauf_0, K_0, tauc_0, v2, digital,maxp/v2,  expansion, eigenvalzeta('chain',Nx2,Ny)['zeta'])['ReMax'],'-.', linewidth=1.5, color='green', label=r'Re$(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx2, Ny,v2))

[lineSigmachain3] = ax1.plot(np.asarray(w3/(2.0*np.pi))*globalFreq(w3, K, tauf, v3, digital,maxp)['tau'], np.asarray((2.0*np.pi)/w3)*solveLinStab_topology_comparison(globalFreq(w3, K, tauf, v3, digital, maxp)['Omeg'],
	globalFreq(w3, K, tauf, v3, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v3, digital, maxp,  expansion, eigenvalzeta('chain',Nx3,Ny)['zeta'])['ReMax'],'--', linewidth=1.5, color='blue', label=r' Re $(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx3, Ny, v3))



# [lineSigmaSquare2d] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	# globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp  ,expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ReMax'], '-', linewidth=1.5, color='blue', label=r'Re$(\lambda)$ ring')


# [lineSigmarin] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
	# globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp  ,expansion, eigenvalzeta('ring',Nx,Ny)['zeta'])['ReMax'], '-', linewidth=1.5, color='green', label=r'Re$\lambda$ ring')





# [lineSigma1] = ax1.plot(globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,al)['Re'],	linewidth=2, color='red', label=r'Re$(\lambda)$')
# [lineGamma1] = ax1.plot(globalFreq(w, K, tauf, v, digital, maxp)['Omeg']*globalFreq(w, K, tauf, v, digital, maxp)['tau'], solveLinStab(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,al)['Im'],	linewidth=2, color='blue', label=r'Im$(\lambda)$')
#
#
#
#
fig2         = plt.figure()
ax2          = fig2.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'solutions to char. eq.,where $\gamma$=Im$(\lambda)$ vs $\tau$');
plt.grid()
plt.xlabel(r'$\omega\tau^{cc}/2\pi (Hz*sec)$', fontsize=18)
plt.ylabel(r'$\gamma/\omega$ ', fontsize=18)


[lineGammachain1] = ax2.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], np.asarray(1/w1)*solveLinStab_topology_comparison(globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg'],
	globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v1, digital,maxp  ,expansion, eigenvalzeta('chain',Nx1,Ny)['zeta'])['ImMax'], '.', ms=1.1 , color='red', label=r'Im$(\lambda)$ chain $N_x=$%d, $N_y=$%d and $v=$%d'%(Nx1, Ny,v1))


[lineGammachain2] = ax2.plot(np.asarray(w2/(2.0*np.pi))*globalFreq(w2, K, tauf, v2, digital, maxp/v2)['tau'], np.asarray(1/w2)*solveLinStab_topology_comparison(globalFreq(w2, K, tauf, v2, digital, maxp/v2)['Omeg'],
	globalFreq(w2, K, tauf, v2, digital,maxp/v2)['tau'], tauf_0, K_0, tauc_0, v2, digital,maxp/v2  ,expansion, eigenvalzeta('chain',Nx2,Ny)['zeta'])['ImMax'], '.', ms=1.1 , color='green', label=r'Im$(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx2, Ny,v2))


[lineGammachain3] = ax2.plot(np.asarray(w3/( 2.0*np.pi))*globalFreq(w3, K, tauf, v3, digital,  maxp)['tau'], np.asarray(1/w3)*solveLinStab_topology_comparison(globalFreq(w3, K, tauf, v3, digital,  maxp)['Omeg'],
	globalFreq(w3, K, tauf, v3, digital,  maxp)['tau'], tauf_0, K_0, tauc_0, v3, digital, maxp  ,expansion, eigenvalzeta('chain',Nx3,Ny)['zeta'])['ImMax'], '.', ms=1.1 , color='blue', label=r'Im$(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx3, Ny,v3))






fig3         = plt.figure()
ax3          = fig3.add_subplot(111)
# plot grid, labels, define intial values

plt.title(r'solutions to char. eq. for different cases of the chain topology , where $\sigma=$Re$(\lambda)$ and $\gamma$=Im$(\lambda)$ vs $\tau$ for $K=$%.3f, $\tau^c=1.0/(0.05*\omega_1)=$%.3f and $\tau^f=$%.3f' %(K ,tauc,tauf));
plt.grid()
plt.xlabel(r'$\Omega\tau^{cc}/2\pi (Hz*sec)$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$ ', fontsize=18)
plt.axhline(y=0, color='black', linestyle='-',linewidth=3)
# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# draw the initial plot
[lineSigmachain1] = ax3.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg']*globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w1)*solveLinStab_topology_comparison(globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg'],
	globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v1, digital,maxp,  expansion, eigenvalzeta('chain',Nx1,Ny)['zeta'])['ReMax'],'-', linewidth=1.5, color='red', label=r'Re$(\lambda)$ chain $N_x=$%d, $N_y=$%d and $v=$%d'%(Nx1, Ny,v1))

[lineSigmachain2] = ax3.plot(np.asarray(w2/(2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital, maxp/v2)['Omeg']*globalFreq(w2, K, tauf, v2, digital,maxp/v2)['tau'], np.asarray((2.0*np.pi)/(w2))*solveLinStab_topology_comparison(globalFreq(w2, K, tauf, v2, digital, maxp/v2)['Omeg'],
	globalFreq(w2, K, tauf, v2, digital, maxp/v2)['tau'], tauf_0, K_0, tauc_0, v2, digital,maxp/v2,  expansion, eigenvalzeta('chain',Nx2,Ny)['zeta'])['ReMax'],'-.', linewidth=1.5, color='green', label=r'Re$(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx2, Ny,v2))

[lineSigmachain3] = ax3.plot(np.asarray(w3/( 2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital,  maxp)['Omeg']*globalFreq(w3, K, tauf, v3, digital,  maxp)['tau'], np.asarray((2.0*np.pi)/w3)*solveLinStab_topology_comparison(globalFreq(w3, K, tauf, v3, digital,  maxp)['Omeg'],
	globalFreq(w3, K, tauf, v3, digital,  maxp)['tau'], tauf_0, K_0, tauc_0, v3, digital, maxp,  expansion, eigenvalzeta('chain',Nx3,Ny)['zeta'])['ReMax'],'--', linewidth=1.5, color='blue', label=r' Re $(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx3, Ny, v3))






fig4         = plt.figure()
ax4          = fig4.add_subplot(111)
# plot grid, labels, define intial values

plt.title(r'solutions to char. eq. for different cases of the chain topology , where $\sigma=$Re$(\lambda)$ and $\gamma$=Im$(\lambda)$ vs $\tau$ for $K=$%.3f, $\tau^c=1.0/(0.05*\omega_1)=$%.3f and $\tau^f=$%.3f' %(K ,tauc,tauf));
plt.grid()
plt.xlabel(r'$\Omega\tau^{cc}/2\pi (Hz*sec)$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$ ', fontsize=18)
plt.axhline(y=0, color='black', linestyle='-',linewidth=3)
# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# draw the initial plot
[lineSigmachain1] = ax4.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg']*globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w1)*solveLinStab_topology_comparison(globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg'],
	globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v1, digital,maxp,  expansion, eigenvalzeta('chain',Nx1,Ny)['zeta'])['ReMax'],'-', linewidth=1.5, color='red', label=r'Re$(\lambda)$ chain $N_x=$%d, $N_y=$%d and $v=$%d'%(Nx1, Ny,v1))

[lineSigmachain2] = ax4.plot(np.asarray(w2/(2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital, maxp/v2)['Omeg']*globalFreq(w2, K, tauf, v2, digital,maxp/v2)['tau'], np.asarray((2.0*np.pi)/(w2))*solveLinStab_topology_comparison(globalFreq(w2, K, tauf, v2, digital, maxp/v2)['Omeg'],
	globalFreq(w2, K, tauf, v2, digital, maxp/v2)['tau'], tauf_0, K_0, tauc_0, v2, digital,maxp/v2,  expansion, eigenvalzeta('chain',Nx2,Ny)['zeta'])['ReMax'],'-.', linewidth=1.5, color='green', label=r'Re$(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx2, Ny,v2))
#
# [lineSigmachain3] = ax3.plot(np.asarray(w3/( 2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital,  maxp)['Omeg']*globalFreq(w3, K, tauf, v3, digital,  maxp)['tau'], np.asarray((2.0*np.pi)/w3)*solveLinStab_topology_comparison(globalFreq(w3, K, tauf, v3, digital,  maxp)['Omeg'],
# 	globalFreq(w3, K, tauf, v3, digital,  maxp)['tau'], tauf_0, K_0, tauc_0, v3, digital, maxp,  expansion, eigenvalzeta('chain',Nx3,Ny)['zeta'])['ReMax'],'--', linewidth=1.5, color='blue', label=r' Re $(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx3, Ny, v3))




fig5         = plt.figure()
ax5          = fig5.add_subplot(111)
# plot grid, labels, define intial values

plt.title(r'solutions to char. eq. for different cases of the chain topology , where $\sigma=$Re$(\lambda)$ and $\gamma$=Im$(\lambda)$ vs $\tau$ for $K=$%.3f, $\tau^c=1.0/(0.05*\omega_1)=$%.3f and $\tau^f=$%.3f' %(K ,tauc,tauf));
plt.grid()
plt.xlabel(r'$\Omega\tau^{cc}/2\pi (Hz*sec)$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$ ', fontsize=18)
plt.axhline(y=0, color='black', linestyle='-',linewidth=3)
# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# draw the initial plot
[lineSigmachain1] = ax5.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg']*globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w1)*solveLinStab_topology_comparison(globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg'],
	globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v1, digital,maxp,  expansion, eigenvalzeta('chain',Nx1,Ny)['zeta'])['ReMax'],'-', linewidth=1.5, color='red', label=r'Re$(\lambda)$ chain $N_x=$%d, $N_y=$%d and $v=$%d'%(Nx1, Ny,v1))

# [lineSigmachain2] = ax5.plot(np.asarray(w2/(2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital, maxp/v2)['Omeg']*globalFreq(w2, K, tauf, v2, digital,maxp/v2)['tau'], np.asarray((2.0*np.pi)/(w2))*solveLinStab_topology_comparison(globalFreq(w2, K, tauf, v2, digital, maxp/v2)['Omeg'],
# 	globalFreq(w2, K, tauf, v2, digital, maxp/v2)['tau'], tauf_0, K_0, tauc_0, v2, digital,maxp/v2,  expansion, eigenvalzeta('chain',Nx2,Ny)['zeta'])['ReMax'],'-.', linewidth=1.5, color='green', label=r'Re$(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx2, Ny,v2))

[lineSigmachain3] = ax5.plot(np.asarray(w3/( 2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital,  maxp)['Omeg']*globalFreq(w3, K, tauf, v3, digital,  maxp)['tau'], np.asarray((2.0*np.pi)/w3)*solveLinStab_topology_comparison(globalFreq(w3, K, tauf, v3, digital,  maxp)['Omeg'],
	globalFreq(w3, K, tauf, v3, digital,  maxp)['tau'], tauf_0, K_0, tauc_0, v3, digital, maxp,  expansion, eigenvalzeta('chain',Nx3,Ny)['zeta'])['ReMax'],'--', linewidth=1.5, color='blue', label=r' Re $(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx3, Ny, v3))






fig6         = plt.figure()
ax6          = fig6.add_subplot(111)
# plot grid, labels, define intial values

plt.title(r'solutions to char. eq. for different cases of the chain topology , where $\sigma=$Re$(\lambda)$ and $\gamma$=Im$(\lambda)$ vs $\tau$ for $K=$%.3f, $\tau^c=1.0/(0.05*\omega_1)=$%.3f and $\tau^f=$%.3f' %(K ,tauc,tauf));
plt.grid()
plt.xlabel(r'$\Omega\tau^{cc}/2\pi (Hz*sec)$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$ ', fontsize=18)
plt.axhline(y=0, color='black', linestyle='-',linewidth=3)
# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# draw the initial plot
# [lineSigmachain1] = ax5.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg']*globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w1)*solveLinStab_topology_comparison(globalFreq(w1, K, tauf, v1, digital, maxp)['Omeg'],
# 	globalFreq(w1, K, tauf, v1, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v1, digital,maxp,  expansion, eigenvalzeta('chain',Nx1,Ny)['zeta'])['ReMax'],'-', linewidth=1.5, color='red', label=r'Re$(\lambda)$ chain $N_x=$%d, $N_y=$%d and $v=$%d'%(Nx1, Ny,v1))

[lineSigmachain2] = ax6.plot(np.asarray(w2/(2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital, maxp/v2)['Omeg']*globalFreq(w2, K, tauf, v2, digital,maxp/v2)['tau'], np.asarray((2.0*np.pi)/(w2))*solveLinStab_topology_comparison(globalFreq(w2, K, tauf, v2, digital, maxp/v2)['Omeg'],
	globalFreq(w2, K, tauf, v2, digital, maxp/v2)['tau'], tauf_0, K_0, tauc_0, v2, digital,maxp/v2,  expansion, eigenvalzeta('chain',Nx2,Ny)['zeta'])['ReMax'],'-.', linewidth=1.5, color='green', label=r'Re$(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx2, Ny,v2))

[lineSigmachain3] = ax6.plot(np.asarray(w3/( 2.0*np.pi))*globalFreq(w1, K, tauf, v1, digital,  maxp)['Omeg']*globalFreq(w3, K, tauf, v3, digital,  maxp)['tau'], np.asarray((2.0*np.pi)/w3)*solveLinStab_topology_comparison(globalFreq(w3, K, tauf, v3, digital,  maxp)['Omeg'],
	globalFreq(w3, K, tauf, v3, digital,  maxp)['tau'], tauf_0, K_0, tauc_0, v3, digital, maxp,  expansion, eigenvalzeta('chain',Nx3,Ny)['zeta'])['ReMax'],'--', linewidth=1.5, color='blue', label=r' Re $(\lambda)$ chain $N_x=$%d , $N_y=$%d and $v=$%d' %(Nx3, Ny, v3))



















#
#
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
# # define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 	global digital
# 	lineOmegStab.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, digital,maxp, expansion)['OmegStab']/np.asarray(w));
# 	lineOmegStab.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, digital,maxp, expansion)['tauStab']);
#
# 	lineOmegUnst.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, digital,maxp, expansion)['OmegUnst']/np.asarray(w));
# 	lineOmegUnst.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 						globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val,
# 						tauc_slider.val, v_slider.val, digital,maxp, expansion)['tauUnst']);
# 	lineSigmachain.set_ydata(np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 		globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp  ,expansion, eigenvalzeta('chain',Nx,Ny)['zeta'])['ReMax']);
# 	lineSigmachain.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
#
# 	lineGammachain.set_ydata(np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 		globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp  ,expansion, eigenvalzeta('chain',Nx,Ny)['zeta'])['ImMax']);
# 	lineGammachain.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
#
# 	# lineSigmaSquare2d.set_ydata(np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 	# 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp  ,expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ReMax']);
# 	# lineSigmaSquare2d.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
# 	#
# 	# lineGammaSquare2d.set_ydata(np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 	# 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp  ,expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ImMax']);
# 	# lineGammaSquare2d.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
# 	#
# 	#
# 	# lineSigmarin.set_ydata(np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 	# 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp  ,expansion, eigenvalzeta('ring',Nx,Ny)['zeta'])['ReMax']);
# 	# lineSigmarin.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
# 	#
# 	# lineGammarin.set_ydata(np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 	# 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp  ,expansion, eigenvalzeta('ring',Nx,Ny)['zeta'])['ImMax']);
# 	# lineGammarin.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
# 	#
# 	#
#
#
# 	# lineSigma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 	# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Re']);
# 	# lineSigma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
# 	#
# 	# lineGamma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
# 	# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Im']);
# 	# lineGamma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
#
# # 	# recompute the ax.dataLim
# 	ax.relim();
# 	ax1.relim();
# 	ax2.relim()
# 	ax3.relim();
# #ax5.relim()
# # 	# update ax.viewLim using the new dataLim
# 	ax.autoscale_view();
# 	ax1.autoscale_view();
# 	ax2.autoscale_view();
# 	ax3.autoscale_view();
# #ax5.autoscale_view()
# 	plt.draw()
# 	fig.canvas.draw_idle();
# 	fig1.canvas.draw_idle();
# 	# fig2.canvas.draw_idle();
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

# #
# def reset(event):
# 	K_slider.reset()
# 	v_slider.reset();
# 	tauf_slider.reset();
# 	wc_slider.reset();
#
# button.on_clicked(reset)
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
ax4.legend()
ax5.legend()
ax6.legend()
plt.show()
