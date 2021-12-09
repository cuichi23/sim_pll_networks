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
w   	= 1.0*2.0*np.pi				#2.0*np.pi*24E9;
Kvco    = 0.4*2.0*np.pi			#2.0*np.pi*250E6;
AkPD	= 1.0
Ga1		= 1.0
tauf 	= 0.0
tauc 	= 1.0/(0.05*2.0*np.pi);
v	 	= 1.0;
c	 	= 3E8;
maxp 	= 55;
inphase	= True
order 	= 1.0
INV		=0.0*np.pi

sync_state=inphase
Nx=8;
Ny=8;

# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
axcolor = 'lightgoldenrodyellow'
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
# tauf_0  = tauf*w/(2.0*np.pi);
# tauc_0  = tauc*(1.0/w);
# K_0     = K/w;
# v_0		= v;
# c_0		= c;

# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# #*******************************************************************************


[lineOmegStab] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'],
 							tauf, Kvco, AkPD, Ga1, tauc, v,  order, maxp, digital, sync_state, INV, expansion)['tauStab'],

						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'],
						 tauf, Kvco, AkPD, Ga1, tauc, v, order,maxp, digital, sync_state, INV, expansion)['OmegStab']/np.asarray(w),

						 '.',ms=1, color='blue',  label=r'Stable')

[lineOmegUnst] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'],
 							tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion)['tauUnst'],
						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'],
						 tauf, Kvco, AkPD, Ga1, tauc, v,  order, maxp, digital, sync_state, INV, expansion)['OmegUnst']/np.asarray(w),
						 'o',ms=1, color='red', label=r'Unstable')
# # [lineOmeg] = ax.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], globalFreq(w, K, tauf, v, digital, maxp)['Omeg']/np.asarray(w), '.', ms=1, color='blue', label=r'$\Omega$')

fig1         = plt.figure()
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values

plt.title(r'solutions to char. eq. for $N_x=$%d, $N_y=$%d, $N=N_x\,N_y=$%d,where $\sigma=$Re$(\lambda)$ and $\gamma$=Im$(\lambda)$ vs $\tau$' %(Nx, Ny, Nx*Ny));
plt.grid()
plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$2\pi\sigma/\omega$ ', fontsize=18)
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# # draw the initial plot
[lineSigma2DSquareperiodic] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ReMax'],'-', linewidth=1.5, color='red', label=r'Re$(\lambda)$ 2DSquareperiodic')

[lineGamma2DSquareperiodic] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ImMax'], '.', ms=1.1 , color='blue', label=r'Im$(\lambda)$ 2DSquareperiodic')



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
plt.title(r'solutions to char. eq. for $N_x=$%d, $N_y=$%d, $N=N_x\,N_y=$%d,where $\gamma$=Im$(\lambda)$ vs $\Omega\tau$' %(Nx, Ny, Nx*Ny));
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
plt.ylabel(r'$\gamma/\omega$ ', fontsize=18)


[lineSigma2DSquareperiodic] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'],
			np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ReMax'],'-', linewidth=1.5, color='red', label=r'Re$(\lambda)$ 2DSquareperiodic')



[lineGamma2DSquareperiodic] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'],
		np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
			globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion,eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ImMax'], '.', ms=1.1 , color='blue', label=r'Im$(\lambda)$ 2DSquareperiodic')

#
# [lineGamma2DSquareperiodic] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp  ,expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ImMax'], '-', linewidth=1.5, color='red', label=r'Im$(\lambda)$ 2DSquareperiodic')

# [lineGammaSquare2d] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp  ,expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ImMax'], '-', linewidth=1.5, color='blue', label=r'Im$(\lambda)$ 2d square')
#
# [lineGammarin] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp  ,expansion, eigenvalzeta('ring',Nx,Ny)['zeta'])['ImMax'], '-', linewidth=1.5, color='green', label=r'Im$(\lambda)$ ring')

# # add two sliders for tweaking the parameters
# # define an axes area and draw a slider in it
#
#
# fig3         = plt.figure()
# ax3          = fig3.add_subplot(111)
# # plot grid, labels, define intial values
# plt.grid()
# plt.xlabel(r'$d/\lambda=\tau \omega/2\pi$', fontsize=18)
# plt.ylabel(r'Re$(\lambda)$, Im$(\lambda)$', fontsize=18)
#
# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later
# # [lineBeta12ax1] = ax1.plot(xrange(K), np.real(solveLinStab(xrange(K), tau_0, tauf_0, K_0, tauc, digital)),
# # 	linewidth=2, color='red', label=r'Re$(\lambda)$')
# # [lineBetaR1ax1] = ax1.plot(xrange(K), np.imag(solveLinStab(xrange(K), tau_0, tauf_0, K_0, tauc, digital)),
# # 	linewidth=2, color='blue', label=r'Im$(\lambda)$')
# [lineSigma2DSquareperiodic] = ax3.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp,  expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ReMax'],	linewidth=2, color='red', label=r'Re$(\lambda)$')
# [lineSigma2DSquareperiodic] = ax3.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, K, tauf, v, digital, maxp)['tau'], np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K, tauf, v, digital, maxp)['Omeg'],
# 	globalFreq(w, K, tauf, v, digital, maxp)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp  ,expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ImMax'],	'.', ms=1.1 , color='blue', label=r'Im$(\lambda)$')
#


	# lineSigmaSquare2d.set_ydata(np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	# 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp  ,expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ReMax']);
	# lineSigmaSquare2d.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	# lineGammaSquare2d.set_ydata(np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	# 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp  ,expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ImMax']);
	# lineGammaSquare2d.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	#
	# lineSigmarin.set_ydata(np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	# 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp  ,expansion, eigenvalzeta('ring',Nx,Ny)['zeta'])['ReMax']);
	# lineSigmarin.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	# lineGammarin.set_ydata(np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	# 	globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,maxp  ,expansion, eigenvalzeta('ring',Nx,Ny)['zeta'])['ImMax']);
	# lineGammarin.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	#


	# lineSigma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Re']);
	# lineSigma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	# lineGamma1.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'],
	# globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Im']);
	# lineGamma1.set_xdata(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg']*globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau']);
	#
	# lineNyq.set_ydata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Im']);
	# lineNyq.set_xdata(solveLinStab(globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['Omeg'], globalFreq(w, K_slider.val, tauf_slider.val, v_slider.val, digital, maxp)['tau'], tauf_slider.val, K_slider.val, tauc_slider.val, v_slider.val, digital,al)['Re']);

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
ax2.legend()
# ax3.legend()
plt.show()
