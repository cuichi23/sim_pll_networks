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
import datetime
now = datetime.datetime.now()
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
digital = True;


# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
w   	= 2.0*np.pi#*24E9            #1.0*2.0*np.pi				#2.0*np.pi*24E9;
Kvco    = 2.0*np.pi*0.1            #*2.4E9 #0.60*2.0*np.pi			#2.0*np.pi*250E6;
AkPD	= 1.0
Ga1		= 1.0
tauf 	= 0.0
tauc 	= 1.0/(2.0*np.pi*0.0001);
v	 	= 512.0;
c	 	= 3E8;
maxp1 	= 1.7e0
maxp2	= 4.99e2
# maxp 	= 1e7
inphase	= True
order 	= 1.0
INV		= 0.0*np.pi

sync_state='in-phase'
Nx=8;
Ny=8;
zetasvalues =  eigenvalzeta('square-open',Nx,Ny)['zeta']#[-1]
color 		= ['blue', 'orange']
# omegatau1= np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp1, sync_state, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp1, sync_state, INV)['tau']
omegatau2= np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp2, sync_state, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp2, sync_state, INV)['tau']
# state1 =globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp1, sync_state, INV)
state2 =globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp2, sync_state, INV)
# sigma1 = np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp1, sync_state, INV)['Omeg'],
    # globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp1, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp1, digital, sync_state, INV, expansion, zetasvalues)['ReMax']
#
sigma2 =np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp2, sync_state, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp2, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp2, digital, sync_state, INV, expansion, zetasvalues)['ReMax']

gamma2 =np.asarray(1.0/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp2, sync_state, INV)['Omeg'],
	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp2, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp2, digital, sync_state, INV, expansion, zetasvalues)['ImMax']

# xmin1 = omegatau1[0]
# xmax1 = omegatau1[-1]
xmin2 = omegatau2[0]
xmax2 = omegatau2[-1]
coupfun='sin'


''' All plots in latex mode '''
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')

plt.rcParams['agg.path.chunksize'] = 10000

''' STYLEPACKS '''
titlefont = {
  'family' : 'serif',
  'color'  : 'black',
  'weight' : 'normal',
  'size'   : 9,
  }

labelfont = {
  'family' : 'sans-serif',
  # 'color'  : 'black',
  'weight' : 'normal',
  'style'  : 'normal',
  'size'   : 36,
  }

annotationfont = {
  'family' : 'monospace',
  'color'  : (0, 0.27, 0.08),
  'weight' : 'normal',
  'size'   : 14,
  }

# plot parameter
axisLabel = 12;
titleLabel= 10;
dpi_val   = 150;
figwidth  = 20;
figheight = 10;

#

# Save data

# f = open('results/data.txt', "w")
# f.write("# x y z \n")        # column names
# np.savetxt(omegatau, np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(synchronstate['Omeg'],
# 	synchronstate['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion, zetasvalues)['ReMax'])

# np.savetxt('results/beginning.txt', np.array([omegatau1, sigma1]).T, fmt="%.16f")


# np.savetxt('results/data_end.txt', np.array([omegatau2, sigma2]).T, fmt="%.16f")



#PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
axcolor = 'lightgoldenrodyellow'
#*******************************************************************************
# fig         = plt.figure()
# ax          = fig.add_subplot(111)
# if digital == True:
# 	plt.title(r'digital case for $\omega$=%.3f' %w);
# 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# else:
# 	plt.title(r'analog case for $\omega$=%.3f' %w);
# # adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.12, bottom=0.25)
# # plot grid, labels, define intial values
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$\Omega/\omega$', fontsize=18)
#
#
# # # draw the initial plot
# # # the 'lineXXX' variables are used for modifying the lines later
#
# # #*******************************************************************************
# [lineOmegStab] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'],  tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion)['tauStab'],
# 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'],  tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion)['OmegStab']/np.asarray(w),
# 						 '.',ms=1, color='blue',  label=r'Stable')
#
# [lineOmegUnst] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'],  tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion)['tauUnst'],
# 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'],  tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion)['OmegUnst']/np.asarray(w),
# 						 'o',ms=1, color='red', label=r'Unstable')
# # [lineOmeg] = ax.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg']/np.asarray(w), '.', ms=1, color='blue', label=r'$\Omega$')

# fig1         = plt.figure()
# ax1          = fig1.add_subplot(111)
# # plot grid, labels, define intial values
#
# plt.title(r'solutions to char. eq. for $N_x=$%d, $N_y=$%d, $N=N_x\,N_y=$%d,where $\sigma=$Re$(\lambda)$ and $\gamma$=Im$(\lambda)$ vs $\tau$' %(Nx, Ny, Nx*Ny));
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$2\pi\sigma/\omega$ ', fontsize=18)
# # resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# # button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
#
# # draw the initial plot
# # [lineSigma2dsquareopen] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
# # 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion, eigenvalzeta('square open',Nx,Ny)['zeta'])['ReMax'],'-', linewidth=1.5, color='red', label=r'Re$(\lambda)$ all2all')
# #
# # [lineGamma2dsquareopen] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
# # 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion, eigenvalzeta('square open',Nx,Ny)['zeta'])['ImMax'], '.', ms=1.1 , color='blue', label=r'Im$(\lambda)$ all2all')
#
#
# [lineSigma2dsquareopen] = ax1.plot(np.asarray(w/(2.0*np.pi))*synchronstate['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(synchronstate['Omeg'],
# 	synchronstate['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion, zetasvalues)['ReMax'],'-', linewidth=1.5, color='red', label=r'Re$(\lambda)$')
#
# [lineGamma2dsquareopen] = ax1.plot(np.asarray(w/(2.0*np.pi))*synchronstate['tau'], np.asarray(1.0/w)*solveLinStab_topology_comparison(synchronstate['Omeg'],
# 	synchronstate['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion, zetasvalues)['ImMax'], '.', ms=1.1 , color='blue', label=r'Im$(\lambda)$')


# [lineSigmaSquare2d] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
	# globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion, eigenvalzeta('square-periodic',Nx,Ny)['zeta'])['ReMax'], '-', linewidth=1.5, color='blue', label=r'Re$(\lambda)$ ring')


# [lineSigmarin] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
	# globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf_0, K_0, tauc_0, v_0, digital,maxp  ,expansion, eigenvalzeta('ring',Nx,Ny)['zeta'])['ReMax'], '-', linewidth=1.5, color='green', label=r'Re$\lambda$ ring')



#
#
# [lineSigma1] = ax1.plot(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf_0, K_0, tauc_0, v_0, digital,al)['Re'],	linewidth=2, color='red', label=r'Re$(\lambda)$')
# [lineGamma1] = ax1.plot(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf_0, K_0, tauc_0, v_0, digital,al)['Im'],	linewidth=2, color='blue', label=r'Im$(\lambda)$')
#

#

fig         = plt.figure()
ax2          = fig.add_subplot(111)
# plot grid, labels, define intial values
# plt.title(r'2-D square lattice with open boundary conditions and $\zeta=-1,2, 1/4$, $N_x=$%d, $N_y=$%d, $N=N_x\,N_y=$%d' %(Nx, Ny, Nx*Ny));
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$',  fontsize=60,labelpad=-5)
plt.ylabel(r' $\frac{ 2\pi\sigma}{\omega}$,  $\frac{\gamma}{\omega}$ ', rotation=90,fontsize=85, labelpad=30)
ax2.set_xlim(xmin2, xmax2)
ax2.tick_params(axis='both', which='major', labelsize=35, pad=1)

#
# for i in range(len(zetasvalues)):
    # [lineSigma2dsquareopen] = ax2.plot(omegatau1, np.asarray((2.0*np.pi)/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp1, sync_state, INV)['Omeg'],
        # globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp1, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp1, digital, sync_state, INV, expansion, zetasvalues[i])['ReMax'],'-', linewidth=4.5, color=color[i], label=r'Re$(\lambda)$ and $\zeta=$%0.2f' %zetasvalues[i])

# ax3          = fig.add_subplot(212)
# # plot grid, labels, define intial values
# # plt.title(r'2-D square lattice with open boundary conditions and $\zeta=-1,2, 1/4$, $N_x=$%d, $N_y=$%d, $N=N_x\,N_y=$%d' %(Nx, Ny, Nx*Ny));
# plt.grid()
# plt.xlabel(r'$\Omega\tau/2\pi$',  fontsize=60,labelpad=-5)
# plt.ylabel(r' $\frac{ 2\pi\sigma}{\omega}$,  $\frac{\gamma}{\omega}$ ', rotation=90,fontsize=85, labelpad=30)
# ax3.set_xlim(xmin2, xmax2)
# ax3.tick_params(axis='both', which='major', labelsize=35, pad=1)
#
# for i in range(len(zetasvalues)):
[lineSigma2dsquareopen] = ax2.plot(omegatau2, sigma2,'-', linewidth=4.5, color=color[0], label=r'$2\pi$Re$(\lambda)\omega$')
[lineGamma2dsquareopen] = ax2.plot(omegatau2, gamma2,'-', linewidth=4.5, color=color[1], label=r'Im$(\lambda)/\omega$')

#
#     #
	# [lineGamma2dsquareopen] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], np.asarray(1/w)*solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['Omeg'],
		# globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, maxp, digital, sync_state, INV, expansion, zetasvalues[i])['ImMax'], '--', linewidth=4.5 , color=color[i], label=r'Im$(\lambda)$ and $\zeta=$%0.2f' %(zetasvalues[i]))
#
#
# fig.set_size_inches(figwidth, figheight)
# plt.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
#
# # if maxp > 1e3:
# if digital == True:
# 	plt.savefig('plts/digital_2D_square_open_sigma_vs_Omega_tau_large%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plts/digital_2D_square_open_sigma_vs_Omega_tau_large%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plts/digital_2D_square_open_sigma_vs_Omega_tau_large%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
# 	if coupfun=='sin':
# 		plt.savefig('plts/analog_sin_2D_square_open_sigma_vs_Omega_tau_large%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plts/analog_sin_2D_square_open_sigma_vs_Omega_tau_large%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plts/analog_sin_2D_square_open_sigma_vs_Omega_tau_large%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfun=='cos':
# 		plt.savefig('plts/analog_cos_2D_square_open_sigma_vs_Omega_tau_large%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plts/analog_cos_2D_square_open_sigma_vs_Omega_tau_large%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plts/analog_cos_2D_square_open_sigma_vs_Omega_tau_large%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)


# ax.legend()
# ax1.legend()
ax2.legend()
# ax3.legend()
plt.show()
