#Linear Stability of Global Frequency (LiStaGloF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq, linStabEq, xrange, alpha, K, coupfunction
import entrainment_of_SLL_lib
from entrainment_of_SLL_lib import xrangetau,  alphaent, solveLinStabEnt
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

# choose digital vs analog
digital = True;
# coupfun= 'cos'
# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion = False;
# choose phase or anti-phase synchroniazed states,

inphase1 = True;
inphase2 = False;

# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider

w    		= (1.0)*2.0*np.pi				#2.0*np.pi*24E9;
# Kvco 		= 0.04*2.0*np.pi			#2.0*np.pi*250E6; for the first plot where the value of Kvco is not on the title of the file, Kvco = 0.05*2.0*np.pi
Kvco 		= 2*0.0500991*2.0*np.pi
AkPD 		= 1.0
Ga1  		= 1.0;
tauf 		= 0.0
tauc 		= 1.0/(2.0*np.pi*16.5E-6);
order		= 1.0
v	 		= 1.0;
c	 		= 3E8;
maxp 		= 22.68; #   (w+Kvco)*tau											# this is a good approximation!  (w+Kvco)*tau
wR        	= 2.0*np.pi*1.0;
slope1		= 1
slope2		= 2
maxtau 		= 1e-6;
K1   		= K(Kvco, AkPD, Ga1)
# xrangetau
INV  		= 10.0*np.pi;
noINV		= 0.0
zeta 		= -1.0

# print(coupfunction(coupfun))


OmegInphase     = globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']
tauInphase      = globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']
delayvecInphase = solveLinStab(OmegInphase, tauInphase, w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)

OmegAntiphase     = globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']
tauAntiphase      = globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']
delayvecAntiphase = solveLinStab(OmegAntiphase, tauAntiphase, w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'


####################################################################################################################################################################################
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
  'size'   : 39,
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
figwidth  = 6;
figheight = 5;





xmin = 0.0
xmax = 5.0
#
# vbar= [1, 4, 16, 128]
# colorbar=['blue','green','orange','purple']
fig         = plt.figure(figsize=(figwidth,figheight))
ax          = fig.add_subplot(111)

#
# if digital == True:
# 	plt.title(r'digital case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# else:
# 	plt.title(r'analog case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
ax.set_xlim(xmin, xmax)
plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60,labelpad=-5)
plt.ylabel(r'$\frac{\Omega}{\omega}$',rotation=0, fontsize=85, labelpad=30)
ax.tick_params(axis='both', which='major', labelsize=55, pad=1)
# ax.set_xlim(0.0, 5.0)
# plt.xlim(-0.1, max(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']))
# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later

# # #*************************************************************************************************************************************************************************
[lineOmegStabIn] = ax.plot(np.asarray(w/(2.0*np.pi))*delayvecInphase['tauStab'], delayvecInphase['OmegStab']/np.asarray(w),
						 '+',ms=7, color='red',  label=r'in-phase stable')# INV On')
#
[lineOmegUnstIn] = ax.plot(np.asarray(w/(2.0*np.pi))*delayvecInphase['tauUnst'], delayvecInphase['OmegUnst']/np.asarray(w),
						 '.',ms=1, color='grey')#, label=r'in-phase unstable')


[lineOmegStabAnti] = ax.plot(np.asarray(w/(2.0*np.pi))*delayvecAntiphase['tauStab'], delayvecAntiphase['OmegStab']/np.asarray(w),
						 '+',ms=7, color='blue',  label=r'antiphase stable')# INV On')
#
[lineOmegUnstAnti] = ax.plot(np.asarray(w/(2.0*np.pi))*delayvecAntiphase['tauUnst'], delayvecAntiphase['OmegUnst']/np.asarray(w),
						 '.',ms=1, color='grey')#, label=r'in-phase unstable')


fig.set_size_inches(20,10)
ax.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
plt.legend(fontsize=25)

if digital == True:
    # print('hallo')
    plt.savefig('plts/digital_Omega_vs_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/digital_Omega_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/digital_Omega_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
    if coupfunction(coupfun)=='sin':
        # print('hallo')
        plt.savefig('plts/analog_sin_Omega_vs_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_sin_Omega_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_sin_Omega_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

    elif coupfunction(coupfun)=='cos':
        # print('hallo')
        plt.savefig('plts/analog_cos_Omega_vs_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_cos_Omega_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_cos_Omega_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    elif coupfunction(coupfun)=='-cos':
        # print('hallo')
        plt.savefig('plts/analog_-cos_Omega_vs_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_-cos_Omega_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_-cos_Omega_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)






#
#
#
#
# fig         = plt.figure(figsize=(figwidth,figheight))
# ax          = fig.add_subplot(111)
#
# #
# # if digital == True:
# # 	plt.title(r'digital case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# # 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# # else:
# # 	plt.title(r'analog case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# # adjust the subplots region to leave some space for the sliders and buttons
# # fig.subplots_adjust(left=0.12, bottom=0.25)
# # plot grid, labels, define intial values
# plt.grid()
# ax.set_xlim(xmin, xmax)
# plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60,labelpad=-5)
# plt.ylabel(r'$\beta$',rotation=0, fontsize=85, labelpad=30)
# ax.tick_params(axis='both', which='major', labelsize=35, pad=1)
# # ax.set_xlim(0.0, 5.0)
# # plt.xlim(-0.1, max(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']))
# # # draw the initial plot
# # # the 'lineXXX' variables are used for modifying the lines later
#
# # # #*************************************************************************************************************************************************************************
# [lineOmegStabIn] = ax.plot(np.asarray(w/(2.0*np.pi))*delayvecInphase['tauStab'], delayvecInphase['betaStab']/np.asarray(w),
# 						 '+',ms=7, color='red',  label=r'in-phase stable')# INV On')
# #
# [lineOmegUnstIn] = ax.plot(np.asarray(w/(2.0*np.pi))*delayvecInphase['tauUnst'], delayvecInphase['betaUnst']/np.asarray(w),
# 						 '.',ms=1, color='grey')#, label=r'in-phase unstable')
#
#
# [lineOmegStabAnti] = ax.plot(np.asarray(w/(2.0*np.pi))*delayvecAntiphase['tauStab'], delayvecAntiphase['betaStab']/np.asarray(w),
# 						 '+',ms=7, color='blue',  label=r'antiphase stable')# INV On')
# #
# [lineOmegUnstAnti] = ax.plot(np.asarray(w/(2.0*np.pi))*delayvecAntiphase['tauUnst'], delayvecAntiphase['betaUnst']/np.asarray(w),
# 						 '.',ms=1, color='grey')#, label=r'in-phase unstable')
#
#
# fig.set_size_inches(20,10)
# ax.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# plt.legend(fontsize=25)
#
# if digital == True:
#     print('hallo')
#     plt.savefig('plts/digital_beta_vs_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#     plt.savefig('plts/digital_beta_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#     plt.savefig('plts/digital_beta_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
#     if coupfunction(coupfun)=='sin':
#         print('hallo')
#         plt.savefig('plts/analog_sin_beta_vs_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#         plt.savefig('plts/analog_sin_beta_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#         plt.savefig('plts/analog_sin_beta_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
#     elif coupfunction(coupfun)=='cos':
#         print('hallo')
#         plt.savefig('plts/analog_cos_beta_vs_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#         plt.savefig('plts/analog_cos_beta_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#         plt.savefig('plts/analog_cos_beta_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#     elif coupfunction(coupfun)=='-cos':
#         print('hallo')
#         plt.savefig('plts/analog_negcos_beta_vs_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#         plt.savefig('plts/analog_negcos_beta_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#         plt.savefig('plts/analog_negcos_beta_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)





#
# #*************************************************************************************************************************************************************************
#
# #
# fig0         = plt.figure(figsize=(figwidth,figheight))
# ax0         = fig0.add_subplot(111)
#
# #
# # if digital == True:
# # 	plt.title(r'digital case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# # 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# # else:
# # 	plt.title(r'analog case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# # adjust the subplots region to leave some space for the sliders and buttons
# # fig.subplots_adjust(left=0.12, bottom=0.25)
# # plot grid, labels, define intial values
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60,labelpad=-5)
# plt.ylabel(r'$\frac{\Omega}{\omega}$',rotation=0, fontsize=85, labelpad=30)
# ax0.tick_params(axis='both', which='major', labelsize=35, pad=1)
# ax0.set_xlim(0.0, 2.5)
#
# [lineOmegStabAnti3] = ax0.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['Omeg'],
# 							globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, noINV, zeta)['tauStab'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, noINV)['Omeg'],
# 						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, noINV, zeta)['OmegStab']/np.asarray(w),
# 						 'o',mfc='none', ms=7, color='blue',  label=r'anti-phase')# stable INV Off')
#
# [lineOmegUnsAnti3] = ax0.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['Omeg'],
# 							globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, noINV, zeta)['tauUnst'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, noINV)['Omeg'],
# 						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, noINV, zeta)['OmegUnst']/np.asarray(w),
# 						 'o', mfc='none', ms=1, color='grey')#  label=r'anti-phase stable INV Off')
#

#
# #*************************************************************************************************************************************************************************
#
#
# fig1         = plt.figure(figsize=(figwidth,figheight))
# ax1          = fig1.add_subplot(111)
#
# # plot grid, labels, define intial values
# # if digital == True:
# # 	plt.title(r'digital case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# # 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# # else:
# # 	plt.title(r'analog case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60,labelpad=-5)
# plt.ylabel(r'$\frac{2\pi\sigma}{\omega}$',rotation=0, fontsize=85, labelpad=30)
# ax1.tick_params(axis='both', which='major', labelsize=35, pad=1)
# # plt.xlim(-0.1, max(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']))
# ax1.set_xlim(0.0, 5.0)
#
# [lineSigmaIn] = ax1.plot(np.asarray(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ReMax'],
# 	 '-',linewidth=4, color='blue', label=r'in-phase')
#
#
# [lineSigmaAnti] = ax1.plot(np.asarray(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['ReMax'],
# 	 ':',linewidth=4, color='blue', label=r'anti-phase')

#
#
#
# # [lineSigmaEnt1] = ax1.plot(np.asarray(w/(2.0*np.pi))*xrangetau(maxtau), np.asarray(2.0*np.pi/w)*solveLinStabEnt(wR, w, tauf, Kvco, AkPD, Ga1, INV, v, tauc, maxtau, digital,slope1)['ReMax'],
# # 	 	'-.',linewidth=4 , color='orange', label=r'solution 1 (entrainment)')
# #
# #
# # [lineSigmaEnt2] = ax1.plot(np.asarray(w/(2.0*np.pi))*xrangetau(maxtau), np.asarray(2.0*np.pi/w)*solveLinStabEnt(wR, w, tauf, Kvco, AkPD, Ga1, INV, v, tauc, maxtau, digital,slope2)['ReMax'],
# # 	 	':',linewidth=4, color='purple', label=r'solution 2 (entrainment)')
# #
# # #
# #
# # # # [lineGammaIn] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# # # # 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ImMax'],
# # # # 	'.', ms=4, color='blue', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
# # #
# # # #
# # # # [lineSigmaAnti] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# # # # 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['ReMax'],
# # # # 	 linewidth=1, color='red')#, label=r'Antiphase Stable$\sigma$=Re$(\lambda)$')
# # # # # #
# # # # [lineGammaAnti] = ax1.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], np.asarray(1.0/globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'])*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# # # # 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['ImMax'],
# # # # 	 '.',ms=1, color='blue', label=r'Antiphase Unstable $\gamma$=Im$(\lambda)$')
# # # #
# # #
# fig1.set_size_inches(20,10)
# ax1.legend(bbox_to_anchor=(0.3415,0.35), prop=labelfont)
# if INV==0.0:
#     plt.savefig('plts/two_mut_sigma_vs_tau_noINV_K_%.2f_v_%.0E_%d_%d_%d.pdf' %(K(Kvco, AkPD, Ga1)/(2.0*np.pi),v,now.year, now.month, now.day), dpi=150, bbox_inches=0)
#     plt.savefig('plts/two_mut_sigma_vs_tau_noINV_K_%.2f_v_%.0E_%d_%d_%d.png' %(K(Kvco, AkPD, Ga1)/(2.0*np.pi),v,now.year, now.month, now.day), dpi=150, bbox_inches=0)
# elif INV==np.pi:
#     plt.savefig('plts/two_mut_sigma_vs_tau_INV_K_%.2f_v_%.0E_%d_%d_%d.pdf' %(K(Kvco, AkPD, Ga1)/(2.0*np.pi),v, now.year, now.month, now.day), dpi=150, bbox_inches=0)
#     plt.savefig('plts/two_mut_sigma_vs_tau_INV_K_%.2f_v_%.0E_%d_%d_%d.png' %(K(Kvco, AkPD, Ga1)/(2.0*np.pi),v, now.year, now.month, now.day), dpi=150, bbox_inches=0)
# plt.show()
#
#
# #
# #
# #
# fig2         = plt.figure(figsize=(figwidth,figheight))
# ax2          = fig2.add_subplot(111)
#
# # plot grid, labels, define intial values
# # if digital == True:
# # 	plt.title(r'digital case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# # 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# # else:
# # 	plt.title(r'analog case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60,labelpad=-5)
# plt.ylabel(r'$\frac{\gamma}{\omega}$',rotation=0, fontsize=85, labelpad=40)
# ax2.tick_params(axis='both', which='major', labelsize=35, pad=1)
# # plt.xlim(-0.1, max(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']))
# # ax2.set_xlim(0.0, 5.0)
#
# [lineSigmaIn1] = ax2.plot(np.asarray(w/(2.0*np.pi))*tauInphase, np.asarray(2.0*np.pi/w)*delayvecInphase['ReMax'],
# 	 linewidth=4, color='red')#, label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
# #
# [lineGammaIn1] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ImMax'],
# 	'.', ms=4, color='blue', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
#
# [lineSigmaAnti] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['ReMax'],
# 	 linewidth=1, color='red', label=r'Antiphase Stable$\sigma$=Re$(\lambda)$')


# [lineGammaIn] = ax2.plot(np.asarray(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['Omeg']/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
	# globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ImMax'],
	# 'd', ms=2, color='blue', label=r'in-phase')
# #
# [lineGammaAnti] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['ImMax'],
# 	 '+', mfc='none',ms=9, color='blue', label=r'anti-phase')
#
# #
#
# [lineGammaEnt1] = ax2.plot(np.asarray(w/(2.0*np.pi))*xrangetau(maxtau), np.asarray(1.0/w)*solveLinStabEnt(wR, w, tauf, Kvco, AkPD, Ga1, INV, v, tauc, maxtau, digital,slope1)['ImMax'],
# 	 	'--',linewidth=4 , color='orange', label=r'solution 1 (entrainment)')
#
#
# [lineGammaEnt2] = ax2.plot(np.asarray(w/(2.0*np.pi))*xrangetau(maxtau), np.asarray(1.0/w)*solveLinStabEnt(wR, w, tauf, Kvco, AkPD, Ga1, INV, v, tauc, maxtau, digital,slope2)['ImMax'],
# 	 	'-.',linewidth=4, color='orange', label=r'solution 2 (entrainment)')
#
# fig2.set_size_inches(20,10)
# ax2.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# # plt.legend(fontsize=25)
# if digital == True:
#     plt.savefig('plts/digital_gamma_vs_tau1%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#     plt.savefig('plts/digital_gamma_vs_tau1%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
#     if coupfunction(coupfun)=='sin'
#         plt.savefig('plts/analog_sin_gamma_vs_tau1%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#         plt.savefig('plts/analog_sin_gamma_vs_tau1%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
#     if coupfunction(coupfun)=='cos'
#         plt.savefig('plts/analog_cos_gamma_vs_tau1%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#         plt.savefig('plts/analog_cos_gamma_vs_tau1%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

# fig5         = plt.figure(figsize=(figwidth,figheight))
# ax5          = fig5.add_subplot(111)
# #ax  = fig.add_subplot(111)
# # plt.title('Entrainment of PLL', fontsize=36)
# # if digital == True:
# # 	plt.title(r'digital case for $\omega$=%.3f' %w);
# # 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# # else:
# # 	plt.title(r'analog case for $\omega$=%.3f' %w);
# # plot grid, labels, define intial values
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=60,labelpad=-5)
# plt.ylabel(r'$\beta$ in [rad]', rotation=90,fontsize=85, labelpad=30)
# ax5.tick_params(axis='both', which='major', labelsize=35, pad=1)
# ax5.set_xlim(0.0, 5.0)
# # draw the initial plot
# # print(len(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# # 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'],
# #     w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauStab']))
#
#
#
# #for i in range(len(delayvecInphase['tauStab'])):
#
# 	#if delayvecInphase['ReMax'][i]<0:
# 		# print(i)
# [lineBeta] = ax5.plot(delayvecInphase['tauStab'], delayvecInphase['betaStab'], 'o', ms=7, color='blue')#, label=r'Inphase')
# 	#elif delayvecInphase['ReMax'][i]>0:
# 		# print('i',i)
# 	#	[lineBeta] = ax5.plot(delayvecInphase['tauUnst'], delayvecInphase['betaUnst'], 'o', ms=2, color='blue')#, label=r'Inphase')
#
# [lineBeta] = ax5.plot(delayvecInphase['tauUnst'], delayvecInphase['betaUnst'], '-', color='blue', label=r'in-phase')
#
#
#
# #for i in range(len(delayvecAntiphase['tauStab'])):
# 	# print(i)
# 	#if delayvecAntiphase['ReMax'][i]<0:
# [lineBeta] = ax5.plot(delayvecAntiphase['tauStab'], delayvecAntiphase['betaStab'], 'o', ms=7, color='red')#, label=r'Antiphase')
# 	#elif delayvecAntiphase['ReMax'][i]>0:
# 	#	[lineBeta] = ax5.plot(delayvecAntiphase['tauUnst'], delayvecAntiphase['betaUnst'], 'o', ms=2, color='red')
#
# [lineBeta] = ax5.plot(delayvecAntiphase['tauUnst'], delayvecAntiphase['betaUnst'], '-', color='red', label=r'anti-phase')
#
# fig5.set_size_inches(20,10)
# plt.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# # [lineBeta2] = ax3.plot(xrange(maxtau), np.mod(beta(wR, w, xrange(maxtau), tauf, K, INV, v, digital, slope1)+np.pi,2*np.pi)-np.pi,
# # 	'-',linewidth=2, markersize=0.75, color='blue', label=r'$\beta$ Unstable')
# # #*******************************************************************************
# #
#
#
# if digital == True:
# 	plt.savefig('plts/digital_betamut_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plts/digital_betamut_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
# 	if coupfunction(coupfun)=='sin':
# 		plt.savefig('plts/analog_sin_betamut_vs_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plts/analog_sin_betamut_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plts/analog_sin_betamut_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfunction(coupfun)=='cos':
# 		plt.savefig('plts/analog_cos_betamut_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plts/analog_cos_betamut_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# plt.savefig('plts/betamut_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# plt.savefig('plts/betamut_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# plt.show()


#
# fig4         = plt.figure(figsize=(figwidth,figheight))
# ax4          = fig4.add_subplot(111)
#
# # plot grid, labels, define intial values
# # if digital == True:
# # 	plt.title(r'digital case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# # 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# # else:
# # 	plt.title(r'analog case for $\omega=2\pi$, INV=%.4f, v=%d'%(INV,v),fontsize=36)#\cdot$%2.4E Hz' %(w/(2.0*np.pi)));
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60,labelpad=3)
# plt.ylabel(r'$\alpha$',rotation=0, fontsize=75, labelpad=30)
# ax4.tick_params(axis='both', which='major', labelsize=35, pad=1)
# # plt.xlim(-0.1, max(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']))
#
# ax4.set_xlim(0.0, 5.0)
# [alphaIn] = ax4.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
#     np.asarray(2.0*np.pi/w)*alpha(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase1, INV),
#      '-',linewidth=4, color='blue', label=r'in-phase (mutual)')
#
# [alphaAnti] = ax4.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'],
#     np.asarray(2.0*np.pi/w)*alpha(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase2, INV),
#      ':',linewidth=4, color='blue', label=r'anti-phase (mutual)')
#
# [linealpha1] = ax4.plot(xrangetau(maxtau), alphaent(wR, w, xrangetau(maxtau), tauf, Kvco, AkPD, Ga1, INV, v, tauc, digital, slope1),
# 	'-.',linewidth=4 , color='orange', label=r'solution 1 (entrainment)')
# [linealpha2] = ax4.plot(xrangetau(maxtau), alphaent(wR, w, xrangetau(maxtau), tauf, Kvco, AkPD, Ga1, INV, v, tauc, digital, slope2),
# 	':',linewidth=4, color='purple', label=r'solution 2 (entrainment)')
# #
# #
# fig4.set_size_inches(20,10)
# ax4.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# # plt.legend(fontsize=25)
# plt.savefig('plts/alpha_vs_tau1%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# plt.savefig('plts/alpha_vs_tau1%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# # [lineSigmaAnti] = ax1.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# # 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['ReMax'],
# # 	 linewidth=4, color='blue', label=r'Antiphase Stable$\sigma$=Re$(\lambda)$')
#
#
# #
#
#
# fig3         = plt.figure(figsize=(figwidth,figheight))
# ax3          = fig3.add_subplot(111)
# plt.rc('legend',fontsize=17)
# # plot grid, labels, define intial values
# plt.title(r'The analytic expression characteristic equation',fontsize=30);
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=26)
# plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\Omega$', fontsize=26)
#
# plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
# ax3.tick_params(axis='both', which='major', labelsize=20)
# for i in range(len(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta) ) ):
#
# 	[lineSigmaIn3] = ax3.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['Re'][i],'+',
# 		markersize=4, color='red')#, label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
# 	[lineGammaIn] = ax3.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['Im'][i],
# 		'd', ms=4, color='black')#, label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
# [lineSigmaIn3] = ax3.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['Re'][0],'+',
# 	markersize=4, color='red', label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
# [lineGammaIn] = ax3.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['Im'][0],
# 	'd', ms=4, color='black', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
# [lineGammaIn] = ax3.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ImMax'],
# 	'.', ms=4, color='blue', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
# fig4         = plt.figure(figsize=(figwidth,figheight))
# ax4          = fig4.add_subplot(111)
# plt.rc('legend',fontsize=17)
# # plot grid, labels, define intial values
# plt.title(r'The analytic expression characteristic equation',fontsize=30);
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=26)
# plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\Omega$', fontsize=26)
#
# plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
# ax4.tick_params(axis='both', which='major', labelsize=20)
# for j in range(len(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta) ) ):
#
# 	[lineSigmaIn4] = ax4.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase2, expansion, INV, zeta)['Re'][j],'+',
# 		markersize=4, color='green')#, label=r'Antiphase Stable $\sigma$=Re$(\lambda)$')
# 	[lineGammaIn4] = ax4.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['Im'][j],
# 		'd', ms=4, color='black')#, label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
# [lineSigmaIn4] = ax4.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase2, expansion, INV, zeta)['Re'][0],'+',
# 	markersize=4, color='green', label=r'Antiphase Stable $\sigma$=Re$(\lambda)$')
# [lineGammaIn2] = ax4.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['Im'][0],
# 	'd', ms=4, color='black', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
#
# for i in range(len(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta) ) ):
#
# 	[lineSigmaIn03] = ax4.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['Re'][i],'+',
# 		markersize=4, color='red')#, label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
#
# [lineSigmaIn03] = ax4.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['Re'][0],'+',
# 	markersize=4, color='red', label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
#
#
#
# [lineGammaIn2] = ax4.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ImMax'],
# 	'.', ms=4, color='blue', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
#


# add two sliders for tweaking the parameters
# define an axes area and draw a slider in it
# v_slider_ax   = fig0.add_axes([0.25, 0.67, 0.65, 0.1], facecolor=axis_color)
# v_slider      = Slider(v_slider_ax, r'$v$', 1, 16, valinit=v, valstep=1)
# # Draw another slider
# tauf_slider_ax  = fig0.add_axes([0.25, 0.45, 0.65, 0.1], facecolor=axis_color)
# tauf_slider     = Slider(tauf_slider_ax, r'$\tau^f$', 0.0, 25.0/w, valinit=tauf, valfmt='%1.2E')
# # Draw another slider
# Ga1_slider_ax = fig0.add_axes([0.25, 0.23, 0.65, 0.1], facecolor=axis_color)
# Ga1_slider    = Slider(Ga1_slider_ax, r'$G^{a,1}$', 0.0001*Ga1, 2.0*Ga1, valinit=Ga1, valstep=0.5)
# # Draw another slider
# tauc_slider_ax  = fig0.add_axes([0.25, 0.04, 0.65, 0.02], facecolor=axis_color)
# tauc_slider     = Slider(tauc_slider_ax, r'$\tau^c$', 0.005*tauc, 3.0*tauc, valinit=tauc, valfmt='%1.2E')
#
# INV_slider_ax  = fig0.add_axes([0.25, 0.01, 0.65, 0.02], facecolor=axis_color)
# INV_slider     = Slider(INV_slider_ax, r'$INV$', 0.0, np.pi, valinit=INV, valstep=np.pi)
#
#
# # define an action for modifying the line when any slider's value changes
# def sliders_on_changed(val):
# 	global digital
#
# 	lineSigmaIn.set_ydata(np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['ReMax']);
#
# 	lineSigmaIn.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau']);
#
# 	lineGammaIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['ImMax']/np.asarray(w));
# 	lineGammaIn.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau']);
#
# 	lineSigmaAnti.set_ydata(solveLinStab(np.asarray(2.0*np.pi/w)*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['ReMax']);
# 	lineSigmaAnti.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
#
# 	lineGammaAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val, tauc_slider.val, v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['ImMax']/np.asarray(w));
# 	lineGammaAnti.set_xdata(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau']);
#
#
#
#
# 	lineOmegStabIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val,Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital, maxp, inphase1, expansion, INV_slider.val, zeta)['OmegStab']/np.asarray(w));
# 	lineOmegStabIn.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['tauStab']);
#
# 	lineOmegUnstIn.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['OmegUnst']/np.asarray(w));
# 	lineOmegUnstIn.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1,INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase1, expansion, INV_slider.val, zeta)['tauUnst']);
#
#
#
#
# 	lineOmegStabAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val,Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital, maxp, inphase2, expansion, INV_slider.val, zeta)['OmegStab']/np.asarray(w));
# 	lineOmegStabAnti.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['tauStab']);
#
# 	lineOmegUnstAnti.set_ydata(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['OmegUnst']/np.asarray(w));
# 	lineOmegUnstAnti.set_xdata(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2,INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['tauUnst']);
#
# 	ax1.set_xlim(-0.1, max(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['tauStab'])-0.01*max(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 											globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 											tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['tauStab']))
#
# 	ax.set_xlim(-0.1, max(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 						tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['tauStab'])-0.01*max(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase2, INV_slider.val)['Omeg'],
# 											globalFreq(w, Kvco, AkPD, Ga1_slider.val, tauf_slider.val, v_slider.val, digital, maxp, inphase1, INV_slider.val)['tau'], tauf_slider.val, Kvco, AkPD, Ga1_slider.val,
# 											tauc_slider.val, v_slider.val, order, digital,maxp, inphase2, expansion, INV_slider.val, zeta)['tauStab']))
#
#
# 	# ax.set_title(r'analog case for $\omega=2\pi$, INV=%.4f, v=%d' %(INV_slider.val,v_slider.val),fontsize=36)
# 	# ax1.set_title(r'analog case for $\omega=2\pi$, INV=%.4f, v=%d' %(INV_slider.val,v_slider.val),fontsize=36)
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
# INV_slider.on_changed(sliders_on_changed)
# add a button for resetting the parameters
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
# # color_radios.on_clicked(color_radios_on_clicked)
# ax.legend()
# ax1.legend()
# ax2.legend()
# ax3.legend()
# ax4.legend()
# ax5.legend()
plt.show()
