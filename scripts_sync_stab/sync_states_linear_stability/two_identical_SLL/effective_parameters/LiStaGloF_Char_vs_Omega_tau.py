#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq,linStabEq,initial_guess, coupfunction, equationSigma, equationGamma, alpha_gen
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
digital = False;
coupfun= 'cos'
# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;

# choose phase or anti-phase synchroniazed states,
inphase1 = True;
inphase2 = False
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider

w    		= 1.0*2.0*np.pi				#2.0*np.pi*24E9;
# Kvco 		= 0.04*2.0*np.pi			#2.0*np.pi*250E6; for the first plot where the value of Kvco is not on the title of the file, Kvco = 0.05*2.0*np.pi
Kvco 		= 0.04*2.0*np.pi
AkPD 		= 1.0
Ga1  		= 1.0;
tauf 		= 0.0
tauc 		= 1.0/(2.0*np.pi*0.2);
order		= 1.0
v	 		= 1.0;
c	 		= 3E8;
maxp 		= 20.68; #   (w+Kvco)*tau											# this is a good approximation!  (w+Kvco)*tau
wR        	= 2.0*np.pi*1.0;
slope1		= 1
slope2		= 2
maxtau 		= 1e0;
# K1   		= K(Kvco, AkPD, Ga1)
# xrangetau
INV  		= 0.0*np.pi;
zeta = -1.0
# coupfun='sin'
syncstateIn   = globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)
syncstateAnti = globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)
StabIn		  = solveLinStab(syncstateIn['Omeg'], syncstateIn['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)
StabAnti	  = solveLinStab(syncstateAnti['Omeg'], syncstateAnti['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)


xmin = 0.0
xmax = 3.0



sigmaLambertW=[];
tausigmaW=[];
OmegsigmaW=[]
# for i in range(len(syncstate['Omeg'])):
# 	gamma=np.max(equationGamma(syncstate['tau'][i], alpha_gen(syncstate['Omeg'][i], syncstate['tau'][i], 0.0, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, INV),
# 	1.0/tauc, zeta) )
# 	if np.isnan( gamma )== False:
# 		sigma= np.max(equationSigma(syncstate['tau'][i], alpha_gen(syncstate['Omeg'][i], syncstate['tau'][i], 0.0, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, INV),
# 				1.0/tauc, zeta, gamma ))
#
# 		if np.isnan( sigma )== False:
# 			sigmaLambertW.append(sigma )
# 			tausigmaW.append( syncstate['tau'][i])
# 			OmegsigmaW.append(syncstate['Omeg'][i])

# print(sigmaLambertW)
# ADD PLOTS
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


####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'

# print(globalFreqINV(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'])
# fig0    = plt.figure(num=0, figsize=(figwidth, figheight), facecolor='w', edgecolor='k')


#*******************************************************************************
fig         = plt.figure(figsize=(figwidth,figheight))
ax          = fig.add_subplot(111)
#*******************************************************************************
if digital == True:
	plt.title(r'digital case for $\omega$=%.3f' %w, fontsize=30);
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'analog case for $\omega$=%.3f' %w , fontsize=30);
# adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$',  fontsize=60,labelpad=-5)
plt.ylabel(r'$\Omega/\omega$', rotation=90,fontsize=85, labelpad=30)

ax.tick_params(axis='both', which='major', labelsize=35, pad=1)
ax.set_xlim(xmin, xmax)

# #*************************************************************************************************************************************************************************
# print('maxtau=', (globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'][-1]))
# print('Omega=',globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<=60000.5)][-1]/(2.0*np.pi))
# print('tau=',  globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'][np.where(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<=60000.5)][-1])



[lineOmegStabIn] = ax.plot(np.asarray(1.0/(2.0*np.pi))*StabIn['tauStab']*StabIn['OmegStab'], StabIn['OmegStab']/np.asarray(w),
				 'o',ms=3, color='blue',  label=r'in-phase Stable')

[lineOmegUnstIn] = ax.plot(np.asarray(1.0/(2.0*np.pi))*StabIn['tauUnst']*StabIn['OmegUnst'], StabIn['OmegUnst']/np.asarray(w),
				'.',ms=3, color='grey', label=r'in-phase Unstable')

[lineOmegStabAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*StabAnti['tauStab']*StabAnti['OmegStab'],StabAnti['OmegStab']/np.asarray(w),
				 'o',ms=3, color='red',  label=r'anti-phase Stable')

[lineOmegUnstAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*StabAnti['tauUnst']*StabAnti['OmegUnst'], StabAnti['OmegUnst']/np.asarray(w),
				'.',ms=3, color='grey', label=r'anti-phase Unstable')



fig.set_size_inches(20,10)
plt.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# [lineBeta2] = ax3.plot(xrange(maxtau), np.mod(beta(wR, w, xrange(maxtau), tauf, K, INV, v, digital, slope1)+np.pi,2*np.pi)-np.pi,
# 	'-',linewidth=2, markersize=0.75, color='blue', label=r'$\beta$ Unstable')
# #*******************************************************************************
#
#
#
# if digital == True:
# 	plt.savefig('plts/digital_sigma_vs_Omega_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plts/digital_sigma_vs_Omega_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
# 	if coupfunction(coupfun)=='sin':
# 		plt.savefig('plts/analog_sin_sigma_vs_Omega_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plts/analog_sin_sigma_vs_Omega_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plts/analog_sin_sigma_vs_Omega_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfunction(coupfun)=='cos':
# 		plt.savefig('plts/analog_cos_betamut_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plts/analog_cos_betamut_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# # plt.savefig('plts/betamut_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# plt.savefig('plts/betamut_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# plt.show()



#
#
# [lineOmegStabAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab'],
# 				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab']/np.asarray(w),
# 				 'o',ms=1, color='blue',  label=r'Antiphase Stable')
#
# [lineOmegUnstAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 				globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst'],
# 				solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst']/np.asarray(w),
# 				'o',ms=1, color='red', label=r'Antiphase Unstable')
#




# #*************************************************************************************************************************************************************************


fig1         = plt.figure(figsize=(figwidth,figheight))
ax1          = fig1.add_subplot(111)
plt.rc('legend',fontsize=17)
# plot grid, labels, define intial values
# plt.title(r'The analytic expression characteristic equation',fontsize=30);
plt.grid()
plt.xlabel(r'$\Omega\tau/2\pi$',  fontsize=60,labelpad=-5)
plt.ylabel(r'$\frac{2\pi\sigma}{\omega}$\\[33pt] $\frac{\mu}{\omega}$', rotation=0,fontsize=85, labelpad=30)
ax1.tick_params(axis='both', which='major', labelsize=55, pad=1)
ax1.set_xlim(xmin, xmax)





plt.axhline(y=0, color='black', linestyle='-',linewidth=1)

[lineSigmaIn] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*syncstateIn['Omeg']*syncstateIn['tau'], np.asarray(2.0*np.pi/w)*StabIn['ReMax'],
	linewidth=4, color='green', label=r'$\frac{2\pi\sigma}{\omega}$')
[lineGammaIn] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*syncstateIn['Omeg']*syncstateIn['tau'], np.asarray(1/w)*StabIn['ImMax'],
	'.', ms=4, color='orange', label=r'$\frac{\mu}{\omega}$')



# [lineSigmaIn] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*np.asarray(tausigmaW)*np.asarray(OmegsigmaW), np.asarray(2.0*np.pi/w)*np.asarray(sigmaLambertW),'--', linewidth=4, color='orange', label=r'$\frac{2\pi\sigma}{\omega}$')

fig1.set_size_inches(20,10)
plt.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# [lineBeta2

if digital == True:
	plt.savefig('plts/digital_sigma_vs_Omega_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plts/digital_sigma_vs_Omega_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plts/digital_sigma_vs_Omega_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plts/analog_sin_sigma_vs_Omega_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plts/analog_sin_sigma_vs_Omega_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plts/analog_sin_sigma_vs_Omega_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plts/analog_cos_sigma_vs_Omega_tau%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plts/analog_cos_sigma_vs_Omega_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plts/analog_cos_sigma_vs_Omega_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

# plt.savefig('plts/betamut_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# plt.savefig('plts/betamut_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
plt.show()



#
# [lineSigmaAnti] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],
# 	linewidth=1, color='red', label=r'Antiphase Stable $\sigma$=Re$(\lambda)$')
# [lineGammaAnti] = ax1.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ImMax'],
# 	 '.',ms=1, color='blue', label=r'Antiphase Unstable $\gamma$=Im$(\lambda)$')

#
# fig2         = plt.figure(figsize=(figwidth,figheight))
# ax2          = fig2.add_subplot(111)
# plt.rc('legend',fontsize=17)
# # plot grid, labels, define intial values
# plt.title(r'The analytic expression characteristic equation',fontsize=30);
# plt.grid()
# plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=26)
# plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\Omega$', fontsize=26)
#
# plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
# ax2.tick_params(axis='both', which='major', labelsize=20)
# [lineSigmaIn2] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ReMax'],
# 	linewidth=4, color='red', label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
# [lineGammaIn2] = ax2.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'])*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ImMax'],
# 	'.', ms=4, color='blue', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
#
# #
# # fig3         = plt.figure(figsize=(figwidth,figheight))
# # ax3          = fig3.add_subplot(111)
# # plt.rc('legend',fontsize=17)
# # # plot grid, labels, define intial values
# # plt.title(r'The analytic expression characteristic equation',fontsize=30);
# # plt.grid()
# # plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=26)
# # plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\Omega$', fontsize=26)
# #
# # plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
# # ax3.tick_params(axis='both', which='major', labelsize=20)
# #
# # for i in range(len(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta) ) ):
# #
# # 	[lineSigmaIn3] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# # 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['Re'][i],'+',
# # 		markersize=4, color='red')#, label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
# # 	[lineGammaIn] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# # 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['Im'][i],
# # 		'd', ms=4, color='black')#, label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
# # [lineSigmaIn3] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# # 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['Re'][0],'+',
# 	markersize=4, color='red', label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
# [lineGammaIn] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['Im'][0],
# 	'd', ms=4, color='black', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
# [lineGammaIn] = ax3.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(1/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ImMax'],
# 	'.', ms=4, color='blue', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
# fig4         = plt.figure(figsize=(figwidth,figheight))
# ax4          = fig4.add_subplot(111)
# plt.rc('legend',fontsize=17)
# # plot grid, labels, define intial values
# plt.title(r'The analytic expression characteristic equation',fontsize=30);
# plt.grid()
# plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=26)
# plt.ylabel(r'$2\pi\sigma/\omega$, $\gamma/\Omega$', fontsize=26)
#
# plt.axhline(y=0, color='green', linestyle='-',linewidth=3)
# ax4.tick_params(axis='both', which='major', labelsize=20)
# for j in range(len(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta) ) ):
#
# 	[lineSigmaIn4] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase2, expansion, INV, zeta)['Re'][j],'+',
# 		markersize=4, color='green')#, label=r'Antiphase Stable $\sigma$=Re$(\lambda)$')
# 	[lineGammaIn4] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['Im'][j],
# 		'd', ms=4, color='black')#, label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
# [lineSigmaIn4] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital,maxp, inphase2, expansion, INV, zeta)['Re'][0],'+',
# 	markersize=4, color='green', label=r'Antiphase Stable $\sigma$=Re$(\lambda)$')
# [lineGammaIn2] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['Im'][0],
# 	'd', ms=4, color='black', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')
#
# for i in range(len(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta) ) ):
#
# 	[lineSigmaIn03] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['Re'][i],'+',
# 		markersize=4, color='red')#, label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
#
# [lineSigmaIn03] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['Re'][0],'+',
# 	markersize=4, color='red', label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
#
#
#
# [lineGammaIn2] = ax4.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ImMax'],
# 	'.', ms=4, color='blue', label=r'Inphase Stable $\gamma$=Im$(\lambda)$')


# ax.legend()
# ax1.legend()
# ax2.legend()
# ax3.legend()
# ax4.legend()

#ax5.legend()
plt.show()
