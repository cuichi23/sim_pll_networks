#Linear Stability of Global Frequency (LiStaGloF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq, linStabEq, xrange, alpha, K, coupfunction, phase_diffsFullEquationsOne, phase_diffsFullEquations, xrangetau, solveLinStabSingleGen, initial_guessbeta, initial_guessOmeg,solveLinStabSingle
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
digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# choose phase or anti-phase synchroniazed states,

inphase1=True;
inphase2=False;
coupfun='sin'
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider

w    = 1.0*2.0*np.pi				#2.0*np.pi*24E9;
Kvco = 0.04*2.0*np.pi			#2.0*np.pi*250E6; for the first plot where the value of Kvco is not on the title of the file, Kvco = 0.05*2.0*np.pi

AkPD = 1.0
Ga1  = 1.0;
tauf = 0.0
tauc = 1.0/(2.0*np.pi*1.0);
order= 1.0
v	 = 1.0;
c	 = 3E8;
maxp = 40;
wR        	= 2.0*np.pi*1.0;
slope1		= 1
slope2		= 2
maxtau = 6.0;
K1   = K(Kvco, AkPD, Ga1)
xrangetau
INV  = 0.0*np.pi;
noINV= 0.0
zeta = -1.0



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
figwidth  = 6;
figheight = 5;

tau=xrangetau(maxtau);
resultsSingle1=[];
for j in range(len(tau)):

	for i in range(len(initial_guessbeta())):
		tempo = phase_diffsFullEquationsOne(w, tau[j], Kvco, AkPD, Ga1, tauf, v, digital, INV, initial_guessbeta()[i], initial_guessOmeg(w, K1)[i])
		# temporary2 = phase_diffsSingleOne(w1, w2, wR[j], tauR1, tauR2, tau12, tauf, Kvco1, Kvco2, AkPD, Ga1, digital, INV, slope2, init1[i], init2[i])

		# if len(temporary1['wRef'])!=0:
			# print(temporary1['betaR1vec'])
		temporaryStab = solveLinStabSingleGen(tempo['Omega'][i], tau[j],tempo['beta'][i],w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp,  expansion, INV, zeta ,initial_guessbeta()[i], initial_guessOmeg(w, K1)[i])

		print(temporaryStab)
		if len(temporaryStab['tauStab'])!=0:
			resultsSingle1.append([ temporaryStab['tauStab'][0],  temporaryStab['betaStab'][0],	temporaryStab['OmegStab'][0] ])

resultsSingle1	  = np.asarray(resultsSingle1);

# print(results1[:,1])
tauUnst			        = resultsSingle1[:,0];
tauinStab			    = resultsSingle1[:,1];
betaUnst     			= resultsSingle1[:,2];
betaStab 			    = resultsSingle1[:,3];
OmegUnst			    = resultsSingle1[:,4];
lineOmegStab            = resultsSingle1[:,5]
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
dp		= 0.015;

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
plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60,labelpad=-5)
plt.ylabel(r'$\frac{\Omega}{\omega}$',rotation=0, fontsize=85, labelpad=30)
ax.tick_params(axis='both', which='major', labelsize=35, pad=1)
ax.set_xlim(0.0, 5.0)
# plt.xlim(-0.1, max(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']))
# # draw the initial plot
# # the 'lineXXX' variables are used for modifying the lines later


# [lineOmegStabIn3] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, noINV, zeta)['tauStab'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, noINV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, noINV, zeta)['OmegStab']/np.asarray(w),
# 						  'o' ,mfc='none', ms=7, color='red',  label=r'in-phase stable INV Off' )
# # #
# [lineOmegUnstIn3] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['Omeg'],
# 								globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, noINV, zeta)['tauUnst'],
# 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['Omeg'],
# 						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, noINV, zeta)['OmegUnst']/np.asarray(w),
# 						 'o', mfc='none', ms=1,  color='grey')# label=r'in-phase unstable no INV')



# # #*************************************************************************************************************************************************************************
# [lineOmegStabIn] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 						globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauStab'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegStab']/np.asarray(w),
# 						 '+',ms=7, color='red',  label=r'in-phase stable')# INV On')
#

# [lineOmegStabIn] = ax.plot(np.asarray(w/(2.0*np.pi))*xrangetau(maxtau),phase_diffsFullEquations(w , maxtau, Kvco, AkPD, Ga1, tauf, v, digital, INV)['beta'],'.')
[lineOmegStabIn] = ax.plot(np.asarray(1/(2.0*np.pi))*OmegUnst*tauUnst,betaUnst,'.')
# [lineOmegUnstIn] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 								globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauUnst'],
# 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegUnst']/np.asarray(w),
						 # '+',ms=1, color='grey')#, label=r'in-phase unstable')


# #
# # [lineOmegUnstAnti] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# # 					     globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauUnst'],
# # 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# # 						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst']/np.asarray(w),
# # 						 'o',ms=1, color='red', label=r'Antiphase Unstable')
# #
# # [lineOmegIn] = ax.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['tau'],
# # 						globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['Omeg']/np.asarray(w),
# # 						'o',ms=7, color='red', label=r'Inphase')


#
fig.set_size_inches(20,10)
ax.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
plt.legend(fontsize=25)

if digital == True:
	plt.savefig('plts/digital_gamma_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plts/digital_gamma_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plts/analog_sin_gamma_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plts/analog_sin_gamma_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plts/analog_cos_gamma_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plts/analog_cos_gamma_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# # #*************************************************************************************************************************************************************************
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
# #
# #
# # [lineSigmaAnti] = ax1.plot(np.asarray(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# # 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['ReMax'],
# # 	 ':',linewidth=4, color='blue', label=r'anti-phase')
#
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
#
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
# ax2.set_xlim(0.0, 5.0)

# [lineSigmaIn1] = ax2.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ReMax'],
# 	 linewidth=4, color='red')#, label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
#
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


# draw the initial plot
# print(len(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'],
#     w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauStab']))

plt.show()
