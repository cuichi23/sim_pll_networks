#Linear Stability of Global Frequency (LiStaGloF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq, linStabEq, xrange, alpha, K, coupfununction
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

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# choose phase or anti-phase synchroniazed states,

inphase1=True;
inphase2=False;
coupfun='cos'
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
INV  = 1.0*np.pi;
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



# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
dp		= 0.015;
# # print(globalFreqINV(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'])
# fig0    = plt.figure(num=0, figsize=(figwidth, figheight), facecolor='w', edgecolor='k')

# solveLinStab(Omega, tau, w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase, expansion, INV, zeta)
# stabListnoINVIn	=np.asarray(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, noINV, zeta)['tauStab'])
# stabListINVIn	=np.asarray(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauStab'])
# stabListnoINVAnti	=np.asarray(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, noINV, zeta)['tauStab'])
# stabListINVAnti	=np.asarray(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauStab'])
# # print()
# # stabListOme =np.asarray(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegStab'])
# # print(stabList)
# # sameOmeg= stabListOme[np.where(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegStab'])]
# # unstabList	=np.asarray(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase, expansion, INV, zeta)['tauUnst'])
# stabminlinoINVIn=[];
# stabmaxlinoINVIn=[];
# stabminliINVIn=[];
# stabmaxliINVIn=[];
# stabminlinoINVAnti=[];
# stabmaxlinoINVAnti=[];
# stabminliINVAnti=[];
# stabmaxliINVAnti=[];
# stabminlinoINVIn.append(stabListnoINVIn[0])
# for i in range(len(stabListnoINVIn)):
# 	# print(i,len(stabminList))
# 	if (stabListnoINVIn[i]-stabListnoINVIn[i-1])>dp:
# 		stabmaxlinoINVIn.append(stabListnoINVIn[i-1]);
# 		stabminlinoINVIn.append(stabListnoINVIn[i]);
#     # if stablist[i]
# 	# else:
# 	#     stabmaxli.append(stabList[-1]);
# 	#     stabminli.append(stabList[0])
# stabmaxlinoINVIn.append(stabListnoINVIn[-1])
#
# stabminliINVIn.append(stabListINVIn[0])
# for i in range(len(stabListINVIn)):
# 	# print(i,len(stabminList))
# 	if (stabListINVIn[i]-stabListINVIn[i-1])>dp:
# 		stabmaxliINVIn.append(stabListINVIn[i-1]);
# 		stabminliINVIn.append(stabListINVIn[i]);
#     # if stablist[i]
# 	# else:
# 	#     stabmaxli.append(stabList[-1]);
# 	#     stabminli.append(stabList[0])
# stabmaxliINVIn.append(stabListINVIn[-1])
#
#
# stabminlinoINVAnti.append(stabListnoINVAnti[0])
# for i in range(len(stabListnoINVAnti)):
# 	# print(i,len(stabminList))
# 	if (stabListnoINVAnti[i]-stabListnoINVAnti[i-1])>dp:
# 		stabmaxlinoINVAnti.append(stabListnoINVAnti[i-1]);
# 		stabminlinoINVAnti.append(stabListnoINVAnti[i]);
#     # if stablist[i]
# 	# else:
# 	#     stabmaxli.append(stabList[-1]);
# 	#     stabminli.append(stabList[0])
# stabmaxlinoINVAnti.append(stabListnoINVAnti[-1])
#
# stabminliINVAnti.append(stabListINVAnti[0])
# for i in range(len(stabListINVAnti)):
# 	# print(i,len(stabminList))
# 	if (stabListINVAnti[i]-stabListINVAnti[i-1])>dp:
# 		stabmaxliINVAnti.append(stabListINVAnti[i-1]);
# 		stabminliINVAnti.append(stabListINVAnti[i]);
#     # if stablist[i]
# 	# else:
# 	#     stabmaxli.append(stabList[-1]);
# 	#     stabminli.append(stabList[0])
# stabmaxliINVAnti.append(stabListINVAnti[-1])
# # print(stabmaxlinoINVIn)
# # print(stabmaxliINVIn)
# # #*******************************************************************************

fig         = plt.figure(figsize=(figwidth,figheight))
ax          = fig.add_subplot(111)

plt.grid()
plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60,labelpad=-5)
plt.ylabel(r'$\frac{\Omega}{\omega}$',rotation=0, fontsize=85, labelpad=30)
ax.tick_params(axis='both', which='major', labelsize=35, pad=1)
ax.set_xlim(0.0, 5.0)


[lineOmegStabIn3] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, noINV, zeta)['tauStab'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, noINV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, noINV, zeta)['OmegStab']/np.asarray(w),
						  'o' ,mfc='none', ms=9, color='red',  label=r'in-phase stable INV off' )
# #
[lineOmegUnstIn3] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['Omeg'],
								globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, noINV, zeta)['tauUnst'],
						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['Omeg'],
						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, noINV, zeta)['OmegUnst']/np.asarray(w),
						 'o', mfc='none', ms=3,  color='grey')# label=r'in-phase unstable no INV')



# #*************************************************************************************************************************************************************************
[lineOmegStabIn] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
						globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauStab'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegStab']/np.asarray(w),
						 'd',ms=9, color='red',  label=r'in-phase stable INV on')
#
[lineOmegUnstIn] = ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
								globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauUnst'],
						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegUnst']/np.asarray(w),
						 'd',ms=3, color='grey')#, label=r'in-phase unstable')
#
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
# # textstr = '\n'.join((r'INV On'))
# ax.text(0.05, 0.85, 'INV 0ff', transform=ax.transAxes, fontsize=36,
#         verticalalignment='top', bbox=props)#, verticalalignment='top')
#
# ax.text(0.25, 0.85, 'INV on', transform=ax.transAxes, fontsize=36,
#         verticalalignment='top', bbox=props)
# ax.text(0.45, 0.85, 'INV off', transform=ax.transAxes, fontsize=36,
#         verticalalignment='top', bbox=props)
#
# ax.text(0.65, 0.85, 'INV on', transform=ax.transAxes, fontsize=36,
#         verticalalignment='top', bbox=props)
#
# ax.text(0.85, 0.85, 'INV off', transform=ax.transAxes, fontsize=36,
#         verticalalignment='top', bbox=props)
#
#

#
# if (len(set(stabmaxlinoINVIn))>1):
#
#
# 	for i in range(len(stabmaxlinoINVIn)):
# 		# print(i)
# 		plt.axvspan(stabminlinoINVIn[i], stabmaxlinoINVIn[i], facecolor='g', alpha=0.5)

# if (len(set(stabmaxliINVIn))>1):
#
#
# 	for i in range(len(stabmaxliINVIn)):
# 		# print(i)
# 		plt.axvspan(stabminliINVIn[i], stabmaxliINVIn[i], facecolor='g', alpha=0.5)
#


#
fig.set_size_inches(20,10)
ax.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# plt.legend(fontsize=25)
if digital == True:
    plt.savefig('plts/Inphase_diff_INV_digital_freq_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/Inphase_diff_INV_digital_freq_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
    if coupfununction(coupfun)=='sin':
        plt.savefig('plts/Inphase_diff_INV_analog_sin_freq_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/Inphase_diff_INV_analog_sin_freq_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

    if coupfununction(coupfun)=='cos':
        plt.savefig('plts/Inphase_diff_INV_analog_cos_freq_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/Inphase_diff_INV_analog_cos_freq_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

fig0         = plt.figure(figsize=(figwidth,figheight))
ax0         = fig0.add_subplot(111)

plt.grid()
plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60,labelpad=-5)
plt.ylabel(r'$\frac{\Omega}{\omega}$',rotation=0, fontsize=85, labelpad=30)
ax0.tick_params(axis='both', which='major', labelsize=35, pad=1)
ax0.set_xlim(0.0, 5.0)

[lineOmegStabAnti3] = ax0.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['Omeg'],
							globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, noINV, zeta)['tauStab'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, noINV)['Omeg'],
						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, noINV, zeta)['OmegStab']/np.asarray(w),
						 'o',mfc='none', ms=9, color='blue',  label=r'anti-phase stable INV off')

[lineOmegUnsAnti3] = ax0.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['Omeg'],
							globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, noINV, zeta)['tauUnst'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, noINV)['Omeg'],
						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, noINV, zeta)['OmegUnst']/np.asarray(w),
						 'o', mfc='none', ms=3, color='grey')#  label=r'anti-phase stable INV Off')

[lineOmegStabAnti] = ax0.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
							globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauStab'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, INV)['Omeg'],
						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab']/np.asarray(w),
						 'd',ms=9, color='blue',  label=r'anti-phase stable INV on')

[lineOmegUnstAnti] = ax0.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
							globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauUnst'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, INV)['Omeg'],
						 globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst']/np.asarray(w),
						 'd',ms=3, color='grey')#,  label=r'anti-phase stable INV On')

#

# [lineOmegAnti] = ax0.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['tau'],
# 			globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, noINV)['Omeg']/np.asarray(w),
# 			'o',ms=7, color='blue', label=r'Antiphase')

# if (len(set(stabmaxlinoINVAnti))>1):
#
#
# 	for i in range(len(stabmaxlinoINVAnti)):
# 		# print(i)
# 		plt.axvspan(stabminlinoINVAnti[i], stabmaxlinoINVAnti[i], facecolor='g', alpha=0.5)

# if (len(set(stabmaxliINVAnti))>1):
#
#
# 	for i in range(len(stabmaxliINVAnti)):
# 		# print(i)
# 		plt.axvspan(stabminliINVAnti[i], stabmaxliINVAnti[i], facecolor='g', alpha=0.5)

# props2 = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
# # textstr = '\n'.join((r'INV On'))
# ax0.text(0.05, 0.85, 'INV 0ff', transform=ax0.transAxes, fontsize=36,
#         verticalalignment='top', bbox=props2)#, verticalalignment='top')
#
# ax0.text(0.25, 0.85, 'INV on', transform=ax0.transAxes, fontsize=36,
#         verticalalignment='top', bbox=props2)
# ax0.text(0.45, 0.85, 'INV off', transform=ax0.transAxes, fontsize=36,
#         verticalalignment='top', bbox=props2)
#
# ax0.text(0.65, 0.85, 'INV on', transform=ax0.transAxes, fontsize=36,
#         verticalalignment='top', bbox=props2)
#
# ax0.text(0.85, 0.85, 'INV off', transform=ax0.transAxes, fontsize=36,
#         verticalalignment='top', bbox=props2)
#
fig0.set_size_inches(20,10)
ax0.legend(bbox_to_anchor=(0.3415,0.35), prop=labelfont)

if digital == True:
    plt.savefig('plts/Antiphase_diff_INV_digital_freq_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/Antiphase_diff_INV_digital_freq_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
    if coupfununction(coupfun)=='sin':
        plt.savefig('plts/Antiphase_diff_INV_analog_sin_freq_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/Antiphase_diff_INV_analog_sin_freq_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

    if coupfununction(coupfun)=='cos':
        plt.savefig('plts/Antiphase_diff_INV_analog_cos_freq_vs_tau%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/Antiphase_diff_INV_analog_cos_freq_vs_tau%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

#*************************************************************************************************************************************************************************



#*************************************************************************************************************************************************************************

plt.show()
