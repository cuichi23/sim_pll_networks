#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq,linStabEq,initial_guess, solveLinStabSingle, K,alpha, coupfunction
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

w    = 1.0*2.0*np.pi				#2.0*np.pi*24E9;
Kvco = 0.4*2.0*np.pi			#2.0*np.pi*250E6;
AkPD = 1.0
Ga1  = 1.0;
tauf = 0.0
tauc = 1.0/(2.0*np.pi*0.014);
order= 1.0
v	 = 1.0;
c	 = 3E8;
maxp = 24#3.82999e5;

INV    = 0.0*np.pi;
zeta   = -1.0
Ga1Max = 5.0*Ga1
fcmax  = 4.5*0.140
G  = np.linspace(1.0E-8, Ga1Max, 500, dtype=np.float64);
# print(G)
fc = np.linspace(1.0E-8, fcmax,500, dtype=np.float64)
figwidth  =	6;
figheight = 6;
# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
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




tau=1.5

# print(globalFreq(w, Kvco, AkPD, G, tauf, v, digital, maxp, inphase1, INV)['tau'])
# print(np.where(globalFreq(w, Kvco, AkPD, G, tauf, v, digital, maxp, inphase1, INV)['tau']<tau)[0][0])
tauin=globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'][np.where(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<tau)]
omegain=globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<tau)]
tauin1=tauin[np.where(tauin>0.0)][-1]
omegain1=omegain[np.where(tauin>0.0)][-1]
tauAn=globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'][np.where(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<tau)]
omegaAn=globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<tau)]
tauAn1=tauAn[np.where(tauAn>0.0)][-1]
omegaAn1=omegaAn[np.where(tauAn>0.0)][-1]
# print(tau1)
print('inphase -> alpha/K:\t\t',alpha(omegain1, tauin1, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase1, INV)/K(Kvco, AkPD, Ga1) )
print('antiphase -> alpha/K:\t\t',alpha(omegaAn1, tauAn1, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase2, INV)/K(Kvco, AkPD, Ga1) )
print('delay:\t\t', tauin1)
print('inphase -> Omega*tau/2pi\t\t:', omegain1*tauin1/(2.0*np.pi))
print('Antiphase -> Omega*tau/2pi\t\t:', omegaAn1*tauin1/(2.0*np.pi))
print('omega*tau/2pi\t\t:', w*tauin1/(2.0*np.pi))
# for i in range(len(G)):
# 	print(i, K(Kvco, AkPD, G[i]), solveLinStabSingle(omega1, tau1, tauf, Kvco, AkPD, G[i], tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'])

fig1         = plt.figure(figsize=(figwidth,figheight))
ax1          = fig1.add_subplot(111)
plt.rc('legend',fontsize=17)
# plot grid, labels, define intial values
plt.title(r'Stability diagram',fontsize=50);
plt.grid()
plt.xlabel(r'$K/2\pi$', fontsize=60,labelpad=-5)
plt.ylabel(r'$\sigma$, $\mu$', fontsize=85, labelpad=30)

plt.axhline(y=0, color='black', linestyle='-',linewidth=1)
ax1.tick_params(axis='both', which='major', labelsize=35)
for i in range(len(G)):
	[lineSigmaIn] = ax1.plot(K(Kvco, AkPD, G[i])/(2.0*np.pi), solveLinStabSingle(omegain1, tauin1, tauf, Kvco, AkPD, G[i], tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'],'+',
	markersize=4, color='green')#, label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
	[lineGammaIn] = ax1.plot(K(Kvco, AkPD, G[i])/(2.0*np.pi), solveLinStabSingle(omegain1,tauin1, tauf, Kvco, AkPD, G[i], tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ImMax'],
		'.', ms=4, color='orange')#, label=r'Inphase Stable $\gamma$=Im$(\lambda)$')

[lineSigmaIn] = ax1.plot(K(Kvco, AkPD, G[0])/(2.0*np.pi), solveLinStabSingle(omegain1, tauin1, tauf, Kvco, AkPD, G[0], tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'],'+',
	markersize=4, color='green', label=r'$\sigma$=Re$(\lambda)$')
[lineGammaIn] = ax1.plot(K(Kvco, AkPD, G[0])/(2.0*np.pi), solveLinStabSingle(omegain1,tauin1, tauf, Kvco, AkPD, G[0], tauc, v, order, digital,maxp, inphase1, expansion, INV, zeta)['ImMax'],
	'.', ms=4, color='orange', label=r'$\gamma$=Im$(\lambda)$')

fig1.set_size_inches(20,10)
ax1.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
plt.legend(fontsize=45)


if digital == True:
    # print('hallo')
    plt.savefig('plts/digital_bif_K_inphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/digital_bif_K_inphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/digital_bif_K_inphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
    if coupfunction(coupfun)=='sin':
        # print('hallo')
        plt.savefig('plts/analog_sin_bif_K_inphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_sin_bif_K_inphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_sin_bif_K_inphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

    elif coupfunction(coupfun)=='cos':
        # print('hallo')
        plt.savefig('plts/analog_cos_bif_K_inphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_cos_bif_K_inphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_cos_bif_K_inphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    elif coupfunction(coupfun)=='negcos':
        # print('hallo')
        plt.savefig('plts/analog_negcos_bif_K_inphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_negcos_bif_K_inphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_negcos_bif_K_inphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)




fig2         = plt.figure(figsize=(figwidth,figheight))
ax2          = fig2.add_subplot(111)
plt.rc('legend',fontsize=17)
# plot grid, labels, define intial values
plt.title(r'Stability diagram',fontsize=30);
plt.grid()
plt.xlabel(r'$K/2\pi$', fontsize=60,labelpad=-5)
plt.ylabel(r'$\sigma$, $\gamma$', fontsize=85, labelpad=30)

plt.axhline(y=0, color='black', linestyle='-',linewidth=1)
ax2.tick_params(axis='both', which='major', labelsize=35)
for i in range(len(G)):
	[lineSigmaAn] = ax2.plot(K(Kvco, AkPD, G[i])/(2.0*np.pi), solveLinStabSingle(omegaAn1, tauAn1, tauf, Kvco, AkPD, G[i], tauc, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],'+',
	markersize=4, color='green')#, label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
	[lineGammaAn] = ax2.plot(K(Kvco, AkPD, G[i])/(2.0*np.pi), solveLinStabSingle(omegaAn1,tauAn1, tauf, Kvco, AkPD, G[i], tauc, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ImMax'],
		'.', ms=4, color='orange')#, label=r'Inphase Stable $\gamma$=Im$(\lambda)$')

[lineSigmaAn] = ax2.plot(K(Kvco, AkPD, G[0])/(2.0*np.pi), solveLinStabSingle(omegaAn1, tauAn1, tauf, Kvco, AkPD, G[0], tauc, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],'+',
	markersize=4, color='green', label=r'$\sigma$=Re$(\lambda)$')
[lineGammaAn] = ax2.plot(K(Kvco, AkPD, G[0])/(2.0*np.pi), solveLinStabSingle(omegaAn1,tauAn1, tauf, Kvco, AkPD, G[0], tauc, v, order, digital,maxp, inphase2, expansion, INV, zeta)['ImMax'],
	'.', ms=4, color='orange', label=r'$\gamma$=Im$(\lambda)$')


fig2.set_size_inches(20,10)
ax2.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
plt.legend(fontsize=25)


if digital == True:
    # print('hallo')
    plt.savefig('plts/digital_bif_K_antiphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/digital_bif_K_antiphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/digital_bif_K_antiphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
    if coupfunction(coupfun)=='sin':
        # print('hallo')
        plt.savefig('plts/analog_sin_bif_K_antiphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_sin_bif_K_antiphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_sin_bif_K_antiphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

    elif coupfunction(coupfun)=='cos':
        # print('hallo')
        plt.savefig('plts/analog_cos_bif_K_antiphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_cos_bif_K_antiphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_cos_bif_K_antiphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    elif coupfunction(coupfun)=='negcos':
        # print('hallo')
        plt.savefig('plts/analog_negcos_bif_K_antiphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_negcos_bif_K_antiphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_negcos_bif_K_antiphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)






fig3         = plt.figure(figsize=(figwidth,figheight))
ax3          = fig3.add_subplot(111)
plt.rc('legend',fontsize=17)
# plot grid, labels, define intial values
plt.title(r'Stability diagram',fontsize=50);
plt.grid()
plt.xlabel(r'$\omega_c/2\pi$', fontsize=60,labelpad=-5)
plt.ylabel(r'$\sigma$, $\mu$', fontsize=85, labelpad=30)

plt.axhline(y=0, color='black', linestyle='-',linewidth=1)
ax3.tick_params(axis='both', which='major', labelsize=35)
for i in range(len(fc)):
	[lineSigmaIn] = ax3.plot(fc[i], solveLinStabSingle(omegain1, tauin1, tauf, Kvco, AkPD, Ga1, 1.0/(2.0*np.pi*fc[i]), v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'],'+',
	markersize=4, color='green')#, label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
	[lineGammaIn] = ax3.plot(fc[i], solveLinStabSingle(omegain1,tauin1, tauf, Kvco, AkPD, Ga1, 1.0/(2.0*np.pi*fc[i]), v, order, digital,maxp, inphase1, expansion, INV, zeta)['ImMax'],
		'.', ms=4, color='orange')#, label=r'Inphase Stable $\gamma$=Im$(\lambda)$')

[lineSigmaIn] = ax3.plot(fc[0], solveLinStabSingle(omegain1, tauin1, tauf, Kvco, AkPD, Ga1, 1.0/(2.0*np.pi*fc[0]), v, order, digital,maxp, inphase1, expansion, INV, zeta)['ReMax'],'+',
	markersize=4, color='green', label=r' $\sigma$=Re$(\lambda)$')
[lineGammaIn] = ax3.plot(fc[0], solveLinStabSingle(omegain1,tauin1, tauf, Kvco, AkPD, Ga1, 1.0/(2.0*np.pi*fc[0]), v, order, digital,maxp, inphase1, expansion, INV, zeta)['ImMax'],
	'.', ms=4, color='orange', label=r'$\mu$=Im$(\lambda)$')



fig3.set_size_inches(20,10)
ax3.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
plt.legend(fontsize=25)


if digital == True:
    # print('hallo')
    plt.savefig('plts/digital_bif_fc_inphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/digital_bif_fc_inphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/digital_bif_fc_inphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
    if coupfunction(coupfun)=='sin':
        # print('hallo')
        plt.savefig('plts/analog_sin_bif_fc_inphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_sin_bif_fc_inphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_sin_bif_fc_inphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

    elif coupfunction(coupfun)=='cos':
        # print('hallo')
        plt.savefig('plts/analog_cos_bif_fc_inphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_cos_bif_fc_inphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_cos_bif_fc_inphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    elif coupfunction(coupfun)=='negcos':
        # print('hallo')
        plt.savefig('plts/analog_negcos_bif_fc_inphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_negcos_bif_fc_inphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_negcos_bif_fc_inphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)


fig4         = plt.figure(figsize=(figwidth,figheight))
ax4          = fig4.add_subplot(111)
plt.rc('legend',fontsize=17)
# plot grid, labels, define intial values
plt.title(r'Stability diagram',fontsize=50);
plt.grid()
plt.xlabel(r'$\omega_c/2\pi$', fontsize=60,labelpad=-5)
plt.ylabel(r'$\sigma$, $\mu$', fontsize=85, labelpad=30)

plt.axhline(y=0, color='black', linestyle='-',linewidth=1)
ax4.tick_params(axis='both', which='major', labelsize=35)
for i in range(len(fc)):
	[lineSigmaAn] = ax4.plot(fc[i], solveLinStabSingle(omegaAn1, tauAn1, tauf, Kvco, AkPD, Ga1, 1.0/(2.0*np.pi*fc[i]), v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],'+',
	markersize=4, color='green')#, label=r'Inphase Stable $\sigma$=Re$(\lambda)$')
	[lineGammaAn] = ax4.plot(fc[i], solveLinStabSingle(omegaAn1,tauAn1, tauf, Kvco, AkPD, Ga1, 1.0/(2.0*np.pi*fc[i]), v, order, digital,maxp, inphase2, expansion, INV, zeta)['ImMax'],
		'.', ms=4, color='orange')#, label=r'Inphase Stable $\gamma$=Im$(\lambda)$')

[lineSigmaAn] = ax4.plot(fc[0], solveLinStabSingle(omegaAn1, tauAn1, tauf, Kvco, AkPD,  Ga1, 1.0/(2.0*np.pi*fc[0]), v, order, digital,maxp, inphase2, expansion, INV, zeta)['ReMax'],'+',
	markersize=4, color='green', label=r'$\sigma$=Re$(\lambda)$')
[lineGammaAn] = ax4.plot(fc[0], solveLinStabSingle(omegaAn1,tauAn1, tauf, Kvco, AkPD,  Ga1, 1.0/(2.0*np.pi*fc[0]), v, order, digital,maxp, inphase2, expansion, INV, zeta)['ImMax'],
	'.', ms=4, color='orange', label=r'$\gamma$=Im$(\lambda)$')

fig4.set_size_inches(20,10)
ax4.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
plt.legend(fontsize=25)


if digital == True:
    # print('hallo')
    plt.savefig('plts/digital_bif_fc_antiphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/digital_bif_fc_antiphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/digital_bif_fc_antiphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
    if coupfunction(coupfun)=='sin':
        # print('hallo')
        plt.savefig('plts/analog_sin_bif_fc_antiphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_sin_bif_fc_antiphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_sin_bif_fc_antiphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

    elif coupfunction(coupfun)=='cos':
        # print('hallo')
        plt.savefig('plts/analog_cos_bif_fc_antiphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_cos_bif_fc_antiphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_cos_bif_fc_antiphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    elif coupfunction(coupfun)=='negcos':
        # print('hallo')
        plt.savefig('plts/analog_negcos_bif_fc_antiphase%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_negcos_bif_fc_antiphase%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/analog_negcos_bif_fc_antiphase%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)


# ax.legend()
# ax1.legend()
# ax2.legend()
# ax3.legend()
# ax4.legend()

#ax5.legend()
plt.show()
