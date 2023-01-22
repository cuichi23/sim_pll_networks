#Linear Stability of Global Frequency (LiStaGloF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq, linStabEq, xrange, alpha, K, coupfununction, xrangedifv
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
coupfun='cos'
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider

w    = 1.0*2.0*np.pi				#2.0*np.pi*24E9;
Kvco = 0.1*2.0*np.pi			#2.0*np.pi*250E6; for the first plot where the value of Kvco is not on the title of the file, Kvco = 0.05*2.0*np.pi

AkPD = 1.0
Ga1  = 1.0;
tauf = 0.0
tauc = 1.0/(2.0*np.pi*0.25);
order= 1.0
v	 = 1.0;
c	 = 3E8;
maxp = 40;
wR        	= 2.0*np.pi*1.0;
slope1		= 1
slope2		= 2
maxtau = 6.0;
K1   = K(Kvco, AkPD, Ga1)
# xrangetau
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



# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
dp		= 0.0015;


# # #*******************************************************************************
vbar= [1, 4, 64]
colorbar=['blue','green','orange']
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

for i in range(len(vbar)):
	# print(bar[i])
	# print(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]])
	[lineclStab11] 	=	ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, vbar[i], digital, maxp, inphase1, noINV)['Omeg'],
							globalFreq(w, Kvco, AkPD, Ga1, tauf, vbar[i], digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc,vbar[i], order, digital, maxp, inphase1, expansion, noINV, zeta)['tauStab'],
							solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, vbar[i], digital, maxp, inphase1, noINV)['Omeg'],
							globalFreq(w, Kvco, AkPD, Ga1, tauf, vbar[i], digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, vbar[i], order, digital, maxp, inphase1, expansion, noINV, zeta)['OmegStab']/np.asarray(w),
							'o' ,mfc='none', ms=7, color= colorbar[i], label=r'$v=$%.0F' %(vbar[i]) )

	[lineclUnsb11] 	=	ax.plot(np.asarray(w/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, vbar[i], digital, maxp, inphase1, noINV)['Omeg'],
							globalFreq(w, Kvco, AkPD, Ga1, tauf, vbar[i], digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc,vbar[i], order, digital, maxp, inphase1, expansion, noINV, zeta)['tauUnst'],
							solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, vbar[i], digital, maxp, inphase1, noINV)['Omeg'],
							globalFreq(w, Kvco, AkPD, Ga1, tauf, vbar[i], digital, maxp, inphase1, noINV)['tau'], w, tauf, Kvco, AkPD, Ga1, tauc, vbar[i], order, digital, maxp, inphase1, expansion, noINV, zeta)['OmegUnst']/np.asarray(w),
							'o' ,mfc='none', ms=1, color='grey')#, label=r'$v=$%.0F' %(vbar[i]) )

#
fig.set_size_inches(20,10)
ax.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
plt.legend(fontsize=25)
if digital == True:
    plt.savefig('plts/different_divisions_digital_gamma_vs_tau1%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
    plt.savefig('plts/different_divisions_digital_gamma_vs_tau1%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
    if coupfununction(coupfun)=='sin':
        plt.savefig('plts/different_divisions_analog_sin_gamma_vs_tau1%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/different_divisions_analog_sin_gamma_vs_tau1%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

    if coupfununction(coupfun)=='cos':
        plt.savefig('plts/different_divisions_analog_cos_gamma_vs_tau1%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
        plt.savefig('plts/different_divisions_analog_cos_gamma_vs_tau1%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
plt.show()
