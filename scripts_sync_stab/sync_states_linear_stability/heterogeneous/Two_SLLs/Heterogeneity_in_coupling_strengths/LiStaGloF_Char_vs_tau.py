#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStabbetain, globalFreq,linStabEq,solveLinStabbetaanti, solveLinStabSingle, K
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
w1   	 	= (1.0 - 0.0*0.02)*2.0*np.pi                                          # intrinsic frequency of PLL1
w2	 	    = (1.0 + 0.0*0.02)*2.0*np.pi                                          # intrinsic frequency of PLL2
wmean	 	= (w1+w2)/2.0
Dw	 	    = w2-w1
Kvco1    	= (0.370+0.0*0.02)*2.0*np.pi                                            # Sensitivity of VCO of PLL1
Kvco2    	= (0.370-0.0*0.02)*2.0*np.pi                                            # Sensitivity of VCO of PLL1
AkPD     	= 1.0
GkLF     	= 1.0
Gvga     	= 1.0		                                                          # Gain of the first adder
K1 		    = K(Kvco1, AkPD, GkLF, Gvga)
K2 		    = K(Kvco2, AkPD, GkLF, Gvga)
Kmean	 	= (K1+K2)/2.0
DK	 	    = K2-K1
tauf 		= 0.0
tauf1       = 0.0
tauf2       = 0.0
tauc1 		= 1.0/((0.055 - 1.0*0.04)*2.0*np.pi);
tauc2 		= 1.0/((0.055 + 1.0*0.04)*2.0*np.pi);
v           = 1                                                                    # the division
c           = 0.63*3E8                                                             # speed of light
maxp        = 10.5
INV         = 0.0*np.pi# tauc1 = 1.0/(2.0*np.pi*0.0146);
# tauc2 = 1.0/(2.0*np.pi*0.0146);
maxp = 10.5;
inphase1 = True;
inphase2 = False;
# syncStateInphase   = globalFreq(wmean, Dw, Kmean, DK, tauf,  digital, maxp, inphase1)
# syncStateAntiphase = globalFreq(wmean, Dw, Kmean, DK, tauf,  digital, maxp, inphase2)

syncStateIn   = globalFreq(wmean, Dw, Kmean, DK, tauf,  digital, maxp, inphase1);
syncStateAnti = globalFreq(wmean, Dw, Kmean, DK, tauf,  digital, maxp, inphase2);
# print(syncStateIn['Omegabetain'])
results1=[]; results2=[]; results1Stab =[]; results2Stab = []; results1Unst =[]; results2Unst = [];
for j in range(len(syncStateIn['Omegabetain'])):
		tempo = solveLinStabSingle(syncStateIn['Omegabetain'][j], syncStateIn['taubetain'][j], tauf, Kvco1, Kvco1, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, syncStateIn['betain'][j], digital, maxp, expansion)
		# print(tempo['OmegStab'])

		if tempo['ImMax']!=999.0:
			results1.append( [ tempo['ReMax'][0], tempo['ImMax'][0] ] )
			if len(tempo['OmegStab'])!=0:
				results1Stab.append( [  tempo['OmegStab'][0],  tempo['tauStab'][0],  tempo['betaStab'][0]])
			if len(tempo['OmegUnst'])!=0:
				results1Unst.append( [  tempo['OmegUnst'][0],  tempo['tauUnst'][0],  tempo['betaUnst'][0]])


for j in range(len(syncStateAnti['taubetaanti'])):
		tempo = solveLinStabSingle(syncStateAnti['Omegabetaanti'][j], syncStateAnti['taubetaanti'][j], tauf, Kvco1, Kvco1, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, syncStateAnti['betaanti'][j], digital, maxp, expansion)
		results2.append( [ tempo['ReMax'][0], tempo['ImMax'][0] ] )
		if len(tempo['OmegStab'])!=0:
			results2Stab.append( [  tempo['OmegStab'][0],  tempo['tauStab'][0],  tempo['betaStab'][0]])
		if len(tempo['OmegUnst'])!=0:
			results2Unst.append( [  tempo['OmegUnst'][0],  tempo['tauUnst'][0],  tempo['betaUnst'][0]])


results1	 	  = np.asarray(results1);
results1Stab	  = np.asarray(results1Stab);
results1Unst	  = np.asarray(results1Unst);

# print(results0[:,1])
sigmain    = results1[:,0];
gammain    = results1[:,1];

OmegStabin = results1Stab[:,0];
tauStabin  = results1Stab[:,1];
betaStabin = results1Stab[:,2];

OmegUnstin = results1Unst[:,0];
tauUnstin  = results1Unst[:,1];
betaUnstin = results1Unst[:,2];


results2	  	  = np.asarray(results2);
results2Stab	  = np.asarray(results2Stab);
results2Unst	  = np.asarray(results2Unst);
# print(results0[:,1])
sigmaanti    = results2[:,0];
gammaanti    = results2[:,1];

OmegStabanti = results2Stab[:,0];
tauStabanti  = results2Stab[:,1];
betaStabanti = results2Stab[:,2];

OmegUnstanti = results2Unst[:,0];
tauUnstanti  = results2Unst[:,1];
betaUnstanti = results2Unst[:,2];

# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
#*******************************************************************************
fig         = plt.figure()
ax          = fig.add_subplot(111)
if digital == True:
	plt.title(r'digital case for $\omega$=%.3f' %wmean);
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'analog case for $\omega$=%.3f' %wmean);
# adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
plt.xlabel(r'$\tau$', fontsize=18)
plt.ylabel(r'$\Omega$', fontsize=18)

# #*******************************************************************************

# print(phasediff(wmean,K, tauf,  digital, maxp, Dw, inphase))
# choose phase or anti-phase synchroniazed states,

# # print( syncState['taubetain'])
# [lineOmeginphase] 	  = ax.plot( syncStateInphase['taubetain'], syncStateInphase['Omegabetain'],'.', linewidth=1, color='blue', label=r'Inphase' )
#
# [lineOmegantiinphase] = ax.plot(syncStateAntiphase['taubetaanti'], syncStateAntiphase['Omegabetaanti'],'.', linewidth=1, color='red', label=r'Antinphase' )
# [lineOmegStabinphase] 	  = ax.plot( tauStabin, OmegStabin, '.', ms=5, color='blue', label=r'Inphase' )
[lineOmegUnstinphase] 	  = ax.plot( tauUnstin, OmegUnstin, '.', ms=1, color='blue', label=r'Inphase' )

# [lineOmegantiinphase] 	  = ax.plot( tauStabanti, OmegStabanti,'.', ms=5, color='red', label=r'Antinphase' )
[lineOmegantiinphase] 	  = ax.plot( tauUnstanti, OmegUnstanti,'.', ms=1, color='red', label=r'Antinphase' )
#
# fig1         = plt.figure()
# ax1          = fig1.add_subplot(111)
# plt.title(r'The phase difference vs $\tau$');
# plt.grid()
# plt.xlabel(r'$\tau$', fontsize=18)
# plt.ylabel(r'$\beta$', fontsize=18)
#
#
# [lineBetainphase] 	  	  = ax1.plot(np.array(tauStabin)*np.array(tauStabin)/(2.0*np.pi), np.mod(betaStabin,2.0*np.pi), '.', ms=10, color='blue', label=r'Inphase' )
# [lineBetaantiphase] 	  = ax1.plot(np.array(tauUnstin)*np.array(OmegUnstin)/(2.0*np.pi), np.mod(betaUnstin,2.0*np.pi), '.', ms=1, color='blue', label=r'Inphase' )

#
# [lineBetaantiinphase] = ax1.plot(tauStabanti, np.mod(betaStabanti,2.0*np.pi), '.', ms=10, color='red', label='Antinphase')
# [lineBetaantiinphase] = ax1.plot(tauUnstanti, np.mod(betaUnstanti,2.0*np.pi), '.', ms=1, color='red', label='Antinphase')

#
fig2         = plt.figure()
ax2          = fig2.add_subplot(111)
plt.title(r'The phase difference vs $\tau$');
plt.grid()
plt.xlabel(r'$\tau$', fontsize=18)
plt.ylabel(r'$\sigma$', fontsize=18)

# # solveLinStabbetain(Omegabetain, taubetain, tauf, Kvco1, Kvco1, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, betain, digital, maxp, expansion):
#


[sigmaIn] 	  	= ax2.plot( syncStateIn['Omegabetain']*syncStateIn['taubetain']/(2.0*np.pi), sigmain, '.', markersize=10, color='blue', label='Inphase')









# [gammaIn] 	  	= ax2.plot( syncStateIn['Omegabetain']*syncStateIn['taubetain']/(2.0*np.pi), solveLinStabbetain(syncStateIn['Omegabetain'], syncStateIn['taubetain'], tauf, Kvco1, Kvco1, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, syncStateIn['betain'], digital, maxp, expansion)['ImbetainMax'], '.', markersize=10, color='red', label='Inphase')

# [lineSigmaAnti] = ax2.plot( syncStateAnti['Omegabetaanti']*syncStateAnti['taubetaanti']/(2.0*np.pi), solveLinStabbetaanti(syncStateAnti['Omegabetaanti'], syncStateAnti['taubetaanti'], tauf, Kvco1, Kvco1, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, syncStateAnti['betaanti'], digital, maxp, expansion)['RebetaantiMax'], '.', color='red', label='Antinphase')


#
# ax.legend()
# ax1.legend()
# ax2.legend()
# ax3.legend()
plt.show()
