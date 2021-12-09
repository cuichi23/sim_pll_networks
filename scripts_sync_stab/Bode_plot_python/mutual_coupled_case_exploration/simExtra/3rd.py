from pylab import *
import sympy
from numpy import pi, sin
import numpy as np
import sympy
from scipy import signal
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()
from itertools import count
import Bode_lib
# from Bode_lib import Holnodel,Hcl,Holwithdel,fMolnodel,fMcl, fMolwithdel
from Bode_lib import solveLinStab, globalFreq, linStabEq,analytical,HopenloopMutuallyCoupledOnePLL,K,LoopGainSteadyState
from Bode_lib import PhaseopenloopMutuallyCoupledOnePLL,PhaseclosedloopMutuallyCoupledOnePLL,HclosedloopMutuallyCoupledOnePLL
from Bode_lib import HopenloopMutuallyCoupledNet, PhaseopenloopMutuallyCoupledNet,HclosedloopMutuallyCoupledNet, PhaseclosedloopMutuallyCoupledNet
from Bode_lib import GainMarginMutuallyCoupledNet,PhaseMarginMutuallyCoupledNet


# plot parameter
axisLabel = 12;
titleLabel= 10;
dpi_val	  = 150;
figwidth  = 6;
figheight = 5;

#
# w1    	  	= 2.0*np.pi*60E9; 			# intrinsic	frequency
# Kvco      	= 2.0*np.pi*(0.1E9); 		# Sensitivity of VCO
# AkPD	  	= 0.162*2.0					# Amplitude of the output of the PD --
# GkLF		= 1.0
# Gvga	  	= 0.5					# Gain of the first adder
# tauf 	  	= 0.0						# tauf = sum of all processing delays in the feedback
# order 	  	= 1.0						# the order of the Loop Filter
# tauc	  	= 1.0/(2.0*np.pi*100E6);
#
#

#
w1    	= 2.0*np.pi*24.25E9;	# intrinsic	frequency
Kvco    = 2.0*np.pi*(757.64E6);	# Sensitivity of VCO
AkPD	= 0.8					# amplitude of the PD -- the factor of 2 here is related to the IMS 2020 paper, AkPD as defined in that paper already includes the 0.5 that we later account for when calculating K
GkLF	= 1.0
Gvga	= 1.0
# Ga1     = 1.0/1.0				# Gain of the first adder
order	= 2.0					# the order of the Loop Filter
tauf    = 0.0					# tauf = sum of all processing delays in the feedback
tauc	= 1.0/(2.0*np.pi*0.965E6);  # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
v		= 32;					# the division
c		= 0.63*3E8				# speed of light
maxp 	= 1500
INV		= 1.0*np.pi
#
#
wref=w1

# digital = True;



# PLL parameters and state information
# wref          = 2.0*np.pi*60E9;
# wvco          = 2.0*np.pi*60E9;                                         # global frequency
# tau1		  = 0.0*1E-10;
# tau_f         = 0.0;
# # beta          = np.pi/2.0;
# G             = 1.0;
# Kvco          = 2.0*np.pi*0.25;
# v           =	1024.0;
tau_c         = tauc
# LF_order	  = 1.0;
#
# K             = G*Kvco*0.5;
# # alpha         = K*np.sin(wref*(tau-tau_f)+beta);
w_cutoff      = 1.0/tau_c
# w             = 2.0*np.pi*f


# choose digital vs analog
digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# choose phase or anti-phase synchroniazed states,

sync_state1='inphase';															# choose between inphase, antiphase, entrainment
sync_state2='antiphase';
sync_state3='entrainment';
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, visor of vider

				#2.0*np.pi*24E9;
# K    = 0.15*2.0*np.pi			#2.0*np.pi*250E6;
# tauf = tau_f
# tauc = tau_c;
c	 = 3E8*(2/3);
# maxp = 1e7;
###### python library for bode plots (lti)
# list_of_coefficients
# system        = signal.lti([alpha], [v*tau_c, v, alpha])
dp		= 0.01;
f             = np.logspace(-2.0,12.0, num=int((maxp+tauf)/dp), endpoint=True, base=10.0)
# f             = np.array([0,100e6,500e6,1e9,1.5e9])
#
# w             = 2.0*np.pi*f
# w, mag, phase1 = signal.bode(system,w)
#
#
# print(Hcl(w_cutoff, wref, wvco, tau1, tau_f, v, tau/_c, K))

fopenloMutC 		= np.vectorize(PhaseopenloopMutuallyCoupledOnePLL)
fclosedloMutC 		= np.vectorize(PhaseclosedloopMutuallyCoupledOnePLL)
fopenloMutCNet 		= np.vectorize(PhaseopenloopMutuallyCoupledNet)
fclosedloMutCNet 	= np.vectorize(PhaseclosedloopMutuallyCoupledNet)
# print(np.argwhere(Hcl(w, wref, wvco, tau, tau_f, v, tau_c, K)<0.999999*Hcl(w_cutoff, wref, wvco, tau, tau_f, v, tau_c, K))[0])
# print(np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K)))<-3.001)[0])
# print(np.where(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K)))>-2.9999999999999 and 20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K)))<-3.000000000000001))
axis_color  = 'lightgoldenrodyellow'
# f = logspace(-3,3) # frequencies from 10**1 to 10**5
# print(fM2(2.0*np.pi*f))
# fig    = plt.figure(num=0, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')

tau1=1e-9

fig1= plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)

# print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
ax0= plt.subplot(211)
fig1.suptitle(r'magnitude and phase-response of a open loop $H_{ol}$ of One PLL in a network two mutually coupled (4gen) PLLs')
# plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
plt.ylabel('dB', fontsize=axisLabel)
plt.xscale('log');


										#
[lineol00] 	=	ax0.plot(f, 20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau']<tau1)][0],
				tau1,
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))),'-', label=r'magnitude,',color='red');

#
# [vlinecl01]	=	ax.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# [vlinecl02]	=	ax.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')

plt.grid();

plt.legend();
ax1 = plt.subplot(212)
plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
plt.xscale('log')

# [line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K), label=r'phase-response', lineWidth=2.5);
[lineol01] 	= 	ax1.plot(f, fopenloMutC(2.0*np.pi*f,
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau']<tau1)][0],
				tau1,
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'phase-response',  color='red');



# [vlinecl03]	=	ax0.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# [vlinecl04]	=	ax0.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')

plt.grid();

plt.legend();

#
#
#
# # ************************************************************************************************************************************
#
# fig2= plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
#
# ax2= plt.subplot(211)
# fig2.suptitle(r'magnitude and phase-response of closed loop $H_{cl}$ of One PLL in a network two mutually coupled (4gen) PLLs')
# # plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# # 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
# plt.ylabel('dB', fontsize=axisLabel)
# plt.xscale('log');
#
#
# 										#
# [linecl02] 	=	ax2.plot(f, 20.0*log10(abs(HclosedloopMutuallyCoupledOnePLL(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))),'-', label=r'magnitude,',color='red');
#
# #
# # [vlinecl01]	=	ax.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# # 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# # 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# # [vlinecl02]	=	ax.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# # 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# # 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')
#
# plt.grid();
#
# plt.legend();
# ax3 = plt.subplot(212)
# plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
# plt.xscale('log')
#
# # [line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K), label=r'phase-response', lineWidth=2.5);
# [linecl03] 	= 	ax3.plot(f, fclosedloMutC(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'phase-response',  color='red');
#
#
#
# # [vlinecl03]	=	ax0.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# # 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# # 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# # [vlinecl04]	=	ax0.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# # 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# # 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')
#
# plt.grid();
#
# plt.legend();
#
#
#
#
#
# # ************************************************************************************************************************************
#
fig3= plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
#
ax4= plt.subplot(211)
tau1=1e-3
fig3.suptitle(r'magnitude and phase-response of open loop $H_{ol,NET}$ of a network two mutually coupled (4gen) PLLs')
plt.title(r'loop bandwidth LB=%0.5E and Loop Gain= %0.5E for K=%0.5E, v=%0.3E, $\omega^c=$%3.0E, $d=$%0.5E' %(f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau']<tau1)][0],
				tau1,
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))>-3.0)][-1], LoopGainSteadyState(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau']<tau1)][0],
				tau1,
				tauf, v, Kvco, AkPD, GkLF, Gvga, digital),K(Kvco, AkPD, GkLF,Gvga)/(2.0*np.pi),v, w_cutoff/(2.0*np.pi), c*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau'][-1]), fontsize=8)

# plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
plt.ylabel('dB', fontsize=axisLabel)
plt.xscale('log');

print('tau=',globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau'][-1])
print('Omage_tau=',globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau'][-1])
print('tau=',tau1)
										#
[linecl04] 	=	ax4.plot(f, 20.0*log10(abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau']<tau1)][0],
				tau1,
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))),'-', label=r'magnitude,',color='red');
print('Loop bandwidth=',f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau']<tau1)][0],
				tau1,
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))>-3.0)][-1])

plt.grid();

plt.legend();
ax5 = plt.subplot(212)
plt.xlabel(r'f in [rad Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
plt.xscale('log')
#
[linecl05] 	= 	ax5.plot(f, fopenloMutCNet(2.0*np.pi*f,
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau']<tau1)][0],
				tau1,
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'phase-response',  color='red');



#
plt.grid();

plt.legend();
#
#
#
# #  ************************************************************************************************************************************
#
# fig4= plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
#
# ax6= plt.subplot(211)
# fig4.suptitle(r'magnitude and phase-response of closed loop $H_{cl,NET}$ of a network two mutually coupled (4gen) PLLs')
# # plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# # # 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
# # plt.ylabel('dB', fontsize=axisLabel)
# # plt.xscale('log');
# #
#
# 										#
# [linecl06] 	=	ax6.plot(f, 20.0*log10(abs(HclosedloopMutuallyCoupledNet(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))),'-', label=r'magnitude,',color='red');

# #
# # [vlinecl01]	=	ax.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# # 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# # 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# # [vlinecl02]	=	ax.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# # 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# # 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')
#
# plt.grid();
#
# plt.legend();
# ax7 = plt.subplot(212)
# plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
# plt.xscale('log')
#
# # [line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K), label=r'phase-response', lineWidth=2.5);
# [linecl07] 	= 	ax7.plot(f, fclosedloMutCNet(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'phase-response',  color='red');
#
#
#
# # [vlinecl03]	=	ax0.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# # 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# # 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# # [vlinecl04]	=	ax0.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# # 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# # 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')
#
# plt.grid();
#
# plt.legend();
#
#
#
#
#
#
#
#
# #  ************************************************************************************************************************************
#
# fig5= plt.figure(num=5, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
#
# ax8= plt.subplot(211)
# fig5.suptitle(r'magnitude and phase-response of closed loop $H_{cl,NET}$ of a network two mutually coupled (4gen) PLLs')
# # plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# # 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
# plt.ylabel('dB', fontsize=axisLabel)
# plt.xscale('log');
#
#
# 										#
# [linecl08] 	=	ax8.plot(f, GainMarginMutuallyCoupledNet(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'-', label=r'magnitude,',color='red');
#
# #
# # [vlinecl01]	=	ax.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# # 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# # 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# # [vlinecl02]	=	ax.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# # 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# # 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')
#
# plt.grid();
#
# plt.legend();
# ax9 = plt.subplot(212)
# plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
# plt.xscale('log')
#
# # [line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K), label=r'phase-response', lineWidth=2.5);
# [linecl09] 	= 	ax9.plot(f,  PhaseMarginMutuallyCoupledNet(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'-', label=r'magnitude,',color='red');
#
#
#
# # [vlinecl03]	=	ax0.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# # 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# # 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# # [vlinecl04]	=	ax0.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# # 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# # 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')
#
# plt.grid();
#
# plt.legend();
#
# #
# #
# #
# # #
# # #
# # #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
#
#
#
#
#
#
#
# # ***********************************************************************************************************************************************************************************************************************************************************************************************
#
#
#




#  ************************************************************************************************************************************
# #
fig6= plt.figure(num=6, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)

#
ax10= plt.subplot(211)
# fig6.suptitle(r'magnitude and phase-response of closed loop $H_{cl,NET}$ of a network two mutually coupled (4gen) PLLs')
# plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
plt.ylabel('GainMargin', fontsize=axisLabel)
plt.xscale('linear')

plt.yscale('linear')
									#

print('tau=',globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'][-1])
print(np.argwhere(fopenloMutCNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)>-179.0))
print(fopenloMutCNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))
[lineOmegStabIn] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
							solveLinStab(wref, w1, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
								globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
								tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, digital, maxp, sync_state1, expansion)['ReMax']/np.asarray(w1),
				 					'o',ms=1, color='purple',  label=r'Inphase Stable')
#
# [GainMargin] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				GainMarginMutuallyCoupledNet(2*np.pi*f[0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'$f=%0.1f$,' %(f[0]),color='red' );
[GainMargin2] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
				GainMarginMutuallyCoupledNet(2*np.pi*f[1],
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'$f=%0.5E$,' %(f[1]),color='green' );

[GainMargin3] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
				GainMarginMutuallyCoupledNet(2*np.pi*f[2],
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'$f=%0.5E$,'%(f[2]),color='blue');
[GainMargin4] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
				GainMarginMutuallyCoupledNet(2*np.pi*f[3],
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'$f=%0.5E$,' %(f[3]),color='black');
[GainMargin5] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
				GainMarginMutuallyCoupledNet(2*np.pi*f[4],
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'$f=%0.5E$,' %(f[4]),color='yellow' );

#
[vlinecl01]	=	ax.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
							'k--',linestyle='--', label=r'$\omega_{gc}$')
[vlinecl02]	=	ax.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')

plt.grid();

plt.legend();
ax11 = plt.subplot(212)
plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
plt.xscale('log')

[line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K), label=r'phase-response', lineWidth=2.5);

[PhaseMargin] = ax11.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],

			GainMarginMutuallyCoupledNet(w[np.argwhere(fclosedloMutCNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
			 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
					tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)<-179.0)][0],
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'magnitude,',color='red');


[vlinecl03]	=	ax0.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
							'k--',linestyle='--', label=r'$\omega_{gc}$')
[vlinecl04]	=	ax0.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')

plt.grid();

plt.legend();

# #
# #
# #
#
#
#




************************************************************************************************************************************

fig7= plt.figure(num=7, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)
tau1=20e-2

ax12= plt.subplot(211)
fig7.suptitle(r'magnitude and phase-response of closed loop $H_{cl,NET}$ of a network two mutually coupled (4gen) PLLs')
# plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
plt.ylabel('dB', fontsize=axisLabel)
plt.xscale('linear')

plt.yscale('linear')

print(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'])
print(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'])

print(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)[-1][-1]])
print(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)[-1][-1]])



[Real] = ax12.plot(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
					1-HopenloopMutuallyCoupledNet(2*np.pi*60e6,
					globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
					globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
					tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital).real,
				'.', label=r'magnitude,',color='red');


[vlinecl01]	=	ax.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
							'k--',linestyle='--', label=r'$\omega_{gc}$')
[vlinecl02]	=	ax.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')

plt.grid();
plt.legend();
# ax13 = plt.subplot(212)
# plt.ylabel(r'$\Im(H)$', fontsize=axisLabel); plt.xlabel(r'$\Re(H)$', fontsize=axisLabel)
# plt.xscale('linear')
#
# plt.yscale('linear')
#
#
#
# [nuqyuist] = ax13.plot(HopenloopMutuallyCoupledNet(2*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)[-1][-1]],
# 				tau1,
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital).real,
#
# 			HopenloopMutuallyCoupledNet(2*np.pi*f,
# 							globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)[-1][-1]],
# 							tau1,
# 							tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital).imag,
# 				'.', label=r'magnitude,',color='red');
#
#
# plt.grid();
#
# plt.legend();

plt.show();
