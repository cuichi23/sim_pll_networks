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
from Bode_lib import solveLinStab, globalFreq, linStabEq, analytical,HopenloopMutuallyCoupledOnePLL,K,LoopGainSteadyState
from Bode_lib import PhaseopenloopMutuallyCoupledOnePLL,PhaseclosedloopMutuallyCoupledOnePLL,HclosedloopMutuallyCoupledOnePLL
from Bode_lib import HopenloopMutuallyCoupledNet, PhaseopenloopMutuallyCoupledNet,HclosedloopMutuallyCoupledNet, PhaseclosedloopMutuallyCoupledNet, HopenloopMutuallyCoupledNet2
from Bode_lib import GainMarginMutuallyCoupledNet,PhaseMarginMutuallyCoupledNet
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# plot parameter
''' All plots in latex mode '''
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
axisLabel = 50;
titleLabel= 10;
dpi_val   = 150;
figwidth  = 10;
figheight = 5;

# ADD PLOTS
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'

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
w1    	= 2.0*np.pi;	# intrinsic	frequency
Kvco    = 2.0*np.pi*(0.8);	# Sensitivity of VCO
AkPD	= 1.0					# amplitude of the PD -- the factor of 2 here is related to the IMS 2020 paper, AkPD as defined in that paper already includes the 0.5 that we later account for when calculating K
GkLF	= 1.0
Gvga	= 1.0
# Ga1     = 1.0/1.0				# Gain of the first adder
order	= 1.0					# the order of the Loop Filter
tauf    = 0.0					# tauf = sum of all processing delays in the feedback
tauc	= 1.0/(2.0*np.pi*0.014);  # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
v		= 1					# the division
c		= 0.63*3E8				# speed of light
maxp 	= 20.5
INV		= 0.0*np.pi
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
dp		= 0.0015;
f             = np.logspace(-8.0, 15.6*1E-1, num=int((tauf+maxp)/dp), endpoint=True, base=10.0)

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
# PhaseMarginin		= (np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'],
# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))/np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'],
# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))
#
# PhaseMarginanti		= (np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'],
# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))/np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'],
# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))
# print(np.argwhere(Hcl(w, wref, wvco, tau, tau_f, v, tau_c, K)<0.999999*Hcl(w_cutoff, wref, wvco, tau, tau_f, v, tau_c, K))[0])
# print(np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K)))<-3.001)[0])
# print(np.where(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K)))>-2.9999999999999 and 20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K)))<-3.000000000000001))
axis_color  = 'lightgoldenrodyellow'
# print(f)
HolNetOnePLL=[]; distance=[]; taudistance=[]; omegdistance=[];
results1 = []; results2 = [];results=[];taustabanti=[];taustabin=[];omegastabanti=[];omegastabin=[];
fgain=[]; taugain=[]; omegagain=[];
# print(HopenloopMutuallyCoupledNet(2.0*np.pi*1, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'],
# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))
# for j in range(len(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'])):
# 	for i in range(len(f)):
#
# 		results2.append( [f[i], globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j], np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))   ])
# 		if np.abs(np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))<=0.001 and np.abs(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))>=0.1:
# 			# print('a')
# 			gain.append(1.0-np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
# 			globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))
# 			taugain.append(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j])
#
# results2	  = np.asarray(results2);
# # print(results0[:,1])
# results2[:,0] = results2[:,0];
# results2[:,1] = results2[:,1];
# results2[:,2] = results2[:,2];
# results2[:,3] = results2[:,3];
# print('Hallo')
sec=[];
for j in range(len(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'])):
	for i in range(len(f)):
		# print('OK',i,j)
		# if np.abs(np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
			# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))<=0.0001:
		results1.append( [f[i],
		globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		np.real(HopenloopMutuallyCoupledNet(f[i]*2.0*np.pi, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
		np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))   ])

		if np.abs(np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i],
		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))<=0.05 and np.real(HopenloopMutuallyCoupledNet(f[i]*2.0*np.pi,
		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital) ) >1.0025 and globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j]!=0.0:

			taustabin.append(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
			omegastabin.append(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j]/(2.0*np.pi))

			#
# for j in range(len(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'])):
# 	for i in range(len(f)):
# 		# print('OK',i,j)
# 		# if np.abs(np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
# 			# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))<=0.0001:
# 		results2.append( [f[i],
# 		globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j],
# 		globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
# 		np.real(HopenloopMutuallyCoupledNet(f[i]*2.0*np.pi, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 		np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))   ])
#
#
# 		if np.abs(np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j],
# 		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))<=0.05 and np.real(HopenloopMutuallyCoupledNet(f[i]*2.0*np.pi,
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j],
# 		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital) ) >1.025 and globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j]!=0.0:
#
# 			taustabanti.append(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j])
# 			omegastabanti.append(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j]/(2.0*np.pi))


#

	# if np.abs(np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
	# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))<=0.0005 and np.abs(1.0-np.real(HopenloopMutuallyCoupledNet(f[i]*2.0*np.pi, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
	# 	globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))) <=0.0005:
	# taugain.append(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
	# 	fgain.append(i)
			# print(i,j)

		# if np.abs(np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))<=0.0005:# and np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		# # globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)) <=1.01:
		# # 	# print('a')
		# 	gain.append(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))
		# 	taugain.append(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
		# 	omegagain.append(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j]/(2.0*np.pi))

		# print(f[i],np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))
	#
results1	  = np.asarray( results1 );

results1[:,0] = results1[:,0];
results1[:,1] = results1[:,1];
results1[:,2] = results1[:,2]/(2.0*np.pi);
results1[:,3] = results1[:,3];
results1[:,4] = results1[:,4];

# results2	  = np.asarray( results2 );
#
# results2[:,0] = results2[:,0];
# results2[:,1] = results2[:,1];
# results2[:,2] = results2[:,2]/(2.0*np.pi);
# results2[:,3] = results2[:,3];
# results2[:,4] = results2[:,4];

# print( results1[:,0], results1[:,1], results1[:,3] )
# print( len( results1[:,0] ), len( results1[:,1] ), len( taugain ) )
# tempresults1 = results1[:,3].reshape( int( len( f ) ), int( len( taugain ) ) )
# print(taustabin)
# print(np.array(taustabin)*np.array(omegastabin))
# print(taustabanti)
# # print(results1[:,1],results1[:,2],results1[:,3])
# print(results1[:,1]*results1[:,2])
# print( tempresults1, len( globalFreq( w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV )['tau'] ) )
#
#
#
print('Hallo 2')
#

# fig1 = plt.figure()
# ax1 = plt.axes(projection='3d')
# plt.title(r'Inphase');
# # plt.ylabel(r'$Re(H(j\omega))$'); plt.zlabel(r'$Im(H(j\omega))$');plt.xlabel(r'$\tau$')
# ax1.set_xlabel(r'$\tau$')
# ax1.set_ylabel(r'$Re(H(j\omega)$')
# ax1.set_zlabel(r'$Im(H(j\omega))$')
#
# ax1.plot3D(results1[:,1], results1[:,2], results1[:,3])
#
# fig2 = plt.figure()
# ax2 = plt.axes(projection='3d')
# plt.title(r'Antiphase');
# # plt.ylabel(r'$Re(H(j\omega))$'); plt.zlabel(r'$Im(H(j\omega))$');plt.xlabel(r'$\tau$')
# ax2.set_xlabel(r'$\tau$')
# ax2.set_ylabel(r'$Re(H(j\omega)$')
# ax2.set_zlabel(r'$Im(H(j\omega))$')
#
# ax2.plot3D(results2[:,1], results2[:,2], results2[:,3])
#

		# if (np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f[i], globalFreq(wref, w1, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
		# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital).any() )==0.0) and (np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j],
		# globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital).any() )==1.0):
		# 	distance.append((np.sqrt( (1.0-np.real(HolNetOnePLL) )**2 +np.imag(HolNetOnePLL)**2 ) ) )
		# 	taudistance.append(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][j])
		# 	omegdistance.append(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][j])
# f = logspace(-3,3) # frequencies from 10**1 to 10**5
# print(fM2(2.0*np.pi*f))
# fig    = plt.figure(num=0, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')

# tau1=	25.9
# print(globalFreq(wref, w1,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1]/(2*np.pi))
# fig1= plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
# # print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
# ax1= plt.subplot(211)
# fig1.suptitle(r'magnitude and phase-response of a open loop $H_{ol}$ of One PLL in a network two mutually coupled PLLs')
# # plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# # 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
#
# plt.title(r' $f_{\Omega}=%0.2f$Hz, $\tau=%.2f$s'%(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1]/(2.0*np.pi),tau1), fontsize=8)
#
# plt.ylabel('dB', fontsize=axisLabel)
# plt.xscale('log');
#
#
# 										#
# [lineol00] 	=	ax1.plot(f, 20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
# 				tau1,
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
#
# # fig2, (, ) = plt.subplots(2)
#
# # print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
#
# ax2 = plt.subplot(212)
# plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
# plt.xscale('log')
#
# # [line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K), label=r'phase-response', lineWidth=2.5);
# [lineol01] 	= 	ax2.plot(f, fopenloMutC(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
# 				tau1,
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
# fig1.set_size_inches(18,8.5)
# # plt.savefig('Plots/Bode_Plot_OnePLL_tau%.2f_tauc%.4f_%d_%d_%d.pdf' %(tau1, tauc, now.year, now.month, now.day), dpi=dpi_value)
# # plt.savefig('Plots/Bode_Plot_OnePLL_tau%.2f_tauc%.4f_%d_%d_%d.png' %(tau1, tauc, now.year, now.month, now.day), dpi=dpi_value)
# #
# #
# #
# # # ************************************************************************************************************************************
# #,
# # # ************************************************************************************************************************************
# #
# fig3= plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # # fig2, (, ) = plt.subplots(2)
# #
# #
# ax4= plt.subplot(211)
# # tau1=1e-3
# fig3.suptitle(r'magnitude and phase-response of open loop $H_{ol,NET}$ of a network two mutually coupled PLLs')
# plt.title(r' $f_{\Omega}=%0.2f$Hz, $\tau=%.2f$s'%(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1]/(2.0*np.pi),tau1), fontsize=8)
#
# # plt.title(r'loop bandwidth LB=%0.5E and Loop Gain= %0.5E for K=%0.5E, v=%0.3E, $\omega^c=$%3.0E, $d=$%0.5E' %(f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
# # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)][0],
# # 				tau1,
# # 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))>-3.0)][-1], LoopGainSteadyState(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)][0],
# # 				tau1,
# # 				tauf, v, Kvco, AkPD, GkLF, Gvga, digital),K(Kvco, AkPD, GkLF,Gvga)/(2.0*np.pi),v, w_cutoff/(2.0*np.pi), c*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'][-1]), fontsize=8)
#
# # plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# # 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
# plt.ylabel('dB', fontsize=axisLabel)
# plt.xscale('log');
#
# # print('tau=',globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][-1])
#
# print('tau=',globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1]/(2*np.pi))
# print('Omage_tau=',globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1]/(2*np.pi))
# print('tau=',tau1)
# 										#
# [linecl04] 	=	ax4.plot(f, 20.0*log10(abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
# 				tau1,
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))),'-', label=r'magnitude,',color='red');
# # print('Loop bandwidth=',f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
# # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)][0],
# # 				tau1,
# # 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))>-3.0)][-1])
#
# plt.grid();
#
# plt.legend();
# ax5 = plt.subplot(212)
# plt.xlabel(r'f in [rad Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
# plt.xscale('log')
# #
# [linecl05] 	= 	ax5.plot(f, fopenloMutCNet(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
# 				tau1,
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'phase-response',  color='red');
#
#
#
# #
# plt.grid();
#
# plt.legend();
#
# fig3.set_size_inches(18,8.5)
# # plt.savefig('Plots/Bode_Plot_Network_tau%.2f_tauc%.4f_%d_%d_%d.pdf' %(tau1, tauc, now.year, now.month, now.day), dpi=dpi_value)
# # plt.savefig('Plots/Bode_Plot_Network_tau%.2f_tauc%.4f_%d_%d_%d.png' %(tau1, tauc, now.year, now.month, now.day), dpi=dpi_value)
#
#
#
# fig4= plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # # fig2, (, ) = plt.subplots(2)
# #
# #
# ax5= plt.subplot(111)
# # tau1=1e-3
# fig4.suptitle(r'NyquistPlot')
# plt.title(r' $f_{\Omega}=%0.2f$Hz, $\tau=%.2f$s'%(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1]/(2.0*np.pi),tau1), fontsize=8)
#
# # plt.title(r'loop bandwidth LB=%0.5E and Loop Gain= %0.5E for K=%0.5E, v=%0.3E, $\omega^c=$%3.0E, $d=$%0.5E' %(f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
# # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)][0],
# # 				tau1,
# # 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)))>-3.0)][-1], LoopGainSteadyState(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)][0],
# # 				tau1,
# # 				tauf, v, Kvco, AkPD, GkLF, Gvga, digital),K(Kvco, AkPD, GkLF,Gvga)/(2.0*np.pi),v, w_cutoff/(2.0*np.pi), c*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'][-1]), fontsize=8)
#
# # plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# # 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
# # plt.ylabel('dB', fontsize=axisLabel)
# # plt.xscale('log');
#
# 				#
# [linecl04] 	=	ax5.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
# 				tau1,
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)), np.imag(  HopenloopMutuallyCoupledNet(2.0*np.pi*f,
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
# 				tau1,
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital) ) )
# # ),'-', label=r'magnitude,',color='red');
#
# # plt.grid();
# # plt.legend();
# # #
# # ax6 = plt.subplot(212)
# # [linecl04] 	=	ax6.plot(np.real(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
# # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
# # 				tau1,
# # 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)), np.imag(  HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
# # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
# # 				tau1,
# # 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital) ) )
# # ),'-', label=r'magnitude,',color='red');
#
# plt.grid();
# plt.legend();
# fig3.set_size_inches(18,8.5)
# plt.savefig('Plots/NyquistPlot_tau%.2f_tauc%.4f_%d_%d_%d.pdf' %(tau1, tauc, now.year, now.month, now.day), dpi=dpi_value)
# plt.savefig('Plots/NyquistPlot_tau%.2f_tauc%.4f_%d_%d_%d.png' %(tau1, tauc, now.year, now.month, now.day), dpi=dpi_value)
#
# #  ************************************************************************************************************************************
#
# fig4= plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
#
#
# #fig         = plt.figure(figsize=(figwidth,figheight))
# ax          = fig.add_subplot(111)
#
fig5= plt.figure(num=3, figsize=(figwidth, figheight))
# # # fig2, (, ) = plt.subplots(2)
# #
# #
ax6= plt.subplot(111)
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=axisLabel);
plt.ylabel(r'$\frac{2\pi\sigma}{\omega}$',rotation=0, fontsize=85, labelpad=30)
plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60,labelpad=-5)
# plt.ylabel(r'$\frac{\Omega}{\omega}$',rotation=0, fontsize=85, labelpad=30)
ax6.tick_params(axis='both', which='major', labelsize=35, pad=1)
#
ax6.set_xlim(0.0, 0.80)
[lineSigmaIn] = ax6.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
		np.asarray(2.0*np.pi/w1)*solveLinStab(wref, w1, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'],
		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
		tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV )['ReMax'],
	    '-',ms=1, color='red', label=r'Inphase Stable $Re(\lambda)$')

# [lineSigmaIn] = ax6.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'],
# 		np.asarray(2.0*np.pi/w1)*solveLinStab(wref, w1, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
# 		tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV )['ReMax'],
# 	    '-',ms=1, color='red', label=r'Inphase Stable $Re$(\lambda)$')


[lineSigmaAnti2] = ax6.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'],
		np.asarray(2.0*np.pi/w1)*solveLinStab(wref, w1, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'],
		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'],
		tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state2, expansion, INV )['ReMax'],
	 	'-',ms=1, color='blue', label=r'Antiphase Stable $\sigma=Re(\lambda)$')




# [l1inecl06] 	=	ax6.plot(np.asarray(taugain)*np.asarray(omegagain),
# 							gain, '.',  label=r'Antiphase $1-Re(H(j\omega))$' )

# [lineGammaIn] = ax6.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
# 		solveLinStab(wref, w1, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
# 		tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV )['ImMax'],'.',ms=1,
# 		color='green', label=r'Inphase Stable $Im$(\lambda)$')

fig5.set_size_inches(20,10)
ax6.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# plt.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
plt.savefig('Plots/sigma_vs_τ_two_mut_%d_%d_%d.pdf' %( now.year, now.month, now.day), dpi=150, bbox_inches=0)
plt.savefig('Plots/sigma_vs_τ_two_mut_%d_%d_%d.png'%( now.year, now.month, now.day), dpi=150, bbox_inches=0)

# plt.savefig('Plots/NyquistPlot.pdf', dpi=150, bbox_inches=0)
# plt.savefig('Plots/NyquistPlot.png', dpi=150, bbox_inches=0)
# fig5.set_size_inches(18,8.5)
# #
# #
# fig6= plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # # # fig2, (, ) = plt.subplots(2)
# # #
# # #
# ax7= plt.subplot(111)
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=axisLabel); plt.ylabel(r'$2\pi\sigma/omega$', fontsize=axisLabel)
# #
# # [linecl06] 	=	ax7.plot(np.asarray(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg']/(2.0*np.pi))*globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'],
# # 				(np.sqrt( (1.0-np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f[-1], globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'],
# # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)) )**2 +(np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'],
# # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)) )**2 )),
# # 				 linewidth=4,  label=r'Antiphase 1-|H(j\omega)|' )
# #
# # 						  # ),'-', label=r'magnitude,',color='red');   2.0*np.pi*f
# # # print(distance[0][0],taudistance)
# # [l1inecl06] 	=	ax7.plot(taugain,
# # 							gain, '.',  label=r'Antiphase $1-Re(H(j\omega))$' )
#
# # # plt.grid();
# # # plt.legend();
# # # #
# # # ax6 = plt.subplot(212)
# # # [linecl04] 	=	ax6.plot(np.real(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
# # # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
# # # 				tau1,
# # # 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)), np.imag(  HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
# # # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
# # # 				tau1,
# # # 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital) ) )
# # # ),'-', label=r'magnitude,',color='red');
# [lineSigmaAnti2] = ax7.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'],
# 		np.asarray(2.0*np.pi/w1)*solveLinStab(wref, w1, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'],
# 		tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state2, expansion, INV )['ReMax'],
# 	 	'-',ms=1, color='red', label=r'Antiphase Stable $\sigma$=Re$(\lambda)$')
# # [lineGammaAnti2] = ax7.plot(np.asarray(w1/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'],
# # 		solveLinStab(wref, w1, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'],
# # 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'],
# # 		tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state2, expansion, INV )['ImMax'],'.', ms=1,
# # 		color='green', label=r'Antiphase Stable $\gamma$=Re$(\lambda)$')
# #
# # plt.grid();
# # plt.legend();
# # # # fig6.set_size_inches(18,8.5)
# # # #
# # # fig8= plt.figure(num=5, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # # # # fig2, (, ) = plt.subplots(2)
# # # #
# # # #
# # # ax8= plt.subplot(111)
# # # [linecl08] 	=	ax8.plot(np.asarray(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg']/(2.0*np.pi))*globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'],
# # # 				PhaseMarginanti ,
# # # 				linewidth=4,  label=r'Antiphase 1-|H(j\omega)|' )
# # # [linecl09] 	=	ax8.plot(np.asarray(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg']/(2.0*np.pi))*globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'],
# # # 				np.abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'],
# # # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)) ,
# # # 				linewidth=4,  label=r'Antiphase 1-|H(j\omega)|' )
# # #
# # #
# # #
# # #
# # # [linecl09] 	=	ax8.plot(np.asarray(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg']/(2.0*np.pi))*globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'],
# # # 				(	PhaseopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'],
# # # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)) ,
# # # 				linewidth=4,  label=r'Antiphase 1-|H(j\omega)|' )				#PhaseopenloopMutuallyCoupledNet(w, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)
# # #
# # #

# fig8= plt.figure(num=7, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
#
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#
# # ax8= plt.subplot(111)
# fig, axs = plt.subplots(2, 2)
# fig.suptitle(r'Nyquist Plot', fontsize=titleLabel)
# plt.rcParams['axes.labelsize'] = 18
# # axs[0, 0].plot(x, y)
# axs[0, 0].set_title(r'τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0]))
# # axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title(r'τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8]))
# # axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title(r'τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][9]))
# # axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title(r'τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][15]))
#
# # ax1.xaxis.set_tick_params(labelsize=18)
# # ax1.yaxis.set_tick_params(labelsize=18)
# # ax2.xaxis.set_tick_params(labelsize=18)
# # ax2.yaxis.set_tick_params(labelsize=18)
# # ax3.xaxis.set_tick_params(labelsize=18)
# # ax3.yaxis.set_tick_params(labelsize=18)
# # ax4.xaxis.set_tick_params(labelsize=18)
# # ax4.yaxis.set_tick_params(labelsize=18)
# # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# # fig.suptitle('Sharing x per column, y per row')
# # ax1.plot(x, y)
# # ax2.plot(x, y**2, 'tab:orange')
# # ax3.plot(x, -y, 'tab:green')
# [li1] 	=	axs[0, 0].plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, color= 'blue', label=r'τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0]))
# plt.grid();
# plt.legend();
# [li2] 	=	axs[0, 1].plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][8],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][8],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, color= 'green', label=r'τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8]))
# plt.grid();
# plt.legend();
# [li3] 	=	axs[1, 0].plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][9],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][9], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][9],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][9], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, color= 'orange', label=r'τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][9]))
# plt.grid();
# plt.legend();
# [li4] 	=	axs[1, 1].plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][15],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][15], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][15],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][15], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, color= 'purple', label=r'τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][15]) )
# plt.grid();
# plt.legend();
# [linl1] =	axs[0, 0].plot(np.real(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'--',ms=1, color= 'blue')
# plt.grid();
# plt.legend();
# [linl2] =	axs[0, 1].plot(np.real(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][8],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][8],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'--',ms=1, color= 'green')
#
# plt.grid();
# plt.legend();
# [linl3] =	axs[1, 0].plot(np.real(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][9],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][9], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][9],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][9], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'--',ms=1, color= 'orange')
# plt.grid();
# plt.legend();
# [linl4] =	axs[1, 1].plot(np.real(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][15],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][15], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][15],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][15], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'--',ms=1, color= 'purple')
# plt.grid();
# plt.legend();
# for ax in axs.flat:
# 	# plt.xlabel(r'$Re(H(i\gamma))$'); plt.ylabel(r'$Im(H(i\gamma))$', fontsize=axisLabel)
#     ax.set(xlabel=r'$Re(H(i\gamma))$', ylabel=r'$Im(H(i\gamma))$')#, fontsize=axisLabel)
# for ax in fig.get_axes():
# 	ax.label_outer();
# 	ax.grid();
# 	# ax.legend();
#
# fig.set_size_inches(17,9.5)
# plt.savefig('Plots/NyquistPlotAllTogether=%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# plt.savefig('Plots/NyquistPlotAllTogether%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

# plt.show();


bar= [0, 8, 20, 30]
colorbar=['blue','green','orange','purple']

fig9= plt.figure(num=7, figsize=(figwidth, figheight))#, dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
#

# plt.title(r'Nyquist Plot', fontsize=titleLabel)
ax9= plt.subplot(111)
plt.xlabel(r'$Re(H(i\gamma))$', fontsize=axisLabel); plt.ylabel(r'$Im(H(i\gamma))$', fontsize=axisLabel)
# ax9.xaxis.set_tick_params(labelsize=24)
# ax9.yaxis.set_tick_params(labelsize=24)
ax9.tick_params(axis='both', which='major', labelsize=35, pad=1)

 # zoom = 2
# axins.tick_params(axis='both', which='major', labelsize=25, pad=1)
# sub_axes = plt.axes([0.6, 0.6, 0.25, 0.25])
# [linecl10] 	=	ax9.plot(np.asarray(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']/(2.0*np.pi))*globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
# 				PhaseMarginin ,
# 				linewidth=4,  label=r'Antiphase 1-|H(j\omega)|' )
# [linecl11] 	=	ax9.plot(np.asarray(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']/(2.0*np.pi))*globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
# 				np.abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)) ,
# 				linewidth=4,  label=r'Antiphase 1-|H(j\omega)|' )
# #
# for i in range(len(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'])):
# # [linecl10] 	=	ax9.plot(np.asarray(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']/(2.0*np.pi))*globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
# # 				PhaseMarginin ,
# # 				linewidth=4,  label=r'Antiphase 1-|H(j\omega)|' )
# 	[linecl11] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][i],
# 					globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][i], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 					np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][i],
# 					globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][i], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 					'-',ms=1)
#
for i in range(len(bar)):
	# print(bar[i])
	# print(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]])
	[linecl11] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[i]],
					globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
					np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[i]],
					globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
					'-',ms=1, color= colorbar[i], label=r'$\tau=$%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]]))
# [linecl11] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, label=r'τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0]))
# [linecl11] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][10],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][10], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][10],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][10], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, label=r'  τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][10]))
# [linecl11] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][4],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][4], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][4],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][4], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, label=r'  τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][4]))
#
# [linecl11] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][8],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][8],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, label=r'  τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8]))
# [linecl11] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][-1],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][-1], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][-1],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][-1], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, label=r' τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][-1]))

	# [linecl17] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[i]],
	# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
	# 				np.imag(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[i]],
	# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
	# 				'--',ms=1, color= colorbar[i], label=r'-f τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]]))
# [linecl17] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'--',ms=1, label=r'-f τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][0]))
# [linecl18] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][10],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][10], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][10],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][10], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'--',ms=1, label=r' -f τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][10]))
# [linecl19] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][4],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][4], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][4],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][4], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'--',ms=1, label=r'-f  τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][4]))
#
# [linecl10] 	=	ax9.plot(np.real(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][8],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][8],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'--',ms=1, label=r'-f  τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][8]))
# [linecl100] =	ax9.plot(np.real(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][-1],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][-1], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(-2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][-1],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][-1], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'--',ms=1, label=r'-f τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][-1]))

# print( (1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][85]*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][85] )
#
# sub_axes.plot(X_detail, Y_detail, '-',ms=1, color= 'blue')
#
axins = zoomed_inset_axes(ax9, zoom=8, loc=2)# ,bbox_transform=ax9.figure.transFigure)
for i in range(len(bar)):
    	axins.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[i]],
    					globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
    					np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[i]],
    					globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
    					'-',ms=1, color= colorbar[i], label=r'$\tau=$%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]]))

# plt.grid()
# # sub region of the original image
x1, x2, y1, y2 = -0.3, 1.42, -0.5, 0.5
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)


# axins2 = inset_axes(axins, width=6, height=3, loc=2)
# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax9, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.xticks(visible=True)
plt.yticks(visible=True)
plt.grid();
# plt.legend();
ax9.legend(bbox_to_anchor=(0.7415,0.75),fontsize=36)
# plt.legend(bbox_to_anchor=(0.7415,0.75),fontsize=36)
fig9.set_size_inches(20,10)
plt.savefig('Plots/NyquistPlot_τ=%.5f_%d_%d_%d.pdf' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]], now.year, now.month, now.day), dpi=150, bbox_inches=0)
plt.savefig('Plots/NyquistPlot_τ=%.5f_%d_%d_%d.png'%(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]], now.year, now.month, now.day), dpi=150, bbox_inches=0)

plt.show();
# plt.close(fig9)
# fig10= plt.figure(num=7, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#


# ax10= plt.subplot(111)
# plt.xlabel(r'$Re(H(i\gamma))$', fontsize=axisLabel); plt.ylabel(r'$Im(H(i\gamma))$', fontsize=axisLabel)
# # [linecl10] 	=	ax9.plot(np.asarray(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']/(2.0*np.pi))*globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
# # 				PhaseMarginin ,
# # 				linewidth=4,  label=r'Antiphase 1-|H(j\omega)|' )
# # [linecl11] 	=	ax9.plot(np.asarray(globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']/(2.0*np.pi))*globalFreq(w1, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
# # 				np.abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'],
# # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)) ,
# # 				linewidth=4,  label=r'Antiphase 1-|H(j\omega)|' )
# #
#
# [linecl14] 	=	ax10.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][0], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][0], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, label=r'Antiphase τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][0]))
# [linecl14] 	=	ax10.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][10],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][10], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][10],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][10], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, label=r'Antiphase  τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][10]))
# [linecl14] 	=	ax10.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][40],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][40], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][40],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][40], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, label=r'Antiphase  τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][40]))
#
# [linecl14] 	=	ax10.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][85],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][85], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][85],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][85], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, label=r'Antiphase  τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][85]))
# [linecl14] 	=	ax10.plot(np.real(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][-1],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][-1], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				np.imag(HopenloopMutuallyCoupledNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['Omeg'][-1],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][-1], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)),
# 				'-',ms=1, label=r'Antiphase τ=%.5f' %(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state2, INV)['tau'][-1]))
# plt.grid();
# plt.legend();
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
# ax7 = plt.subplot(212)
# plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
# plt.xscale('log')
#
# # [line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K), label=r'phase-response', lineWidth=2.5);N
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
#
#
#
#
#  ************************************************************************************************************************************
# #
# fig6= plt.figure(num=6, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
# #
# ax10= plt.subplot(211)
# fig6.suptitle(r'magnitude and phase-response of closed loop $H_{cl,NET}$ of a network two mutually coupled (4gen) PLLs')
# # plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# # 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
# plt.ylabel('GainMargin', fontsize=axisLabel)
# plt.xscale('linear')
#
# plt.yscale('linear')
# 									#
#
# print('tau=',globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'][-1])
# print(np.argwhere(fopenloMutCNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)>-179.0))
# print(fopenloMutCNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital))
# [lineOmegStabIn] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 							solveLinStab(wref, w1, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 								globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 								tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, digital, maxp, sync_state1, expansion)['ReMax']/np.asarray(w1),
# 				 					'o',ms=1, color='purple',  label=r'Inphase Stable')
# #
# # [GainMargin] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# # 				GainMarginMutuallyCoupledNet(2*np.pi*f[0],
# # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# # 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# # 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'$f=%0.1f$,' %(f[0]),color='red' );
# [GainMargin2] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				GainMarginMutuallyCoupledNet(2*np.pi*f[1],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'$f=%0.5E$,' %(f[1]),color='green' );
#
# [GainMargin3] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				GainMarginMutuallyCoupledNet(2*np.pi*f[2],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'$f=%0.5E$,'%(f[2]),color='blue');
# [GainMargin4] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				GainMarginMutuallyCoupledNet(2*np.pi*f[3],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'$f=%0.5E$,' %(f[3]),color='black');
# [GainMargin5] = ax10.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				GainMarginMutuallyCoupledNet(2*np.pi*f[4],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'$f=%0.5E$,' %(f[4]),color='yellow' );
#
# #
# [vlinecl01]	=	ax.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# [vlinecl02]	=	ax.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')
#
# plt.grid();
#
# plt.legend();
# ax11 = plt.subplot(212)
# plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
# plt.xscale('log')
#
# [line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, v, tau_c, K), label=r'phase-response', lineWidth=2.5);
#
# [PhaseMargin] = ax11.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
#
# 			GainMarginMutuallyCoupledNet(w[np.argwhere(fclosedloMutCNet(2.0*np.pi*f, globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 			 		globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 					tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital)<-179.0)][0],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 				globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital),'.', label=r'magnitude,',color='red');
#
#
# [vlinecl03]	=	ax0.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# [vlinecl04]	=	ax0.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')
#
# plt.grid();
#
# plt.legend();
#
# # #
# # #
# # #
# #
# #
# #




# ************************************************************************************************************************************
#
# fig7= plt.figure(num=7, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
# tau1=20e-2
#
# ax12= plt.subplot(211)
# fig7.suptitle(r'magnitude and phase-response of closed loop $H_{cl,NET}$ of a network two mutually coupled (4gen) PLLs')
# # plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
# # 															wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, v, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K)), fontsize=8)
# plt.ylabel('dB', fontsize=axisLabel)
# plt.xscale('linear')
#
# plt.yscale('linear')
									#
# print(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'])
# print(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'])
#
# print(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)[-1][-1]])
# print(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']<tau1)[-1][-1]])
#

#
# [Real] = ax12.plot(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau']*globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 					1-HopenloopMutuallyCoupledNet(2*np.pi*60e6,
# 					globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['Omeg'],
# 					globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1)['tau'],
# 					tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital).real,
# 				'.', label=r'magnitude,',color='red');


# [vlinecl01]	=	ax.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]],
# 							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, v, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
# 							'k--',linestyle='--', label=r'$\omega_{gc}$')
# [vlinecl02]	=	ax.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]],
# 							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, v, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))[0]]),
# 							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')

# plt.grid();
# plt.legend();
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
