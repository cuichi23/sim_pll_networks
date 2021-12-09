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
import bode_lib
# from Bode_lib import Holnodel,Hcl,Holwithdel,fMolnodel,fMcl, fMolwithdel
from bode_lib import solveLinStab, globalFreq, linStabEq, K, coupfunction

from bode_lib import HopenLoopMutuallyCoupledOnePLL1, LoopGainSteadyState1, PhaseopenloopMutuallyCoupledOnePLL1,  PhaseclosedloopMutuallyCoupledOnePLL1, HclosedLoopMutuallyCoupledOnePLL1, GainMarginMutuallyCoupledOne1, PhaseMarginMutuallyCoupledOne1
from bode_lib import  HopenLoopMutuallyCoupledOnePLL2, LoopGainSteadyState2, PhaseopenloopMutuallyCoupledOnePLL2,  PhaseclosedloopMutuallyCoupledOnePLL2, HclosedLoopMutuallyCoupledOnePLL2, GainMarginMutuallyCoupledOne2, PhaseMarginMutuallyCoupledOne2
from bode_lib import HopenLoopMutuallyCoupledNet, PhaseopenloopMutuallyCoupledNet,HclosedloopMutuallyCoupledNet, PhaseclosedloopMutuallyCoupledNet, HopenLoopMutuallyCoupledNet2
from bode_lib import GainMarginMutuallyCoupledNet,PhaseMarginMutuallyCoupledNet
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

####################################################################################################################################################################################
coupfun='cos'

####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
#
w1   = (1.0 - 0.0*0.02)*2.0*np.pi                                                   # intrinsic frequency of PLL1
w2	 = (1.0 + 0.0*0.02)*2.0*np.pi                                                   # intrinsic frequency of PLL2
wmean=(w1+w2)/2.0

Dw	 = w2-w1
Kvco1    = 2.0*np.pi*(0.8);                                                      # Sensitivity of VCO of PLL1
Kvco2    = 2.0*np.pi*(0.8);                                                      # Sensitivity of VCO of PLL1
AkPD    = 1.0                                                                   # amplitude of the PD -- the factor of 2 here is related to the IMS 2020 paper, AkPD as defined in that paper already includes the 0.5 that we later account for when calculating K
GkLF    = 1.0
Gvga    = 1.0
Ga1     = 1.0                                                                   # Gain of the first adder
Kmean=(K(Kvco1, AkPD, GkLF, Gvga)+K(Kvco2, AkPD, GkLF, Gvga))/2.0
DK	 = K(Kvco2, AkPD, GkLF, Gvga)-K(Kvco1, AkPD, GkLF, Gvga)

order   = 1.0                                                                   # the order of the Loop Filter
tauf    = 0.0                                                                   # tauf = sum of all processing delays in the feedback
tauc1    = 1.0/(2.0*np.pi*0.0146);                                               # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
tauc2    = 1.0/(2.0*np.pi*0.0146);
v       = 1                                                                     # the division
c       = 0.63*3E8                                                              # speed of light
maxp    = 10.5
INV     = 0.0*np.pi
model   = 'Nonlinear'




wref=w
w_cutoff      = 1.0/tauc

# choose digital vs analog
digital = False;

# choose full expression of the characteristic equation vs the expansion of 3d Order, False is the Full, True is the expansion
expansion=False;

# choose phase or anti-phase synchroniazed states,
sync_state1='inphase';                                                          # choose between inphase, antiphase, entrainment
sync_state2='antiphase';
sync_state3='entrainment';

# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, visor of vider

				#2.0*np.pi*24E9;
# K    = 0.15*2.0*np.pi            #2.0*np.pi*250E6;
# tauf = tau_f
# tauc = tau_c;
c     = 3E8*(2/3);
# maxp = 1e7;
###### python library for bode plots (lti)
# list_of_coefficients
# system        = signal.lti([alpha], [v*tau_c, v, alpha])
df 		   = 0.0001;
f  		   = np.logspace(-8, 6, num=int((tauf+maxp)/df), endpoint=True, base=10.0)
dK 		   = 0.001;
Kvcovalues = np.arange(0.0001*2.0*np.pi, 0.8*2.0*np.pi, dK)

Kvcovalues2plot   = [0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi,  0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
tauvalues2plot 	  = [0*w/(2*np.pi), 0.25*w/(2*np.pi), 0.5*w/(2*np.pi), 0.75*w/(2*np.pi), 0.9*w/(2*np.pi), 1.25*w/(2*np.pi)]
delayEntry_forNet = 3;
bar = [];

for i in range(len(tauvalues2plot)):
	tempo = np.where( globalFreq( wmean, Dw, Kmean, DK, tauf,  digital, maxp, sync_state )['tau'][:]>tauvalues2plot[i] )[0][0]
	# print('tempo:', tempo)
	bar.append( tempo )

####################################################################################################################################################################################

PhaseOnePLL        = np.vectorize(PhaseopenloopMutuallyCoupledOnePLL)
PhaseNet           = np.vectorize(PhaseopenloopMutuallyCoupledNet)

# syncState = globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV);

##Here we calculate and save to lists the gain and phase crossover frequencies and gain and phase margin as well as functions of the coupling strength
###########

#
pm_vs_K_singlePLL 		= [];
gm_vs_K_singlePLL 		= [];
Kvalue            		= [];
phaseCrossOver    		= [];
KNetvalue         		= [];
gm_vs_K_Net       		= [];
gcf_vs_K_Net      		= [];
pm_vs_K_Net       		= [];
GM_Net            		= [];
gm_vs_K_singlePLL_Lin	= [];
KOnevalueLin       		= [];
GM_Net_Lin         		= [];
phaseCrossOver_Lin 		= [];
KNetvalue_Lin      		= [];
pm_vs_K_singlePLL_Lin 	= [];
pm_vs_K_Net_Lin			= [];
gcf_vs_K_Net_Lin		= [];

for j in range(len(Kvcovalues)):

	print('Progress:',j,'/',len(Kvcovalues))

	syncState = globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV);

	tempo1 	  = np.where(syncState['tau'][:]>tauvalues2plot[delayEntry_forNet])[0][0]


####################### Single PLL

	gfc_vs_K_singlePLL=f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Nonlinear'))) <=0.0001)][0]

	pm_vs_K_singlePLL.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*gfc_vs_K_singlePLL, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Nonlinear'))


	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Nonlinear')) <=0.7).any():

		pfc_vs_K_singlePLL=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Nonlinear')) <=0.7)][0]
		# print(GM)
		gm_vs_K_singlePLL.append(GainMarginMutuallyCoupledOne(2.0*np.pi*pfc_vs_K_singlePLL, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Nonlinear'))
		Kvalue.append(K(Kvcovalues[j], AkPD, GkLF, Gvga))



####################### Network

	gcf_vs_K_Net.append(f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Nonlinear'))) <=0.0001)][-1] )

	pm_vs_K_Net.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_K_Net[-1], syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Nonlinear'))


	phaseNet = PhaseNet(2.0*np.pi*f, syncState['Omeg'][tempo1],	syncState['tau'][tempo1], tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Nonlinear')
	index1   = np.where(phaseNet >=170)[0][0]
	#print('index1: ', index1)
	#print('phaseNet[index1:]:', phaseNet[index1:])
	index2 = np.where(phaseNet[index1:] <= 0.0)[0][0]
	#print('index2: ', index2)
	GMNet  = f[index1+index2]
	#print('found GMNet:', GMNet, ' at index:', index1+index2)
	#Kvcovalues2plot			= [0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi, 0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
	if ( np.abs(Kvcovalues[j]-Kvcovalues2plot[0]) <= 0.01 and len(phaseCrossOver) == 0 ):
		phaseCrossOver.append(GMNet);
	GM_Net.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
														tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Nonlinear') )
	KNetvalue.append(K(Kvcovalues[j], AkPD, GkLF, Gvga))


	######################################################################################################## LINEAR




	####################### Single PLL
	gfc_vs_K_singlePLL_Lin = f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
																		tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Linear'))) <= 0.0001)][0]

	pm_vs_K_singlePLL_Lin.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*gfc_vs_K_singlePLL_Lin, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
																		tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Linear'))


	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
																		tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Linear')) <=0.7).any():

		pfc_vs_K_singlePLL_Lin=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,	syncState['Omeg'][tempo1], syncState['tau'][tempo1],
																		tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Linear')) <=0.7)][0]

		gm_vs_K_singlePLL_Lin.append(GainMarginMutuallyCoupledOne(2.0*np.pi*pfc_vs_K_singlePLL_Lin, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
																		tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Linear'))

		KOnevalueLin.append(K(Kvcovalues[j], AkPD, GkLF, Gvga))

	####################### Network

	gcf_vs_K_Net_Lin.append(f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Linear'))) <=0.0001)][-1] )

	pm_vs_K_Net_Lin.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_K_Net[-1], syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Linear'))

	phaseNet_Lin = PhaseNet(2.0*np.pi*f, syncState['Omeg'][tempo1],	syncState['tau'][tempo1], tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Linear')
	index1_Lin   = np.where(phaseNet_Lin >=170)[0][0]
	#print('index1: ', index1)
	#print('phaseNet[index1:]:', phaseNet[index1:])
	index2_Lin = np.where(phaseNet_Lin[index1_Lin:] <= 0.0)[0][0]
	#print('index2: ', index2)
	GMNet_Lin  = f[index1_Lin+index2_Lin]
	#print('found GMNet:', GMNet, ' at index:', index1+index2)
	#Kvcovalues2plot			= [0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi, 0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
	if ( np.abs(Kvcovalues[j]-Kvcovalues2plot[0]) <= 0.01 and len(phaseCrossOver_Lin) == 0 ):
		phaseCrossOver_Lin.append(GMNet_Lin);
	GM_Net_Lin.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
														tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, 'Linear') )
	KNetvalue_Lin.append(K(Kvcovalues[j], AkPD, GkLF, Gvga))



# ADD PLOTS
####################################################################################################################################################################################
colorbar=['blue','green','orange','purple', 'cyan', 'black']
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1]/(2*np.pi))
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1])
##################################################################################################################################################################



#########################################################################################################################################
#Bode plots for different K
#########################################################################################################################################


#########################################################################################################################################

fig1= plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig1.set_size_inches(20,10)

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #

ax1= plt.subplot(211)
plt.axhline(y=0, color='r', linestyle='-.')
plt.xscale('log');
for i in range(len(Kvcovalues2plot)):
	[lineol00]     =    ax1.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvcovalues2plot[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[delayEntry_forNet]],
		globalFreq(wref, w, Kvcovalues2plot[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[delayEntry_forNet]],
		tauf, tauc, v, Kvcovalues2plot[i], AkPD, GkLF, Gvga,  digital, model))),'-',ms=1, color= colorbar[i], label=r' K=%.5f' %(K(Kvcovalues2plot[i], AkPD, GkLF, Gvga)/(2.0*np.pi)))

plt.grid();
ax1.set_xlim(f[0], f[-1])
plt.ylabel(r'$20\textrm{log}\left|H_k^\textrm{ OL}(j\gamma)\right|$',  rotation=90, fontsize=36, labelpad=30)
ax1.legend(fontsize=15);
ax1.tick_params(axis='both', which='major', labelsize=25, pad=1)

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #

ax2= plt.subplot(212)
plt.xscale('log')
plt.axhline(y=-180, color='r', linestyle='-.')
					# PhaseclosedloopMutuallyCoupledOnePLL
for i in range(len(Kvcovalues2plot)):
	[lineol00]     =    ax2.plot(f, PhaseOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvcovalues2plot[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[delayEntry_forNet]],
		globalFreq(wref, w, Kvcovalues2plot[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[delayEntry_forNet]],
		tauf, tauc, v, Kvcovalues2plot[i], AkPD, GkLF, Gvga,  digital, model),'.',ms=1, color= colorbar[i], label=r'K=%.5f' %(K(Kvcovalues2plot[i], AkPD, GkLF, Gvga)))

plt.grid();
ax2.set_xlim(f[0], f[-1])
plt.ylabel(r'\angle H_k^\textrm{ \LARGE OL}(j\gamma) ',  rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
ax2.tick_params(axis='both', which='major', labelsize=25, pad=1)

# ax2.legend(fontsize=15)
fig1.set_size_inches(20,10)
if digital == True:
	plt.savefig('plots/digital_bode_plot_One_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One_different_K_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_bode_plot_One_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_One_different_K_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_One_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_bode_plot_One_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One_different_K_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

#########################################################################################################################################

#########################################################################################################################################

fig2= plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig2.set_size_inches(20,10)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax3= plt.subplot(211)
plt.axhline(y=0, color='r', linestyle='-.')
if phaseCrossOver:
	print('Plot PCF for first K value!')
	plt.axvline(x=phaseCrossOver, color='y', linestyle='-', linewidth=1, alpha=0.5)
plt.xscale('log');
for i in range(len(Kvcovalues2plot)):
	[lineol00]     =    ax3.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f,
		globalFreq(wref, w, Kvcovalues2plot[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[delayEntry_forNet]],
		globalFreq(wref, w, Kvcovalues2plot[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[delayEntry_forNet]],
		tauf, tauc, v, Kvcovalues2plot[i], AkPD, GkLF, Gvga,  digital, model))),'-',ms=1, color= colorbar[i], label=r' K=%.5f' %(K(Kvcovalues2plot[i], AkPD, GkLF, Gvga)/(2.0*np.pi)))

plt.grid();
plt.ylabel(r'$20\textrm{log}\left|H_k^\textrm{ OL}(j\gamma)\right|$',  rotation=90, fontsize=36, labelpad=30)
ax3.legend(fontsize=15);
ax3.set_xlim(f[0], f[-1])
ax3.tick_params(axis='both', which='major', labelsize=25, pad=1)

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #

ax4= plt.subplot(212)
plt.xscale('log')
plt.axhline(y=0, color='r', linestyle='-.')
if phaseCrossOver:
	print('Plot PCF for first K value!')
	plt.axvline(x=phaseCrossOver, color='y', linestyle='-', linewidth=1, alpha=0.5)

for i in range(len(Kvcovalues2plot)):
	[lineol00]     =    ax4.plot(f, PhaseNet(2.0*np.pi*f,
		globalFreq(wref, w, Kvcovalues2plot[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[delayEntry_forNet]],
		globalFreq(wref, w, Kvcovalues2plot[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[delayEntry_forNet]],
		tauf, tauc, v, Kvcovalues2plot[i], AkPD, GkLF, Gvga,  digital, model),'.',ms=1, color= colorbar[i], label=r'K=%.5f' %(K(Kvcovalues2plot[i], AkPD, GkLF, Gvga)))

plt.grid();
plt.ylabel(r'\angle H_k^\textrm{ \LARGE OL}(j\gamma)', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
ax4.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax4.set_xlim(f[0], f[-1])
ax4.legend(fontsize=15)
fig2.set_size_inches(20,10)
if digital == True:
	plt.savefig('plots/digital_bode_plot_Net_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_Net_different_K_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_Net_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_bode_plot_Net_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_Net_different_K_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_Net_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_bode_plot_Net_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_Net_different_K_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_Net_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)



#########################################################################################################################################
#Margins vs K
#########################################################################################################################################


fig3 = plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig3.set_size_inches(20,10)
ax5 = plt.subplot(211)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
Kvec = K(Kvcovalues, AkPD, GkLF, Gvga)/(2.0*np.pi);
[line1] = ax5.plot(Kvec, np.array(pm_vs_K_singlePLL), '.', color='blue');

plt.ylabel(r'phase margin', rotation=90, fontsize=36, labelpad=30)
ax5.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
ax5.legend()
ax5.set_xlim(Kvec[0], Kvec[-1])

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax6= plt.subplot(212)
[line2] = ax6.plot(np.array(Kvalue)/(2.0*np.pi), np.array(gm_vs_K_singlePLL), '.', color='blue');
plt.ylabel(r'gain margin', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$K$', fontsize=40, labelpad=-5)
ax6.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax6.legend()
ax6.set_xlim(Kvec[0], Kvec[-1])
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_OneVsK%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_OneVsK%d_%d_%d.svg' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_OneVsK%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_OneVsK%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_OneVsK%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_OneVsK%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_OneVsK%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_OneVsK%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_OneVsK%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)


#########################################################################################################################################

#########################################################################################################################################

fig4= plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig4.set_size_inches(20,10)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax7= plt.subplot(211)
[line1]     =    ax7.plot(Kvec, np.array(pm_vs_K_Net), '.', color='blue');
plt.ylabel(r'phase margin', rotation=90, fontsize=36, labelpad=30)
ax7.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
ax7.set_xlim(Kvec[0], Kvec[-1])
ax7.legend()

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #

ax8= plt.subplot(212)
[line2]     =    ax8.plot(np.array(KNetvalue)/(2.0*np.pi), np.array(GM_Net),'.',color='blue');
ax8.set_xlim(Kvec[0], Kvec[-1])
plt.ylabel(r'gain margin',  rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$K$', fontsize=40,labelpad=-5)
ax8.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax8.legend()
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_NetVsK%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_NetVsK%d_%d_%d.svg' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_NetVsK%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_NetVsK%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_NetVsK%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_NetVsK%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_NetVsK%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_NetVsK%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_NetVsK%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)




#########################################################################################################################################

#########################################################################################################################################




fig5 = plt.figure(num=5, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig5.set_size_inches(20,10)
ax9 = plt.subplot(211)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
Kvec = K(Kvcovalues, AkPD, GkLF, Gvga)/(2.0*np.pi);
[line1] = ax9.plot(Kvec, np.array(pm_vs_K_singlePLL_Lin), '.', color='blue');

plt.ylabel(r'phase margin', rotation=90, fontsize=36, labelpad=30)
ax9.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
ax9.legend()
ax9.set_xlim(Kvec[0], Kvec[-1])

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax10= plt.subplot(212)
[line2] = ax10.plot(np.array(KOnevalueLin)/(2.0*np.pi), np.array(gm_vs_K_singlePLL_Lin), '.', color='blue');
plt.ylabel(r'gain margin', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$K$', fontsize=40, labelpad=-5)
ax10.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax10.legend()
ax10.set_xlim(Kvec[0], Kvec[-1])
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_OneVsK_Lin%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_OneVsK_Lin%d_%d_%d.svg' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_OneVsK_Lin%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_OneVsK_Lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_OneVsK_Lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_OneVsK_Lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_OneVsK_Lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_OneVsK_Lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_OneVsK_Lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)



#########################################################################################################################################

#########################################################################################################################################



fig6= plt.figure(num=6, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig6.set_size_inches(20,10)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax11= plt.subplot(211)
[line1]     =    ax11.plot(Kvec, np.array(pm_vs_K_Net_Lin), '.', color='blue');
plt.ylabel(r'phase margin', rotation=90, fontsize=36, labelpad=30)
ax11.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
ax11.set_xlim(Kvec[0], Kvec[-1])
ax11.legend()

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #

ax12= plt.subplot(212)
[line2]     =    ax12.plot(np.array(KNetvalue_Lin)/(2.0*np.pi), np.array(GM_Net_Lin),'.',color='blue');
ax12.set_xlim(Kvec[0], Kvec[-1])
plt.ylabel(r'gain margin',  rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$K$', fontsize=40,labelpad=-5)
ax12.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax12.legend()
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_NetVsK_Lin%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_NetVsK_Lin%d_%d_%d.svg' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_NetVsK_Lin%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_NetVsK_Lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_NetVsK_Lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_NetVsK_Lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_NetVsK_Lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_NetVsK_Lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_NetVsK_Lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)


plt.show();
