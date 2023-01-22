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
from bode_lib import solveLinStab, globalFreq, linStabEq, analytical,HopenLoopMutuallyCoupledOnePLL,K,LoopGainSteadyState, coupfunction, K
from bode_lib import PhaseopenloopMutuallyCoupledOnePLL,PhaseclosedloopMutuallyCoupledOnePLL,HclosedLoopMutuallyCoupledOnePLL, GainMarginMutuallyCoupledOne, PhaseMarginMutuallyCoupledOne
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

axis_color  = 'lightgoldenrodyellow'
#
w       = 2.0*np.pi;                                                            # intrinsic    frequency
Kvco    = 2.0*np.pi*(0.8);                                                      # Sensitivity of VCO
####################################################################################################################################################################################
AkPD    = 1.0                                                                   # amplitude of the PD -- the factor of 2 here is related to the IMS 2020 paper, AkPD as defined in that paper already includes the 0.5 that we later account for when calculating K
GkLF    = 1.0
Gvga    = 1.0
Ga1     = 1.0                                                                   # Gain of the first adder
order   = 1.0                                                                   # the order of the Loop Filter
tauf    = 0.0                                                                   # tauf = sum of all processing delays in the feedback
tauc    = 1.0/(2.0*np.pi*0.0146);                                               # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
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

dwc		  = 0.00005 #0.00005;
wcvalues  = np.arange(0.0001*2.0*np.pi, 0.5*2.0*np.pi, dwc)
df        = 0.001 #0.001;
f         = np.logspace(-8, 4, num=int((tauf+maxp)/df), endpoint=True, base=10.0)

####################################################################################################################################################################################

PhaseOnePLL        = np.vectorize(PhaseopenloopMutuallyCoupledOnePLL)
PhaseNet           = np.vectorize(PhaseopenloopMutuallyCoupledNet)

syncState = globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV);

wcvalues2plot	= [0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi,  0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
tauvalues2plot  = [0*w/(2*np.pi), 0.33*w/(2*np.pi), 0.5*w/(2*np.pi), 0.75*w/(2*np.pi), 1.0*w/(2*np.pi), 2.15*w/(2*np.pi)]
bar = [];
for i in range(len(tauvalues2plot)):
	tempo = np.where(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][:]>tauvalues2plot[i])[0][0]
	# print('tempo:', tempo)
	bar.append( tempo )
	# print('bar[i]', bar[i])

# pick for which value of the delay to compute the plots
tempo1 = np.where(syncState['tau'][:] > tauvalues2plot[3])[0][0]

#Here we calculate and save to lists the gain and phase crossover frequencies and gain and phase margin as well as functions of the cutoff frequency
#########
pm_vs_wc_singlePLL=[];
gm_vs_wc_singlePLL=[];
wcvalueSingle=[];
gcf_vs_wc_singlePLL=[];

gcf_vs_wc_Net=[];
pm_vs_wc_Net =[];
wcvalueNet=[];
GM_Net=[]; phaseCrossOver = [];
gm_vs_wc_singlePLL_Lin	= [];
wcvalueSingleLin   		= [];
GM_Net_Lin         		= [];
phaseCrossOver_Lin 		= [];
pm_vs_wc_singlePLL_Lin 	= [];
pm_vs_wc_Net_Lin		= [];
gcf_vs_wc_Net_Lin		= [];
wcvalueNet_Lin          = []
freqMaxGainIn=[]; freqMaxGainAn=[]; maxGainIn=[]; maxGainAn=[];
print('plotting for OmegTau=', syncState['tau'][tempo1]*syncState['Omeg'][tempo1]/(2.0*np.pi))
for j in range(len(wcvalues)):

	print('Progress:',j,'/',len(wcvalues))

	gcf_vs_wc_singlePLL.append(f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
											tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model))) <=0.0001)][0])

	pm_vs_wc_singlePLL.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*gcf_vs_wc_singlePLL[-1], syncState['Omeg'][tempo1], syncState['tau'][tempo1],
											tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model))

	#gcf_vs_wc_Net.append(f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
	#										tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model))) <=0.0001)][-1] )

	tempHopenNet = HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1], tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model)
	gcf_vs_wc_Net.append(f[np.where(20.0*log10((abs(tempHopenNet))) <= 0.0001)][-1] )
	maxGainIn.append(np.max(20.0*log10(abs(tempHopenNet))))
	freqMaxGainIn.append(f[np.where(20.0*log10(abs(tempHopenNet))==maxGainIn[-1])[0][0]]); #print('freqMaxGainIn[-1]', freqMaxGainIn[-1])
	#print('np.where(20.0*log10(abs(tempHopenNet))==maxGainIn[-1])[0][0]', np.where(20.0*log10(abs(tempHopenNet))==maxGainIn[-1])[0][0])

	pm_vs_wc_Net.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_wc_Net[-1], syncState['Omeg'][tempo1], syncState['tau'][tempo1],
											tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model))

	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
											tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model)) <=0.7).any():

		GMDifwc=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
											tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model)) <=0.7)][0]
		# print(GM)
		gm_vs_wc_singlePLL.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GMDifwc, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
											tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model))
		wcvalueSingle.append(wcvalues[j])

	phaseNet = PhaseNet(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1], tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model)
	index1   = np.where(np.abs(phaseNet) > 1E-10)[0][0]
	#print('index1: ', index1)
	#print('phaseNet[index1:]:', phaseNet[index1:])
	if np.any(phaseNet[index1:]<0):												# catch the zero delay case where the phase never goes below zero
		if phaseNet[index1] > 0:
			index2 = np.where(phaseNet[index1:] < 0)[0][0]
			GMNet  = f[index1+index2]
		elif phaseNet[index1] < 0:
			index2 = np.where(phaseNet[index1:] > 0)[0][0]
			index3 = np.where(phaseNet[(index1+index2):] < 0)[0][0]
			GMNet  = f[index1+index2+index3]
		else:
			print('Error! Debug.'); sys.exit();
	elif np.min(phaseNet[index1:]) < treshold_detect_zero_delay:
		index2 = np.where(phaseNet[index1:] < treshold_detect_zero_delay)[0][0]
		GMNet  = f[index1+index2]
	else:
		print('Phase associated to zero delay does not go below threshold, increase frequency range of Bode plot or change treshold!');
		print('Smallest value for phase of zero delay case:', np.min(phaseNet[index1:]))
		sys.exit();

	#index1   = np.where(phaseNet >=150)[0][0]
	# print('index1: ', index1)
	# print('phaseNet[index1:]:', phaseNet[index1:])
	#index2 = np.where(phaseNet[index1:] <= 0.0)[0][0]
	# if syncState['tau'][j] > 0:
	# 	index2 = np.where(phaseNet[index1:] <= 0.0)[0][0]
	# else:
	# 	index2 = np.where(phaseNet[index1:] <= 0.0)[0]
	# print('index2: ', index2)
	#GMNet  = f[index1+index2]
	# print('found GMNet:', GMNet, ' at index:', index1+index2)
	#Kvcobar			= [0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi, 0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
	if ( np.abs(wcvalues[j]-wcvalues2plot[0]) <= dwc and len(phaseCrossOver)==0 ):
		# print('Save phaseCrossOver for one value of wc! PCF:', GMNet, ' for wcvalues[j]', wcvalues[j],' wcvalues2plot[0]:', wcvalues2plot[0])
		#sys.exit()
		phaseCrossOver.append(GMNet);

	GM_Net.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet,
		syncState['Omeg'][tempo1], syncState['tau'][tempo1], tauf,  1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model) )
	wcvalueNet.append(wcvalues[j])


	# if (abs(phaseNet) <=0.01).any():
	# 	GMNet = f[np.where(abs(phaseNet) <=0.01)][-1]
		# print(GM)
	# 	GM_Net.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet,
	# 		syncState['Omeg'][tempo1], syncState['tau'][tempo1], tauf,  1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model) )
	# 	wcvalueNet.append(wcvalues[j])

	######################################################################################################## LINEAR


	####################### Single PLL
	gfc_vs_wc_singlePLL_Lin = f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
																		tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))) <= 0.0001)][0]

	pm_vs_wc_singlePLL_Lin.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*gfc_vs_wc_singlePLL_Lin, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
																		tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))


	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
																		tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear')) <=0.7).any():

		pfc_vs_wc_singlePLL_Lin=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,	syncState['Omeg'][tempo1], syncState['tau'][tempo1],
																		tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear')) <=0.7)][0]

		gm_vs_wc_singlePLL_Lin.append(GainMarginMutuallyCoupledOne(2.0*np.pi*pfc_vs_wc_singlePLL_Lin, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
																		tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))

		wcvalueSingleLin.append(wcvalues[j])

	####################### Network

	gcf_vs_wc_Net_Lin.append(f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))) <=0.0001)][-1] )

	pm_vs_wc_Net_Lin.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_wc_Net[-1], syncState['Omeg'][tempo1], syncState['tau'][tempo1],
									tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))

	phaseNet_Lin = PhaseNet(2.0*np.pi*f, syncState['Omeg'][tempo1],	syncState['tau'][tempo1], tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear')
	index1_Lin   = np.where(phaseNet_Lin >=170)[0][0]
	#print('index1: ', index1)
	#print('phaseNet[index1:]:', phaseNet[index1:])
	index2_Lin = np.where(phaseNet_Lin[index1_Lin:] <= 0.0)[0][0]
	#print('index2: ', index2)
	GMNet_Lin  = f[index1_Lin+index2_Lin]
	#print('found GMNet:', GMNet, ' at index:', index1+index2)
	#Kvcovalues2plot			= [0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi, 0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
	if ( np.abs(wcvalues[j]-wcvalues2plot[0]) <= 0.01 and len(phaseCrossOver_Lin) == 0 ):
		phaseCrossOver_Lin.append(GMNet_Lin);
	GM_Net_Lin.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
														1.0/wcvalues[j], tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear') )
	wcvalueNet_Lin.append(wcvalues[j])



	#
	# PMLin = f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
	# 	tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))) <= 0.0001)][0]
	#
	# pm_vs_tau_singlePLL_Lin.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*PMLin, syncStateInphase['Omeg'][j],
	# 	syncStateInphase['tau'][j],
	# 	tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))
	#
	#
	# if (abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
	# 			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear')) <=0.7).any():
	#
	# 	GMLin=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,	syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
	# 			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear')) <=0.7)][0]
	# 	# print(GM)
	# 	GM_One_Lin.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GMLin, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
	# 		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))
	#
	# 	tauOne_Lin.append(syncStateInphase['tau'][j])
	# 	OmegaOne_Lin.append(syncStateInphase['Omeg'][j])

# ADD PLOTS
####################################################################################################################################################################################

colorbar=['blue','green','orange','purple', 'cyan', 'black']
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1]/(2*np.pi))
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1])
##################################################################################################################################################################
# #
#
#
fig1= plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig1.set_size_inches(20,10)
# ## ## ## ## ## ## ## ## ## ## # Subplot# ## ## ## ## ## ## ## ## ## ## #

ax1= plt.subplot(211)
ax1.set_xlim(0.0, wcvalues[-1]/(2.0*np.pi))
[line1]     =    ax1.plot(np.array(wcvalues)/(2.0*np.pi), np.array(pm_vs_wc_singlePLL), '.', ms=1.5, color='blue');
plt.ylabel(r'phase margin', rotation=90, fontsize=60, labelpad=30)
ax1.tick_params(axis='both', which='major', labelsize=35, pad=1)
plt.grid();
# ax1.legend(bbox_to_anchor=(0.2,0.75), prop=labelfont)

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax2= plt.subplot(212)
ax2.set_xlim(0.0, wcvalues[-1]/(2.0*np.pi))
[line66] = ax2.plot(np.array(wcvalueSingle)/(2.0*np.pi), np.array(gm_vs_wc_singlePLL), '.', ms=1.5, color='blue');

plt.ylabel(r'gain margin', rotation=90, fontsize=60, labelpad=30)
plt.xlabel(r'$f_c$', fontsize=60, labelpad=10 )
ax2.tick_params(axis='both', which='major', labelsize=35, pad=1)
# ax2.legend()
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_OneVswc%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_OneVswc%d_%d_%d.svg' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_OneVswc%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_OneVswc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_OneVswc%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_OneVswc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_OneVswc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_OneVswc%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_OneVswc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

#########################################################################################################################################

fig2= plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig2.set_size_inches(20,10)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax3= plt.subplot(211)
plt.axhline(y=0, color='r', linestyle='-.')
ax3.set_xlim(f[0], f[-1])
plt.xscale('log');
for i in range(len(wcvalues2plot)):
	[lineol00]     =    ax3.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
		tauf, (1.0/wcvalues2plot[i]), v, Kvco, AkPD, GkLF, Gvga,  digital, model))),'-', ms=1.5, color=colorbar[i], label=r'fc=%.5f' %( wcvalues2plot[i]/(2.0*np.pi)))

plt.grid();
plt.ylabel(r'$20\log\left|H_k^\textrm{\LARGE OL}(j\gamma)\right|$', rotation=90, fontsize=60, labelpad=30)
#ax3.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont);
ax3.tick_params(axis='both', which='major', labelsize=35, pad=1)

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax4= plt.subplot(212)
ax4.set_xlim(f[0], f[-1])
plt.xscale('log')
plt.axhline(y=-180, color='r', linestyle='-.')
for i in range(len(wcvalues2plot)):
	[lineol00]     =    ax4.plot(f, PhaseOnePLL(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
		tauf, (1.0/wcvalues2plot[i]), v, Kvco, AkPD, GkLF, Gvga,  digital, model), '.', ms=1.5, color= colorbar[i], label=r' fc=%.5f' %(wcvalues2plot[i]/(2.0*np.pi)))

plt.grid();
plt.ylabel(r'phase', rotation=90, fontsize=60, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=60, labelpad=-5)
ax4.tick_params(axis='both', which='major', labelsize=35, pad=1)

# ax4.legend(fontsize=15)
fig2.set_size_inches(20,10)
if digital == True:
	plt.savefig('plots/digital_bode_plot_One_different_wc_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One_different_wc_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One_different_wc_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_bode_plot_One_different_wc_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_One_different_wc_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_sin_bode_plot_One_different_wc_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_bode_plot_One_different_wc_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One_different_wc_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One_different_wc_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

#######################################################################################################################################

fig3= plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig3.set_size_inches(20,10)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax5= plt.subplot(211)

ax5.set_xlim(0.0, wcvalues[-1]/(2.0*np.pi))
[line1] = ax5.plot(np.array(wcvalues)/(2.0*np.pi), np.array(pm_vs_wc_Net), '.', ms=1.5, color='blue');
plt.ylabel(r'phase margin', rotation=90, fontsize=60, labelpad=30)
ax5.tick_params(axis='both', which='major', labelsize=35, pad=1)
plt.grid();
#ax5.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax6= plt.subplot(212)
ax6.set_xlim(0.0, wcvalues[-1]/(2.0*np.pi))
[line2] = ax6.plot(np.array(wcvalueNet)/(2.0*np.pi), np.array(GM_Net), '.', ms=1.5, color='blue');
plt.ylabel(r'gain margin',  rotation=90, fontsize=60, labelpad=30)
plt.xlabel(r'$f_c$', fontsize=60, labelpad=10)
ax6.tick_params(axis='both', which='major', labelsize=35, pad=1)
#ax6.legend()
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_Net_vs_wc%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_Net_vs_wc%d_%d_%d.svg' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_Net_vs_wc%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_Net_vs_wc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_Net_vs_wc%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_Net_vs_wc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_Net_vs_wc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_Net_vs_wc%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_Net_vs_wc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

##################################################################################################################################################

fig4= plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig4.set_size_inches(20,10)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax7= plt.subplot(211)
plt.axhline(y=0, color='r', linestyle='-.')
ax7.set_xlim(f[0], f[-1])
if phaseCrossOver:
	print('Plot PCF for first wc value!')
	plt.axvline(x=phaseCrossOver, color='y', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xscale('log');
for i in range(len(wcvalues2plot)):
	[lineol00]     =    ax7.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
		tauf, (1.0/wcvalues2plot[i]), v, Kvco, AkPD, GkLF, Gvga,  digital, model))), '.', ms=1.5, color= colorbar[i], label=r'fc=%.5f' %(wcvalues2plot[i]/(2.0*np.pi)))

plt.grid();
plt.ylabel(r'$20\textrm{log}\left|H_k^\textrm{\LARGE OL}(j\gamma)\right|$',  rotation=90, fontsize=60, labelpad=30)
ax7.tick_params(axis='both', which='major', labelsize=35, pad=1)

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax8= plt.subplot(212)
ax8.set_xlim(f[0], f[-1])
plt.xscale('log')
plt.axhline(y=0, color='r', linestyle='-.')
if phaseCrossOver:
	print('Plot PCF for first wc value!')
	plt.axvline(x=phaseCrossOver, color='y', linestyle='-', linewidth=0.5, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-.')
for i in range(len(wcvalues2plot)):
	[lineol00]     =    ax8.plot(f, PhaseNet(2.0*np.pi*f, syncState['Omeg'][tempo1], syncState['tau'][tempo1],
		tauf, (1.0/wcvalues2plot[i]), v, Kvco, AkPD, GkLF, Gvga,  digital, model), '.', ms=1.5, color= colorbar[i], label=r' fc=%.5f' %(wcvalues2plot[i]/(2.0*np.pi)))
plt.grid();
plt.ylabel(r'phase', rotation=90, fontsize=60, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=60, labelpad=-5)
ax8.tick_params(axis='both', which='major', labelsize=35, pad=1)
ax8.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont);

# ax8.legend(fontsize=15)
fig4.set_size_inches(20,10)
if digital == True:
	plt.savefig('plots/digital_bode_plot_Net_vs_wc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_Net_vs_wc%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_Net_vs_wc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_bode_plot_Net_vs_wc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_Net_vs_wc%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_Net_vs_wc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_bode_plot_Net_vs_wc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_Net_vs_wc%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_Net_vs_wc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)


#########################################################################################################################################

#########################################################################################################################################



fig5 = plt.figure(num=5, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig5.set_size_inches(20,10)
ax9 = plt.subplot(211)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
# Kvec = K(Kvcovalues, AkPD, GkLF, Gvga)/(2.0*np.pi);
[line1] = ax9.plot(wcvalueSingleLin, np.array(pm_vs_wc_singlePLL_Lin), '.', color='blue');

plt.ylabel(r'phase margin', rotation=90, fontsize=60, labelpad=30)
ax9.tick_params(axis='both', which='major', labelsize=35, pad=1)
plt.grid();
ax9.legend()
ax9.set_xlim(wcvalues[0]/(2.0*np.pi), wcvalues[-1]/(2.0*np.pi))

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax10= plt.subplot(212)
[line2] = ax10.plot(np.array(wcvalueSingleLin)/(2.0*np.pi), np.array(gm_vs_wc_singlePLL_Lin), '.', color='blue');
plt.ylabel(r'gain margin', rotation=90, fontsize=60, labelpad=30)
plt.xlabel(r'$f_c$', fontsize=60, labelpad=-5)
ax10.tick_params(axis='both', which='major', labelsize=35, pad=1)
ax10.legend()
ax10.set_xlim(wcvalues[0]/(2.0*np.pi), wcvalues[-1]/(2.0*np.pi))
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_OneVswc_Lin%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_OneVswc_Lin%d_%d_%d.svg' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_OneVswc_Lin%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_OneVswc_Lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_OneVswc_Lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_OneVswc_Lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_OneVswc_Lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_OneVswc_Lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_OneVswc_Lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)


#########################################################################################################################################

#########################################################################################################################################


fig6= plt.figure(num=6, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig6.set_size_inches(20,10)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax11= plt.subplot(211)
[line1]     =    ax11.plot(wcvalueNet_Lin, np.array(pm_vs_wc_Net_Lin), '.', color='blue');
plt.ylabel(r'phase margin', rotation=90, fontsize=60, labelpad=30)
ax11.tick_params(axis='both', which='major', labelsize=35, pad=1)
plt.grid();
ax11.set_xlim(wcvalues[0]/(2.0*np.pi), wcvalues[-1]/(2.0*np.pi))
ax11.legend()

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #

ax12= plt.subplot(212)
[line2]     =    ax12.plot(np.array(wcvalueNet_Lin)/(2.0*np.pi), np.array(GM_Net_Lin),'.',color='blue');
ax12.set_xlim(wcvalues[0]/(2.0*np.pi), wcvalues[-1]/(2.0*np.pi))
plt.ylabel(r'gain margin',  rotation=90, fontsize=60, labelpad=30)
plt.xlabel(r'$f_c$', fontsize=60, labelpad=-5)
ax12.tick_params(axis='both', which='major', labelsize=35, pad=1)
ax12.legend()
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_NetVswc_Lin%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_NetVswc_Lin%d_%d_%d.svg' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_NetVswc_Lin%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_NetVswc_Lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_NetVswc_Lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_NetVswc_Lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_NetVswc_Lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_NetVswc_Lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_NetVswc_Lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)


################################################################################################################################################################## sigma & gain margin vs Omega tau

fig7= plt.figure(num=7, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig7.set_size_inches(20,10)
# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax13= plt.subplot(211)

ax13.set_xlim(0.0, wcvalues[-1]/(2.0*np.pi))
#[line1] = ax13.plot(np.array(wcvalues)/(2.0*np.pi), maxGainAn, '-', linewidth=1.5, color='red');
[line1] = ax13.plot(np.array(wcvalues)/(2.0*np.pi), np.array(maxGainIn), '-', linewidth=1.5, color='blue');
plt.grid();
plt.ylabel(r'$20\log\left|H_k^\textrm{\LARGE OL}(j\gamma)\right|_\textrm{max}$', rotation=90, fontsize=60, labelpad=30)
ax13.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
ax13.tick_params(axis='both', which='major', labelsize=35, pad=1)
#ax5.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)

# ## ## ## ## ## ## ## ## ## ## # Subplot # ## ## ## ## ## ## ## ## ## ## #
ax14= plt.subplot(212)
ax14.set_xlim(0.0, wcvalues[-1]/(2.0*np.pi))
#print('freqMaxGainIn', freqMaxGainIn)
#[line3] = ax14.plot(np.array(wcvalueNet)/(2.0*np.pi), freqMaxGainAn, '-', linewidth=1.5, color='red');
[line4] = ax14.plot(np.array(wcvalueNet)/(2.0*np.pi), freqMaxGainIn, '-', linewidth=1.5, color='blue');
#ax14.set_ylim(-1., np.max(freqMaxGainIn)) #maxGainIn, maxGainAn,
plt.ylabel(r'$f^\textrm{pert}$', rotation=90, fontsize=60, labelpad=30)
plt.xlabel(r'$f_c$', fontsize=60, labelpad=5)
plt.grid();
#ax6.legend()
plt.grid();

if digital == True:
	plt.savefig('plots/digital_gamma_freqMaxG_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_gamma_freqMaxG_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_gamma_freqMaxG_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_gamma_freqMaxG_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_gamma_freqMaxG_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_gamma_freqMaxG_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_gamma_freqMaxG_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_gamma_freqMaxG_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_gamma_freqMaxG_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

# fig7 = plt.figure(num=7, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig7.set_size_inches(20,10)
#
# ax13 = plt.subplot(211)
# OmegTauIn = np.array(OmegaNet)*np.array(tauNet)/(2.0*np.pi)
# OmegTauAn = np.array(OmegaNet_AP)*np.array(tauNet_AP)/(2.0*np.pi)
# ax13.set_xlim(min(OmegTauIn[0],OmegTauAn[0]), max(OmegTauIn[-1],OmegTauAn[-1]))
# [lineSigmaIn] = ax13.plot(np.asarray(syncStateInphase['Omeg']/(2.0*np.pi))*syncStateInphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateInphase['Omeg'],
# 					syncStateInphase['tau'], tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ImMax'], '-', linewidth=1.5, color='blue', label=r'in-phase')
# [lineSigmaAn] = ax13.plot(np.asarray(syncStateAnphase['Omeg']/(2.0*np.pi))*syncStateAnphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateAnphase['Omeg'],
# 					syncStateAnphase['tau'], tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state2, expansion, INV)['ImMax'], '-', linewidth=1.5, color='red', label=r'anti-phase')
# plt.grid();
# plt.ylabel(r'$\frac{\gamma}{\omega}$', rotation=0, fontsize=60, labelpad=30)
# ax13.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# ax13.tick_params(axis='both', which='major', labelsize=35, pad=1)
#
# ax14 = plt.subplot(212)
# ax14.set_xlim(min(OmegTauIn[0],OmegTauAn[0]), max(OmegTauIn[-1],OmegTauAn[-1]))
# ax14.tick_params(axis='both', which='major', labelsize=35, pad=1)
# #[lineoNet00]  = ax14.plot(OmegTauIn, np.array(maxGainIn), '--', linewidth=1.5, label=r'max gain dB (in-phase)', color='blue');
# #[lineoNet01]  = ax14.plot(OmegTauAn, np.array(maxGainAn), '--', linewidth=1.5, label=r'max gain dB (anti-phase)', color='red');
# [lineoNet02]  = ax14.plot(OmegTauIn, np.array(freqMaxGainIn), '-', linewidth=1.5, label=r'angle shoulder (in-phase)', color='blue');
# #try:
# [lineoNet03]  = ax14.plot(OmegTauAn, np.array(freqMaxGainAn), '-', linewidth=1.5, label=r'angle shoulder (anti-phase)', color='red');
# #except:
# #	print('Problem plotting!')
# ax14.set_ylim(-1., np.max(np.array([freqMaxGainIn, freqMaxGainAn]))) #maxGainIn, maxGainAn,
# plt.ylabel(r'$f^\textrm{pert}$', rotation=90, fontsize=60, labelpad=30)
# plt.xlabel(r'$\Omega\tau/2 \pi$', fontsize=60, labelpad=5)
# plt.grid();
#




plt.show();
