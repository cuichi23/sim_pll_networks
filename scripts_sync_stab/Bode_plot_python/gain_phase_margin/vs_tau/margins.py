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
  'size'   : 16,
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
w       = 2.0*np.pi;                                                            # intrinsic    frequency
Kvco    = 2.0*np.pi*(0.8);                                                      # Sensitivity of VCO
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

wref	 = w
w_cutoff = 1.0/tauc
treshold_detect_zero_delay = 0.5												# treshold to detect phase crossover frequency for the zero delay case!
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
df        = 0.00025#0.0005 #0.00025;
fmin      = -16
fmax      =  11
f         = np.logspace(fmin, fmax, num=int((tauf+maxp)/df), endpoint=True, base=10.0)

tauvalues2plot = [0*w/(2*np.pi), 0.025*w/(2*np.pi), 0.05*w/(2*np.pi), 0.1*w/(2*np.pi), 0.2*w/(2*np.pi), 0.3*w/(2*np.pi)]
#tauvalues2plot = [0*w/(2*np.pi), 0.25*w/(2*np.pi), 0.5*w/(2*np.pi), 0.75*w/(2*np.pi), 1.0*w/(2*np.pi), 1.25*w/(2*np.pi)]
####################################################################################################################################################################################

gcf_vs_tau_singlePLL=[]; gcf_vs_tau_singlePLL_AP=[];
#gamma_phase_margin_One=[];
pm_vs_tau_singlePLL=[]; pm_vs_tau_singlePLL_AP=[];
GM_One=[]; GM_One_AP=[];
tauOne=[]; tauOne_AP=[];
#gamma_phase_margin_Net=[];
pm_vs_tau_Net=[]; pm_vs_tau_Net_AP=[];
GM_Net=[]; GM_Net_AP=[]; GM_Net_test=[]; x_test=[];
tauNet=[]; tauNet_AP=[];
OmegaNet=[]; OmegaNet_AP=[];
OmegaOne=[]; OmegaOne_AP=[];
#gain_cross_freq_One=[];
gcf_vs_tau_Net=[]; gcf_vs_tau_Net_AP=[];
phase_cross_freq_One=[]; phase_cross_freq_One_AP=[];
GM_One_Lin=[];
pm_vs_tau_singlePLL_Lin=[];
tauOne_Lin=[];OmegaOne_Lin=[];
phaseCrossOver=[]; phaseCrossOver_AP=[]; phaseCrossOver_Lin=[];
freqMaxGainIn=[]; freqMaxGainAn=[]; maxGainIn=[]; maxGainAn=[];

PhaseOnePLL        = np.vectorize(PhaseopenloopMutuallyCoupledOnePLL)
PhaseNet           = np.vectorize(PhaseopenloopMutuallyCoupledNet)

syncStateInphase = globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV);
syncStateAnphase = globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state2, INV);
inphasetaus = syncStateInphase['tau']; antiphasetaus = syncStateAnphase['tau'];
treshold_find_tau_value_inphase = np.max(sort(inphasetaus)[1:]-sort(inphasetaus)[0:-1]);
treshold_find_tau_value_anphase = np.max(sort(antiphasetaus)[1:]-sort(antiphasetaus)[0:-1]);
print('tau-resolution in-phase   (mean, min max): ', np.mean(sort(inphasetaus)[1:]-sort(inphasetaus)[0:-1]), np.min(sort(inphasetaus)[1:]-sort(inphasetaus)[0:-1]), treshold_find_tau_value_inphase)
print('tau-resolution anti-phase (mean, min max): ', np.mean(sort(antiphasetaus)[1:]-sort(antiphasetaus)[0:-1]), np.min(sort(antiphasetaus)[1:]-sort(antiphasetaus)[0:-1]), treshold_find_tau_value_anphase)
#Here we calculate and save to lists the gain and phase crossover frequencies and gain and phase margin as functions of the delay.
for j in range(len(syncStateInphase['tau'])):

	print('Progress:',j,'/',len(syncStateInphase['tau']))

	######################################################################################################## INPHASE

	gcf_vs_tau_singlePLL.append(f[np.where(20.0*log10((abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1)))) <= 0.0001)][0])

	pm_vs_tau_singlePLL.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*gcf_vs_tau_singlePLL[-1], syncStateInphase['Omeg'][j],
		syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1))

	# tempHopenNet = HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1)
	# gcf_vs_tau_Net.append(f[np.where(20.0*log10((abs(tempHopenNet))) <= 0.0001)][-1] )
	# maxGainIn.append(np.max(20.0*log10(abs(tempHopenNet))))
	# freqMaxGainIn.append(f[np.where(20.0*log10(abs(tempHopenNet))==maxGainIn[-1])[0][0]]); #print('freqMaxGainIn[-1]', freqMaxGainIn[-1])

	# pm_vs_tau_Net.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_tau_Net[-1], syncStateInphase['Omeg'][j],
		# syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1))

	phase_cross_freq_One.append(f[np.where((abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1))) < 0.7 )] )

	phase = PhaseOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1)
	if (abs(180.0+phase) <= 0.7 ).any():
		GM = f[np.where(abs(180.0+phase) <= 0.7)][0]

		GM_One.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GM, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1))
		tauOne.append(syncStateInphase['tau'][j])
		OmegaOne.append(syncStateInphase['Omeg'][j])

	# index1 = 0; index2 = 0; index3 = 0; index4 = 0; index5 = 0;					# reset for safety
	# phaseNet = PhaseNet(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1)
	# index1   = np.where(np.abs(phaseNet) > 1E-10)[0][0]                         # find first phase value that deviates more than the treshold from zero
	# #print('index1: ', index1)
	# #print('phaseNet[index1:]:', phaseNet[index1:])
	# if np.any(phaseNet[index1:]<0):												# catch the zero delay case where the phase never goes below zero
	# 	if phaseNet[index1] > 0:
	# 		index2 = np.where(phaseNet[index1:] < 0)[0][0]                      # where is the phase below zero the first time from index1 to the end
	# 		GMNet  = f[index1+index2]                                           # save the frequency at this point
	# 		# find next encirclement
	# 		if syncStateInphase['tau'][j] > 0.5:
	# 			index3 = np.where(phaseNet[(index1+index2):] > 0)[0][0]
	# 			index4 = np.where(phaseNet[(index1+index2+index3):] < 0)[0][0]
	# 			GMNet2 = f[index1+index2+index3+index4]
	# 	elif phaseNet[index1] < 0:
	# 		index2 = np.where(phaseNet[index1:] > 0)[0][0]
	# 		index3 = np.where(phaseNet[(index1+index2):] < 0)[0][0]
	# 		GMNet  = f[index1+index2+index3]
	# 		# find next encirclement
	# 		if syncStateInphase['tau'][j] > 0.5:
	# 			index4 = np.where(phaseNet[(index1+index2+index3):] > 0)[0][0]
	# 			#print('np.where(phaseNet[(index1+index2+index3):] > 0)[0][0]:', np.where(phaseNet[(index1+index2+index3):] > 0)[0][0])
	# 			index5 = np.where(phaseNet[(index1+index2+index3+index4):] < 0)[0][0]
	# 			#print('index5 = np.where(phaseNet[(index1+index2+index3+index4):] < 0)[0][0]:', index5 = np.where(phaseNet[(index1+index2+index3+index4):] < 0)[0][0])
	# 			GMNet2 = f[index1+index2+index3+index4+index5]
	# 	else:
	# 		print('Error! Debug.'); sys.exit();
	# elif np.min(phaseNet[index1:]) < treshold_detect_zero_delay:				# this is the case for zero delay
	# 	index2 = np.where(phaseNet[index1:] < treshold_detect_zero_delay)[0][0]
	# 	GMNet  = f[index1+index2]
		#GMNet2 = f[0]
	# else:
	# 	print('Phase associated to zero delay does not go below threshold, increase frequency range of Bode plot or change treshold!');
	# 	print('Smallest value for phase of zero delay case:', np.min(phaseNet[index1:]))
	# 	sys.exit();
    #
	# if ( np.abs(syncStateInphase['tau'][j]-tauvalues2plot[3]) <= 1.1*treshold_find_tau_value_inphase and len(phaseCrossOver)==0 ):
	# 	print('Save phaseCrossOver for one value of the delay! PCF:', GMNet)
	# 	phaseCrossOver.append(GMNet);
    #
	# GM_Net.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1))
	# if syncStateInphase['tau'][j] > 0.5:
	# 	GM_Net_test.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet2, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1))
	# 	x_test.append(syncStateInphase['tau'][j]*syncStateInphase['Omeg'][j]/(2*np.pi))
	# tauNet.append(syncStateInphase['tau'][j])
	# OmegaNet.append(syncStateInphase['Omeg'][j])

	######################################################################################################## ANTIPHASE

	gcf_vs_tau_singlePLL_AP.append(f[np.where(20.0*log10((abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncStateAnphase['Omeg'][j], syncStateAnphase['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2)))) <= 0.0001)][0])

	pm_vs_tau_singlePLL_AP.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*gcf_vs_tau_singlePLL_AP[-1], syncStateAnphase['Omeg'][j], syncStateAnphase['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2))

	# tempHopenNet = HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncStateAnphase['Omeg'][j], syncStateAnphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2)
	# gcf_vs_tau_Net_AP.append(f[np.where(20.0*log10((abs(tempHopenNet))) <= 0.0001)][-1] )
	# maxGainAn.append(np.max(20.0*log10(abs(tempHopenNet))))
	# freqMaxGainAn.append(f[np.where(20.0*log10(abs(tempHopenNet))==maxGainAn[-1])[0][0]]); #print('freqMaxGainIn[-1]', freqMaxGainIn[-1])

	# pm_vs_tau_Net_AP.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_tau_Net_AP[-1], syncStateAnphase['Omeg'][j], syncStateAnphase['tau'][j],
	# 	tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2))

	phase_cross_freq_One_AP.append(f[np.where((abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncStateAnphase['Omeg'][j], syncStateAnphase['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2))) < 0.7 )] )

	phase_AP = PhaseOnePLL(2.0*np.pi*f, syncStateAnphase['Omeg'][j], syncStateAnphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2)
	if (abs(180.0+phase_AP) <= 0.7 ).any():
		GM_AP = f[np.where(abs(180.0+phase_AP) <= 0.7)][0]

		GM_One_AP.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GM_AP, syncStateAnphase['Omeg'][j], syncStateAnphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2))
		tauOne_AP.append(syncStateAnphase['tau'][j])
		OmegaOne_AP.append(syncStateAnphase['Omeg'][j])
    #
	# phaseNet_AP = PhaseNet(2.0*np.pi*f, syncStateAnphase['Omeg'][j], syncStateAnphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2)
	# index1_AP   = np.where(np.abs(phaseNet_AP) > 1E-10)[0][0]
	# #print('index1: ', index1)
	# #print('phaseNet[index1:]:', phaseNet[index1:])
	# if np.any(phaseNet_AP[index1_AP:] < 0):										# catch the zero delay case where the phase never goes below zero
	# 	if phaseNet_AP[index1_AP] > 0:
	# 		index2_AP = np.where(phaseNet_AP[index1_AP:] < 0)[0][0]
	# 		GMNet_AP  = f[index1_AP+index2_AP]
	# 	elif phaseNet_AP[index1_AP] < 0:
	# 		index2_AP = np.where(phaseNet_AP[index1_AP:] > 0)[0][0]
	# 		index3_AP = np.where(phaseNet_AP[(index1_AP+index2_AP):] < 0)[0][0]
	# 		GMNet_AP  = f[index1_AP+index2_AP+index3_AP]
	# 	else:
	# 		print('Error! Debug.'); sys.exit();
	# elif np.min(phaseNet_AP[index1_AP:]) < treshold_detect_zero_delay:
	# 	index2_AP = np.where(phaseNet_AP[index1_AP:] < treshold_detect_zero_delay)[0][0]
	# 	GMNet_AP  = f[index1_AP+index2_AP]
	# else:
	# 	print('Phase associated to zero delay does not go below threshold, increase frequency range of Bode plot or change treshold!'); sys.exit();
    #
	# if ( np.abs(syncStateAnphase['tau'][j]-tauvalues2plot[3]) <= 1.1*treshold_find_tau_value_anphase and len(phaseCrossOver_AP)==0 ):
	# 	print('Save phaseCrossOver for one value of the delay! PCF:', GMNet_AP)
	# 	phaseCrossOver_AP.append(GMNet_AP);
    #
	# GM_Net_AP.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet_AP, syncStateAnphase['Omeg'][j], syncStateAnphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2))
	# tauNet_AP.append(syncStateAnphase['tau'][j])
	# OmegaNet_AP.append(syncStateAnphase['Omeg'][j])

	######################################################################################################## LINEAR

	PMLin = f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1))) <= 0.0001)][0]
    #
	pm_vs_tau_singlePLL_Lin.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*PMLin, syncStateInphase['Omeg'][j],
		syncStateInphase['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1))
    #
    #
	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1)) <=0.7).any():

		GMLin=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,	syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1)) <=0.7)][0]
		# print(GM)
		GM_One_Lin.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GMLin, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1))
    
		tauOne_Lin.append(syncStateInphase['tau'][j])
		OmegaOne_Lin.append(syncStateInphase['Omeg'][j])

bar = [];
for i in range(len(tauvalues2plot)):
	tempo = np.where(syncStateInphase['tau'][:]>tauvalues2plot[i])[0][0]
	# print('tempo:', tempo)
	bar.append( tempo )

# ADD PLOTS
####################################################################################################################################################################################
colorbar=['blue','green','orange','purple', 'cyan', 'black']
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1]/(2*np.pi))
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1])
##################################################################################################################################################################
fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig1.set_size_inches(20,10)

ax1 = plt.subplot(211)
ax1.set_xlim(f[0], f[-1])
plt.axhline(y=0, color='r', linestyle='-.')
plt.xscale('log');
for i in range(len(bar)):
	[lineol00] = ax1.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][bar[i]], syncStateInphase['tau'][bar[i]],
								tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1))), '-', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateInphase['tau'][bar[i]]))
plt.grid();
plt.ylabel(r'$20\textrm{log}\left|H_k^\textrm{\LARGE OL}(j\gamma)\right|$', rotation=90, fontsize=60, labelpad=30)
ax1.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont);
ax1.tick_params(axis='both', which='major', labelsize=35, pad=1)

ax2 = plt.subplot(212)
ax2.set_xlim(f[0], f[-1])
plt.xscale('log')
plt.axhline(y=-180, color='r', linestyle='-.')
for i in range(len(bar)):
	[lineol00] = ax2.plot(f, PhaseOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][bar[i]], syncStateInphase['tau'][bar[i]],
								tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1), '.', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateInphase['tau'][bar[i]]))
plt.grid();
plt.ylabel(r'$\angle\,H_\textrm{net}^\textrm{\LARGE OL}(j\gamma)$', rotation=90, fontsize=60, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=60, labelpad=-5)
ax2.tick_params(axis='both', which='major', labelsize=35, pad=1)
# ax2.legend(fontsize=15)
if digital == True:
	plt.savefig('plots/digital_bode_plot_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_bode_plot_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_One%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_bode_plot_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

##################################################################################################################################################################
#
# fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2.set_size_inches(20,10)
#
# ax3 = plt.subplot(211)
# ax3.set_xlim(f[0], f[-1])
# plt.xscale('log');
# plt.axhline(y=0, color='r', linestyle='-.')
# if phaseCrossOver:
# 	print('Plot PCF for first tau value! PCF:', phaseCrossOver)
# 	plt.axvline(x=phaseCrossOver, color='y', linestyle='-', linewidth=0.5, alpha=0.5)
# for i in range(len(bar)):
# 	[lineol00] = ax3.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncStateInphase['Omeg'][bar[i]], syncStateInphase['tau'][bar[i]],
# 								tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1))), '-', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateInphase['tau'][bar[i]]))
# plt.grid();
# plt.ylabel(r'$20\textrm{log} \left|H_\textrm{net}^\textrm{\LARGE OL}(j\gamma)\right|$', rotation=90, fontsize=60, labelpad=30)
# ax3.tick_params(axis='both', which='major', labelsize=35, pad=1)
# ax3.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont);
#
# ax4 = plt.subplot(212)
# plt.xscale('log');
# ax4.set_xlim(f[0], f[-1])
# plt.axhline(y=0, color='r', linestyle='-.')
# if phaseCrossOver:
# 	print('Plot PCF for first tau value!')
# 	plt.axvline(x=phaseCrossOver, color='y', linestyle='-', linewidth=0.5, alpha=0.5)
# for i in range(len(bar)):
# 	[lineol00] = ax4.plot(f, PhaseNet(2.0*np.pi*f, syncStateInphase['Omeg'][bar[i]], syncStateInphase['tau'][bar[i]],
# 								tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1), '.', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateInphase['tau'][bar[i]]))
# plt.grid();
# plt.ylabel(r'$\angle\,H_\textrm{net}^\textrm{\LARGE OL}(j\gamma)$', rotation=90, fontsize=60, labelpad=30)
# plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=60, labelpad=-5)
# ax4.tick_params(axis='both', which='major', labelsize=35, pad=1)
# # ax4.legend(fontsize=15)
# if digital == True:
# 	plt.savefig('plots/digital_bode_plot_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plots/digital_bode_plot_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plots/digital_bode_plot_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
# 	if coupfunction(coupfun)=='sin':
# 		plt.savefig('plots/analog_sin_bode_plot_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_sin_bode_plot_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_sin_bode_plot_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfunction(coupfun)=='cos':
# 		plt.savefig('plots/analog_cos_bode_plot_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_cos_bode_plot_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_cos_bode_plot_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

##################################################################################################################################################################

fig3 = plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig3.set_size_inches(20,10)

ax5 = plt.subplot(211)
OmegTauIn = syncStateInphase['tau']*syncStateInphase['Omeg']/(2.0*np.pi)
OmegTauAn = syncStateAnphase['tau']*syncStateAnphase['Omeg']/(2.0*np.pi)
ax5.set_xlim(min(OmegTauIn[0],OmegTauAn[0]), max(OmegTauIn[-1],OmegTauAn[-1]))
[line004] = ax5.plot(OmegTauIn, np.array(pm_vs_tau_singlePLL), 'd', ms=1.5, color='blue', label=r'in-phase (nonlin)');
[line044] = ax5.plot(OmegTauAn, np.array(pm_vs_tau_singlePLL_AP), '.', ms=1.5, color='red', label=r'anti-phase (nonlin)');
[line444] = ax5.plot(OmegTauIn, np.array(pm_vs_tau_singlePLL_Lin), '.', ms=1.5, color='black', label=r'linear');
plt.ylabel(r'phase margin', rotation=90, fontsize=60, labelpad=30)
ax5.tick_params(axis='both', which='major', labelsize=35, pad=1)
plt.grid();
ax5.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)

ax6 = plt.subplot(212)
ax6.set_xlim(min(OmegTauIn[0],OmegTauAn[0]), max(OmegTauIn[-1],OmegTauAn[-1]))
[lineol06] = ax6.plot(np.array(tauOne)*np.array(OmegaOne)/(2.0*np.pi), np.array(GM_One), 'd', ms=1.5, color='blue', label=r'in-phase (nonlin)');
[lineol06] = ax6.plot(np.array(tauOne_AP)*np.array(OmegaOne_AP)/(2.0*np.pi), np.array(GM_One_AP), '.', ms=1.5, color='red', label=r'anti-phase (nonlin)');
[lineol66] = ax6.plot(np.array(tauOne_Lin)*np.array(OmegaOne_Lin)/(2.0*np.pi), np.array(GM_One_Lin), '-', color='black', label=r'nonlinear');
plt.ylabel(r'gain margin', rotation=90, fontsize=60, labelpad=30)
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=60, labelpad=5)
ax6.tick_params(axis='both', which='major', labelsize=35, pad=1)
# ax6.legend()
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_One%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_One%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_One%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

##################################################################################################################################################################
#
# fig4 = plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig4.set_size_inches(20,10)
#
# ax7 = plt.subplot(211)
# OmegTauIn = np.array(OmegaNet)*np.array(tauNet)/(2.0*np.pi)
# OmegTauAn = np.array(OmegaNet_AP)*np.array(tauNet_AP)/(2.0*np.pi)
# ax7.set_xlim(min(OmegTauIn[0],OmegTauAn[0]), max(OmegTauIn[-1],OmegTauAn[-1]))
# [lineNet07]   = ax7.plot(OmegTauIn, np.array(pm_vs_tau_Net), 'd', ms=1.5, label=r'phase margin net (in-phase)', color='blue');
# [lineNet77]   = ax7.plot(OmegTauAn, np.array(pm_vs_tau_Net_AP), '.', ms=1.5, label=r'phase margin net (anti-phase)', color='red');
# #[lineSigmaIn] = ax7.plot(np.asarray(syncStateInphase['Omeg']/(2.0*np.pi))*syncStateInphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateInphase['Omeg'], syncStateInphase['tau'],
# #								tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ReMax'], '-', linewidth=1, color='green', label=r'in-phase')
# plt.grid();
# plt.ylabel(r'phase margin', rotation=90, fontsize=60, labelpad=30)
# ax7.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# ax7.tick_params(axis='both', which='major', labelsize=35, pad=1)
#
# ax8 = plt.subplot(212)
# ax8.set_xlim(min(OmegTauIn[0],OmegTauAn[0]), max(OmegTauIn[-1],OmegTauAn[-1]))
# ax8.tick_params(axis='both', which='major', labelsize=35, pad=1)
# [lineoNet08]  = ax8.plot(OmegTauIn, np.array(GM_Net), '.', ms=1.5, label=r'gain margin net (in-phase)', color='blue');
# [lineoNet88]  = ax8.plot(OmegTauAn, np.array(GM_Net_AP), '.', ms=1.5, label=r'gain margin net (anti-phase)', color='red');
# [lineSigmaIn] = ax8.plot(np.asarray(syncStateInphase['Omeg']/(2.0*np.pi))*syncStateInphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateInphase['Omeg'],
# 					syncStateInphase['tau'], tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ReMax'], '-', linewidth=1, color='green', label=r'in-phase')
# [lineSigmaAn] = ax8.plot(np.asarray(syncStateAnphase['Omeg']/(2.0*np.pi))*syncStateAnphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateAnphase['Omeg'],
# 					syncStateAnphase['tau'], tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state2, expansion, INV)['ReMax'], '-', linewidth=1, color='red', label=r'anti-phase')
# plt.ylabel(r'gain margin', rotation=90, fontsize=60, labelpad=30)
# plt.xlabel(r'$\Omega\tau/2 \pi$', fontsize=60, labelpad=5)
# plt.grid();
#
# if digital == True:
# 	plt.savefig('plots/digital_Margins_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plots/digital_Margins_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plots/digital_Margins_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
# 	if coupfunction(coupfun)=='sin':
# 		plt.savefig('plots/analog_sin_Margins_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_sin_Margins_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_sin_Margins_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	if coupfunction(coupfun)=='cos':
# 		plt.savefig('plots/analog_cos_Margins_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_cos_Margins_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_cos_Margins_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# ##################################################################################################################################################################
#
# fig5 = plt.figure(num=5, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig5.set_size_inches(20,10)
#
# ax9 = plt.subplot(211)
# ax9.set_xlim(f[0], f[-1])
# plt.xscale('log');
# plt.axhline(y=0, color='r', linestyle='-.')
# if phaseCrossOver_AP:
# 	print('Plot PCF for first tau value! PCF:', phaseCrossOver_AP)
# 	plt.axvline(x=phaseCrossOver_AP, color='y', linestyle='-', linewidth=0.5, alpha=0.5)
# for i in range(len(bar)):
# 	[lineol00] = ax9.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncStateAnphase['Omeg'][bar[i]], syncStateAnphase['tau'][bar[i]],
# 								tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2))), '-', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateAnphase['tau'][bar[i]]))
# plt.grid();
# plt.ylabel(r'$20\textrm{log} \left|H_\textrm{net}^\textrm{\LARGE OL}(j\gamma)\right|$', rotation=90, fontsize=60, labelpad=30)
# ax9.tick_params(axis='both', which='major', labelsize=35, pad=1)
# ax9.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont);
#
# ax10 = plt.subplot(212)
# plt.xscale('log');
# ax10.set_xlim(f[0], f[-1])
# plt.axhline(y=0, color='r', linestyle='-.')
# if phaseCrossOver_AP:
# 	print('Plot PCF for first tau value!')
# 	plt.axvline(x=phaseCrossOver_AP, color='y', linestyle='-', linewidth=0.5, alpha=0.5)
# for i in range(len(bar)):
# 	[lineol00] = ax10.plot(f, PhaseNet(2.0*np.pi*f, syncStateAnphase['Omeg'][bar[i]], syncStateAnphase['tau'][bar[i]],
# 								tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state2), '.', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateAnphase['tau'][bar[i]]))
# plt.grid();
# plt.ylabel(r'$\angle\,H_\textrm{net}^\textrm{\LARGE OL}(j\gamma)$', rotation=90, fontsize=60, labelpad=30)
# plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=60, labelpad=-5)
# ax10.tick_params(axis='both', which='major', labelsize=35, pad=1)
# # ax10.legend(fontsize=15)
# if digital == True:
# 	plt.savefig('plots/digital_AP_bode_plot_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plots/digital_AP_bode_plot_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plots/digital_AP_bode_plot_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
# 	if coupfunction(coupfun)=='sin':
# 		plt.savefig('plots/analog_AP_sin_bode_plot_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_AP_sin_bode_plot_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_AP_sin_bode_plot_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfunction(coupfun)=='cos':
# 		plt.savefig('plots/analog_AP_cos_bode_plot_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_AP_cos_bode_plot_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_AP_cos_bode_plot_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# ################################################################################################################################################################## sigma & gain margin vs Omega tau
#
# fig6 = plt.figure(num=6, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig6.set_size_inches(20,10)
#
# ax11 = plt.subplot(211)
# OmegTauIn = np.array(OmegaNet)*np.array(tauNet)/(2.0*np.pi)
# OmegTauAn = np.array(OmegaNet_AP)*np.array(tauNet_AP)/(2.0*np.pi)
# ax11.set_xlim(min(OmegTauIn[0],OmegTauAn[0]), max(OmegTauIn[-1],OmegTauAn[-1]))
# [lineSigmaIn] = ax11.plot(np.asarray(syncStateInphase['Omeg']/(2.0*np.pi))*syncStateInphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateInphase['Omeg'],
# 					syncStateInphase['tau'], tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ReMax'], '-', linewidth=1.5, color='blue', label=r'in-phase')
# [lineSigmaAn] = ax11.plot(np.asarray(syncStateAnphase['Omeg']/(2.0*np.pi))*syncStateAnphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateAnphase['Omeg'],
# 					syncStateAnphase['tau'], tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state2, expansion, INV)['ReMax'], '-', linewidth=1.5, color='red', label=r'anti-phase')
# plt.grid();
# plt.ylabel(r'$\frac{2\pi\sigma}{\omega}$', rotation=90, fontsize=60, labelpad=30)
# ax11.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# ax11.tick_params(axis='both', which='major', labelsize=35, pad=1)
#
# ax12 = plt.subplot(212)
# ax12.set_xlim(min(OmegTauIn[0],OmegTauAn[0]), max(OmegTauIn[-1],OmegTauAn[-1]))
# ax12.tick_params(axis='both', which='major', labelsize=35, pad=1)
# #[lineoNet08]  = ax12.plot(OmegTauIn, np.array(GM_Net), '.', ms=1.5, label=r'gain margin net (in-phase)', color='blue');
# #[lineoNet88]  = ax12.plot(OmegTauAn, np.array(GM_Net_AP), '.', ms=1.5, label=r'gain margin net (anti-phase)', color='red');
# [lineoNet08]  = ax12.plot(OmegTauIn, np.array(GM_Net), '-', linewidth=1.5, label=r'gain margin net (in-phase)', color='blue');
# [lineoNet09]  = ax12.plot(x_test, np.array(GM_Net_test), '-', linewidth=1.5, label=r'gain margin net (test)', color='grey');
# [lineoNet88]  = ax12.plot(OmegTauAn, np.array(GM_Net_AP), '-', linewidth=1.5, label=r'gain margin net (anti-phase)', color='red');
#
# plt.ylabel(r'gain margin', rotation=90, fontsize=60, labelpad=30)
# plt.xlabel(r'$\Omega\tau/2 \pi$', fontsize=60, labelpad=5)
# plt.grid();
#
# if digital == True:
# 	plt.savefig('plots/digital_sigma_GM_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plots/digital_sigma_GM_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plots/digital_sigma_GM_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
# 	if coupfunction(coupfun)=='sin':
# 		plt.savefig('plots/analog_sin_sigma_GM_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_sin_sigma_GM_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_sin_sigma_GM_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfunction(coupfun)=='cos':
# 		plt.savefig('plots/analog_cos_sigma_GM_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_cos_sigma_GM_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_cos_sigma_GM_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# ################################################################################################################################################################## sigma & gain margin vs Omega tau
#
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
# #ax14.set_ylim(-0.1+np.min(np.array([freqMaxGainIn, freqMaxGainAn])), 1.05*np.max(np.array([freqMaxGainIn, freqMaxGainAn]))) #maxGainIn, maxGainAn,
# plt.ylabel(r'$f^\textrm{pert}$', rotation=90, fontsize=60, labelpad=30)
# plt.xlabel(r'$\Omega\tau/2 \pi$', fontsize=60, labelpad=5)
# plt.grid();
#
# if digital == True:
# 	plt.savefig('plots/digital_gamma_freqMaxG_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plots/digital_gamma_freqMaxG_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('plots/digital_gamma_freqMaxG_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
# 	if coupfunction(coupfun)=='sin':
# 		plt.savefig('plots/analog_sin_gamma_freqMaxG_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_sin_gamma_freqMaxG_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_sin_gamma_freqMaxG_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfunction(coupfun)=='cos':
# 		plt.savefig('plots/analog_cos_gamma_freqMaxG_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_cos_gamma_freqMaxG_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('plots/analog_cos_gamma_freqMaxG_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

plt.show();

# #*******************************************************************************
# fig         = plt.figure(figsize=(figwidth,figheight))
# ax          = fig.add_subplot(111)
#
#
# # adjust the subplots region to leave some space for the sliders and buttons
# # fig.subplots_adjust(left=0.12, bottom=0.25)
# # plot grid, labels, define intial values
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$',fontsize=60, labelpad=-5)
# plt.ylabel(r'$\frac{\Omega}{\omega}$',rotation=0, fontsize=85, labelpad=30)
# ax.tick_params(axis='both', which='major', labelsize=35, pad=1)
# ax.set_xlim(0.0, 5.0)
# ax.set_ylim(0.2, 1.15)
# # plt.xlim(-0.1, max(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']))
# # # draw the initial plot
# # # the 'lineXXX' variables are used for modifying the lines later
#
#
# # #*************************************************************************************************************************************************************************
# [NonlineOmegStabIn] = ax.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 						 globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, INV)['Omeg']/np.asarray(w),
# 						 '-',lineWidth=4, color='blue',  label=r'nonlinear in-phase (mutual)')
#
#
# [NonlineOmegStabAnti] = ax.plot(np.asarray(w/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'],
# 						 globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, INV)['Omeg']/np.asarray(w),
# 						 ':',lineWidth=4, color='blue',  label=r'nonlinear antiphase (mutual)')
#
#
# [lineOmegStabIn] = ax.plot(np.asarray(w/(2.0*np.pi))*LinearglobalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 						 LinearglobalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, INV)['Omeg']/np.asarray(w),
# 						 '--',lineWidth=4, color='red',  label=r'linear (mutual)')
# wref=np.empty(len(xrangetau(maxtau)));wref.fill(wR)
# # draw the initial plot
# [lineBeta] = ax.plot(np.asarray(w/(2.0*np.pi))*xrangetau(maxtau), wref/w,
# 	'-.',linewidth=4, markersize=0.75, color='orange', label=r'entrainment case')
# fig.set_size_inches(20,10)
# ax.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# # plt.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
# plt.savefig('Plots/Nonlinear_vs_linear%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# plt.savefig('Plots/Nonlinear_vs_linear%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
