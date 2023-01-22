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
Kvco    = 2.0*np.pi*(0.370);                                                      # Sensitivity of VCO
AkPD    = 1.0                                                                   # amplitude of the PD -- the factor of 2 here is related to the IMS 2020 paper, AkPD as defined in that paper already includes the 0.5 that we later account for when calculating K
GkLF    = 1.0
Gvga    = 1.0
Ga1     = 1.0                                                                   # Gain of the first adder
order   = 1.0                                                                   # the order of the Loop Filter
tauf    = 0.0                                                                   # tauf = sum of all processing delays in the feedback
tauc    = 1.0/(2.0*np.pi*0.055);                                               # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
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
df        = 0.00125#0.0005 #0.00025;
fmin      = -4
fmax      =  1
f         = np.logspace(fmin, fmax, num=int((tauf+maxp)/df), endpoint=True, base=10.0)

tauvalues2plot = [0*w/(2*np.pi), 0.025*w/(2*np.pi), 0.05*w/(2*np.pi), 0.1*w/(2*np.pi), 0.2*w/(2*np.pi), 0.3*w/(2*np.pi)]
#tauvalues2plot = [0*w/(2*np.pi), 0.25*w/(2*np.pi), 0.5*w/(2*np.pi), 0.75*w/(2*np.pi), 1.0*w/(2*np.pi), 1.25*w/(2*np.pi)]
####################################################################################################################################################################################

GM_Net_Lin=[];  GM_Net_test_Lin=[]; x_test_Lin=[];
gcf_vs_tau_Net_Lin=[]
GM_One_Lin=[];
pm_vs_tau_singlePLL_Lin=[];
tauOne_Lin=[];OmegaOne_Lin=[];
tauNet_Lin=[];OmegaNet_Lin=[];
phaseCrossOver_Lin=[];
freqMaxGainIn=[]; freqMaxGainAn=[]; maxGainIn=[]; maxGainAn=[];
pm_vs_tau_Net_Lin =[]
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

	######################################################################################################## LINEAR
	tempHopenNet = HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1)

	gcf_vs_tau_Net_Lin.append(f[np.where(20.0*log10((abs(tempHopenNet))) <= 0.0001)][-1] )


	pm_vs_tau_Net_Lin.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_tau_Net_Lin[-1], syncStateInphase['Omeg'][j],
		syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear', sync_state1))
	PMLin = f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1))) <= 0.0001)][0]

	pm_vs_tau_singlePLL_Lin.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*PMLin, syncStateInphase['Omeg'][j],
		syncStateInphase['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1))


	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1)) <=0.7).any():

		GMLin=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,	syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
				tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1)) <=0.7)][0]
		# print(GM)
		GM_One_Lin.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GMLin, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1))

		tauOne_Lin.append(syncStateInphase['tau'][j])
		OmegaOne_Lin.append(syncStateInphase['Omeg'][j])



	index1 = 0; index2 = 0; index3 = 0; index4 = 0; index5 = 0;					# reset for safety
	phaseNet_Lin = PhaseNet(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1)
	index1   = np.where(np.abs(phaseNet_Lin) > 1E-10)[0][0]
                     # find first phase value that deviates more than the treshold from zero
	#print('index1: ', index1)
	#print('phaseNet[index1:]:', phaseNet[index1:])
	if np.any(phaseNet_Lin[index1:]<0):												# catch the zero delay case where the phase never goes below zero
		if phaseNet_Lin[index1] > 0:
			index2 = np.where(phaseNet_Lin[index1:] < 0)[0][0]                      # where is the phase below zero the first time from index1 to the end
			GMNet_Lin  = f[index1+index2]
	                                # save the frequency at this point
			# find next encirclement
			if syncStateInphase['tau'][j] > 0.5:
				index3 = np.where(phaseNet_Lin[(index1+index2):] > 0)[0][0]
				index4 = np.where(phaseNet_Lin[(index1+index2+index3):] < 0)[0][0]
				GMNet2_Lin = f[index1+index2+index3+index4]
		elif phaseNet_Lin[index1] < 0:
			index2 = np.where(phaseNet_Lin[index1:] > 0)[0][0]
			if len(np.where(phaseNet_Lin[(index1+index2):] < 0)[0])!=0:
				index3 = np.where(phaseNet_Lin[(index1+index2):] < 0)[0][0]

				GMNet_Lin  = f[index1+index2+index3]
			else:
				GMNet_Lin  = f[index1+index2]
			# find next encirclement
			if syncStateInphase['tau'][j] > 0.5:
				index4 = np.where(phaseNet_Lin[(index1+index2+index3):] > 0)[0][0]
				#print('np.where(phaseNet[(index1+index2+index3):] > 0)[0][0]:', np.where(phaseNet[(index1+index2+index3):] > 0)[0][0])
				index5 = np.where(phaseNet_Lin[(index1+index2+index3+index4):] < 0)[0][0]
				#print('index5 = np.where(phaseNet[(index1+index2+index3+index4):] < 0)[0][0]:', index5 = np.where(phaseNet[(index1+index2+index3+index4):] < 0)[0][0])
				GMNet2_Lin = f[index1+index2+index3+index4+index5]
		else:
			print('Error! Debug.'); sys.exit();
	elif np.min(phaseNet_Lin[index1:]) < treshold_detect_zero_delay:				# this is the case for zero delay
		index2 = np.where(phaseNet_Lin[index1:] < treshold_detect_zero_delay)[0][0]
		GMNet_Lin  = f[index1+index2]
		#GMNet2 = f[0]
	else:
		print('Phase associated to zero delay does not go below threshold, increase frequency range of Bode plot or change treshold!');
		print('Smallest value for phase of zero delay case:', np.min(phaseNet[index1:]))
		sys.exit();

	if ( np.abs(syncStateInphase['tau'][j]-tauvalues2plot[3]) <= 1.1*treshold_find_tau_value_inphase and len(phaseCrossOver_Lin)==0 ):
		print('Save phaseCrossOver for one value of the delay! PCF:', GMNet_Lin)
		phaseCrossOver_Lin.append(GMNet_Lin);

	GM_Net_Lin.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet_Lin, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1))
	if syncStateInphase['tau'][j] > 0.5:
		GM_Net_test_Lin.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet2_Lin, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear', sync_state1))
		x_test_Lin.append(syncStateInphase['tau'][j]*syncStateInphase['Omeg'][j]/(2*np.pi))
	tauNet_Lin.append(syncStateInphase['tau'][j])
	OmegaNet_Lin.append(syncStateInphase['Omeg'][j])





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

##################################################################################################################################################################

fig4 = plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig4.set_size_inches(20,10)

ax7 = plt.subplot(211)
OmegTau_Lin = np.array(OmegaNet_Lin)*np.array(tauNet_Lin)/(2.0*np.pi)

ax7.set_xlim(OmegTau_Lin[0], OmegTau_Lin[-1])
[lineNet07]   = ax7.plot(OmegTau_Lin, np.array(pm_vs_tau_Net_Lin), 'd', ms=1.5, label=r'phase margin net (linear)', color='blue');
#[lineSigmaIn] = ax7.plot(np.asarray(syncStateInphase['Omeg']/(2.0*np.pi))*syncStateInphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateInphase['Omeg'], syncStateInphase['tau'],
#								tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ReMax'], '-', linewidth=1, color='green', label=r'in-phase')
plt.grid();
plt.ylabel(r'phase margin', rotation=90, fontsize=60, labelpad=30)
ax7.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
ax7.tick_params(axis='both', which='major', labelsize=35, pad=1)

ax8 = plt.subplot(212)
ax8.set_xlim(OmegTau_Lin[0], OmegTau_Lin[-1])
ax8.tick_params(axis='both', which='major', labelsize=35, pad=1)
[lineoNet08]  = ax8.plot(OmegTau_Lin, np.array(GM_Net_Lin), '.', ms=1.5, label=r'gain margin net (linear)', color='blue');

# [lineSigmaIn] = ax8.plot(np.asarray(syncStateInphase['Omeg']/(2.0*np.pi))*syncStateInphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateInphase['Omeg'],
# 					syncStateInphase['tau'], tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ReMax'], '-', linewidth=1, color='green', label=r'in-phase')
# [lineSigmaAn] = ax8.plot(np.asarray(syncStateAnphase['Omeg']/(2.0*np.pi))*syncStateAnphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateAnphase['Omeg'],
# 					syncStateAnphase['tau'], tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state2, expansion, INV)['ReMax'], '-', linewidth=1, color='red', label=r'anti-phase')
plt.ylabel(r'gain margin', rotation=90, fontsize=60, labelpad=30)
plt.xlabel(r'$\Omega\tau/2 \pi$', fontsize=60, labelpad=5)
plt.grid();

if digital == True:
	plt.savefig('plots/digital_Margins_Net_lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_Net_lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_Net_lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_Net_lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_Net_lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_Net_lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_Net_lin%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_Net_lin%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_Net_lin%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

##################################################################################################################################################################

################################################################################################################################################################## sigma & gain margin vs Omega tau


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
